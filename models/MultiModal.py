import torch
import torch.nn as nn
import torch.nn.functional as F
from .TextEncoder import TextEncoder


# ------------------ 融合用的Transformer --------------------
class FusionTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, num_layers=2, dropout=0.5):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.encoder(x)


# -----------------------------------------------------------
#                    MultimodalMultitaskModel
# -----------------------------------------------------------
class MultimodalMultitaskModel(nn.Module):
    """
    多模态肺癌复发预测 + 结节长短径回归 + 跨模态对齐对比损失 + 遮挡预测
    ViT 3D和BERT最后的隐藏层输出拼接，加入CLS token，输入外部Transformer，CLS token作融合特征。
    """

    def __init__(self,
                 vit_3d_model,
                 bert_model_name="hfl/chinese-roberta-wwm-ext",
                 image_dim=384,
                 text_feature_dim=384,
                 demographic_dim=3,
                 fusion_dim=384,
                 num_classes_recurrence=2,
                 dropout=0.5,
                 fusion_transformer_heads=6,
                 fusion_transformer_layers=6,
                 align_dim=128,  # 对比学习共享维度
                 contrastive_tau=0.07,  # 温度系数
                 entity_vocab_size=1000  # 医疗实体词表大小
                 ):
        super().__init__()

        # --- 编码器 --------------------------------------------------
        self.image_encoder = vit_3d_model
        self.text_encoder = TextEncoder(
            bert_model_name=bert_model_name,
            output_dim=text_feature_dim,
            dropout=0.5)

        self.modality_embeddings = nn.Embedding(3, fusion_dim)

        # --- 投影到统一维度 -----------------------------------------
        self.image_proj = nn.Linear(image_dim, fusion_dim) if image_dim != fusion_dim else nn.Identity()
        self.text_proj = nn.Linear(text_feature_dim, fusion_dim)

        # --- 融合用CLS token和Transformer ---------------------------
        self.fusion_cls_token = nn.Parameter(torch.randn(1, 1, fusion_dim))
        self.fusion_transformer = FusionTransformer(
            embed_dim=fusion_dim, num_heads=fusion_transformer_heads, num_layers=fusion_transformer_layers, dropout=0.5
        )

        self.fusion_norm = nn.LayerNorm(fusion_dim)

        # --- 人口学投影 ---------------------------------------------
        self.demographic_projection = nn.Sequential(
            nn.Linear(demographic_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 384),
        )

        # --- 下游任务头 ---------------------------------------------
        feature_len = fusion_dim + 384  # 外部CLS token + 人口学
        self.recurrence_classifier = nn.Sequential(
            nn.Linear(feature_len, 384),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(384, num_classes_recurrence),
        )

        self.axis_regressor = nn.Sequential(
            nn.Linear(feature_len, 256),
            nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 2)
        )

        # ========= 对比对齐模块 =========
        self.fused_proj = nn.Linear(fusion_dim, align_dim)
        self.text2_proj = nn.Linear(fusion_dim, align_dim)
        self.contrastive_tau = contrastive_tau

        # ========= 遮挡预测模块 =========
        self.entity_vocab_size = entity_vocab_size

        # 遮挡预测分类头：基于文本特征预测被遮挡的实体
        self.masked_entity_predictor = nn.Sequential(
            nn.Linear(text_feature_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, entity_vocab_size)
        )

        # 遮挡预测损失函数
        self.masked_entity_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

        print(f"✅ 遮挡预测模块已初始化，实体词表大小: {entity_vocab_size}")

    def freeze_encoders(self, freeze_image_encoder=False, freeze_text_encoder=False):
        """冻结编码器主干，但保持提示向量可训练"""
        # 冻结ViT3D编码器
        if freeze_image_encoder:
            for name, param in self.image_encoder.named_parameters():
                if 'class_prompts' not in name and 'cls_token' not in name:
                    param.requires_grad = False
                    print(f"已冻结: {name}")
                else:
                    print(f"保持可训练: {name}")

        # 冻结BERT编码器
        if freeze_text_encoder:
            k = 10  # 仅解冻 encoder.layer.10 和 layer.11
            for name, param in self.text_encoder.bert.named_parameters():
                if name.startswith("encoder.layer."):
                    layer_num = int(name.split('.')[2])
                    if layer_num < k:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                else:
                    param.requires_grad = False
            print(f"[BERT冻结策略] 冻结 encoder.layer.0 ~ layer.{k - 1}, 解冻 layer.{k} ~ layer.11")

    # ---------------------------------------------------------------
    # 对比损失 (InfoNCE，双向)
    # ---------------------------------------------------------------
    def _contrastive_loss(self, x, y):
        """
        x: [B, D]  fused_features
        y: [B, D]  enhanced_text_features2
        """
        B = x.size(0)
        x = F.normalize(self.fused_proj(x), dim=-1)  # [B, D']
        y = F.normalize(self.text2_proj(y), dim=-1)  # [B, D']

        logits = x @ y.T / self.contrastive_tau  # [B, B]
        labels = torch.arange(B, device=x.device)

        loss_xy = F.cross_entropy(logits, labels)
        loss_yx = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss_xy + loss_yx)

    def normalize_features(self, features):
        return F.layer_norm(features, features.shape[-1:])

    def forward(self, image, text_inputs, demographic,
                masked_text_inputs=None, target_entity_ids=None):
        """
        前向传播

        参数:
            image: 图像输入 [B, C, H, W, D]
            text_inputs: dict, 必须包含 ['input_ids'] 和 ['attention_mask']
            demographic: 人口学特征 [B, demographic_dim]
            masked_text_inputs: dict, 遮挡后的文本输入（可选）
            target_entity_ids: Tensor [B, max_masked], 被遮挡的实体ID（可选）

        返回:
            dict: 包含所有任务的输出和损失
        """
        # ========== 1) 图像编码 ==========
        _, img_tokens, initial_image_features = self.image_encoder(image)
        num_cls = self.image_encoder.num_cls
        img_tokens = img_tokens[:, num_cls:]
        img_tokens = self.image_proj(img_tokens)  # [B, N_patch, fusion_dim]

        # ========== 2) 文本编码（原始文本） ==========
        text_tokens, text_cls, _ = self.text_encoder(
            input_ids=text_inputs['input_ids'],
            attention_mask=text_inputs['attention_mask'])  # [B, N_text, text_feature_dim]
        text_tokens = self.text_proj(text_tokens)  # [B, N_text, fusion_dim]

        B, N_img, D_img = img_tokens.shape
        _, N_text, D_text = text_tokens.shape

        # ========== 3) 人口学特征 ==========
        demo_feat = self.demographic_projection(demographic)  # [B, 384]

        # ========== 4) 融合Transformer ==========
        fusion_cls = self.fusion_cls_token.expand(B, -1, -1)  # [B, 1, fusion_dim]
        fusion_input = torch.cat([fusion_cls, img_tokens, text_tokens], dim=1)

        fusion_out = self.fusion_transformer(fusion_input)  # [B, 1+N_patch+N_text, fusion_dim]
        fusion_cls = fusion_out[:, 0]  # [B, fusion_dim]
        fusion_feature = torch.cat([fusion_cls, demo_feat], dim=1)  # [B, fusion_dim + 384]

        # ========== 5) 主任务预测 ==========
        recurrence_logits = self.recurrence_classifier(fusion_feature)
        axis_preds = self.axis_regressor(fusion_feature)

        contrastive_loss = 0  # 如需可调用self._contrastive_loss等

        # ========== 6) 遮挡预测任务 ==========
        masked_entity_loss = torch.tensor(0.0, device=image.device)
        predicted_entities = None

        if masked_text_inputs is not None and target_entity_ids is not None:
            # 6.1) 编码遮挡后的文本
            masked_text_tokens, masked_text_cls, _ = self.text_encoder(
                input_ids=masked_text_inputs['input_ids'],
                attention_mask=masked_text_inputs['attention_mask']
            )  # [B, N_text, text_feature_dim]

            # 6.2) 使用[CLS] token进行实体预测
            masked_logits = self.masked_entity_predictor(masked_text_cls)  # [B, entity_vocab_size]

            # 6.3) 预测结果（取第一个被遮挡的实体）
            predicted_entities = torch.argmax(masked_logits, dim=-1)  # [B]

            # 6.4) 计算损失（只计算第一个被遮挡实体的损失）
            target_ids = target_entity_ids[:, 0]  # [B]

            # 只对有效目标（!= -1）计算损失
            valid_mask = (target_ids != -1)
            if valid_mask.sum() > 0:
                masked_entity_loss = self.masked_entity_loss_fn(
                    masked_logits[valid_mask],
                    target_ids[valid_mask]
                )

        # ========== 7) 返回所有输出 ==========
        return {
            'recurrence_logits': recurrence_logits,  # [B, num_classes]
            'axis_preds': axis_preds,  # [B, 2]
            'contrastive_loss': contrastive_loss,  # 标量
            'masked_entity_loss': masked_entity_loss,  # 标量
            'predicted_entities': predicted_entities,  # [B] 或 None
            'fusion_feature': fusion_feature  # [B, feature_len]
        }