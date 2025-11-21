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
    多模态肺癌复发预测 + 结节长短径回归 + 跨模态对齐对比损失
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
                 align_dim=128,          # 对比学习共享维度
                 contrastive_tau=0.07
                 ):  # 温度系数
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
        feature_len = fusion_dim + 384 # 外部CLS token + 人口学
        self.recurrence_classifier = nn.Sequential(
            nn.Linear(feature_len, 384),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(384, num_classes_recurrence),
            # nn.Linear(feature_len, num_classes_recurrence),
        )

        self.axis_regressor = nn.Sequential(
            nn.Linear(feature_len, 256),
            nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 2)
        )

        # ========= 新增：对比对齐模块 =========
        self.fused_proj   = nn.Linear(fusion_dim, align_dim)
        self.text2_proj   = nn.Linear(fusion_dim, align_dim)
        self.contrastive_tau = contrastive_tau
        # =====================================

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
            k = 10 # 仅解冻 encoder.layer.10 和 layer.11
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
        x = F.normalize(self.fused_proj(x), dim=-1)   # [B, D']
        y = F.normalize(self.text2_proj(y), dim=-1)   # [B, D']

        logits = x @ y.T / self.contrastive_tau       # [B, B]
        labels = torch.arange(B, device=x.device)

        loss_xy = F.cross_entropy(logits, labels)
        loss_yx = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss_xy + loss_yx)

    def normalize_features(self, features):
        return F.layer_norm(features, features.shape[-1:])

    def forward(self, image, text_inputs, demographic):
        """
        text_inputs: dict, 必须包含 ['input_ids'] 和 ['attention_mask']，每个为tokenizer后的字典
        """
        # 1) 图像token序列（不取CLS，全部patch token）
        _, img_tokens, initial_image_features, _ = self.image_encoder(image)  # [B, N_patch, image_dim]
        num_cls = self.image_encoder.num_cls
        # img_tokens = self.image_proj(img_tokens)   # [B, N_patch, fusion_dim]
        img_tokens = img_tokens[:, num_cls:]
        img_tokens = self.image_proj(img_tokens)
        # 2) 文本token序列（不取[CLS]，全部token）
        text_tokens, _, _ = self.text_encoder(
            input_ids=text_inputs['input_ids'],
            attention_mask=text_inputs['attention_mask'])  # [B, N_text, text_feature_dim]
        text_tokens = self.text_proj(text_tokens)          # [B, N_text, fusion_dim]

        B, N_img, D_img = img_tokens.shape
        _, N_text, D_text = text_tokens.shape

        # 5) 人口学特征
        demo_feat = self.demographic_projection(demographic)  # [B, 128]
        # demo_feat = demo_feat.unsqueeze(1)  # [B, 1, fusion_dim]
        # demo_feat = self.normalize_features(demo_feat)
        # 3) 加CLS token拼接
        fusion_cls = self.fusion_cls_token.expand(B, -1, -1)  # [B, 1, fusion_dim]


        # 拼接所有token
        fusion_input = torch.cat([fusion_cls, img_tokens, text_tokens], dim=1)
        # fusion_input = self.fusion_norm(fusion_input)

        # 4) 融合Transformer
        fusion_out = self.fusion_transformer(fusion_input)     # [B, 1+N_patch+N_text, fusion_dim]
        pooled_feat = fusion_out.mean(dim=1)
        fusion_cls = fusion_out[:,0]
        fusion_feature = torch.cat([fusion_cls, demo_feat],dim=1) # 取CLS


        recurrence_logits = self.recurrence_classifier(fusion_feature)
        axis_preds = self.axis_regressor(fusion_feature)

        contrastive_loss = 0  # 如需可调用self._contrastive_loss等

        return recurrence_logits, axis_preds, contrastive_loss, fusion_feature