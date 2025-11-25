import torch
import torch.nn as nn
import torch.nn.functional as F
from .TextEncoder import TextEncoder
from .EntitySeqDecoder import EntitySeqDecoder  # 新增


class FusionTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, num_layers=2, dropout=0.5):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.encoder(x)


class MultimodalMultitaskModel(nn.Module):
    """
    多模态多任务模型 + 实体序列生成 decoder
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
                 align_dim=128,
                 contrastive_tau=0.07,
                 entity_vocab_size=1000,
                 # 新增: decoder 参数
                 use_entity_decoder=False,
                 decoder_num_layers=3,
                 decoder_num_heads=6,
                 decoder_max_seq_len=10):
        super().__init__()

        self.use_entity_decoder = use_entity_decoder

        self.image_encoder = vit_3d_model
        self.text_encoder = TextEncoder(
            bert_model_name=bert_model_name,
            output_dim=text_feature_dim,
            dropout=dropout
        )

        self.modality_embeddings = nn.Embedding(3, fusion_dim)

        self.image_proj = nn.Linear(image_dim, fusion_dim) if image_dim != fusion_dim else nn.Identity()
        self.text_proj = nn.Linear(text_feature_dim, fusion_dim)

        self.fusion_cls_token = nn.Parameter(torch.randn(1, 1, fusion_dim))
        self.fusion_transformer = FusionTransformer(
            embed_dim=fusion_dim,
            num_heads=fusion_transformer_heads,
            num_layers=fusion_transformer_layers,
            dropout=dropout
        )
        self.fusion_norm = nn.LayerNorm(fusion_dim)

        self.demographic_projection = nn.Sequential(
            nn.Linear(demographic_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 384),
        )

        feature_len = fusion_dim + 384
        self.recurrence_classifier = nn.Sequential(
            nn.Linear(feature_len, 384),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(384, num_classes_recurrence),
        )

        self.axis_regressor = nn.Sequential(
            nn.Linear(feature_len, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        self.fused_proj = nn.Linear(fusion_dim, align_dim)
        self.text2_proj = nn.Linear(fusion_dim, align_dim)
        self.contrastive_tau = contrastive_tau

        # ============== 实体序列 decoder ==============
        self.use_entity_decoder = use_entity_decoder

        if use_entity_decoder:
            num_entities = entity_vocab_size  # 这里的 entity_vocab_size 就是 masker.vocab_size
            self.entity_decoder = EntitySeqDecoder(
                d_model=fusion_dim,
                num_layers=decoder_num_layers,
                num_heads=decoder_num_heads,
                dim_feedforward=fusion_dim * 4,
                num_entities=num_entities,
                dropout=dropout,
                max_seq_len=decoder_max_seq_len
            )
            print(f"✅ 实体序列 decoder 已启用, num_entities={num_entities}")

        print(f"✅ MultimodalMultitaskModel 初始化完成")

    def freeze_encoders(self,
                        freeze_image_encoder: bool = False,
                        freeze_text_encoder: bool = False,
                        bert_unfrozen_last_k: int = 2):
        if freeze_image_encoder:
            for name, param in self.image_encoder.named_parameters():
                param.requires_grad = False
            print("✅ 已冻结 ViT3D 图像编码器全部参数")

        if freeze_text_encoder:
            for param in self.text_encoder.bert.parameters():
                param.requires_grad = False

            num_layers = self.text_encoder.bert.config.num_hidden_layers
            k = max(1, bert_unfrozen_last_k)
            start = max(0, num_layers - k)

            for name, param in self.text_encoder.bert.named_parameters():
                if name.startswith("encoder.layer."):
                    layer_num = int(name.split('.')[2])
                    if layer_num >= start:
                        param.requires_grad = True

            print(f"✅ BERT 微调策略：只微调 encoder.layer.{start} ~ encoder.layer.{num_layers - 1}")

    def _contrastive_loss(self, x, y):
        if x is None or y is None:
            dev = x.device if x is not None else y.device
            return torch.tensor(0.0, device=dev)

        B = x.size(0)
        x = F.normalize(self.fused_proj(x), dim=-1)
        y = F.normalize(self.text2_proj(y), dim=-1)

        logits = x @ y.T / self.contrastive_tau
        labels = torch.arange(B, device=x.device)

        loss_xy = F.cross_entropy(logits, labels)
        loss_yx = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss_xy + loss_yx)

    def normalize_features(self, features):
        return F.layer_norm(features, features.shape[-1:])

    def forward(self,
                image,
                text_inputs,
                demographic,
                masked_text_inputs=None,
                target_entity_ids=None):
        """
        参数:
            image: [B, C, H, W, D]
            text_inputs: dict
            demographic: [B, demographic_dim]
            masked_text_inputs: dict (遮挡文本编码)
            target_entity_ids: [B, K] 被 mask 实体 ID 序列
        """
        # 1) 图像编码
        _, img_tokens, _ = self.image_encoder(image)
        num_cls = self.image_encoder.num_cls
        img_tokens = img_tokens[:, num_cls:]
        img_tokens = self.image_proj(img_tokens)  # [B, N_img, fusion_dim]

        # 2) 文本编码(原文本, 用于主任务)
        text_seq, text_cls, _ = self.text_encoder(
            input_ids=text_inputs['input_ids'],
            attention_mask=text_inputs['attention_mask'],
            token_type_ids=text_inputs.get('token_type_ids', None)
        )
        text_tokens = self.text_proj(text_seq)  # [B, L, fusion_dim]

        B, N_img, _ = img_tokens.shape
        _, N_text, _ = text_tokens.shape

        # 3) 人口学特征
        demo_feat = self.demographic_projection(demographic)  # [B, 384]

        # 4) 融合 Transformer (主任务用)
        fusion_cls = self.fusion_cls_token.expand(B, -1, -1)
        fusion_input = torch.cat([fusion_cls, img_tokens, text_tokens], dim=1)
        fusion_out = self.fusion_transformer(fusion_input)
        fusion_cls = fusion_out[:, 0]
        fusion_feature = torch.cat([fusion_cls, demo_feat], dim=1)

        # 5) 主任务
        recurrence_logits = self.recurrence_classifier(fusion_feature)
        axis_preds = self.axis_regressor(fusion_feature)

        contrastive_loss = torch.tensor(0.0, device=image.device)

        # 6) 实体序列生成 decoder
        decoder_loss = torch.tensor(0.0, device=image.device)
        generated_entities = None

        if self.use_entity_decoder and masked_text_inputs is not None and target_entity_ids is not None:
            # 编码遮挡文本
            masked_seq, masked_cls, _ = self.text_encoder(
                input_ids=masked_text_inputs['input_ids'],
                attention_mask=masked_text_inputs['attention_mask'],
                token_type_ids=masked_text_inputs.get('token_type_ids', None)
            )
            masked_tokens = self.text_proj(masked_seq)  # [B, L_mask, fusion_dim]

            # 构建 decoder 的 encoder context: 图像 + 遮挡文本
            decoder_context = torch.cat([img_tokens, masked_tokens], dim=1)  # [B, N_img+L_mask, D]

            # decoder forward
            logits, loss = self.entity_decoder(
                encoder_context=decoder_context,
                target_entity_ids=target_entity_ids,
                context_mask=None
            )

            decoder_loss = loss if loss is not None else torch.tensor(0.0, device=image.device)

            # 推理时可拿 logits 解码出实体序列(暂不实现, 训练时不需要)

        return {
            'recurrence_logits': recurrence_logits,
            'axis_preds': axis_preds,
            'contrastive_loss': contrastive_loss,
            'decoder_loss': decoder_loss,  # 新增
            'fusion_feature': fusion_feature
        }