import torch
import torch.nn as nn
import torch.nn.functional as F

from .TextEncoder import TextEncoder
from .EntitySeqDecoder import EntitySeqDecoder


class FusionTransformer(nn.Module):
    """
    原有的 fusion token + TransformerEncoder 方式
    """
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


class CrossModalFusion(nn.Module):
    """
    传统多模态融合方式（不使用单独 fusion token，但仍有注意力）：
    - 图像自注意力
    - 文本自注意力
    - 图 -> 文 本 cross-attention
    - 文 -> 图 像 cross-attention
    最后对图/文 token 池化得到全局特征，并返回 img_global, text_global。
    """

    def __init__(self, dim, num_heads=4, dropout=0.5):
        super().__init__()
        # 自注意力
        self.img_self_attn = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.text_self_attn = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        # cross-attention
        self.attn_img_to_text = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.attn_text_to_img = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_img = nn.LayerNorm(dim)
        self.norm_text = nn.LayerNorm(dim)

    def forward(self, img_tokens, text_tokens):
        """
        img_tokens:  [B, N_img, D]
        text_tokens: [B, N_txt, D]

        return:
            img_global:   [B, D]
            text_global:  [B, D]
        """
        # 自注意力
        img_feat = self.img_self_attn(img_tokens)     # [B, N_img, D]
        text_feat = self.text_self_attn(text_tokens)  # [B, N_txt, D]

        # 文本作为 query，看图像 (text->img)
        text2img, _ = self.attn_text_to_img(
            query=text_feat,
            key=img_feat,
            value=img_feat,
            need_weights=False
        )  # [B, N_txt, D]

        # 图像作为 query，看文本 (img->text)
        img2text, _ = self.attn_img_to_text(
            query=img_feat,
            key=text_feat,
            value=text_feat,
            need_weights=False
        )  # [B, N_img, D]

        # 残差 + LN
        text_fused = self.norm_text(text_feat + text2img)
        img_fused = self.norm_img(img_feat + img2text)

        # 池化得到全局特征
        img_global = img_fused.mean(dim=1)    # [B, D]
        text_global = text_fused.mean(dim=1)  # [B, D]

        return img_global, text_global


class MultimodalMultitaskModel(nn.Module):
    """
    多模态多任务模型 + 实体序列生成 decoder
    支持两种融合方式：
    - fusion_type="cls": 使用 fusion token + TransformerEncoder
    - fusion_type="cross_attn": 使用传统多模态注意力融合（无 fusion token）
    """

    def __init__(self,
                 vit_3d_model,
                 bert_model_name: str = "hfl/chinese-roberta-wwm-ext",
                 tokenizer=None,
                 image_dim: int = 384,
                 text_feature_dim: int = 384,
                 demographic_dim: int = 3,
                 fusion_dim: int = 384,
                 num_classes: int = 2,
                 dropout: float = 0.5,
                 fusion_transformer_heads: int = 6,
                 fusion_transformer_layers: int = 6,
                 align_dim: int = 128,
                 contrastive_tau: float = 0.07,
                 entity_vocab_size: int = 1000,
                 use_entity_decoder: bool = False,
                 decoder_num_layers: int = 3,
                 decoder_num_heads: int = 6,
                 decoder_max_seq_len: int = 10,
                 fusion_type: str = "cross_attn"):
        super().__init__()

        assert fusion_type in ["cls", "cross_attn"], \
            "fusion_type 必须是 'cls' 或 'cross_attn'"
        self.fusion_type = fusion_type
        self.use_entity_decoder = use_entity_decoder

        # 图像 & 文本编码器
        self.image_encoder = vit_3d_model
        self.text_encoder = TextEncoder(
            bert_model_name=bert_model_name,
            output_dim=text_feature_dim,
            dropout=dropout
        )

        self.image_proj = nn.Linear(image_dim, fusion_dim) if image_dim != fusion_dim else nn.Identity()
        self.text_proj = nn.Linear(text_feature_dim, fusion_dim)

        # ====== 融合结构 ======
        # 方式一：fusion token 方式
        self.fusion_cls_token = nn.Parameter(torch.randn(1, 1, fusion_dim))
        self.fusion_transformer = FusionTransformer(
            embed_dim=fusion_dim,
            num_heads=fusion_transformer_heads,
            num_layers=fusion_transformer_layers,
            dropout=dropout
        )
        self.fusion_norm = nn.LayerNorm(fusion_dim)

        # 方式二：传统 cross-modal 融合（无 fusion token）
        self.cross_modal_fusion = CrossModalFusion(
            dim=fusion_dim,
            num_heads=fusion_transformer_heads,
            dropout=dropout
        )

        # 人口学特征投影
        self.demographic_projection = nn.Sequential(
            nn.Linear(demographic_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 384),
        )

        # 不同融合方式，送入下游任务的 feature_len 不同
        if self.fusion_type == "cls":
            # [fusion_cls + demo]
            feature_len = fusion_dim + 384
        else:
            # [img_global + text_global + demo]
            feature_len = fusion_dim * 2 + 384

        # 分类 head
        self.classifier = nn.Sequential(
            nn.Linear(feature_len, 384),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(384, num_classes),
        )

        # 轴向回归 head
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

        # 对比学习投影头
        self.fused_proj = nn.Linear(fusion_dim, align_dim)
        self.text2_proj = nn.Linear(fusion_dim, align_dim)
        self.contrastive_tau = contrastive_tau

        # ============== 实体序列 decoder ==============
        if use_entity_decoder:
            num_entities = entity_vocab_size
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

        print(f"✅ MultimodalMultitaskModel 初始化完成，fusion_type={self.fusion_type}")

    # =================== 冻结编码器 ===================
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

    # =================== 对比学习 ===================
    def _contrastive_loss(self, x, y):
        """
        x, y: [B, D] 的图像/文本全局特征
        """
        if x is None or y is None:
            dev = x.device if x is not None else y.device
            return torch.tensor(0.0, device=dev)

        B = x.size(0)
        x = F.normalize(self.fused_proj(x), dim=-1)
        y = F.normalize(self.text2_proj(y), dim=-1)

        logits = x @ y.T / self.contrastive_tau   # [B, B]
        labels = torch.arange(B, device=x.device)

        loss_xy = F.cross_entropy(logits, labels)
        loss_yx = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss_xy + loss_yx)

    @staticmethod
    def normalize_features(features):
        return F.layer_norm(features, features.shape[-1:])

    # =================== 前向传播 ===================
    def forward(self,
                image,
                text_inputs,
                demographic,
                masked_text_inputs=None,
                target_entity_ids=None):
        """
        参数:
            image: [B, C, H, W, D]
            text_inputs: dict, 包含 input_ids / attention_mask / token_type_ids
            demographic: [B, demographic_dim]
            masked_text_inputs: dict (实体遮挡后的文本)
            target_entity_ids: [B, K] 被 mask 实体 ID 序列
        """
        device = image.device

        # 1) 图像编码
        _, img_tokens, _ = self.image_encoder(image)
        num_cls = self.image_encoder.num_cls
        img_tokens = img_tokens[:, num_cls:]               # 去掉自身的 cls
        img_tokens = self.image_proj(img_tokens)           # [B, N_img, fusion_dim]

        # 2) 文本编码
        text_seq, text_cls, _ = self.text_encoder(
            input_ids=text_inputs['input_ids'],
            attention_mask=text_inputs['attention_mask'],
            token_type_ids=text_inputs.get('token_type_ids', None)
        )
        text_tokens = self.text_proj(text_seq)             # [B, N_txt, fusion_dim]
        text_cls = self.text_proj(text_cls)               # [B, 384]
        B = img_tokens.size(0)
        demo_feat = self.demographic_projection(demographic)   # [B, 384]

        # ======= 融合部分 =======
        if self.fusion_type == "cls":
            # 方式一：fusion token + TransformerEncoder
            fusion_cls = self.fusion_cls_token.expand(B, 1, -1)          # [B, 1, D]
            fusion_input = torch.cat([fusion_cls, img_tokens, text_tokens], dim=1)
            fusion_out = self.fusion_transformer(fusion_input)           # [B, 1+N_img+N_txt, D]
            fusion_cls = fusion_out[:, 0]                                # [B, D]
            fusion_feature = torch.cat([fusion_cls, demo_feat], dim=1)   # [B, D+384]

            # 对比学习：图像全局特征用简单 mean pool
            img_global_for_cl = img_tokens.mean(dim=1)                   # [B, D]
            contrastive_loss = self._contrastive_loss(img_global_for_cl, text_cls)

        else:
            # 方式二：传统 cross-modal 融合（有注意力，无 fusion token）
            img_global, text_global = self.cross_modal_fusion(img_tokens, text_tokens)  # [B,D], [B,D]
            fusion_feature = torch.cat([img_global, text_global, demo_feat], dim=1)     # [B, 2D+384]
            # 对比学习：用 img_global 和 text_cls（也可以 text_global，看你偏好）
            contrastive_loss = self._contrastive_loss(img_global, text_cls)

        # 5) 主任务
        logits = self.classifier(fusion_feature)           # 分类
        axis_preds = self.axis_regressor(fusion_feature)   # 轴向回归

        # 6) 实体序列生成 decoder
        decoder_loss = torch.tensor(0.0, device=device)
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

            _, loss = self.entity_decoder(
                encoder_context=decoder_context,
                target_entity_ids=target_entity_ids,
                context_mask=None
            )
            decoder_loss = loss if loss is not None else torch.tensor(0.0, device=device)

        return {
            'logits': logits,
            'axis_preds': axis_preds,
            'contrastive_loss': contrastive_loss,
            'decoder_loss': decoder_loss,
            'fusion_feature': fusion_feature
        }