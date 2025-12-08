import torch
import torch.nn as nn
import torch.nn.functional as F

from .TextEncoder import TextEncoder
from .EntitySeqDecoder import EntitySeqDecoder


class FusionTransformer(nn.Module):
    """Fusion token + TransformerEncoder 融合方式"""

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
    双向交叉注意力融合模块
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
        # 交叉注意力
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
        return: img_global [B, D], text_global [B, D]
        """
        # 自注意力
        img_feat = self.img_self_attn(img_tokens)
        text_feat = self.text_self_attn(text_tokens)

        # 双向交叉注意力
        text2img, _ = self.attn_text_to_img(
            query=text_feat, key=img_feat, value=img_feat, need_weights=False
        )
        img2text, _ = self.attn_img_to_text(
            query=img_feat, key=text_feat, value=text_feat, need_weights=False
        )

        # 残差 + LayerNorm
        text_fused = self.norm_text(text_feat + text2img)
        img_fused = self.norm_img(img_feat + img2text)

        # 全局池化
        img_global = img_fused.mean(dim=1)
        text_global = text_fused.mean(dim=1)

        return img_global, text_global


class MultimodalMultitaskModel(nn.Module):
    """
    多模态多任务模型 V2 - 解决梯度解耦问题

    核心改进：
    1. 训练时使用遮挡文本进行融合和分类，迫使模型依赖图像补全信息
    2.  Decoder和分类器共享同一份文本编码，梯度互通
    3. 推理时使用完整文本，获得最佳性能
    """

    def __init__(self,
                 vit_3d_model,
                 bert_model_name: str = "hfl/chinese-roberta-wwm-ext",
                 tokenizer=None,
                 image_dim: int = 384,
                 text_feature_dim: int = 384,
                 demographic_dim: int = 4,
                 fusion_dim: int = 384,
                 num_classes: int = 2,
                 dropout: float = 0.5,
                 fusion_transformer_heads: int = 6,
                 fusion_transformer_layers: int = 6,
                 align_dim: int = 128,
                 contrastive_tau: float = 0.07,
                 entity_vocab_size: int = 20,
                 use_entity_decoder: bool = True,
                 decoder_num_layers: int = 3,
                 decoder_num_heads: int = 6,
                 decoder_max_seq_len: int = 10,
                 fusion_type: str = "cross_attn",
                 demo_hidden_dim: int = 128,
                 demo_output_dim: int = 384):
        super().__init__()

        assert fusion_type in ["cls", "cross_attn"], \
            "fusion_type 必须是 'cls' 或 'cross_attn'"

        self. fusion_type = fusion_type
        self. use_entity_decoder = use_entity_decoder
        self.fusion_dim = fusion_dim
        self.demo_output_dim = demo_output_dim  # 【修复】避免硬编码

        # =================== 编码器 ===================
        self.image_encoder = vit_3d_model
        self. text_encoder = TextEncoder(
            bert_model_name=bert_model_name,
            output_dim=text_feature_dim,
            dropout=dropout
        )

        # 特征投影层
        self.image_proj = nn.Linear(image_dim, fusion_dim) if image_dim != fusion_dim else nn.Identity()
        self.text_proj = nn.Linear(text_feature_dim, fusion_dim)

        # =================== 融合模块 ===================
        if self.fusion_type == "cls":
            self.fusion_cls_token = nn.Parameter(torch.randn(1, 1, fusion_dim))
            self. fusion_transformer = FusionTransformer(
                embed_dim=fusion_dim,
                num_heads=fusion_transformer_heads,
                num_layers=fusion_transformer_layers,
                dropout=dropout
            )
            feature_len = fusion_dim + self.demo_output_dim  # 【修复】使用变量替代硬编码
        else:
            self.cross_modal_fusion = CrossModalFusion(
                dim=fusion_dim,
                num_heads=fusion_transformer_heads,
                dropout=dropout
            )
            feature_len = fusion_dim * 2 + self.demo_output_dim  # 【修复】使用变量替代硬编码

        # =================== 人口学特征投影 ===================
        self.demographic_projection = nn.Sequential(
            nn. Linear(demographic_dim, demo_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn. Linear(demo_hidden_dim, self.demo_output_dim),
        )

        # =================== 任务头 ===================
        # 分类头
        self.classifier = nn.Sequential(
            nn. Linear(feature_len, 384),
            nn. LayerNorm(384),
            nn. ReLU(),
            nn.Dropout(dropout),
            nn. Linear(384, num_classes),
        )

        # 轴向回归头
        self.axis_regressor = nn.Sequential(
            nn. Linear(feature_len, 256),
            nn. LayerNorm(256),
            nn. ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn. LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        # =================== 对比学习 ===================
        self. fused_proj = nn.Linear(fusion_dim, align_dim)
        self.text_align_proj = nn.Linear(fusion_dim, align_dim)
        self.contrastive_tau = contrastive_tau

        # =================== 实体序列Decoder ===================
        if use_entity_decoder:
            self.entity_decoder = EntitySeqDecoder(
                d_model=fusion_dim,
                num_layers=decoder_num_layers,
                num_heads=decoder_num_heads,
                dim_feedforward=fusion_dim * 4,
                num_entities=entity_vocab_size,
                dropout=dropout,
                max_seq_len=decoder_max_seq_len
            )
            print(f"✅ 实体序列 Decoder 已启用, entity_vocab_size={entity_vocab_size}")

        print(f"✅ MultimodalMultitaskModel V2 初始化完成")
        print(f"  - fusion_type: {fusion_type}")
        print(f"  - use_entity_decoder: {use_entity_decoder}")
        print(f"  - feature_len: {feature_len}")
        print(f"  - demo_output_dim: {self.demo_output_dim}")

    def freeze_encoders(self,
                        freeze_image_encoder: bool = False,
                        freeze_text_encoder: bool = False,
                        bert_unfrozen_last_k: int = 2):
        """冻结编码器参数"""
        if freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            print("✅ 已冻结 ViT3D 图像编码器")

        if freeze_text_encoder:
            for param in self.text_encoder.bert.parameters():
                param.requires_grad = False

            # 解冻最后k层
            num_layers = self.text_encoder.bert.config.num_hidden_layers
            start = max(0, num_layers - bert_unfrozen_last_k)

            # 【修复】修正字符串匹配逻辑
            for name, param in self.text_encoder.bert.named_parameters():
                if name.startswith("encoder.layer."):
                    layer_num = int(name.split('.')[2])
                    if layer_num >= start:
                        param.requires_grad = True

            print(f"✅ BERT 冻结，解冻 layer.{start} ~ layer. {num_layers - 1}")

    def _encode_image(self, image):
        """图像编码"""
        _, img_tokens, _ = self.image_encoder(image)
        num_cls = self.image_encoder.num_cls
        img_tokens = img_tokens[:, num_cls:]  # 去掉cls tokens
        img_tokens = self.image_proj(img_tokens)
        return img_tokens

    def _encode_text(self, text_inputs):
        """文本编码"""
        text_seq, text_cls, _ = self.text_encoder(
            input_ids=text_inputs['input_ids'],
            attention_mask=text_inputs['attention_mask'],
            token_type_ids=text_inputs. get('token_type_ids', None)
        )
        text_tokens = self.text_proj(text_seq)
        text_cls = self.text_proj(text_cls)
        return text_tokens, text_cls

    def _fuse_features(self, img_tokens, text_tokens, demo_feat):
        """特征融合"""
        B = img_tokens.size(0)

        if self.fusion_type == "cls":
            fusion_cls = self.fusion_cls_token.expand(B, 1, -1)
            fusion_input = torch.cat([fusion_cls, img_tokens, text_tokens], dim=1)
            fusion_out = self.fusion_transformer(fusion_input)
            fusion_cls_out = fusion_out[:, 0]
            fusion_feature = torch.cat([fusion_cls_out, demo_feat], dim=1)
            img_global = img_tokens. mean(dim=1)
            text_global = text_tokens.mean(dim=1)
        else:
            img_global, text_global = self.cross_modal_fusion(img_tokens, text_tokens)
            fusion_feature = torch.cat([img_global, text_global, demo_feat], dim=1)

        return fusion_feature, img_global, text_global

    def _contrastive_loss(self, img_feat, text_feat):
        """对比学习损失"""
        if img_feat is None or text_feat is None:
            device = img_feat.device if img_feat is not None else text_feat.device
            return torch.tensor(0.0, device=device)

        B = img_feat. size(0)
        img_feat = F.normalize(self.fused_proj(img_feat), dim=-1)
        text_feat = F. normalize(self.text_align_proj(text_feat), dim=-1)

        logits = img_feat @ text_feat.T / self.contrastive_tau
        labels = torch.arange(B, device=img_feat.device)

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F. cross_entropy(logits.T, labels)

        return 0.5 * (loss_i2t + loss_t2i)

    def forward(self,
                image,
                text_inputs,
                demographic,
                masked_text_inputs=None,
                target_entity_ids=None,
                enable_contrastive_grad: bool = True):
        """
        前向传播 - 解决梯度解耦的核心逻辑

        训练时 (masked_text_inputs is not None):
            - 使用 masked_text_inputs 进行融合和分类
            - 同一份 text_tokens 用于分类和 Decoder
            - 梯度互通，Decoder 帮助学习更好的表示

        推理时 (masked_text_inputs is None):
            - 使用完整的 text_inputs 进行融合和分类

        参数:
            image: [B, C, H, W, D] 3D CT图像
            text_inputs: dict, 完整文本的tokenized输入
            demographic: [B, demographic_dim] 人口学特征
            masked_text_inputs: dict, 遮挡后文本的tokenized输入（训练时提供）
            target_entity_ids: [B, K] 被遮挡的实体ID序列（训练时提供）
            enable_contrastive_grad: bool, 是否允许对比学习梯度回传到文本编码器

        返回:
            dict: {
                'logits': 分类logits,
                'axis_preds': 轴向回归预测,
                'contrastive_loss': 对比损失,
                'decoder_loss': Decoder损失,
                'fusion_feature': 融合特征
            }
        """
        device = image.device
        B = image.size(0)

        # =================== 1. 图像编码（始终相同）===================
        img_tokens = self._encode_image(image)  # [B, N_img, D]

        # =================== 2. 人口学特征 ===================
        demo_feat = self.demographic_projection(demographic)  # [B, demo_output_dim]

        # =================== 3. 文本编码（关键改动！）===================
        # 训练时：使用遮挡文本进行融合，迫使模型依赖图像
        # 推理时：使用完整文本获得最佳性能

        use_masked_text = (self.training and
                           self.use_entity_decoder and
                           masked_text_inputs is not None)

        if use_masked_text:
            # 【训练模式】使用遮挡文本 - 分类和Decoder共享同一份编码！
            text_tokens, text_cls = self._encode_text(masked_text_inputs)

            # 【修复】可选择是否允许对比损失的梯度回传到文本编码器
            if enable_contrastive_grad:
                _, original_text_cls = self._encode_text(text_inputs)
            else:
                with torch.no_grad():
                    _, original_text_cls = self._encode_text(text_inputs)
        else:
            # 【推理模式】使用完整文本
            text_tokens, text_cls = self._encode_text(text_inputs)
            original_text_cls = text_cls

        # =================== 4. 特征融合 ===================
        fusion_feature, img_global, text_global = self._fuse_features(
            img_tokens, text_tokens, demo_feat
        )

        # =================== 5. 主任务：分类和回归 ===================
        logits = self.classifier(fusion_feature)
        axis_preds = self.axis_regressor(fusion_feature)

        # =================== 6. 对比学习损失 ===================
        # 使用图像全局特征和完整文本CLS进行对比
        contrastive_loss = self._contrastive_loss(img_global, original_text_cls)

        # =================== 7.  Decoder损失（核心！）===================
        decoder_loss = torch.tensor(0.0, device=device)

        if self.use_entity_decoder and self.training and target_entity_ids is not None:
            # 构建Decoder上下文：图像tokens + 遮挡文本tokens
            # 注意：这里的text_tokens和融合用的是同一份！梯度共享！
            decoder_context = torch. cat([img_tokens, text_tokens], dim=1)

            _, loss = self.entity_decoder(
                encoder_context=decoder_context,
                target_entity_ids=target_entity_ids,
                context_mask=None
            )
            decoder_loss = loss if loss is not None else torch. tensor(0.0, device=device)

        return {
            'logits': logits,
            'axis_preds': axis_preds,
            'contrastive_loss': contrastive_loss,
            'decoder_loss': decoder_loss,
            'fusion_feature': fusion_feature
        }

    @torch.no_grad()
    def inference(self, image, text_inputs, demographic):
        """
        推理专用接口 - 使用完整文本
        """
        # 【修复】保存原始训练状态，推理后恢复
        was_training = self. training
        self. eval()

        try:
            img_tokens = self._encode_image(image)
            text_tokens, text_cls = self._encode_text(text_inputs)
            demo_feat = self. demographic_projection(demographic)

            fusion_feature, _, _ = self._fuse_features(img_tokens, text_tokens, demo_feat)

            logits = self.classifier(fusion_feature)
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            return {
                'logits': logits,
                'probs': probs,
                'preds': preds
            }
        finally:
            # 恢复原始状态
            if was_training:
                self.train()

    @torch.no_grad()
    def predict_entities(self, image, masked_text_inputs):
        """
        实体预测接口 - 给定图像和遮挡文本，预测被遮挡的实体
        """
        if not self.use_entity_decoder:
            raise RuntimeError("Entity decoder is not enabled")

        # 【修复】保存原始训练状态，推理后恢复
        was_training = self.training
        self.eval()

        try:
            img_tokens = self._encode_image(image)
            text_tokens, _ = self._encode_text(masked_text_inputs)

            decoder_context = torch. cat([img_tokens, text_tokens], dim=1)
            generated, _ = self.entity_decoder(
                encoder_context=decoder_context,
                target_entity_ids=None  # 推理模式
            )

            return generated
        finally:
            # 恢复原始状态
            if was_training:
                self.train()