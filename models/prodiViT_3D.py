import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models import VisionTransformer
from transformers import Dinov2ForImageClassification
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

class ViT3D(VisionTransformer):
    def __init__(self,
                 num_classes=3,
                 dim=384,
                 depth=4,
                 heads=8,
                 mlp_dim=512,
                 dropout=0.25,
                 emb_dropout=0.5,
                 patch_size=16,
                 image_size=None,
                 crop_size=None,
                 in_chans=1,
                 pool='max',
                 cpt_num=3,
                 mlp_num=3,
                 use_flow=False):
        super().__init__(img_size=image_size, patch_size=patch_size, in_chans=in_chans,
                         num_classes=num_classes, embed_dim=dim, depth=depth, num_heads=heads)
        self.pool = pool
        self.num_classes = num_classes

        # 处理图像尺寸
        if crop_size:
            self.image_size = crop_size
        else:
            self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size, image_size)

        # 确保patch_size是标量
        self.patch_dim = patch_size

        # 计算每个维度的补丁数量
        self.d_patches = self.image_size[2] // self.patch_dim
        self.h_patches = self.image_size[0] // self.patch_dim
        self.w_patches = self.image_size[1] // self.patch_dim
        self.num_patches = self.d_patches * self.h_patches * self.w_patches

        # 创建3D补丁嵌入
        patch_size_3d = (self.patch_dim, self.patch_dim, self.patch_dim)
        self.patch_embedding = nn.Conv3d(in_chans, dim,
                                         kernel_size=patch_size_3d,
                                         stride=patch_size_3d)

        # 修改位置编码以匹配非立方体补丁数量
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1 + cpt_num, dim))

        # # 添加类别令牌
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # 添加类原型向量
        self.class_prompts = nn.Parameter(torch.randn(1, cpt_num, dim))
        self.num_cls = 1 + cpt_num

        mlp_input_dim = dim*2

        # 创建MLP分类头
        if mlp_num == 1:
            self.mlp_head = nn.Sequential(
                nn.Linear(mlp_input_dim, num_classes)
            )
        elif mlp_num == 2:
            self.mlp_head = nn.Sequential(
                nn.Linear(mlp_input_dim, mlp_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(mlp_dim, num_classes)
            )


        # 存储注意力相关数据
        self.attention_weights = []  # 用于存储注意力权重
        self.hooks = []  # 用于存储hook句柄
        self.original_input = None  # 用于存储原始输入

        print(
            f"模型初始化完成：裁剪尺寸={self.image_size}, 补丁数量={self.num_patches} ({self.h_patches}×{self.w_patches}×{self.d_patches})")


    def forward(self, x):
        """
        前向传播
        参数:
            x: 输入张量，形状为(batch_size, channels, height, width, depth)
        """
        batch_size, channels, height, width, depth = x.shape
        # 生成3D补丁嵌入
        # conv3D 需要将输入形状转换为[N, C, D, H, W]
        x = x.permute(0, 1, 4, 2, 3)  # 转换为[N, C, D, H, W]
        x = self.patch_embedding(x)  # (batch_size, dim, h', w', d')

        # 展平补丁并转置为序列形式
        x = x.flatten(2).transpose(1, 2)  # (batch_size, num_patches, dim)

        # 添加类别提示向量
        class_prompts = self.class_prompts.expand(batch_size, -1, -1)
        x = torch.cat((class_prompts, x), dim=1)

        # 添加分类令牌
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embedding
        x = self.blocks(x)

        # 提取类别提示向量
        class_prompt_features = x[:, 1:self.num_cls]
        similarity_loss = self.prompt_cosine_similarity_loss(0.3, 0.7)

        # 提取分类令牌和补丁令牌（不包括类别提示向量）
        cls_token_out = x[:, 0]  # (batch_size, dim)
        class_prompts_pooled = class_prompt_features

        # 根据池化方法聚合特征
        if self.pool == 'mean':
            if 0 not in class_prompts_pooled.shape:
                class_prompts_pooled = torch.mean(class_prompt_features, dim=1)  # (batch_size, dim)
        elif self.pool == 'max':
            if 0 not in class_prompts_pooled.shape:
                class_prompts_pooled = torch.max(class_prompt_features, dim=1)[0]  # (batch_size, dim)

        combined_features = torch.cat([class_prompts_pooled,cls_token_out], dim=1)
        # 最终分类头
        out = self.mlp_head(combined_features)
        return out, x, None, similarity_loss

    def load_pretrained_dino(self, path):
        """加载预训练的DINOv2权重"""
        try:
            basic_model = Dinov2ForImageClassification.from_pretrained(
                path,
                num_labels=self.num_classes,
                ignore_mismatched_sizes=True
            )
            self.load_state_dict(basic_model.state_dict(), strict=False)
            # 对于加载的预训练模型，直接把CSL随机初始化。
            print("成功加载DINOv2预训练权重")
        except Exception as e:
            print(f"加载预训练权重失败: {str(e)}")


