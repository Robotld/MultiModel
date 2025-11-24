import torch
import torch.nn as nn
from transformers import BertModel, AutoTokenizer


class TextEncoder(nn.Module):
    """基于预训练中文BERT的病理或CT报告编码器（无关键词模式）"""

    def __init__(self, bert_model_name=r"E:\workplace\3D\pred_model\bert_chinese",
                 output_dim=384, dropout=0.5):
        super().__init__()
        self.bert_model_name = bert_model_name
        self.output_dim = output_dim

        # 加载预训练BERT模型
        self.bert = BertModel.from_pretrained(bert_model_name, ignore_mismatched_sizes=True)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

        # 线性层和Dropout层
        self.feature_fc = nn.Linear(self.bert.config.hidden_size, output_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        修正后的前向传播
        """
        # 1. 获取BERT输出
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )

        # 2. 获取[CLS]标记的表示
        cls_output = outputs.last_hidden_state[:, 0]  # [B, 768]

        # 3. 特征投影
        text_features = self.feature_fc(cls_output)  # [B, 384]
        text_features = self.dropout_layer(text_features)

        # ✅ 修正：将第二个返回值从 None 改为 text_features (或 cls_output)
        # 这样 MultiModal.py 中的 masked_text_cls 就会拿到一个 Tensor，而不是 None
        return outputs.last_hidden_state, text_features, text_features