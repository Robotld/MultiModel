import torch
import torch.nn as nn
from transformers import BertModel, AutoTokenizer

class TextEncoder(nn.Module):
    """基于预训练中文BERT的病理或CT报告编码器（无关键词模式）"""

    def __init__(self, bert_model_name=r"E:\workplace\3D\pred_model\bert_chinese",
                 output_dim=384, dropout=0.5):
        """
        参数:
            bert_model_name: 预训练BERT模型名称或路径
            output_dim: 输出特征维度
            dropout: Dropout比率
        """
        super().__init__()

        self.bert_model_name = bert_model_name
        self.output_dim = output_dim

        # 加载预训练BERT模型
        self.bert = BertModel.from_pretrained(bert_model_name, ignore_mismatched_sizes=True)
        # 创建对应的tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

        # 线性层和Dropout层，用于处理BERT输出
        self.feature_fc = nn.Linear(self.bert.config.hidden_size, output_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        前向传播
        参数:
            input_ids: 输入token IDs (批处理)
            attention_mask: 注意力掩码 (批处理)
            token_type_ids: token类型IDs（可选）
        返回:
            enhanced_text_features: 增强后的文本特征 [B, output_dim]
            keyword_features: None（无关键词）
            original_text_features: 原始文本特征 [B, output_dim]
        """
        # 1. 获取BERT输出及[CLS]特征
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        # 获取[CLS]标记的表示作为文本特征
        cls_output = outputs.last_hidden_state[:, 0]

        text_features = self.feature_fc(cls_output)  # [B, output_dim]
        text_features = self.dropout_layer(text_features)
        # 无关键词特征，直接返回None
        return outputs.last_hidden_state, None, text_features