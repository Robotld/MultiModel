import torch
import torch.nn as nn
from transformers import BertModel, AutoTokenizer


class TextEncoder(nn.Module):
    """基于预训练中文BERT的病理或CT报告编码器"""

    def __init__(self, bert_model_name=r"E:\workplace\3D\pred_model\bert_chinese",
                 output_dim=384, dropout=0.5):
        super().__init__()
        self.bert_model_name = bert_model_name
        self.output_dim = output_dim

        # 加载预训练BERT模型
        self.bert = BertModel.from_pretrained(bert_model_name, ignore_mismatched_sizes=True)
        # ✅ 使用 Fast tokenizer，支持 return_offsets_mapping
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name, use_fast=True)

        self.feature_fc = nn.Linear(self.bert.config.hidden_size, output_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        cls_output = outputs.last_hidden_state[:, 0]
        text_features = self.feature_fc(cls_output)
        text_features = self.dropout_layer(text_features)
        return outputs.last_hidden_state, text_features, text_features