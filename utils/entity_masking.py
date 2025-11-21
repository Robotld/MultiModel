# utils/entity_masking.py
import random
import re
import torch


class MedicalEntityMasker:
    """
    医学实体遮挡工具类
    """

    def __init__(self, mask_token="[MASK]", mask_ratio=0.5):
        self.mask_token = mask_token
        self.mask_ratio = mask_ratio

        # 肺癌浸润性相关关键实体词典
        self.key_entities = {
            "density": ["磨玻璃", "实性", "混合密度", "GGO", "纯磨玻璃", "部分实性"],
            "edge": ["分叶", "毛刺", "光滑", "不规则", "边缘清晰", "边缘模糊"],
            "size": ["结节", "肿块", "病灶", "占位"],
            "location": ["上叶", "下叶", "中叶", "右肺", "左肺", "胸膜"],
            "texture": ["空泡", "支气管充气征", "血管穿行", "钙化"],
            "invasiveness": ["浸润", "原位", "微浸润", "AIS", "MIA", "IAC"]
        }

        # 扁平化所有实体
        self.all_entities = []
        self.entity_to_category = {}
        for category, entities in self.key_entities.items():
            for entity in entities:
                self.all_entities.append(entity)
                self.entity_to_category[entity] = category

        # 构建实体到ID的映射
        self.entity2id = {entity: idx for idx, entity in enumerate(self.all_entities)}
        self.id2entity = {idx: entity for entity, idx in self.entity2id.items()}
        self.vocab_size = len(self.all_entities)

    def extract_entities(self, text):
        """
        从文本中提取关键医学实体
        返回: [(entity_text, start_pos, end_pos), ...]
        """
        found_entities = []
        # 按长度降序排序，优先匹配长实体（避免"磨玻璃结节"只匹配到"结节"）
        sorted_entities = sorted(self.all_entities, key=len, reverse=True)

        for entity in sorted_entities:
            # 使用正则避免重复匹配
            for match in re.finditer(re.escape(entity), text):
                start, end = match.span()
                # 避免重复区间
                overlap = any(s <= start < e or s < end <= e
                              for _, s, e in found_entities)
                if not overlap:
                    found_entities.append((entity, start, end))

        # 按位置排序
        found_entities.sort(key=lambda x: x[1])
        return found_entities

    def mask_entities(self, text, entities=None):
        """
        遮挡文本中的关键实体
        返回:
            - masked_text: 遮挡后的文本
            - target_entities: 被遮挡的实体列表（保持原始顺序）
            - target_ids: 对应的实体ID列表
        """
        if entities is None:
            entities = self.extract_entities(text)

        if len(entities) == 0:
            # 无实体，返回原文本
            return text, [], []

        # 随机选择要遮挡的实体
        num_to_mask = max(1, int(len(entities) * self.mask_ratio))
        entities_to_mask = random.sample(entities, num_to_mask)
        entities_to_mask.sort(key=lambda x: x[1])  # 按位置排序

        # 从后往前替换，避免位置偏移
        masked_text = text
        target_entities = []
        target_ids = []

        for entity_text, start, end in reversed(entities_to_mask):
            masked_text = masked_text[:start] + self.mask_token + masked_text[end:]
            target_entities.insert(0, entity_text)  # 保持原始顺序
            if entity_text in self.entity2id:
                target_ids.insert(0, self.entity2id[entity_text])
            else:
                target_ids.insert(0, -1)  # 未知实体

        return masked_text, target_entities, target_ids

    def batch_mask(self, texts):
        """
        批量遮挡
        返回:
            - masked_texts: 遮挡后的文本列表
            - all_target_ids: [B, max_num_masked] 被遮挡实体ID的张量（padding=-1）
        """
        masked_texts = []
        all_target_ids = []
        max_len = 0

        for text in texts:
            masked_text, _, target_ids = self.mask_entities(text)
            masked_texts.append(masked_text)
            all_target_ids.append(target_ids)
            max_len = max(max_len, len(target_ids))

        # Padding到相同长度
        padded_ids = []
        for ids in all_target_ids:
            padded = ids + [-1] * (max_len - len(ids))
            padded_ids.append(padded)

        return masked_texts, torch.tensor(padded_ids, dtype=torch.long)