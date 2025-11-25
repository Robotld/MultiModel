import random
import re
import torch


class MedicalEntityMasker:
    """
    医学实体遮挡工具类（多实体 + token 级支持）

    功能：
    - 从文本中抽取预定义医学实体
    - 按比例随机遮挡若干实体（20%/30%/40%可选）
    - 返回：
        * 遮挡后的文本
        * 被遮挡实体ID列表
        * 被遮挡实体的字符级 span（起止索引）
    """

    def __init__(self,
                 mask_token: str = "[MASK]",
                 mask_ratio: float = 0.5,
                 mask_ratio_choices=None):
        """
        参数:
            mask_token: 遮挡用 token
            mask_ratio: 若未提供 mask_ratio_choices，则使用该固定比例
            mask_ratio_choices: 多档候选比例列表，例如 [0.2, 0.3, 0.4]；
                                若不为 None，则每条样本随机选其中一个，覆盖 mask_ratio。
        """
        self.mask_token = mask_token
        self.mask_ratio = mask_ratio
        self.mask_ratio_choices = mask_ratio_choices

        # 肺癌浸润性相关关键实体词典
        self.key_entities = {
            "density": ["磨玻璃", "实性", "混合密度", "GGO", "纯磨玻璃", "部分实性"],
            "edge": ["分叶", "毛刺", "光滑", "不规则", "边缘清晰", "边缘模糊"],
            "size": ["结节", "肿块", "病灶", "占位"],
            "location": ["上叶", "下叶", "中叶", "右肺", "左肺", "胸膜"],
            "texture": ["空泡", "支气管充气征", "血管穿行", "钙化"],
            "invasiveness": ["浸润", "原位", "微浸润", "AIS", "MIA", "IAC"]
        }

        # 扁平化实体
        self.all_entities = []
        self.entity_to_category = {}
        for category, entities in self.key_entities.items():
            for entity in entities:
                self.all_entities.append(entity)
                self.entity_to_category[entity] = category

        # 实体 ↔ ID
        self.entity2id = {entity: idx for idx, entity in enumerate(self.all_entities)}
        self.id2entity = {idx: entity for entity, idx in self.entity2id.items()}
        self.vocab_size = len(self.all_entities)

    # ---------------- 基础工具 ----------------

    def _choose_mask_ratio(self) -> float:
        if self.mask_ratio_choices:
            return random.choice(self.mask_ratio_choices)
        return self.mask_ratio

    def extract_entities(self, text: str):
        """
        从文本中提取关键医学实体
        返回: [(entity_text, start_pos, end_pos), ...]
        """
        found_entities = []
        sorted_entities = sorted(self.all_entities, key=len, reverse=True)

        for entity in sorted_entities:
            for match in re.finditer(re.escape(entity), text):
                start, end = match.span()
                overlap = any(s <= start < e or s < end <= e for _, s, e in found_entities)
                if not overlap:
                    found_entities.append((entity, start, end))

        found_entities.sort(key=lambda x: x[1])
        return found_entities

    # ---------------- 单样本遮挡 ----------------

    def mask_entities(self, text: str):
        """
        在文本中按比例遮挡多个实体
        返回:
            masked_text: 遮挡后的文本
            target_ids:  List[int] 被遮挡实体ID列表
            spans:       List[(start, end)] 被遮挡实体的原始字符级区间
        """
        entities = self.extract_entities(text)
        if len(entities) == 0:
            return text, [], []

        ratio = self._choose_mask_ratio()
        num_to_mask = max(1, int(len(entities) * ratio))
        num_to_mask = min(num_to_mask, len(entities))

        # 随机采样要遮挡的实体
        entities_to_mask = random.sample(entities, num_to_mask)
        entities_to_mask.sort(key=lambda x: x[1])  # 按起始位置排序

        masked_text = text
        target_ids = []
        spans = []

        # 从后往前替换，避免索引偏移
        for entity_text, start, end in reversed(entities_to_mask):
            masked_text = masked_text[:start] + self.mask_token + masked_text[end:]
            if entity_text in self.entity2id:
                eid = self.entity2id[entity_text]
            else:
                eid = -1
            target_ids.insert(0, eid)
            spans.insert(0, (start, end))

        return masked_text, target_ids, spans

    # ---------------- 批量遮挡 ----------------

    def batch_mask(self, texts):
        """
        批量遮挡
        返回:
            masked_texts: List[str]
            target_ids:   LongTensor [B, K]，padding = -1
            spans:        LongTensor [B, K, 2]，padding = -1（字符级起止）
        """
        masked_texts = []
        all_target_ids = []
        all_spans = []
        max_len = 0

        for text in texts:
            masked_text, target_ids, spans = self.mask_entities(text)
            masked_texts.append(masked_text)
            all_target_ids.append(target_ids)
            all_spans.append(spans)
            max_len = max(max_len, len(target_ids))

        if max_len == 0:
            # 整个 batch 都无实体
            B = len(texts)
            target_ids_tensor = torch.full((B, 1), -1, dtype=torch.long)
            spans_tensor = torch.full((B, 1, 2), -1, dtype=torch.long)
            return masked_texts, target_ids_tensor, spans_tensor

        padded_ids = []
        padded_spans = []
        for ids, spans in zip(all_target_ids, all_spans):
            ids = ids if len(ids) > 0 else []
            spans = spans if len(spans) > 0 else []

            pad_len = max_len - len(ids)
            if pad_len > 0:
                ids = ids + [-1] * pad_len
                spans = spans + [(-1, -1)] * pad_len

            padded_ids.append(ids)
            padded_spans.append(spans)

        target_ids_tensor = torch.tensor(padded_ids, dtype=torch.long)          # [B, K]
        spans_tensor = torch.tensor(padded_spans, dtype=torch.long)            # [B, K, 2]

        return masked_texts, target_ids_tensor, spans_tensor