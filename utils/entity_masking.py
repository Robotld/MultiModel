"""
Medical entity masking for text augmentation and prediction tasks
"""
import re
import random
import torch
import numpy as np


class MedicalEntityMasker:
    """
    医疗实体遮挡工具，用于遮挡医学报告中的关键实体进行预测训练
    """
    
    def __init__(self, mask_ratio=0.5, mask_token='[MASK]'):
        """
        初始化遮挡器
        
        Args:
            mask_ratio: 遮挡比例 (0-1)
            mask_token: 遮挡标记
        """
        self.mask_ratio = mask_ratio
        self.mask_token = mask_token
        
        # 医学实体词表 - 常见的医学术语和实体
        self.entity_patterns = [
            # 解剖位置
            r'(左|右|双)(上|中|下)肺',
            r'(左|右|双)肺(上|中|下)叶',
            r'肺(上|中|下)叶',
            r'(主|叶|段)支气管',
            r'纵隔',
            r'胸膜',
            r'肺门',
            r'气管',
            
            # 病变类型
            r'结节',
            r'肿块',
            r'浸润',
            r'钙化',
            r'空洞',
            r'实变',
            r'磨玻璃影',
            r'肺不张',
            r'积液',
            r'气胸',
            
            # 测量和描述
            r'\d+\.?\d*\s*(mm|cm|毫米|厘米)',
            r'直径约?\s*\d+\.?\d*\s*(mm|cm)',
            r'大小约?\s*\d+\.?\d*\s*[×x]\s*\d+\.?\d*\s*(mm|cm)',
            
            # 病理描述
            r'腺癌',
            r'鳞癌',
            r'小细胞癌',
            r'(恶性|良性)',
            r'转移',
            r'复发',
            
            # 其他医学术语
            r'淋巴结',
            r'纤维化',
            r'炎症',
            r'水肿',
        ]
        
        # 编译正则表达式
        self.compiled_patterns = [re.compile(pattern) for pattern in self.entity_patterns]
        
        # 构建实体词表
        self.entity_vocab = {}
        self.vocab_size = 0
        self._build_vocab()
    
    def _build_vocab(self):
        """构建实体词表"""
        # 预定义一些常见医学实体
        common_entities = [
            '结节', '肿块', '浸润', '钙化', '积液',
            '上叶', '中叶', '下叶', '左肺', '右肺',
            '腺癌', '鳞癌', '转移', '纵隔', '胸膜',
            '磨玻璃影', '实变', '空洞', '淋巴结', '纤维化'
        ]
        
        for idx, entity in enumerate(common_entities):
            self.entity_vocab[entity] = idx
        
        self.vocab_size = len(self.entity_vocab)
    
    def extract_entities(self, text):
        """
        从文本中提取医学实体
        
        Args:
            text: 输入文本
            
        Returns:
            list of tuples: [(entity_text, start, end), ...]
        """
        entities = []
        
        for pattern in self.compiled_patterns:
            for match in pattern.finditer(text):
                entity_text = match.group()
                start = match.start()
                end = match.end()
                entities.append((entity_text, start, end))
        
        # 去重并按位置排序
        entities = sorted(set(entities), key=lambda x: x[1])
        
        return entities
    
    def mask_entities(self, text, entities=None):
        """
        遮挡文本中的实体
        
        Args:
            text: 输入文本
            entities: 要遮挡的实体列表，如果为None则自动提取
            
        Returns:
            masked_text: 遮挡后的文本
            masked_entities: 被遮挡的实体列表
        """
        if entities is None:
            entities = self.extract_entities(text)
        
        if len(entities) == 0:
            return text, []
        
        # 随机选择要遮挡的实体
        num_to_mask = max(1, int(len(entities) * self.mask_ratio))
        entities_to_mask = random.sample(entities, num_to_mask)
        
        # 按位置排序（从后往前遮挡，避免位置偏移）
        entities_to_mask = sorted(entities_to_mask, key=lambda x: x[1], reverse=True)
        
        masked_text = text
        masked_entities = []
        
        for entity_text, start, end in entities_to_mask:
            # 替换为遮挡标记
            masked_text = masked_text[:start] + self.mask_token + masked_text[end:]
            masked_entities.append(entity_text)
        
        return masked_text, masked_entities
    
    def get_entity_id(self, entity_text):
        """
        获取实体的ID
        
        Args:
            entity_text: 实体文本
            
        Returns:
            int: 实体ID，如果不在词表中返回-1
        """
        # 尝试完全匹配
        if entity_text in self.entity_vocab:
            return self.entity_vocab[entity_text]
        
        # 尝试部分匹配
        for vocab_entity, entity_id in self.entity_vocab.items():
            if vocab_entity in entity_text or entity_text in vocab_entity:
                return entity_id
        
        return -1  # 未知实体
    
    def batch_mask(self, texts, max_masked=3):
        """
        批量遮挡文本
        
        Args:
            texts: 文本列表
            max_masked: 每个样本最多遮挡的实体数
            
        Returns:
            masked_texts: 遮挡后的文本列表
            target_ids: 目标实体ID张量 (batch_size, max_masked)
        """
        masked_texts = []
        target_ids_list = []
        
        for text in texts:
            masked_text, masked_entities = self.mask_entities(text)
            masked_texts.append(masked_text)
            
            # 获取被遮挡实体的ID
            entity_ids = [self.get_entity_id(entity) for entity in masked_entities[:max_masked]]
            
            # 填充到固定长度
            while len(entity_ids) < max_masked:
                entity_ids.append(-1)  # -1 表示填充
            
            target_ids_list.append(entity_ids[:max_masked])
        
        # 转换为张量
        target_ids_tensor = torch.tensor(target_ids_list, dtype=torch.long)
        
        return masked_texts, target_ids_tensor
    
    def decode_entity(self, entity_id):
        """
        将实体ID解码为文本
        
        Args:
            entity_id: 实体ID
            
        Returns:
            str: 实体文本
        """
        for entity_text, eid in self.entity_vocab.items():
            if eid == entity_id:
                return entity_text
        return '[UNK]'


# Example usage
if __name__ == '__main__':
    masker = MedicalEntityMasker(mask_ratio=0.5)
    
    sample_text = "患者左肺上叶见结节影，大小约15mm×12mm，边缘清晰，密度均匀。纵隔淋巴结未见明显肿大。"
    
    print(f"原始文本: {sample_text}")
    print(f"\n实体词表大小: {masker.vocab_size}")
    
    # 提取实体
    entities = masker.extract_entities(sample_text)
    print(f"\n提取的实体: {entities}")
    
    # 遮挡实体
    masked_text, masked_entities = masker.mask_entities(sample_text)
    print(f"\n遮挡后文本: {masked_text}")
    print(f"被遮挡的实体: {masked_entities}")
    
    # 批量遮挡
    texts = [sample_text, "右肺下叶见肿块影，伴有胸腔积液。"]
    masked_texts, target_ids = masker.batch_mask(texts)
    print(f"\n批量遮挡:")
    for i, (mt, ti) in enumerate(zip(masked_texts, target_ids)):
        print(f"  样本{i+1}: {mt}")
        print(f"  目标ID: {ti}")
