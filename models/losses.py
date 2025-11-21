import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOSS_REGISTRY = {
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "FocalLoss": None,  # 将在下面定义
    "None": None  # 无
}


class FocalLoss(nn.Module):
    def __init__(self, gamma=2., alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class MultitaskLoss(nn.Module):
    """
    多任务损失函数：
    1. 复发预测（分类）
    2. 结节长短径回归
    3. 相似度/对比损失
    4. 遮挡关键词预测（新增）
    """

    def __init__(self,
                 recurrence_weight=1.0,
                 axis_weight=1.0,
                 masked_entity_weight=0.8,  # 🔥 新增：遮挡预测权重
                 use_focal_loss=False):  # 🔥 可选：是否用Focal Loss
        """
        参数:
            recurrence_weight: 复发预测任务的权重
            axis_weight: 长短径回归任务的权重
            similarity_weight: 相似度/对比损失的权重
            masked_entity_weight: 遮挡关键词预测的权重（新增）
            use_focal_loss: 是否对复发预测使用Focal Loss
        """
        super().__init__()
        self.recurrence_weight = recurrence_weight
        self.axis_weight = axis_weight
        self.masked_entity_weight = masked_entity_weight  # 🔥 新增

        # 复发预测损失
        if use_focal_loss:
            self.recurrence_criterion = FocalLoss(gamma=2.0, alpha=0.25)
        else:
            self.recurrence_criterion = nn.CrossEntropyLoss(
                weight=torch.tensor([1.0, 1.0], device='cuda')
            )

        # 长短径回归损失
        self.axis_criterion = nn.MSELoss()

        # 🔥 遮挡关键词预测损失（交叉熵）
        # 注意：ignore_index=-1，忽略padding位置
        self.masked_entity_criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self,
                recurrence_logits,
                axis_preds,
                recurrence_labels,
                axis_labels,
                masked_entity_logits=None,  # 🔥 新增：遮挡预测的logits
                target_entity_ids=None):  # 🔥 新增：真实实体ID
        """
        计算多任务损失

        参数:
            recurrence_logits: 复发预测的输出 [B, 2]
            axis_preds: 结节长短径预测 [B, 2]
            recurrence_labels: 复发标签 [B]
            axis_labels: 真实长短径 [B, 2]
            similarity_loss: 跨模态相似度/对比损失（标量或Tensor）
            masked_entity_logits: 遮挡实体预测logits [B, entity_vocab_size]（可选）
            target_entity_ids: 真实实体ID [B, max_num_masked]（可选）

        返回:
            total_loss: 总损失
            recurrence_loss: 复发预测损失
            axis_loss: 回归损失
            similarity_loss: 相似度损失
            masked_entity_loss: 遮挡预测损失（新增）
        """
        # 1. 复发预测损失
        recurrence_loss = self.recurrence_criterion(
            recurrence_logits,
            recurrence_labels.long()
        )

        # 2. 长短径回归损失（目前未启用，可以根据需要调整）
        axis_loss = self.axis_criterion(axis_preds, axis_labels.float()) if axis_preds is not None else 0.0

        # 🔥 4. 遮挡关键词预测损失（新增）
        masked_entity_loss = torch.tensor(0.0, device=recurrence_logits.device)

        if masked_entity_logits is not None and target_entity_ids is not None:
            # target_entity_ids: [B, max_num_masked]
            # masked_entity_logits: [B, entity_vocab_size]

            # 取第一个被遮挡的实体作为目标（简化版）
            # 如果需要预测多个实体，可以扩展为序列预测
            valid_mask = (target_entity_ids[:, 0] != -1)  # [B]

            if valid_mask.sum() > 0:
                # 只计算有效样本的损失
                valid_logits = masked_entity_logits[valid_mask]  # [N, vocab_size]
                valid_targets = target_entity_ids[:, 0][valid_mask]  # [N]

                masked_entity_loss = self.masked_entity_criterion(
                    valid_logits,
                    valid_targets.long()
                )

        # 5. 总损失
        total_loss = (
                self.recurrence_weight * recurrence_loss +
                self.axis_weight * axis_loss +
                self.masked_entity_weight * masked_entity_loss  # 🔥 新增
        )

        return (
            total_loss,
            recurrence_loss,
            axis_loss,  # 改为返回实际值而非0
            masked_entity_loss  # 🔥 新增返回值
        )


# 🔥 进阶版：多实体预测损失（可选）
class MultitaskLossAdvanced(MultitaskLoss):
    """
    进阶版：支持预测多个被遮挡的实体
    """

    def forward(self,
                recurrence_logits,
                axis_preds,
                recurrence_labels,
                axis_labels,
                masked_entity_logits=None,  # [B, max_num_masked, vocab_size]
                target_entity_ids=None):  # [B, max_num_masked]
        """
        支持预测多个被遮挡实体的版本
        """
        # 前面的损失计算与基础版相同
        recurrence_loss = self.recurrence_criterion(recurrence_logits, recurrence_labels.long())
        axis_loss = self.axis_criterion(axis_preds, axis_labels.float()) if axis_preds is not None else 0.0


        # 🔥 多实体预测损失
        masked_entity_loss = torch.tensor(0.0, device=recurrence_logits.device)

        if masked_entity_logits is not None and target_entity_ids is not None:
            # masked_entity_logits: [B, max_num_masked, vocab_size]
            # target_entity_ids: [B, max_num_masked]

            B, max_num, vocab_size = masked_entity_logits.shape

            # 展平成 [B * max_num, vocab_size]
            logits_flat = masked_entity_logits.reshape(-1, vocab_size)
            targets_flat = target_entity_ids.reshape(-1)

            # 只计算非padding位置的损失
            valid_mask = (targets_flat != -1)

            if valid_mask.sum() > 0:
                masked_entity_loss = self.masked_entity_criterion(
                    logits_flat[valid_mask],
                    targets_flat[valid_mask].long()
                )

        # 总损失
        total_loss = (
                self.recurrence_weight * recurrence_loss +
                self.axis_weight * axis_loss +
                self.masked_entity_weight * masked_entity_loss
        )

        return total_loss, recurrence_loss, axis_loss, masked_entity_loss


# 更新注册表
LOSS_REGISTRY["FocalLoss"] = FocalLoss
LOSS_REGISTRY["MultitaskLoss"] = MultitaskLoss
LOSS_REGISTRY["MultitaskLossAdvanced"] = MultitaskLossAdvanced