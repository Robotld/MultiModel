import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOSS_REGISTRY = {
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "FocalLoss": None,
    "None": None
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
    ❌ 去掉：相似度/对比损失（已移到模型内部）
    """

    def __init__(self,
                 recurrence_weight=1.0,
                 axis_weight=1.0,
                 use_focal_loss=False):
        """
        参数:
            recurrence_weight: 复发预测任务的权重
            axis_weight: 长短径回归任务的权重
            use_focal_loss: 是否对复发预测使用Focal Loss
        """
        super().__init__()
        self.recurrence_weight = recurrence_weight
        self.axis_weight = axis_weight

        # 复发预测损失
        if use_focal_loss:
            self.recurrence_criterion = FocalLoss(gamma=2.0, alpha=0.25)
        else:
            self.recurrence_criterion = nn.CrossEntropyLoss(
                weight=torch.tensor([1.0, 1.0], device='cuda')
            )

        # 长短径回归损失
        self.axis_criterion = nn.MSELoss()

    def forward(self,
                recurrence_logits,
                axis_preds,
                recurrence_labels,
                axis_labels):
        """
        计算多任务损失（❌ 去掉 similarity_loss）

        参数:
            recurrence_logits: 复发预测的输出 [B, 2]
            axis_preds: 结节长短径预测 [B, 2]
            recurrence_labels: 复发标签 [B]
            axis_labels: 真实长短径 [B, 2]

        返回:
            total_loss: 总损失
            recurrence_loss: 复发预测损失
            axis_loss: 回归损失
        """
        # 1. 复发预测损失
        recurrence_loss = self.recurrence_criterion(
            recurrence_logits,
            recurrence_labels.long()
        )

        # 2. 长短径回归损失
        axis_loss = self.axis_criterion(axis_preds, axis_labels.float()) if axis_preds is not None else 0.0

        # 3. 总损失（❌ 不包含 similarity_loss）
        total_loss = (
                self.recurrence_weight * recurrence_loss +
                self.axis_weight * axis_loss
        )

        return total_loss, recurrence_loss, axis_loss


# 更新注册表
LOSS_REGISTRY["FocalLoss"] = FocalLoss
LOSS_REGISTRY["MultitaskLoss"] = MultitaskLoss