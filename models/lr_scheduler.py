"""
学习率调度器模块
实现各种学习率预热和调度策略
"""

import math
from torch.optim.lr_scheduler import LRScheduler


class WarmupScheduler(LRScheduler):
    """
    带预热功能的学习率调度器，可以包装任何其他scheduler
    """

    def __init__(self, optimizer, warmup_epochs, base_scheduler=None, warmup_type='linear',
                 last_epoch=-1, verbose=False):
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.warmup_type = warmup_type  # 'linear' 或 'exponential'
        self.finished_warmup = False
        super(WarmupScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epochs:
            if not self.finished_warmup:
                self.finished_warmup = True
                print(f"预热阶段结束，切换到基础学习率调度器")

            if self.base_scheduler:
                # 如果已经完成预热，使用基础调度器
                if hasattr(self.base_scheduler, '_last_lr'):
                    return self.base_scheduler._last_lr
                return self.base_scheduler.get_lr()
            return self.base_lr

        # 处于预热阶段
        warmup_factor = self.get_warmup_factor(self.last_epoch, self.warmup_epochs, self.warmup_type)
        return [base_lr * warmup_factor for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished_warmup and self.base_scheduler:
            self.base_scheduler.step(epoch)
        else:
            super(WarmupScheduler, self).step(epoch)

    @staticmethod
    def get_warmup_factor(current_step, warmup_steps, warmup_type='linear'):
        if warmup_type == 'linear':
            # 线性预热
            alpha = current_step / warmup_steps
            return min(1.0, alpha)
        elif warmup_type == 'exponential':
            # 指数预热
            return 1.0 - math.exp(-(current_step + 1) / warmup_steps)
        else:
            raise ValueError(f"Unsupported warmup_type: {warmup_type}")