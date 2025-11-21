import os
import random

import numpy as np
import torch


def set_seed(seed=42):
    """设置所有随机数生成器的种子"""
    random.seed(seed)  # Python内置随机模块
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU
    os.environ['PYTHONHASHSEED'] = str(seed)  # Python哈希种子
    torch.backends.cudnn.deterministic = True  # 确保每次返回相同的卷积算法
    torch.backends.cudnn.benchmark = False  # 禁用cudnn自动基准测试