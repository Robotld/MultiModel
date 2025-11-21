import torch
from torch import nn


def freeze_encoder_keep_prompts(model):
    """冻结编码器参数，只保留CLS token和类原型向量可训练"""
    # 默认冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 解冻CLS token
    model.cls_token.requires_grad = True

    # 解冻类原型向量和位置编码
    if hasattr(model, 'class_prompts'):
        model.class_prompts.requires_grad = True

    # 可选：保持分类头可训练
    for param in model.mlp_head.parameters():
        param.requires_grad = True



def progressive_unfreeze(model, current_epoch, total_epochs, initial_lr=1e-4, finetune_lr=5e-6):
    """
    按比例渐进式解冻class_prompts参数

    参数:
        model: 模型实例
        current_epoch: 当前训练轮次
        total_epochs: 总训练轮次
        initial_lr: 分类头学习率
        finetune_lr: 提示参数微调学习率

    返回:
        updated: 是否更新了模型状态
        optimizer: 如果更新了状态，返回新的优化器；否则返回None
    """
    updated = False

    # 查找class_prompts参数
    class_prompts_param = None
    class_prompts_name = None
    for name, param in model.named_parameters():
        if 'class_prompts' in name:
            class_prompts_param = param
            class_prompts_name = name
            break

    if class_prompts_param is None:
        print("警告：未找到class_prompts参数")
        return False, None

    # 获取prompts数量
    cpt_num = class_prompts_param.shape[1]

    # 定义解冻阶段和每个阶段解冻的提示向量数量
    stages = {
        0: 0,  # 初始阶段：不解冻class_prompts
        int(total_epochs * 0.2): int(cpt_num * 0.25),  # 20%进度：解冻25%的prompts
        int(total_epochs * 0.4): int(cpt_num * 0.5),  # 40%进度：解冻50%的prompts
        int(total_epochs * 0.6): int(cpt_num * 0.75),  # 60%进度：解冻75%的prompts
        int(total_epochs * 0.8): cpt_num  # 80%进度：解冻全部prompts
    }

    # 初始阶段：冻结所有参数，只训练分类头
    if current_epoch == 0:
        print("初始阶段：冻结所有参数，只训练分类头和CLS")
        model.cls_token.requires_grad = True
        for name, param in model.named_parameters():
            if 'mlp_head' in name or 'head' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        updated = True

    # 按阶段解冻部分class_prompts
    elif current_epoch in stages and current_epoch > 0:
        # 获取当前阶段需要解冻的提示向量数量
        num_to_unfreeze = stages[current_epoch]

        if num_to_unfreeze > 0:
            # 创建渐进式解冻掩码
            # 我们需要保存原有的requires_grad状态
            old_requires_grad = class_prompts_param.requires_grad

            # 先将整个参数设置为不需要梯度
            class_prompts_param.requires_grad = False

            # 然后手动为部分提示向量创建梯度
            # 注意：这需要使用PyTorch的高级参数操作
            if hasattr(model, class_prompts_name):
                # 获取当前属性
                current_prompts = getattr(model, class_prompts_name.split('.')[-1])

                # 创建新的class_prompts参数，只有部分需要梯度
                new_prompts = nn.Parameter(current_prompts.clone())

                # 设置部分提示向量需要梯度
                # 注意：只解冻前num_to_unfreeze个提示向量
                if num_to_unfreeze < cpt_num:
                    new_prompts.requires_grad_(False)
                    new_prompts[:, :num_to_unfreeze, :].requires_grad_(True)
                else:
                    new_prompts.requires_grad_(True)

                # 更新模型中的参数
                setattr(model, class_prompts_name.split('.')[-1], new_prompts)

                print(
                    f"第{current_epoch}轮：解冻{num_to_unfreeze}/{cpt_num}个class_prompts ({num_to_unfreeze / cpt_num * 100:.1f}%)")
                updated = True

    # 如果状态已更新，创建新的优化器
    if updated:
        # 收集参数组
        prompts_params = [p for n, p in model.named_parameters()
                          if 'class_prompts' in n and p.requires_grad]
        head_params = [p for n, p in model.named_parameters()
                       if ('mlp_head' in n or 'head' in n) and p.requires_grad]

        # 设置不同学习率
        params = []
        if prompts_params:
            params.append({'params': prompts_params, 'lr': finetune_lr})
        if head_params:
            params.append({'params': head_params, 'lr': initial_lr})

        # 创建优化器
        optimizer = torch.optim.Adam(params, weight_decay=1e-5)

        # 输出当前可训练参数统计
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"可训练参数比例: {trainable}/{total} ({trainable / total * 100:.2f}%)")

        return True, optimizer

    return False, None