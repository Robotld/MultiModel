import os
import time
import numpy as np
import pandas as pd
import torch
from torch import optim
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, confusion_matrix
from transformers import BertTokenizer
# 假设这些是从您的项目中导入的
from models import WarmupScheduler, MultitaskLoss
from utils import save_training_history, save_config, calculate_metrics
from utils.entity_masking import MedicalEntityMasker


# --- Helper Functions 保持不变 ---
def save_best_model_if_improved(model, metric_value, metric_name, train_dir, epoch, fold):
    print(f"\n保存新的最佳模型，{metric_name}: {metric_value:.4f}，在第{epoch + 1}轮")
    save_path = f'{train_dir}/best_{metric_name}_model_{fold}.pth'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    return save_path


def save_metrics_to_csv(metrics_dict, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    processed_dict = {}
    for key, value in metrics_dict.items():
        processed_dict[key] = value.tolist() if isinstance(value, np.ndarray) else value
    df = pd.DataFrame([processed_dict])
    df.to_csv(file_path, index=False, float_format='%.6f')
    return metrics_dict


# --- 核心重构部分 ---

def one_epoch_multimodal_train(model, data_loader, optimizer, criterion, device,
                               train=True, scaler=None, use_amp=False, max_grad_norm=1.0,
                               entity_masker=None, tokenizer=None, use_masked_prediction=False):
    """
    重构后的单 Epoch 训练/评估函数：
    1. 统一了 Train/Eval 的前向传播流程，消除冗余。
    2. 优化了 AMP 和 梯度裁剪 的逻辑。
    """
    model.train() if train else model.eval()

    # 初始化统计变量
    losses = {'total': 0.0, 'rec': 0.0, 'axis': 0.0, 'masked': 0.0, 'masked_acc': 0.0}
    all_probs, all_labels = [], []
    features_list = []

    loader = tqdm(data_loader, desc=f"{'训练中' if train else '评估中'}")

    # 使用 set_grad_enabled 统一上下文，只在 train 时开启梯度计算
    with torch.set_grad_enabled(train):
        for batch_data in loader:
            # 1. 数据搬运
            images = batch_data["image"].to(device)
            text_inputs = {k: v.to(device) for k, v in batch_data["text"].items()}
            demographics = batch_data["demographic"].to(device)
            rec_labels = batch_data["label"].to(device)
            axis_labels = batch_data["nodule_axis_label"].to(device)

            # 2. 动态遮挡处理 (仅在需要时)
            masked_inputs, target_ids = None, None
            if use_masked_prediction and entity_masker and tokenizer:
                # 注意：通常验证集也可以计算此 Loss 作为监控，但如果不想在验证时 Mask，可加 if train: 判断
                if "original_text" in batch_data:
                    masked_txt, target_ids_ts = entity_masker.batch_mask(batch_data["original_text"])
                    masked_inputs = tokenizer(masked_txt, padding=True, truncation=True, max_length=512,
                                              return_tensors='pt')
                    masked_inputs = {k: v.to(device) for k, v in masked_inputs.items()}
                    target_ids = target_ids_ts.to(device)

            # 3. 前向传播 (封装 AMP 上下文)
            # 如果不使用 AMP，autocast 上下文没有任何副作用，可以安全包裹
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(
                    images, text_inputs, demographics,
                    masked_text_inputs=masked_inputs,
                    target_entity_ids=target_ids
                )

                rec_logits = outputs['recurrence_logits']
                axis_preds = outputs['axis_preds']
                # 获取各项 Loss
                main_loss, rec_loss_item, axis_loss_item = criterion(rec_logits, axis_preds, rec_labels, axis_labels)

                # 组合 Loss
                loss = main_loss + outputs['contrastive_loss'] + outputs['masked_entity_loss']

            # 4. 反向传播与优化 (仅训练)
            if train:
                optimizer.zero_grad()
                if scaler:
                    scaler.scale(loss).backward()
                    if max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

            # 5. 统计与记录
            batch_size = images.size(0)  # 实际上 len(loader) 已经足够，这里简单累加
            losses['total'] += loss.item()
            losses['rec'] += rec_loss_item.item()
            losses['masked'] += outputs['masked_entity_loss'].item() if isinstance(outputs['masked_entity_loss'],
                                                                                   torch.Tensor) else 0

            # 特征收集
            if outputs['fusion_feature'] is not None:
                features_list.append(outputs['fusion_feature'].detach().cpu().numpy())

            # 分类指标收集
            valid_mask = (rec_labels != -1)
            if valid_mask.any():
                probs = torch.softmax(rec_logits[valid_mask], dim=1).detach().cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(rec_labels[valid_mask].cpu().numpy())

            # 遮挡准确率
            if use_masked_prediction and outputs['predicted_entities'] is not None and target_ids is not None:
                valid_ent = (target_ids[:, 0] != -1)
                if valid_ent.any():
                    acc = (outputs['predicted_entities'][valid_ent] == target_ids[:, 0][
                        valid_ent]).float().mean().item()
                    losses['masked_acc'] += acc

    # 6. Epoch 结束汇总
    num_batches = len(data_loader)
    avg_loss = {k: v / num_batches for k, v in losses.items()}

    if features_list:
        features_concat = np.vstack(features_list)
        print(
            f"[{'Train' if train else 'Val'}] 特征分布: Mean={np.mean(features_concat):.4f}, Std={np.std(features_concat):.4f}")

    metrics_dict = {
        'probabilities': np.array(all_probs),
        'labels': np.array(all_labels) if all_labels else np.array([]),
    }

    return avg_loss['total'], metrics_dict, {
        'total_loss': avg_loss['total'],
        'recurrence_loss': avg_loss['rec'],
        'masked_entity_loss': avg_loss['masked'],
        'masked_entity_acc': avg_loss['masked_acc']
    }


def train(model, train_loader, val_loader, config, fold, device, args,
          warmup_epochs=3, warmup_type='linear', max_grad_norm=1.0, train_dir=None,
          best_f1=0, best_auc=0, use_masked_prediction=True):
    os.makedirs(train_dir, exist_ok=True)
    save_config(config, os.path.join(train_dir, 'config.yaml'))

    # 初始化 Mask 工具
    entity_masker, tokenizer = None, None
    if use_masked_prediction:
        entity_masker = MedicalEntityMasker(mask_ratio=0.5)
        tokenizer = BertTokenizer.from_pretrained(config.model.get('bert_model_name', 'hfl/chinese-roberta-wwm-ext'))
        print(f"✅ 启用遮挡关键词预测")

    criterion = MultitaskLoss(
        recurrence_weight=config.losses['MultitaskLoss']['recurrence_weight'],
        axis_weight=config.losses['MultitaskLoss']['axis_weight'],
    )

    optimizer = getattr(optim, config.optimizer["name"])(model.parameters(), **config.get_optimizer_params(model))
    base_scheduler = getattr(optim.lr_scheduler, config.scheduler["name"])(optimizer, **config.get_scheduler_params())
    scheduler = WarmupScheduler(optimizer, warmup_epochs=warmup_epochs, base_scheduler=base_scheduler,
                                warmup_type=warmup_type)

    use_amp = (device.type == 'cuda') and getattr(args, 'use_amp', False)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp: print("启用混合精度训练 (AMP)")

    # ⚠️ 关键修改：移除了极其危险的 "验证集全量缓存到GPU" 逻辑 ⚠️
    # 显存应该只存当前 Batch，不要存整个 Dataset
    print(f"验证集样本数: {len(val_loader.dataset)} (不进行显存预缓存)")

    train_metrics_history, val_metrics_history = [], []
    best_val_metrics, best_train_metrics = None, None
    best_val_auc, best_val_f1 = 0, 0
    best_f1_model_state, best_auc_model_state = None, None

    # 早停机制
    patience = getattr(args, 'patience', 10)
    patience_counter = 0

    for epoch in range(args.epochs):
        start_time = time.time()

        # --- Train ---
        train_loss, train_metrics, train_loss_dict = one_epoch_multimodal_train(
            model, train_loader, optimizer, criterion, device,
            train=True, scaler=scaler, use_amp=use_amp, max_grad_norm=max_grad_norm,
            entity_masker=entity_masker, tokenizer=tokenizer, use_masked_prediction=True
        )

        # --- Val ---
        val_loss, val_metrics, val_loss_dict = one_epoch_multimodal_train(
            model, val_loader, optimizer, criterion, device,
            train=False, scaler=scaler, use_amp=use_amp, max_grad_norm=max_grad_norm,
            entity_masker=entity_masker, tokenizer=tokenizer, use_masked_prediction=False
        )

        # 计算指标
        train_metrics = calculate_metrics(train_metrics, config.data['num_classes'])
        val_metrics = calculate_metrics(val_metrics, config.data['num_classes'])

        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        # 记录历史
        train_metrics_history.append(train_metrics)
        val_metrics_history.append(val_metrics)

        # 打印日志
        print(f'\nFold {fold + 1}, Epoch {epoch + 1}/{args.epochs} - {epoch_time:.2f}s, LR: {current_lr:.6e}')
        print(
            f" Train | Loss: {train_loss:.4f} | AUC: {train_metrics['auc']:.4f} | F1: {train_metrics['micro_f1']:.4f}")
        print(f" Val   | Loss: {val_loss:.4f} | AUC: {val_metrics['auc']:.4f} | F1: {val_metrics['micro_f1']:.4f}")

        if use_masked_prediction:
            print(
                f" Mask  | Train Acc: {train_loss_dict['masked_entity_acc']:.4f} | Val Acc: {val_loss_dict['masked_entity_acc']:.4f}")

        # 保存最佳模型逻辑
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_val_metrics = val_metrics.copy()
            best_auc_model_state = model.state_dict().copy()
            save_best_model_if_improved(model, val_metrics['auc'], "auc", train_dir, epoch, fold)
            patience_counter = 0
        else:
            patience_counter += 1

        if val_metrics['micro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['micro_f1']
            best_f1_model_state = model.state_dict().copy()

        if train_metrics['auc'] > getattr(best_train_metrics, 'get', lambda k, v: 0)('auc', 0):  # safe check
            best_train_metrics = train_metrics.copy()

        scheduler.step()

        # Early Stopping check could go here
        if patience_counter >= patience:
            print(f"早停触发：验证集 AUC 未提升持续 {patience} 轮")
            break

    # 保存最终结果
    if best_val_metrics:
        best_dir = os.path.join(train_dir, 'best_metrics')
        save_metrics_to_csv(best_val_metrics, os.path.join(best_dir, f'best_val_metrics_fold_{fold}.csv'))
        if best_train_metrics:
            save_metrics_to_csv(best_train_metrics, os.path.join(best_dir, f'best_train_metrics_fold_{fold}.csv'))

    save_training_history(train_dir, fold, train_metrics_history, val_metrics_history)

    return best_val_f1, best_val_auc, best_f1_model_state, best_auc_model_state