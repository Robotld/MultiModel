import os
import time
import numpy as np
import pandas as pd
import torch
from torch import optim
from tqdm import tqdm
from transformers import AutoTokenizer

from models import WarmupScheduler, MultitaskLoss
from utils import save_training_history, save_config, calculate_metrics
from utils.entity_masking import MedicalEntityMasker


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


def one_epoch_multimodal_train(
        model,
        data_loader,
        optimizer,
        criterion,
        device,
        train: bool = True,
        scaler=None,
        use_amp: bool = False,
        max_grad_norm: float = 1.0,
        entity_masker: MedicalEntityMasker = None,
        tokenizer=None,
        use_entity_decoder: bool = False,
        decoder_weight: float = 0.5,
        contrastive_weight: float = 1.0,  # 新增：对比学习权重，默认1.0，和模型里保持一致
):
    model.train() if train else model.eval()

    # 新增 contrastive 统计
    losses = {'total': 0.0, 'cls': 0.0, 'axis': 0.0, 'decoder': 0.0, 'contrastive': 0.0}
    all_probs, all_labels = [], []
    features_list = []

    loader = tqdm(data_loader, desc="训练中" if train else "评估中")

    with torch.set_grad_enabled(train):
        for batch_data in loader:
            images = batch_data["image"].to(device)
            text_inputs = {k: v.to(device) for k, v in batch_data["text"].items()}
            demographics = batch_data["demographic"].to(device)
            cls_labels = batch_data["label"].to(device)
            axis_labels = batch_data["nodule_axis_label"].to(device)

            masked_inputs, target_ids = None, None
            if use_entity_decoder and entity_masker is not None and tokenizer is not None:
                if "original_text" in batch_data:
                    original_texts = batch_data["original_text"]
                    # batch_mask 返回 (masked_texts, target_ids [B,K], spans [B,K,2])
                    masked_txt, target_ids_ts, _ = entity_masker.batch_mask(original_texts)

                    masked_inputs = tokenizer(
                        masked_txt,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors='pt'
                    )
                    masked_inputs = {k: v.to(device) for k, v in masked_inputs.items()}
                    target_ids = target_ids_ts.to(device)  # [B, K]

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(
                    images,
                    text_inputs,
                    demographics,
                    masked_text_inputs=masked_inputs,
                    target_entity_ids=target_ids,
                )

                logits = outputs['logits']
                axis_preds = outputs['axis_preds']

                main_loss, cls_loss_item, axis_loss_item = criterion(
                    logits,
                    axis_preds,
                    cls_labels,
                    axis_labels
                )

                contrastive_loss = outputs['contrastive_loss']  # 模型中已经算好
                decoder_loss = outputs['decoder_loss']

                # 组合 loss: 主任务 + 对比学习 + decoder 辅助任务
                loss = (
                    main_loss
                    + contrastive_weight * contrastive_loss
                    + decoder_weight * decoder_loss
                )

            if train:
                optimizer.zero_grad()
                if scaler is not None:
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

            # ====== 统计各项 loss ======
            losses['total'] += float(loss.item())
            losses['cls'] += float(cls_loss_item.item())
            if isinstance(axis_loss_item, torch.Tensor):
                losses['axis'] += float(axis_loss_item.item())
            if isinstance(decoder_loss, torch.Tensor):
                losses['decoder'] += float(decoder_loss.item())
            if isinstance(contrastive_loss, torch.Tensor):
                losses['contrastive'] += float(contrastive_loss.item())

            # ====== 记录特征用于分析 ======
            if outputs.get('fusion_feature', None) is not None:
                features_list.append(outputs['fusion_feature'].detach().cpu().numpy())

            # ====== 分类概率 & 标签 ======
            valid_mask = (cls_labels != -1)
            if valid_mask.any():
                probs = torch.softmax(logits[valid_mask], dim=1).detach().cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(cls_labels[valid_mask].cpu().numpy())

    num_batches = len(data_loader)
    avg_loss = {k: v / num_batches for k, v in losses.items()}

    if features_list:
        features_concat = np.vstack(features_list)
        print(
            f"[{'Train' if train else 'Val'}] 特征分布: "
            f"Mean={np.mean(features_concat):.4f}, Std={np.std(features_concat):.4f}"
        )

    metrics_dict = {
        'probabilities': np.array(all_probs),
        'labels': np.array(all_labels) if all_labels else np.array([]),
    }

    loss_dict = {
        'total_loss': avg_loss['total'],
        'cls_loss': avg_loss['cls'],
        'axis_loss': avg_loss['axis'],
        'decoder_loss': avg_loss['decoder'],
        'contrastive_loss': avg_loss['contrastive'],
    }

    return avg_loss['total'], metrics_dict, loss_dict


def train(
        model,
        tokenizer,
        train_loader,
        val_loader,
        config,
        fold,
        device,
        args,
        warmup_epochs: int = 3,
        warmup_type: str = 'linear',
        max_grad_norm: float = 1.0,
        train_dir: str = None,
        best_f1: float = 0.0,
        best_auc: float = 0.0,
        use_entity_decoder: bool = True,
        decoder_weight: float = 0.5,
        contrastive_weight: float = 0.5,  # 新增：全局控制对比损失权重
):
    os.makedirs(train_dir, exist_ok=True)
    save_config(config, os.path.join(train_dir, 'config.yaml'))

    entity_masker = None
    if use_entity_decoder:
        entity_masker = MedicalEntityMasker(mask_ratio_choices=[0.4])
        print(f"✅ 启用实体序列生成 decoder，遮挡比例: {entity_masker.mask_ratio_choices}")

    criterion = MultitaskLoss(
        classification_weight=config.losses['MultitaskLoss']['classification_weight'],
        axis_weight=config.losses['MultitaskLoss']['axis_weight'],
    )

    optimizer = getattr(optim, config.optimizer["name"])(
        model.parameters(),
        **config.get_optimizer_params(model)
    )
    base_scheduler = getattr(optim.lr_scheduler, config.scheduler["name"])(
        optimizer,
        **config.get_scheduler_params()
    )
    scheduler = WarmupScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        base_scheduler=base_scheduler,
        warmup_type=warmup_type
    )

    use_amp = (device.type == 'cuda') and getattr(args, 'use_amp', False)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("启用混合精度训练 (AMP)")

    print(f"验证集样本数: {len(val_loader.dataset)}")

    train_metrics_history, val_metrics_history = [], []
    best_val_metrics, best_train_metrics = None, None
    best_val_auc, best_val_f1 = 0.0, 0.0
    best_f1_model_state, best_auc_model_state = None, None

    patience = getattr(args, 'patience', 10)
    patience_counter = 0

    for epoch in range(args.epochs):
        start_time = time.time()

        train_loss, train_metrics_raw, train_loss_dict = one_epoch_multimodal_train(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            train=True,
            scaler=scaler,
            use_amp=use_amp,
            max_grad_norm=max_grad_norm,
            entity_masker=entity_masker,
            tokenizer=tokenizer,
            use_entity_decoder=use_entity_decoder,
            decoder_weight=decoder_weight,
            contrastive_weight=contrastive_weight,
        )

        val_loss, val_metrics_raw, val_loss_dict = one_epoch_multimodal_train(
            model=model,
            data_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            train=False,
            scaler=scaler,
            use_amp=use_amp,
            max_grad_norm=max_grad_norm,
            entity_masker=None,
            tokenizer=tokenizer,
            use_entity_decoder=False,  # 验证时不计算 decoder loss，但仍然有 contrastive_loss（如果你想关掉也可以设权重为0）
            decoder_weight=0,
            contrastive_weight=contrastive_weight,
        )

        train_metrics = calculate_metrics(train_metrics_raw, config.data['num_classes'])
        val_metrics = calculate_metrics(val_metrics_raw, config.data['num_classes'])

        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        train_metrics_history.append(train_metrics)
        val_metrics_history.append(val_metrics)

        print(
            f'\nFold {fold + 1}, Epoch {epoch + 1}/{args.epochs} '
            f'- {epoch_time:.2f}s, LR: {current_lr:.6e}'
        )
        print(
            f" Train | Loss: {train_loss:.4f} "
            f"| AUC: {train_metrics['auc']:.4f} "
            f"| F1: {train_metrics['micro_f1']:.4f}"
        )
        print(
            f" Val   | Loss: {val_loss:.4f} "
            f"| AUC: {val_metrics['auc']:.4f} "
            f"| F1: {val_metrics['micro_f1']:.4f}"
        )

        if use_entity_decoder:
            print(
                f" Decoder | Train Loss: {train_loss_dict['decoder_loss']:.4f} "
                f"| Val Loss: {val_loss_dict['decoder_loss']:.4f}"
            )

        print(
            f" Contrastive | Train Loss: {train_loss_dict['contrastive_loss']:.4f} "
            f"| Val Loss: {val_loss_dict['contrastive_loss']:.4f}"
        )

        # 早停 & best model
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

        if best_train_metrics is None or train_metrics['auc'] > best_train_metrics.get('auc', 0):
            best_train_metrics = train_metrics.copy()

        scheduler.step()

        if patience_counter >= patience:
            print(f"早停触发：验证集 AUC 未提升持续 {patience} 轮")
            break

    if best_val_metrics is not None:
        best_dir = os.path.join(train_dir, 'best_metrics')
        save_metrics_to_csv(best_val_metrics, os.path.join(best_dir, f'best_val_metrics_fold_{fold}.csv'))
        if best_train_metrics is not None:
            save_metrics_to_csv(best_train_metrics, os.path.join(best_dir, f'best_train_metrics_fold_{fold}.csv'))

    save_training_history(train_dir, fold, train_metrics_history, val_metrics_history)

    return best_val_f1, best_val_auc, best_f1_model_state, best_auc_model_state