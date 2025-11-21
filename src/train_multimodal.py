import os
import time
import numpy as np
import pandas as pd
import torch
from torch import optim
from tqdm import tqdm
from models import WarmupScheduler
from sklearn.metrics import roc_curve, auc, confusion_matrix
from models import MultitaskLoss
from utils import save_training_history, save_config, calculate_metrics
from utils.Visualizer import *
from utils.entity_masking import MedicalEntityMasker  # 新增：导入遮挡工具
from transformers import BertTokenizer  # 新增：导入tokenizer


def save_best_model_if_improved(model, metric_value, metric_name, train_dir, epoch, fold):
    print(f"\n保存新的最佳模型，{metric_name}: {metric_value:.4f}，在第{epoch + 1}轮")
    save_path = f'{train_dir}/best_{metric_name}_model_{fold}.pth'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    return save_path


def save_metrics_to_csv(metrics_dict, file_path):
    """
    保存指标字典到CSV文件，确保大型数组不被截断
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    processed_dict = {}
    for key, value in metrics_dict.items():
        if isinstance(value, np.ndarray):
            processed_dict[key] = value.tolist()  # 转为list
        else:
            processed_dict[key] = value

    # 创建DataFrame
    df = pd.DataFrame([processed_dict])
    # 保存为CSV，设置大字段限制
    df.to_csv(file_path, index=False, float_format='%.6f')

    print(f"指标已保存至: {file_path}")
    return metrics_dict


def one_epoch_multimodal_train(model, data_loader, optimizer, criterion, device,
                               train=True, scaler=None, use_amp=False, max_grad_norm=1.0,
                               entity_masker=None, tokenizer=None, use_masked_prediction=False):
    """
    训练或评估一个epoch（适配主模型和损失函数）
    新增：遮挡关键词预测功能
    """
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    recurrence_loss_total = 0.0
    axis_loss_total = 0.0
    similarity_loss_total = 0.0
    masked_entity_loss_total = 0.0  # 新增：遮挡预测损失
    masked_entity_acc_total = 0.0  # 新增：遮挡预测准确率

    all_rec_probs = []
    all_rec_labels = []
    all_axis_preds = []
    all_axis_labels = []

    loader = tqdm(data_loader, desc=f"{'训练中' if train else '评估中'}")
    features = []

    for batch_data in loader:
        # 解包多模态数据
        ct_number = batch_data["ct_number"]
        images = batch_data["image"].to(device)
        text_inputs = {k: v.to(device) for k, v in batch_data["text"].items()}
        demographics = batch_data["demographic"].to(device)
        recurrence_labels = batch_data["label"].to(device)
        axis_labels = batch_data["nodule_axis_label"].to(device)

        # ========== 新增：生成遮挡文本（训练时） ==========
        masked_text_inputs = None
        target_entity_ids = None
        if use_masked_prediction and entity_masker is not None and tokenizer is not None:
            # 获取原始文本（需要从batch_data中提取）
            original_texts = batch_data.get("original_text", None)  # 假设Dataset返回原始文本

            if original_texts is not None:
                # 批量遮挡
                masked_texts, target_ids_tensor = entity_masker.batch_mask(original_texts)

                # Tokenize遮挡后的文本
                masked_text_inputs = tokenizer(
                    masked_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                masked_text_inputs = {k: v.to(device) for k, v in masked_text_inputs.items()}
                target_entity_ids = target_ids_tensor.to(device)
        # ================================================

        if train:
            optimizer.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    # 前向传播（传入遮挡文本）
                    outputs = model(
                        images, text_inputs, demographics,
                        masked_text_inputs=masked_text_inputs,  # 新增
                        target_entity_ids=target_entity_ids  # 新增
                    )

                    # 解包输出
                    recurrence_logits = outputs['recurrence_logits']
                    axis_preds = outputs['axis_preds']
                    similarity_loss = outputs['similarity_loss']
                    contrastive_loss = outputs['contrastive_loss']
                    masked_entity_loss = outputs['masked_entity_loss']  # 新增

                    # 计算损失
                    loss, rec_loss, axis_loss, sim_loss = criterion(
                        recurrence_logits, axis_preds, recurrence_labels, axis_labels, similarity_loss
                    )
                    loss = loss + contrastive_loss + masked_entity_loss  # 新增遮挡损失

                scaler.scale(loss).backward()
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                # 前向传播（传入遮挡文本）
                outputs = model(
                    images, text_inputs, demographics,
                    masked_text_inputs=masked_text_inputs,  # 新增
                    target_entity_ids=target_entity_ids  # 新增
                )

                # 解包输出
                recurrence_logits = outputs['recurrence_logits']
                axis_preds = outputs['axis_preds']
                similarity_loss = outputs['similarity_loss']
                contrastive_loss = outputs['contrastive_loss']
                masked_entity_loss = outputs['masked_entity_loss']  # 新增
                predicted_entities = outputs['predicted_entities']  # 新增
                feature = outputs['fusion_feature']

                # 计算损失
                loss, rec_loss, axis_loss, sim_loss = criterion(
                    recurrence_logits, axis_preds, recurrence_labels, axis_labels, similarity_loss
                )
                loss = loss + contrastive_loss + masked_entity_loss  # 新增遮挡损失

                loss.backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
        else:
            with torch.no_grad():
                # 前向传播（传入遮挡文本）
                outputs = model(
                    images, text_inputs, demographics,
                    masked_text_inputs=masked_text_inputs,
                    target_entity_ids=target_entity_ids
                )

                recurrence_logits = outputs['recurrence_logits']
                axis_preds = outputs['axis_preds']
                similarity_loss = outputs['similarity_loss']
                contrastive_loss = outputs['contrastive_loss']
                masked_entity_loss = outputs['masked_entity_loss']
                predicted_entities = outputs['predicted_entities']
                feature = outputs['fusion_feature']

                loss, rec_loss, axis_loss, sim_loss = criterion(
                    recurrence_logits, axis_preds, recurrence_labels, axis_labels, similarity_loss
                )
                loss = loss + contrastive_loss + masked_entity_loss

        # ========== 新增：计算遮挡预测准确率 ==========
        if use_masked_prediction and predicted_entities is not None and target_entity_ids is not None:
            valid_mask = (target_entity_ids[:, 0] != -1)  # 只看第一个被遮挡的实体
            if valid_mask.sum() > 0:
                correct = (predicted_entities[valid_mask] == target_entity_ids[:, 0][valid_mask]).float()
                masked_entity_acc = correct.mean().item()
                masked_entity_acc_total += masked_entity_acc
        # ==========================================

        features.append(feature.detach().cpu().numpy())
        total_loss += loss.item()
        recurrence_loss_total += rec_loss.item()
        similarity_loss_total += similarity_loss.item() if isinstance(similarity_loss,
                                                                      torch.Tensor) else similarity_loss
        masked_entity_loss_total += masked_entity_loss.item() if isinstance(masked_entity_loss, torch.Tensor) else 0

        # 分类预测、概率、标签
        valid_rec_mask = (recurrence_labels != -1)
        if valid_rec_mask.sum() > 0:
            rec_probs = torch.softmax(recurrence_logits[valid_rec_mask], dim=1).detach().cpu().numpy()
            all_rec_probs.extend(rec_probs)
            all_rec_labels.extend(recurrence_labels[valid_rec_mask].cpu().numpy())

    features = np.vstack(features)
    print(f"特征分布 - 均值: {np.mean(features):.4f}, 标准差: {np.std(features):.4f}")

    total_loss /= len(data_loader)
    recurrence_loss_total /= len(data_loader)
    axis_loss_total /= len(data_loader)
    similarity_loss_total /= len(data_loader)
    masked_entity_loss_total /= len(data_loader)
    masked_entity_acc_total /= len(data_loader)

    metrics_dict = {
        'probabilities': np.array(all_rec_probs),
        'labels': np.array(all_rec_labels) if all_rec_labels else np.array([]),
    }

    loss_dict = {
        'total_loss': total_loss,
        'recurrence_loss': recurrence_loss_total,
        'similarity_loss': similarity_loss_total,
        'masked_entity_loss': masked_entity_loss_total,  # 新增
        'masked_entity_acc': masked_entity_acc_total  # 新增
    }

    return total_loss, metrics_dict, similarity_loss_total, loss_dict  # 修改返回值


def train(model, train_loader, val_loader, config, fold, device, args,
          warmup_epochs=3, warmup_type='linear', max_grad_norm=1.0, train_dir=None,
          best_f1=0, best_auc=0, use_masked_prediction=False):  # 新增参数
    """
    多模态模型训练主循环（适配联合任务：复发分类+长短径回归+遮挡预测）
    """

    os.makedirs(train_dir, exist_ok=True)
    config_save_path = os.path.join(train_dir, 'config.yaml')
    save_config(config, config_save_path)

    # ========== 新增：初始化遮挡工具 ==========
    entity_masker = None
    tokenizer = None
    if use_masked_prediction:
        entity_masker = MedicalEntityMasker(mask_ratio=0.5)
        tokenizer = BertTokenizer.from_pretrained(
            config.model.get('bert_model_name', 'hfl/chinese-roberta-wwm-ext')
        )
        print(f"✅ 启用遮挡关键词预测，实体词表大小: {entity_masker.vocab_size}")
    # =========================================

    criterion = MultitaskLoss(
        recurrence_weight=config.losses['MultitaskLoss']['recurrence_weight'],
        axis_weight=config.losses['MultitaskLoss']['axis_weight'],
    )

    optimizer = getattr(optim, config.optimizer["name"])(
        model.parameters(), **config.get_optimizer_params(model)
    )

    base_scheduler = getattr(optim.lr_scheduler, config.scheduler["name"])(
        optimizer, **config.get_scheduler_params()
    )

    scheduler = WarmupScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        base_scheduler=base_scheduler,
        warmup_type=warmup_type
    )

    print(f"启用学习率预热: {warmup_epochs} 个epoch, 类型: {warmup_type}")
    print(f"启用梯度裁剪，最大范数: {max_grad_norm}")

    use_amp = device.type == 'cuda' and args.use_amp
    scaler = None
    if use_amp:
        scaler = torch.cuda.amp.GradScaler(
            init_scale=2. ** 16,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000
        )
        print("启用混合精度训练")

    # 缓存验证集（代码保持不变）
    print("缓存验证数据到GPU内存以加速评估...")
    val_data_cached = []
    val_size = len(val_loader.dataset)
    cache_threshold = getattr(args, 'cache_threshold', 10000)
    should_cache = val_size <= cache_threshold or getattr(args, 'force_cache_val', False)

    if should_cache:
        try:
            for batch_data in tqdm(val_loader, desc="缓存验证数据"):
                cached_batch = {
                    "ct_number": batch_data["ct_number"],
                    "image": batch_data["image"].to(device),
                    "text": batch_data["text"],
                    "demographic": batch_data["demographic"].to(device),
                    "label": batch_data["label"],
                    "nodule_axis_label": batch_data["nodule_axis_label"].to(device),
                    "original_text": batch_data.get("original_text", None)  # 新增：保留原始文本
                }
                val_data_cached.append(cached_batch)
            print(f"已缓存全部 {val_size} 个验证样本")
        except RuntimeError as e:
            print(f"缓存验证数据时出错（可能是GPU内存不足）: {e}")
            print("将直接使用验证数据加载器")
            val_data_cached = val_loader
    else:
        print(f"验证集样本数 {val_size} 超过阈值 {cache_threshold}，跳过缓存以节省内存")
        val_data_cached = val_loader

    train_metrics_history = []
    val_metrics_history = []
    best_train_metrics = None
    best_val_metrics = None
    best_val_auc = 0
    best_val_f1 = 0
    best_train_auc = 0
    best_f1_model_state = None
    best_auc_model_state = None

    patience = getattr(args, 'patience', 5)
    patience_counter = 0
    last_best_epoch = 0

    for epoch in range(args.epochs):
        start_time = time.time()

        # 训练
        train_loss, train_metrics, train_sim_loss, train_loss_dict = one_epoch_multimodal_train(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            train=True,
            scaler=scaler,
            use_amp=use_amp,
            max_grad_norm=max_grad_norm,
            entity_masker=entity_masker,  # 新增
            tokenizer=tokenizer,  # 新增
            use_masked_prediction=use_masked_prediction  # 新增
        )

        # 验证
        val_loss, val_metrics, val_sim_loss, val_loss_dict = one_epoch_multimodal_train(
            model=model,
            data_loader=val_data_cached,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            train=False,
            scaler=scaler,
            use_amp=use_amp,
            max_grad_norm=max_grad_norm,
            entity_masker=entity_masker,  # 新增
            tokenizer=tokenizer,  # 新增
            use_masked_prediction=use_masked_prediction  # 新增
        )

        current_lr = optimizer.param_groups[0]['lr']
        if epoch == warmup_epochs - 1:
            print("\n预热阶段结束，切换到基础学习率调度器")
        epoch_time = time.time() - start_time

        # 计算指标
        val_metrics = calculate_metrics(val_metrics, config.data['num_classes'])
        train_metrics = calculate_metrics(train_metrics, config.data['num_classes'])

        improved = False
        if val_metrics['auc'] > best_val_auc:
            best_val_metrics = val_metrics.copy()
            best_val_auc = val_metrics['auc']
            best_auc_model_state = model.state_dict().copy()
            last_best_epoch = epoch
            patience_counter = 0
            improved = True
            save_best_model_if_improved(model, val_metrics['auc'], "auc", train_dir, epoch, fold)
        else:
            patience_counter += 1

        if val_metrics['micro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['micro_f1']
            best_f1_model_state = model.state_dict().copy()

        if train_metrics['auc'] > best_train_auc:
            best_train_auc = train_metrics['auc']
            best_train_metrics = train_metrics.copy()

        train_metrics_history.append(train_metrics)
        val_metrics_history.append(val_metrics)

        # ========== 新增：打印遮挡预测指标 ==========
        print(f'\nFold {fold + 1}, Epoch {epoch + 1}/{args.epochs} - 耗时: {epoch_time:.2f}s, LR: {current_lr:.6f}')
        print(f' Train Loss: {train_loss:.4f}, ACC: {train_metrics["accuracy"]:.4f} AUC: {train_metrics["auc"]:.4f}')
        print(f'   Val Loss: {val_loss:.4f}, ACC: {val_metrics["accuracy"]:.4f} AUC: {val_metrics["auc"]:.4f}')
        print(f'train F1:{train_metrics["micro_f1"]:.4f}')
        print(f'Best Val F1: {best_val_f1:.4f}, Best Val AUC: {best_val_auc:.4f}')

        if use_masked_prediction:
            print(f'\n--- 遮挡预测指标 ---')
            print(
                f'Train Masked Loss: {train_loss_dict["masked_entity_loss"]:.4f}, Acc: {train_loss_dict["masked_entity_acc"]:.4f}')
            print(
                f'Val   Masked Loss: {val_loss_dict["masked_entity_loss"]:.4f}, Acc: {val_loss_dict["masked_entity_acc"]:.4f}')
        # ==========================================

        scheduler.step()

    # 保存最佳指标
    if best_val_metrics:
        best_metrics_file = os.path.join(train_dir, f'best_metrics')
        os.makedirs(best_metrics_file, exist_ok=True)
        save_metrics_to_csv(best_val_metrics, best_metrics_file + f'/best_val_metrics_fold_{fold}.csv')
        save_metrics_to_csv(best_train_metrics, best_metrics_file + f'/best_train_metrics_fold_{fold}.csv')

    save_training_history(train_dir, fold, train_metrics_history, val_metrics_history)

    return best_val_f1, best_val_auc, best_f1_model_state, best_auc_model_state