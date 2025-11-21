
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve,
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_recall_fscore_support, classification_report
)

import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve

def fit_ovr_thresholds(probs, y, method="youden"):
    # probs: [N, C], y: [N], C=3
    C = probs.shape[1]
    th = np.zeros(C)
    for c in range(C):
        y_bin = (y == c).astype(int)
        p = probs[:, c]
        if method == "youden":
            fpr, tpr, t = roc_curve(y_bin, p)
            idx = np.argmax(tpr - fpr); th[c] = t[idx]
        elif method == "f1":
            prec, rec, t = precision_recall_curve(y_bin, p)
            f1 = 2*prec[:-1]*rec[:-1]/np.clip(prec[:-1]+rec[:-1], 1e-12, None)
            th[c] = t[np.argmax(f1)]
        else:
            th[c] = 0.5
    return th

def predict_with_thresholds(probs, th, favor_class0=True):
    # 若至少一个类满足 p_i >= th_i，选 (p_i - th_i) 最大的类
    # 否则回退为普通 argmax；可选地对类0再加一点偏置
    margin = probs - th[None, :]
    mask = (probs >= th[None, :])
    pred = np.where(mask.any(axis=1), np.argmax(np.where(mask, margin, -1e9), axis=1),
                    np.argmax(probs, axis=1))
    if favor_class0:
        # 轻微偏置类0：相当于把类0阈值再“降一点”
        bias = np.array([0.02, 0.0, 0.0])
        pred = np.argmax(probs + bias[None, :], axis=1)
    return pred


def calculate_metrics(metrics_dict, num_classes, threshold_method='argmax', ovr_th=None):
    """
    计算分类（多分类）与回归（如长短径）的评估指标，添加了PPV、NPV、敏感性和特异性

    metrics_dict: 需包含
        - 'probabilities': (N, num_classes) 分类概率
        - 'labels': (N,) 分类标签
        - 'axis_preds': (N, 2) 长短径预测值
        - 'axis_labels': (N, 2) 长短径真实值
    """
    # --- 分类指标 ---
    if 'probabilities' in metrics_dict and 'labels' in metrics_dict:
        probs = metrics_dict['probabilities']  # (N, num_classes)形状
        labels = metrics_dict['labels']

        if threshold_method == 'argmax':
            predictions = np.argmax(probs, axis=1)
        elif threshold_method == 'ovr_threshold':
            if ovr_th is None:
                ovr_th = fit_ovr_thresholds(probs, labels, method='youden')  # or 'f1'
            # print("最佳阈值",ovr_th)
            predictions = predict_with_thresholds(probs, ovr_th, favor_class0=True)
        else:
            predictions = np.argmax(probs, axis=1)

        # 多分类混淆矩阵
        cm = confusion_matrix(labels, predictions)

        # 计算多分类评估指标
        accuracy = accuracy_score(labels, predictions)

        # 计算每个类别的precision, recall, f1
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None
        )

        # 宏平均和微平均
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro'
        )

        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            labels, predictions, average='micro'
        )

        # 多分类AUC (one-vs-rest方式)
        if num_classes > 2:
            auc_scores = []
            for i in range(num_classes):
                # 将当前类别视为正类，其他类别视为负类
                binary_labels = (labels == i).astype(int)
                auc_scores.append(roc_auc_score(binary_labels, probs[:, i]))
            auc_score = np.mean(auc_scores)  # 平均所有类别的AUC
        else:
            auc_score = roc_auc_score(labels, probs[:, 1]) if len(np.unique(labels)) > 1 else 0

        # 添加PPV、NPV、敏感性和特异性的计算
        # 对于每个类别分别计算（一对其余策略）
        class_ppv = []  # 对应precision
        class_npv = []
        class_sensitivity = []  # 对应recall
        class_specificity = []

        for i in range(num_classes):
            # 二分类混淆矩阵计算（当前类为正类，其他类为负类）
            binary_labels = (labels == i).astype(int)
            binary_preds = (predictions == i).astype(int)

            tn, fp, fn, tp = confusion_matrix(binary_labels, binary_preds).ravel()

            # PPV (Precision)
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            class_ppv.append(ppv)

            # NPV
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            class_npv.append(npv)

            # 敏感性 (Recall/Sensitivity)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            class_sensitivity.append(sensitivity)

            # 特异性 (Specificity)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            class_specificity.append(specificity)

        # 计算宏平均和微平均（可选）
        macro_ppv = np.mean(class_ppv)
        macro_npv = np.mean(class_npv)
        macro_sensitivity = np.mean(class_sensitivity)
        macro_specificity = np.mean(class_specificity)

        # 更新指标字典
        metrics_dict.update({
            'prob': probs,
            'labels': labels,
            'predictions': predictions,
            'confusion_matrix': cm,
            'accuracy': accuracy,
            'class_precision': precision.tolist(),
            'class_recall': recall.tolist(),
            'class_f1': f1.tolist(),
            'class_support': support.tolist(),
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'auc': auc_score,
            # 添加新指标
            'class_ppv': class_ppv,  # 各类别PPV
            'class_npv': class_npv,  # 各类别NPV
            'class_sensitivity': class_sensitivity,  # 各类别敏感性
            'class_specificity': class_specificity,  # 各类别特异性
            'macro_ppv': macro_ppv,  # 宏平均PPV
            'macro_npv': macro_npv,  # 宏平均NPV
            'macro_sensitivity': macro_sensitivity,  # 宏平均敏感性
            'macro_specificity': macro_specificity,  # 宏平均特异性
        })

        # 对于二分类问题，添加额外的便捷指标
        if num_classes == 2:
            tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
            metrics_dict.update({
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn,
                'ppv': class_ppv[1],  # 通常第1类为正类
                'npv': class_npv[0],  # 通常第0类为负类
                'sensitivity': class_sensitivity[1],  # 第1类敏感性
                'specificity': class_specificity[0],  # 第0类特异性
                'threshold': 0.5 if threshold_method == 'fixed_0.5' else None
            })

    # --- 回归指标 --- (保持不变)
    if 'axis_preds' in metrics_dict and 'axis_labels' in metrics_dict:
        axis_preds = metrics_dict['axis_preds']
        axis_labels = metrics_dict['axis_labels']
        # 允许有无效值过滤
        valid_mask = ~np.isnan(axis_labels).any(axis=1)
        axis_preds = axis_preds[valid_mask]
        axis_labels = axis_labels[valid_mask]
        if axis_preds.shape[0] > 0:
            mse = mean_squared_error(axis_labels, axis_preds)
            mae = mean_absolute_error(axis_labels, axis_preds)
            r2 = r2_score(axis_labels, axis_preds)
            metrics_dict.update({
                'axis_preds': axis_preds,
                'axis_labels': axis_labels,
                'axis_mse': mse,
                'axis_mae': mae,
                'axis_r2': r2
            })

    return metrics_dict