import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from matplotlib.font_manager import FontProperties


def plot_training_history(train_dir, fold, train_metrics_history, val_metrics_history):
    """生成训练历史图表并保存"""
    epochs = range(1, len(train_metrics_history) + 1)

    # 配置图表全局参数
    plt.figure(figsize=(15, 12), constrained_layout=True)
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    plt.rcParams.update({'font.size': 10})  # 减小字体尺寸

    # 标题和标签文本
    titles = ['Loss Curve', 'Accuracy Curve', 'F1 Score Curve', 'AUC Curve']
    y_labels = ['Loss', 'Accuracy', 'F1 Score', 'AUC']
    metrics = ['loss', 'accuracy', 'f1', 'auc']

    # 创建子图
    for i, (title, y_label, metric) in enumerate(zip(titles, y_labels, metrics)):
        plt.subplot(2, 2, i + 1)
        plt.plot(epochs, [m[metric] for m in train_metrics_history], 'b-', label='Train')
        plt.plot(epochs, [m[metric] for m in val_metrics_history], 'r-', label='Validation')
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel(y_label)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # 添加高亮最佳值
        val_values = [m[metric] for m in val_metrics_history]
        if metric != 'loss':  # 对于accuracy, f1, auc找最大值
            best_epoch = val_values.index(max(val_values)) + 1
            best_value = max(val_values)
        else:  # 对于loss找最小值
            best_epoch = val_values.index(min(val_values)) + 1
            best_value = min(val_values)

        plt.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5)
        plt.scatter(best_epoch, best_value, s=100, c='g', marker='*')
        plt.annotate(f'Best: {best_value:.4f}',
                     xy=(best_epoch, best_value),
                     xytext=(best_epoch + 0.5, best_value),
                     fontsize=9)

    plt.tight_layout()

    # 保存图表
    plot_path = os.path.join(train_dir, f'training_plot_fold_{fold}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"训练图表已保存至: {plot_path}")

    return plot_path


def save_training_history(train_dir, fold, train_metrics_history, val_metrics_history):
    """保存训练历史记录为多种格式"""
    # 1. 保存为CSV
    csv_path = os.path.join(train_dir, f'training_history_fold_{fold}.csv')

    # 创建数据框
    import pandas as pd

    data = {
        'epoch': list(range(1, len(train_metrics_history) + 1)),
        'train_loss': [m.get('loss', float('nan')) for m in train_metrics_history],
        'train_acc': [m.get('accuracy', float('nan')) for m in train_metrics_history],
        'train_f1': [m.get('f1', float('nan')) for m in train_metrics_history],
        'train_auc': [m.get('auc', float('nan')) for m in train_metrics_history],
        'train_sensitivity': [m.get('sensitivity', float('nan')) for m in train_metrics_history],
        'train_specificity': [m.get('specificity', float('nan')) for m in train_metrics_history],
        'train_ppv': [m.get('ppv', float('nan')) for m in train_metrics_history],
        'train_npv': [m.get('npv', float('nan')) for m in train_metrics_history],
        'val_loss': [m.get('loss', float('nan')) for m in val_metrics_history],
        'val_acc': [m.get('accuracy', float('nan')) for m in val_metrics_history],
        'val_f1': [m.get('f1', float('nan')) for m in val_metrics_history],
        'val_auc': [m.get('auc', float('nan')) for m in val_metrics_history],
        'val_sensitivity': [m.get('sensitivity', float('nan')) for m in val_metrics_history],
        'val_specificity': [m.get('specificity', float('nan')) for m in val_metrics_history],
        'val_ppv': [m.get('ppv', float('nan')) for m in val_metrics_history],
        'val_npv': [m.get('npv', float('nan')) for m in val_metrics_history],
        'val_threshold': [m.get('threshold', 0.5) for m in val_metrics_history],
        'train_threshold': [m.get('threshold', 0.5) for m in train_metrics_history]
    }

    # 转换为DataFrame并保存
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False, float_format='%.4f')

    # 2. 生成可视化图表
    try:
        if data['epoch'][0] > 1:
            plot_training_history(train_dir, fold, train_metrics_history, val_metrics_history)
    except Exception as e:
        print(f"生成图表时出错: {str(e)}")

