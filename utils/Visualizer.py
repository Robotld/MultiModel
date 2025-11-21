import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
from sklearn.calibration import calibration_curve
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from scipy import stats

# 为医学期刊设计的专业配色方案
MEDICAL_COLORS = {
    'primary': ['#0072B2', '#009E73', '#D55E00', '#CC79A7', '#F0E442'],  # 主要曲线
    'secondary': ['#56B4E9', '#E69F00', '#882255', '#117733', '#332288'],  # 次要曲线
    'highlight': '#E41A1C',  # 高亮/强调色
    'neutral': '#999999',  # 中性/参考色
    'binary': ['#0072B2', '#D55E00'],  # 二分类问题
    'categorical': [  # 多分类问题
        '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442',
        '#56B4E9', '#E69F00', '#882255', '#117733', '#332288'
    ],
    'sequential': ['#FEE5D9', '#FCBBA1', '#FC9272', '#FB6A4A', '#EF3B2C', '#CB181D', '#99000D'],  # 连续性数据
    'diverging': ['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#FDDBC7', '#F4A582', '#D6604D', '#B2182B']  # 对比数据
}


# 自定义医学期刊热图配色
def medical_cmap(cmap_type='blues'):
    """
    返回适合医学期刊的自定义colormap

    参数:
        cmap_type: 'blues', 'reds', 'greens', 'purples', 'oranges', 'grays'

    返回:
        matplotlib colormap对象
    """
    cmaps = {
        'blues': ["#FFFFFF", "#EBF4F8", "#D7E8F0", "#C3DDE9", "#AFD1E1", "#9BC6D9", "#87BAD2", "#73AECA", "#5FA2C3",
                  "#4A97BB", "#368BB4", "#2280AC", "#0E74A5", "#00689D"],
        'reds': ["#FFFFFF", "#FFF5F0", "#FEE0D2", "#FCBBA1", "#FC9272", "#FB6A4A", "#EF3B2C", "#E31A1C", "#D00C0E",
                 "#BD0026", "#A50F15", "#8C0D0E", "#730B0B", "#5A0808"],
        'greens': ["#FFFFFF", "#F7FCF5", "#E5F5E0", "#D3ECD1", "#C1E3C3", "#AFDBB4", "#9DD3A6", "#8BCA97", "#79C289",
                   "#67BA7A", "#55B26C", "#43AA5E", "#31A14F", "#1F9841"],
        'purples': ["#FFFFFF", "#FCFBFD", "#EFEDF5", "#E2DEEB", "#D4D0E2", "#C7C1D8", "#B9B3CE", "#ACA4C5", "#9E95BB",
                    "#9186B1", "#8378A8", "#7669AE", "#685AA4", "#5A4C9A"],
        'oranges': ["#FFFFFF", "#FFF5EB", "#FEE6CE", "#FDD8B0", "#FDCA92", "#FDC073", "#FDB255", "#FDA736", "#FC9918",
                    "#F78C00", "#E57E00", "#D37100", "#C16300", "#AF5600"],
        'grays': ["#FFFFFF", "#F7F7F7", "#E5E5E5", "#D3D3D3", "#C1C1C1", "#AFAFAF", "#9D9D9D", "#8B8B8B", "#797979",
                  "#676767", "#555555", "#434343", "#313131", "#1F1F1F"]
    }

    if cmap_type.lower() not in cmaps:
        cmap_type = 'blues'

    return LinearSegmentedColormap.from_list(
        f"medical_{cmap_type}", cmaps[cmap_type.lower()]
    )

def set_publication_style():
    """设置更符合医学顶刊的图表样式"""
    # 基础样式
    plt.style.use('default')  # 先重置样式，确保一致性

    # 高质量字体设置
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.weight'] = 'normal'
    plt.rcParams['svg.fonttype'] = 'none'  # 确保文本在SVG中可编辑

    # 线条和轴的设置
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.labelcolor'] = '#333333'
    plt.rcParams['xtick.color'] = '#333333'
    plt.rcParams['ytick.color'] = '#333333'
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8
    plt.rcParams['xtick.minor.width'] = 0.5
    plt.rcParams['ytick.minor.width'] = 0.5
    plt.rcParams['xtick.major.size'] = 3.0
    plt.rcParams['ytick.major.size'] = 3.0
    plt.rcParams['xtick.minor.size'] = 1.5
    plt.rcParams['ytick.minor.size'] = 1.5

    # 文字大小设置 - 精确匹配期刊要求
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9

    # 图例样式
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.markerscale'] = 1.0
    plt.rcParams['legend.handlelength'] = 1.5

    # 线条样式
    plt.rcParams['lines.linewidth'] = 1.2
    plt.rcParams['lines.markersize'] = 4

    # 图像尺寸和分辨率（根据期刊要求）
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['figure.figsize'] = (5.5, 4.5)  # 单栏图常用尺寸

    # 网格线和背景
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = ':'


def plot_learning_curves(train_metrics, val_metrics, save_path=None):
    """
    绘制训练过程中的学习曲线，符合医学期刊标准

    参数:
        train_metrics: 包含训练指标的字典列表，每个元素对应一个epoch
        val_metrics: 包含验证指标的字典列表，每个元素对应一个epoch
        save_path: 保存路径，如果为None则显示图像而不保存
    """
    metrics = ['loss', 'accuracy', 'f1', 'auc']
    epochs = range(1, len(train_metrics) + 1)

    # 设置最佳指标方向
    best_direction = {'loss': 'min', 'accuracy': 'max', 'f1': 'max', 'auc': 'max'}

    plt.figure(figsize=(16, 12))

    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 1)

        if metric in train_metrics[0] and metric in val_metrics[0]:
            train_values = [m.get(metric, 0) for m in train_metrics]
            val_values = [m.get(metric, 0) for m in val_metrics]

            # 绘制曲线
            plt.plot(epochs, train_values, 'b-o', label=f'Training',
                     linewidth=1.5, markersize=3, alpha=0.8)
            plt.plot(epochs, val_values, 'r-o', label=f'Validation',
                     linewidth=1.5, markersize=3, alpha=0.8)

            # 标记训练集最佳值
            if best_direction[metric] == 'min':
                best_idx = np.argmin(train_values)
                best_value = min(train_values)
            else:
                best_idx = np.argmax(train_values)
                best_value = max(train_values)

            best_epoch = epochs[best_idx]
            plt.plot(best_epoch, best_value, 'bo', markersize=6,
                     markeredgecolor='black', markeredgewidth=1)
            plt.annotate(f'{best_value:.3f}', (best_epoch, best_value),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)

            # 标记验证集最佳值
            if best_direction[metric] == 'min':
                best_idx = np.argmin(val_values)
                best_value = min(val_values)
            else:
                best_idx = np.argmax(val_values)
                best_value = max(val_values)

            best_epoch = epochs[best_idx]
            plt.plot(best_epoch, best_value, 'ro', markersize=6,
                     markeredgecolor='black', markeredgewidth=1)
            plt.annotate(f'{best_value:.3f}', (best_epoch, best_value),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)

            # 设置图表属性
            plt.title(f'{metric.upper()} Curve', fontsize=14)
            plt.xlabel('Epochs', fontsize=12)
            plt.ylabel(metric.upper(), fontsize=12)
            plt.grid(True, linestyle=':', alpha=0.3)
            plt.legend(loc='best')

            # 为loss设置下限为0，为其他指标设置0-1范围
            if metric == 'loss':
                ylim = plt.ylim()
                plt.ylim(0, ylim[1])
            elif metric in ['accuracy', 'f1', 'auc']:
                plt.ylim(-0.05, 1.05)

    plt.suptitle('Training and Validation Metrics', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为顶部标题留出空间

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_roc_curve(metrics_dict_list, labels=None, title="ROC Curve",
                            figsize=(5, 4.5), save_path=None, grid=True,
                            show_legend=True, legend_loc='lower right'):
    """改进版ROC曲线图表"""
    set_publication_style()

    fig, ax = plt.subplots(figsize=figsize)

    # 精致背景网格
    if grid:
        ax.grid(which='major', linestyle=':', linewidth=0.5, color='#DDDDDD', alpha=0.7)

    # 绘制参考线
    ax.plot([0, 1], [0, 1], linestyle='--', linewidth=1, color='#777777',
            alpha=0.7, label='_nolegend_')

    # 确保列表格式统一
    if not isinstance(metrics_dict_list, list):
        metrics_dict_list = [metrics_dict_list]

    if labels is None:
        labels = [f'Model {i + 1}' for i in range(len(metrics_dict_list))]

    # 使用专业配色
    colors = MEDICAL_COLORS['primary']

    # 绘制各曲线
    for i, metrics_dict in enumerate(metrics_dict_list):
        probs = metrics_dict['probabilities'][:, 1]
        true_labels = metrics_dict['labels']

        fpr, tpr, _ = roc_curve(true_labels, probs)
        roc_auc = auc(fpr, tpr)

        # 改进线条样式
        ax.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=1.5,
                marker=None, alpha=0.9,
                label=f'{labels[i]} (AUC = {roc_auc:.3f})')

    # 精确设置坐标轴
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    # 添加更专业的标签
    ax.set_xlabel('1 - Specificity', fontweight='normal')
    ax.set_ylabel('Sensitivity', fontweight='normal')

    # 优化刻度
    ax.tick_params(which='both', direction='out', length=3)

    # 设置图例
    if show_legend:
        ax.legend(loc=legend_loc, frameon=False, fontsize=9)

    # 添加标题
    if title:
        ax.set_title(title, fontsize=12, pad=10)

    # 移除顶部和右侧边框
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=600, transparent=False)
        plt.close()
    else:
        plt.show()


def plot_pr_curve(metrics_dict_list, labels=None, title="Precision-Recall Curve",
                 figsize=(5, 4.5), save_path=None):
    """
    绘制精确率-召回率曲线，对不平衡数据集特别有用

    参数:
        metrics_dict_list: 包含预测概率和真实标签的字典列表
        labels: 每条曲线的标签列表
        title: 图表标题
        figsize: 图表尺寸
        save_path: 保存路径，如不提供则显示图表
    """

    plt.figure(figsize=figsize)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f']

    if not isinstance(metrics_dict_list, list):
        metrics_dict_list = [metrics_dict_list]

    if labels is None:
        labels = [f'Model {i+1}' for i in range(len(metrics_dict_list))]

    # 确保标签列表长度匹配
    labels = labels[:len(metrics_dict_list)]

    for i, metrics_dict in enumerate(metrics_dict_list):
        probs = metrics_dict['probabilities'][:, 1]  # 取正类概率
        true_labels = metrics_dict['labels']

        precision, recall, _ = precision_recall_curve(true_labels, probs)
        pr_auc = auc(recall, precision)

        # 计算基线水平 (即正类比例)
        baseline = np.mean(true_labels)

        plt.plot(recall, precision, color=colors[i % len(colors)], lw=1.5,
                label=f'{labels[i]} (AUPRC = {pr_auc:.3f})')

    # 绘制基线
    plt.axhline(y=baseline, color='r', linestyle='--', alpha=0.5,
                label=f'Baseline (Prevalence = {baseline:.3f})')

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='best')

    # 添加网格线
    plt.grid(True, linestyle=':', alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


from matplotlib.colors import LinearSegmentedColormap


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def plot_confusion_matrix(conf_matrix=None, metrics_dict=None, labels=None,
                          normalize=False, title='Confusion Matrix',
                          figsize=(5, 4), save_path=None,
                          color_scheme='blue', annotate_counts=True):
    """
    绘制专业美观的混淆矩阵热图

    参数:
        conf_matrix: 混淆矩阵数组
        metrics_dict: 包含tn,fp,fn,tp的字典
        labels: 类别标签
        normalize: 是否归一化
        title: 图表标题
        figsize: 图表尺寸
        save_path: 保存路径
        color_scheme: 颜色方案 ('blue', 'red_blue', 'green', 'purple', 'viridis')
        annotate_counts: 是否在归一化时额外标注原始计数
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    # If no matrix provided but metrics are available, build a 2x2 matrix
    if conf_matrix is None and metrics_dict is not None:
        tn = metrics_dict['tn']
        fp = metrics_dict['fp']
        fn = metrics_dict['fn']
        tp = metrics_dict['tp']
        conf_matrix = np.array([[tn, fp], [fn, tp]])

    if conf_matrix is None:
        raise ValueError("Must provide conf_matrix or metrics_dict")

    if labels is None:
        if conf_matrix.shape[0] == 2:  # Binary classification
            labels = ['Negative', 'Positive']
        else:  # Multi-class
            labels = [f'Class {i}' for i in range(conf_matrix.shape[0])]

    # Save raw counts for annotation
    raw_counts = conf_matrix.copy()

    # Normalize if requested
    if normalize:
        conf_matrix = conf_matrix.astype(float) / conf_matrix.sum(axis=1, keepdims=True)
        fmt = '.2f'
    else:
        fmt = 'd'

    # Color scheme options
    color_schemes = {
        'blue': ["#f7fbff", "#08306b"],  # Blue gradient
        'red_blue': ["#67a9cf", "#f7f7f7", "#ef8a62"],  # Red-blue diverging
        'green': ["#edf8e9", "#005a32"],  # Green gradient
        'purple': ["#f2f0f7", "#54278f"],  # Purple gradient
        'viridis': None  # Use built-in viridis
    }

    # Set Seaborn style
    sns.set_style("white")
    sns.set_context("paper", font_scale=0.9)  # 减小整体字体比例

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Select color scheme
    if color_scheme in color_schemes and color_schemes[color_scheme]:
        colors = color_schemes[color_scheme]
        custom_cmap = LinearSegmentedColormap.from_list(f"{color_scheme}_gradient", colors)
    else:
        custom_cmap = 'viridis'  # Built-in matplotlib scheme

    # Set annotation format
    if normalize and annotate_counts:
        # Show both percentage and raw counts
        annot_func = lambda i, j: f"{conf_matrix[i, j]:.2f}\n({raw_counts[i, j]})"
        annot = True
        fmt = ''
    else:
        annot = True
        annot_func = None

    # Draw the heatmap
    im = sns.heatmap(
        conf_matrix,
        cmap=custom_cmap,
        annot=annot,
        fmt=fmt,
        annot_kws={"fontsize": 9, "fontweight": "medium"},  # 减小注释字体
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        cbar=True,
        linewidths=0,  # 移除单元格之间的空隙
        ax=ax
    )

    # If using custom format function
    if annot_func:
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                text = annot_func(i, j)
                # Adjust text color based on cell color
                cell_value = conf_matrix[i, j]
                text_color = 'white' if cell_value > 0.5 else 'black'
                im.text(j + 0.5, i + 0.5, text, ha='center', va='center',
                        fontsize=8, color=text_color)  # 减小自定义注释字体

        # Clear original annotations
        for t in im.texts:
            t.set_visible(False)

    # Set colorbar
    cbar = im.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8)  # 减小色条刻度字体
    if normalize:
        cbar.set_label('Proportion', fontsize=9, labelpad=6)  # 减小色条标签字体

    # 去除坐标轴标签
    # ax.set_xlabel('Predicted Label', fontsize=11, labelpad=10)
    # ax.set_ylabel('True Label', fontsize=11, labelpad=10)

    # 设置标题字体更小
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

    # 减小刻度标签字体
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8, rotation=90, va="center")

    # 简化边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#aaaaaa')
        spine.set_linewidth(0.6)

    # 布局紧凑
    plt.tight_layout()

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()






def plot_decision_curve_analysis(metrics_dict_list, labels=None,
                                title="Decision Curve Analysis",
                                figsize=(6, 4.5), save_path=None):
    """
    绘制决策曲线分析图 (Decision Curve Analysis)，医学顶刊常用

    参数:
        metrics_dict_list: 包含预测概率和真实标签的字典列表
        labels: 每条曲线的标签列表
        title: 图表标题
        figsize: 图表尺寸
        save_path: 保存路径
    """

    plt.figure(figsize=figsize)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    thresholds = np.linspace(0, 1, 101)[1:-1]  # 0.01 to 0.99

    if not isinstance(metrics_dict_list, list):
        metrics_dict_list = [metrics_dict_list]

    if labels is None:
        labels = [f'Model {i+1}' for i in range(len(metrics_dict_list))]

    # 确保标签列表长度匹配
    labels = labels[:len(metrics_dict_list)]

    # 计算治疗所有人的净获益（水平线）
    all_patients = []

    for i, metrics_dict in enumerate(metrics_dict_list):
        probs = metrics_dict['probabilities'][:, 1]  # 取正类概率
        true_labels = metrics_dict['labels']
        all_patients.append(np.mean(true_labels))

    # 使用第一个模型的正类率
    pt_prevalence = np.mean(all_patients)
    y_all = np.ones(len(thresholds)) * pt_prevalence

    # 绘制两条参考线
    plt.plot(thresholds, y_all, 'k--', lw=1, alpha=0.5, label='Treat All')
    plt.plot(thresholds, np.zeros_like(thresholds), 'k-', lw=1, alpha=0.5, label='Treat None')

    # 为每个模型计算并绘制决策曲线
    for i, metrics_dict in enumerate(metrics_dict_list):
        probs = metrics_dict['probabilities'][:, 1]
        true_labels = metrics_dict['labels']

        net_benefit = []

        for threshold in thresholds:
            # 阈值处的决策
            pred_positive = (probs >= threshold).astype(int)

            # 真阳性和假阳性
            tp = np.sum((pred_positive == 1) & (true_labels == 1))
            fp = np.sum((pred_positive == 1) & (true_labels == 0))
            n = len(true_labels)

            # 计算净获益
            if np.sum(pred_positive) == 0:
                # 如果没有预测为阳性的样本
                nb = 0
            else:
                nb = (tp/n) - (fp/n) * (threshold/(1-threshold))

            net_benefit.append(nb)

        plt.plot(thresholds, net_benefit, color=colors[i % len(colors)],
                 lw=1.5, label=labels[i])

    plt.xlim([0, 1])
    plt.xlabel('Threshold Probability')
    plt.ylabel('Net Benefit')
    plt.title(title)
    plt.legend(loc='best')

    # 添加网格线
    plt.grid(True, linestyle=':', alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_metrics_comparison(metrics_dict_list, metric_names=None,
                           labels=None, title="Performance Metrics Comparison",
                           figsize=(7, 5), save_path=None):
    """
    绘制多个模型多指标对比条形图

    参数:
        metrics_dict_list: 包含多个模型评估指标的字典列表
        metric_names: 要比较的指标名称列表，默认为['auc', 'sensitivity', 'specificity', 'precision', 'f1']
        labels: 每个模型的标签
        title: 图表标题
        figsize: 图表尺寸
        save_path: 保存路径
    """

    if metric_names is None:
        metric_names = ['auc', 'sensitivity', 'specificity', 'precision', 'f1']

    if not isinstance(metrics_dict_list, list):
        metrics_dict_list = [metrics_dict_list]

    if labels is None:
        labels = [f'Model {i+1}' for i in range(len(metrics_dict_list))]

    # 确保标签列表长度匹配
    labels = labels[:len(metrics_dict_list)]

    # 准备数据
    metric_values = []
    for metrics_dict in metrics_dict_list:
        values = [metrics_dict.get(metric, np.nan) for metric in metric_names]
        metric_values.append(values)

    # 将数据转换为DataFrame
    df = pd.DataFrame(metric_values, columns=metric_names, index=labels)

    # 绘制条形图
    fig, ax = plt.subplots(figsize=figsize)

    # 获取条形间距
    n_bars = len(df)
    n_metrics = len(metric_names)
    bar_width = 0.8 / n_metrics
    r = np.arange(n_bars)

    # 为每个指标绘制条形
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f']
    for i, metric in enumerate(metric_names):
        position = r + i * bar_width
        bars = ax.bar(position, df[metric], bar_width, alpha=0.7,
                      color=colors[i % len(colors)], label=metric.capitalize())

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    # 设置x轴标签位置和标题
    ax.set_xticks(r + bar_width * (n_metrics-1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Performance')
    ax.set_title(title)

    # 添加图例
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1))

    # 添加网格线
    plt.grid(True, axis='y', linestyle=':', alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_threshold_impact(metrics_dict, threshold_range=None,
                         metrics_to_plot=None, title="Threshold Impact Analysis",
                         figsize=(6, 4.5), save_path=None):
    """
    绘制阈值对各性能指标影响的曲线图

    参数:
        metrics_dict: 包含预测概率和真实标签的字典
        threshold_range: 阈值范围，默认为np.linspace(0.05, 0.95, 19)
        metrics_to_plot: 要绘制的指标列表，默认为['sensitivity', 'specificity', 'f1', 'precision']
        title: 图表标题
        figsize: 图表尺寸
        save_path: 保存路径
    """


    if threshold_range is None:
        threshold_range = np.linspace(0.05, 0.95, 19)

    if metrics_to_plot is None:
        metrics_to_plot = ['sensitivity', 'specificity', 'f1', 'precision']

    # 提取预测概率和真实标签
    probs = metrics_dict['probabilities'][:, 1]  # 取正类概率
    true_labels = metrics_dict['labels']

    # 计算每个阈值下的性能指标
    results = {metric: [] for metric in metrics_to_plot}

    for threshold in threshold_range:
        predictions = (probs >= threshold).astype(int)

        # 计算混淆矩阵元素
        tn = np.sum((predictions == 0) & (true_labels == 0))
        fp = np.sum((predictions == 1) & (true_labels == 0))
        fn = np.sum((predictions == 0) & (true_labels == 1))
        tp = np.sum((predictions == 1) & (true_labels == 1))

        # 计算各指标
        if 'sensitivity' in metrics_to_plot:
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            results['sensitivity'].append(sensitivity)

        if 'specificity' in metrics_to_plot:
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            results['specificity'].append(specificity)

        if 'precision' in metrics_to_plot:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            results['precision'].append(precision)

        if 'f1' in metrics_to_plot:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            results['f1'].append(f1)

    # 绘制曲线
    plt.figure(figsize=figsize)

    colors = {'sensitivity': '#1f77b4', 'specificity': '#ff7f0e',
              'precision': '#2ca02c', 'f1': '#d62728'}

    for metric in metrics_to_plot:
        plt.plot(threshold_range, results[metric], 'o-', lw=1.5,
                 label=metric.capitalize(), color=colors.get(metric, None))

    # 标记出最佳阈值
    if 'f1' in metrics_to_plot:
        best_idx = np.argmax(results['f1'])
        best_threshold = threshold_range[best_idx]
        plt.axvline(x=best_threshold, color='gray', linestyle='--', alpha=0.7,
                   label=f'Best F1 @ {best_threshold:.2f}')

    plt.xlim([0, 1])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('Decision Threshold')
    plt.ylabel('Performance')
    plt.title(title)
    plt.legend(loc='best')

    # 添加网格线
    plt.grid(True, linestyle=':', alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_prediction_distribution(metrics_dict, bins=20,
                                title="Prediction Probability Distribution",
                                figsize=(6, 4), save_path=None):
    """
    绘制预测概率分布直方图，分离正例和负例

    参数:
        metrics_dict: 包含预测概率和真实标签的字典
        bins: 直方图的箱数
        title: 图表标题
        figsize: 图表尺寸
        save_path: 保存路径
    """
    set_publication_style()

    probs = metrics_dict['probabilities'][:, 1]  # 取正类概率
    true_labels = metrics_dict['labels']
    threshold = metrics_dict.get('threshold', 0.5)

    # 分离正例和负例
    pos_probs = probs[true_labels == 1]
    neg_probs = probs[true_labels == 0]

    plt.figure(figsize=figsize)

    # 绘制分布
    plt.hist(neg_probs, bins=bins, alpha=0.6, color='#ff7f0e',
             label=f'Negative (n={len(neg_probs)})', density=True)
    plt.hist(pos_probs, bins=bins, alpha=0.6, color='#1f77b4',
             label=f'Positive (n={len(pos_probs)})', density=True)

    # 标记阈值线
    plt.axvline(x=threshold, color='red', linestyle='--', alpha=0.7,
               label=f'Threshold = {threshold:.2f}')

    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend(loc='best')

    # 添加网格线
    plt.grid(True, linestyle=':', alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import matplotlib.gridspec as gridspec


def plot_calibration_curves(metrics_dict_list, labels=None, figsize=(6, 5), save_path=None):
    """
    只绘制校准曲线 (Calibration Curve)，对比不同模型的预测置信度与真实观测概率。

    参数:
        metrics_dict_list: 包含预测概率和真实标签的字典或字典列表。
                           每个字典应包含:
                           {
                               'probabilities': shape为 [N, 2] 的概率预测 (含负类和正类),
                               'labels': shape为 [N, ] 的二分类标签(0或1)
                           }
        labels: 模型名称列表, 与metrics_dict_list对应。
        figsize: 图像大小。
        save_path: 若指定路径, 则保存图像并关闭, 否则直接 plt.show()。
    """
    # 若只传了一个字典, 转成列表保持统一处理
    if not isinstance(metrics_dict_list, list):
        metrics_dict_list = [metrics_dict_list]

    # 若未指定labels, 默认给出 "Model 1", "Model 2", ...
    if labels is None:
        labels = [f'Model {i + 1}' for i in range(len(metrics_dict_list))]

    # 创建画布
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(gs[0, 0])

    # 绘制对角线(完美匹配参考线)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.7, label='Perfectly Calibrated')

    # 颜色列表, 可根据需要增减
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # 逐个模型进行绘制
    for i, metrics_dict in enumerate(metrics_dict_list):
        probs = metrics_dict['probabilities'][:, 1]  # 正类概率 (第二列)
        labels_ = metrics_dict['labels']

        # 计算校准曲线
        prob_true, prob_pred = calibration_curve(labels_, probs, n_bins=10)

        ax.plot(prob_pred, prob_true, marker='o', markersize=4,
                color=colors[i % len(colors)], lw=1.5, label=labels[i])

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Observed Frequency')
    ax.set_title('Calibration Curve')
    ax.legend(loc='best')
    ax.grid(True, linestyle=':', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()
