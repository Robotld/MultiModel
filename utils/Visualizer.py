"""
Visualization utilities for training metrics and results
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns


def plot_training_curves(train_history, val_history, save_path=None):
    """
    Plot training and validation curves for various metrics
    
    Args:
        train_history: List of dictionaries containing training metrics
        val_history: List of dictionaries containing validation metrics
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(train_history) + 1)
    
    # Plot loss
    if 'loss' in train_history[0]:
        axes[0, 0].plot(epochs, [h.get('loss', 0) for h in train_history], 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, [h.get('loss', 0) for h in val_history], 'r-', label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    # Plot accuracy
    if 'accuracy' in train_history[0]:
        axes[0, 1].plot(epochs, [h.get('accuracy', 0) for h in train_history], 'b-', label='Train Acc')
        axes[0, 1].plot(epochs, [h.get('accuracy', 0) for h in val_history], 'r-', label='Val Acc')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Plot F1 score
    if 'micro_f1' in train_history[0]:
        axes[1, 0].plot(epochs, [h.get('micro_f1', 0) for h in train_history], 'b-', label='Train F1')
        axes[1, 0].plot(epochs, [h.get('micro_f1', 0) for h in val_history], 'r-', label='Val F1')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('Training and Validation F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Plot AUC
    if 'auc' in train_history[0]:
        axes[1, 1].plot(epochs, [h.get('auc', 0) for h in train_history], 'b-', label='Train AUC')
        axes[1, 1].plot(epochs, [h.get('auc', 0) for h in val_history], 'r-', label='Val AUC')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].set_title('Training and Validation AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存到: {save_path}")
    
    plt.close()


def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存到: {save_path}")
    
    plt.close()


def plot_roc_curve(labels, probabilities, save_path=None):
    """
    Plot ROC curve
    
    Args:
        labels: True labels
        probabilities: Predicted probabilities for positive class
        save_path: Path to save the plot (optional)
    """
    fpr, tpr, _ = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC曲线已保存到: {save_path}")
    
    plt.close()


def save_all_visualizations(train_history, val_history, metrics, save_dir, fold=None):
    """
    Save all visualizations for a training run
    
    Args:
        train_history: List of training metrics per epoch
        val_history: List of validation metrics per epoch
        metrics: Final metrics dictionary with confusion matrix, labels, probabilities
        save_dir: Directory to save visualizations
        fold: Fold number (optional)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fold_suffix = f'_fold_{fold}' if fold is not None else ''
    
    # Plot training curves
    plot_training_curves(
        train_history, 
        val_history, 
        save_path=os.path.join(save_dir, f'training_curves{fold_suffix}.png')
    )
    
    # Plot confusion matrix if available
    if 'confusion_matrix' in metrics and metrics['confusion_matrix'] is not None:
        class_names = ['Non-recurrence', 'Recurrence']  # Default class names
        plot_confusion_matrix(
            metrics['confusion_matrix'],
            class_names,
            save_path=os.path.join(save_dir, f'confusion_matrix{fold_suffix}.png')
        )
    
    # Plot ROC curve if labels and probabilities are available
    if 'labels' in metrics and 'probabilities' in metrics:
        labels = metrics['labels']
        probabilities = metrics['probabilities']
        if len(labels) > 0 and len(probabilities) > 0:
            # For binary classification, use positive class probability
            if probabilities.ndim > 1 and probabilities.shape[1] == 2:
                prob_positive = probabilities[:, 1]
            else:
                prob_positive = probabilities
            
            plot_roc_curve(
                labels,
                prob_positive,
                save_path=os.path.join(save_dir, f'roc_curve{fold_suffix}.png')
            )
