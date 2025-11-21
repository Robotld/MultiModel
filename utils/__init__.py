"""
Utility functions for MultiModel project
"""
import os
import argparse
import yaml
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)


class ConfigManager:
    """Configuration manager for loading and accessing YAML configs"""
    
    def __init__(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
    def __getitem__(self, key):
        """Allow dictionary-style access"""
        return self.config[key]
    
    def get(self, key, default=None):
        """Get value with default"""
        return self.config.get(key, default)
    
    def __getattr__(self, key):
        """Allow attribute-style access"""
        if key == 'config':
            return object.__getattribute__(self, key)
        return self.config.get(key)
    
    def get_optimizer_params(self, model):
        """Get optimizer parameters from config"""
        params = self.config['optimizer']['params'].copy()
        return params
    
    def get_scheduler_params(self):
        """Get scheduler parameters from config"""
        return self.config['scheduler']['params'].copy()


def parse_args(config):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='多模态肺癌复发预测训练')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=config.training.get('num_epochs', 10),
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=config.training.get('batch_size', 32),
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=config.optimizer['params'].get('lr', 0.0001),
                        help='学习率')
    parser.add_argument('--device', type=str, default=config.training.get('device', 'cuda'),
                        help='训练设备')
    parser.add_argument('--num_workers', type=int, default=config.training.get('num_workers', 4),
                        help='数据加载线程数')
    
    # Model parameters
    parser.add_argument('--frozen1', type=bool, default=config.training.get('frozen1', False),
                        help='是否冻结图像编码器')
    parser.add_argument('--frozen2', type=bool, default=config.training.get('frozen2', False),
                        help='是否冻结文本编码器')
    
    # Training options
    parser.add_argument('--use_amp', type=bool, default=config.training.get('use_amp', True),
                        help='是否使用混合精度训练')
    parser.add_argument('--cache_threshold', type=int, default=10000,
                        help='验证集缓存阈值')
    parser.add_argument('--force_cache_val', type=bool, default=False,
                        help='强制缓存验证集')
    parser.add_argument('--patience', type=int, default=5,
                        help='早停的耐心值')
    
    args = parser.parse_args()
    return args


def update_config_from_args(config, args):
    """Update configuration with command line arguments"""
    if hasattr(args, 'epochs'):
        config.config['training']['num_epochs'] = args.epochs
    if hasattr(args, 'batch_size'):
        config.config['training']['batch_size'] = args.batch_size
    if hasattr(args, 'lr'):
        config.config['optimizer']['params']['lr'] = args.lr
    if hasattr(args, 'device'):
        config.config['training']['device'] = torch.device(args.device)
    if hasattr(args, 'num_workers'):
        config.config['training']['num_workers'] = args.num_workers
    
    return config


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"随机种子设置为: {seed}")


def save_config(config, save_path):
    """Save configuration to YAML file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert config to dictionary if it's ConfigManager
    if isinstance(config, ConfigManager):
        config_dict = config.config
    else:
        config_dict = config
    
    # Convert torch.device to string for YAML serialization
    def convert_device(obj):
        if isinstance(obj, torch.device):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: convert_device(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_device(item) for item in obj]
        return obj
    
    config_dict = convert_device(config_dict)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    print(f"配置已保存到: {save_path}")


def calculate_metrics(metrics_dict, num_classes=2):
    """Calculate classification metrics from predictions and labels"""
    probabilities = metrics_dict['probabilities']
    labels = metrics_dict['labels']
    
    if len(labels) == 0:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'micro_f1': 0.0,
            'macro_f1': 0.0,
            'auc': 0.0,
            'confusion_matrix': np.zeros((num_classes, num_classes))
        }
    
    # Get predictions from probabilities
    predictions = np.argmax(probabilities, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    
    # For binary classification
    if num_classes == 2:
        precision = precision_score(labels, predictions, average='binary', zero_division=0)
        recall = recall_score(labels, predictions, average='binary', zero_division=0)
        f1 = f1_score(labels, predictions, average='binary', zero_division=0)
        
        # AUC score using positive class probability
        try:
            auc = roc_auc_score(labels, probabilities[:, 1])
        except:
            auc = 0.0
    else:
        # For multi-class
        precision = precision_score(labels, predictions, average='macro', zero_division=0)
        recall = recall_score(labels, predictions, average='macro', zero_division=0)
        f1 = f1_score(labels, predictions, average='macro', zero_division=0)
        
        try:
            auc = roc_auc_score(labels, probabilities, multi_class='ovr', average='macro')
        except:
            auc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Calculate micro and macro F1
    micro_f1 = f1_score(labels, predictions, average='micro', zero_division=0)
    macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'predictions': predictions,
        'probabilities': probabilities,
        'labels': labels
    }


def save_training_history(train_dir, fold, train_metrics_history, val_metrics_history):
    """Save training history to CSV files"""
    os.makedirs(train_dir, exist_ok=True)
    
    # Convert metrics history to DataFrame
    train_df = pd.DataFrame(train_metrics_history)
    val_df = pd.DataFrame(val_metrics_history)
    
    # Save to CSV
    train_csv_path = os.path.join(train_dir, f'train_history_fold_{fold}.csv')
    val_csv_path = os.path.join(train_dir, f'val_history_fold_{fold}.csv')
    
    # Drop non-serializable columns if they exist
    for col in ['confusion_matrix', 'predictions', 'probabilities', 'labels']:
        if col in train_df.columns:
            train_df = train_df.drop(columns=[col])
        if col in val_df.columns:
            val_df = val_df.drop(columns=[col])
    
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    
    print(f"训练历史已保存到: {train_csv_path} 和 {val_csv_path}")
