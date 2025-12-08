"""
多模态肺癌复发预测/亚型分类 — 独立测试脚本
- 支持从 .pth 加载整模型或 state_dict
- 计算分类与回归指标
- 导出逐样本预测表
"""

import os
import argparse
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)


# ==== 你项目里的模块 ====
from models import create_transforms, ViT3D
from models import MultimodalDataset
from models import MultimodalMultitaskModel
from models import MultitaskLoss

from src.train_multimodal import save_metrics_to_csv
from src.train_multimodal import one_epoch_multimodal_train
from utils.calculate_metrics import calculate_metrics
from utils import ConfigManager, set_seed


# ---------- 工具函数 ----------
def _maybe_strip_prefix(state_dict, prefixes=("module.", "model.", "net.")):
    """去掉常见的前缀，便于DDP/封装加载"""
    if not isinstance(state_dict, dict):
        return state_dict
    keys = list(state_dict.keys())
    if len(keys) == 0:
        return state_dict
    for p in prefixes:
        if keys[0].startswith(p):
            return {k[len(p):]: v for k, v in state_dict.items()}
    return state_dict


def build_vit3d_from_config(cfg):
    return ViT3D(
        num_classes=2,
        image_size=cfg.model["params"]["image_size"],
        crop_size=cfg.training["crop_size"],
        patch_size=cfg.model["params"]["patch_size"],
        dim=cfg.model["params"]["dim"],
        depth=cfg.model["params"]["depth"],
        heads=cfg.model["params"]["heads"],
        mlp_dim=cfg.model["params"]["mlp_dim"],
        pool=cfg.model["params"]["pool"],
        cpt_num=cfg.model["params"]["cpt_num"],
        mlp_num=cfg.model["params"]["mlp_num"],
    )


def build_mm_model_from_config(cfg, vit3d_model):
    return MultimodalMultitaskModel(
        vit_3d_model=vit3d_model,
        image_dim=cfg.model["params"]["dim"],
        bert_model_name=cfg.model['bert_model_name'],
        text_feature_dim=cfg.model['text_feature_dim'],
        demographic_dim=cfg.model['demographic_dim'],
        fusion_dim=cfg.model['fusion_dim'],
        num_classes=cfg.data['num_classes'],
        fusion_transformer_heads=cfg.model['fusion_transformer_heads'],
        fusion_transformer_layers=cfg.model['fusion_transformer_layers'],
        dropout=cfg.model['dropout'],
        use_entity_decoder=True,
        decoder_num_layers=cfg.training['decoder_num_layers'],
        decoder_num_heads=cfg.training['decoder_num_heads'],
        decoder_max_seq_len=20,
        fusion_type=cfg.model['fusion_type'],
    )


def robust_load_weights(pth_path, model, device="cpu", strict=False):
    """
    既兼容 torch.save(model) 的整模型 .pth，
    也兼容 torch.save(model.state_dict()) 的权重 .pth。
    """
    obj = torch.load(pth_path, map_location=device)

    # 情况A：直接是模块（整模型保存）
    if isinstance(obj, torch.nn.Module):
        # 直接拿来用：把obj的权重搬到当前 model
        sd = obj.state_dict()
        sd = _maybe_strip_prefix(sd)
        missing, unexpected = model.load_state_dict(sd, strict=strict)
        print(f"[Load:Module] missing={len(missing)} unexpected={len(unexpected)}")
        return

    # 情况B：是纯 state_dict 或包了一层字典
    if isinstance(obj, dict):
        # 常见包裹字段名尝试
        candidates = ["state_dict", "model_state_dict", "model", "net", "weights"]
        state_dict = None
        for k in candidates:
            if k in obj and isinstance(obj[k], dict):
                state_dict = obj[k]
                break
        if state_dict is None:
            state_dict = obj

        state_dict = _maybe_strip_prefix(state_dict)
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        print(f"[Load:state_dict] missing={len(missing)} unexpected={len(unexpected)}")
        if len(missing) or len(unexpected):
            print("  - missing keys (前10):", list(missing)[:10])
            print("  - unexpected keys (前10):", list(unexpected)[:10])
        return

    raise RuntimeError(f"未知的 .pth 格式: {type(obj)}")



# ---------- 主流程 ----------
def main():
    cfg = ConfigManager(r'E:\workplace\MultiModel\config\config_test.yaml')

    parser = argparse.ArgumentParser("Multimodal Test")
    parser.add_argument("--weights", type=str, default=r'E:\workplace\MultiModel\src\mul_train_output\train_multimodal_stratified_20251206_151951\best_auc_model_3.pth',
                        help=".pth 权重文件（支持整模型或state_dict）")
    parser.add_argument("--test_csv", type=str, default=None,
                        help="独立测试集 CSV（若不提供则用配置里的 data.csv_path）")
    parser.add_argument("--batch_size", type=int, default=None, help="测试 batch_size（覆盖配置）")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument('--output_dir', type=str, nargs='+', default=cfg.data["output_dir"],
                        help='test output dir')
    parser.add_argument("--strict", action="store_true", help="加载权重时严格匹配键名")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()


    set_seed(args.seed)
    device = cfg.training.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available() and device.startswith("cuda"):
        device = "cpu"
    print(f"使用设备: {device}")

    # 输出目录
    ts = time.strftime("%Y%m%d_%H%M%S")
    # Transforms/Tokenizer
    _,  test_transforms = create_transforms(cfg, args)
    bert_model_name = cfg.model['tokenizer_path']
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    # 数据集（测试集）
    test_csv = args.test_csv or cfg.data['csv_path']
    print(f"加载测试 CSV: {test_csv}")
    test_set = MultimodalDataset(
        csv_files=test_csv,
        transform=test_transforms,
        text_tokenizer=tokenizer,
        max_length=cfg.data.get('max_text_length', 512)
    )
    print(f"测试样本数: {len(test_set.samples)}")

    bs = args.batch_size or cfg.training.get('batch_size', 2)
    test_loader = DataLoader(
        test_set, batch_size=bs, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # 构建模型骨架 & 加载权重
    vit3d = build_vit3d_from_config(cfg)
    model = build_mm_model_from_config(cfg, vit3d)
    model.to(device)

    # 如果你在训练时还加载过ViT3D的预训练权重，这里无需再单独加载；
    # 直接 robust_load_weights 会一并覆盖到位
    print(f"加载权重: {args.weights}")
    robust_load_weights(args.weights, model, device=device, strict=args.strict)

    criterion = MultitaskLoss(
        classification_weight=cfg.losses['MultitaskLoss']['classification_weight'],
        axis_weight=cfg.losses['MultitaskLoss']['axis_weight'],
    )

    test_loss, test_metrics, test_sim_loss = one_epoch_multimodal_train(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        optimizer=None,
        device=device,
        train=False,
        scaler=None,
        use_amp=False,
        max_grad_norm=1.0,

    )

    # th = np.array([0.28033251, 0.33480483, 0.44875756])
    test_metrics = calculate_metrics(test_metrics, cfg.data['num_classes'])


    print("\n===== 测试指标 =====")

    print(
        f'   Test Loss: {test_loss:.4f}, ACC: {test_metrics["accuracy"]:.4f} AUC: {test_metrics["auc"]:.4f}')
    print(
        f'   Test   F1: {test_metrics["micro_f1"]:.4f}, f1_0: {test_metrics["class_f1"][0]:.4f}, f1_1: {test_metrics["class_f1"][1]:.4f}')

    for k, v in test_metrics.items():
        if k == "confusion_matrix":
            print(f"{k}:\n{np.array(v)}")
        elif k not in {'probabilities', 'logits', 'labels', 'prob', 'predictions'}:
            print(f"{k}: {v}")

        # 创建输出目录
    existing_dirs = [d for d in os.listdir(args.output_dir) if
                     d.startswith('test_') and os.path.isdir(os.path.join(args.output_dir, d))]
    next_num = 1
    if existing_dirs:
        existing_nums = [int(d.split('_')[1]) for d in existing_dirs if d.split('_')[1].isdigit()]
        if existing_nums:
            next_num = max(existing_nums) + 1
    output_dir = os.path.join(args.output_dir, f'test_{next_num}')

    save_metrics_to_csv(test_metrics, os.path.join(output_dir, 'test_metrics.csv'))



if __name__ == "__main__":
    main()
