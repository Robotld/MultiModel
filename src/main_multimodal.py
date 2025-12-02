"""
多模态肺癌复发预测与亚型分类多任务训练脚本
整合CT图像、病理报告和人口学特征进行端到端训练
"""
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    CropForegroundd, Resized, NormalizeIntensityd, ToTensord
)

# 导入自定义模块
from models import create_transforms, MulCrossValidator, ViT3D
from models import MultimodalDataset
from models import MultimodalMultitaskModel
from utils import update_config_from_args, parse_args, ConfigManager, set_seed
from utils.entity_masking import MedicalEntityMasker  # ✅ 新增
from train_multimodal import train


def main():
    # 记录开始时间
    start_time = time.time()

    # 加载配置
    config = ConfigManager('../config/config_multimodal.yaml')

    # 解析命令行参数
    args = parse_args(config)
    config = update_config_from_args(config, args)

    set_seed(config.training['random_seed'])

    # 优化CUDA性能
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.empty_cache()

    print(f"使用设备: {config.training['device']}")

    # 创建输出目录
    os.makedirs(config.data['output_dir'], exist_ok=True)

    # 根据交叉验证类型命名输出目录
    cv_type = "clustering" if config.cross_validation.get('use_clustering', False) else "stratified"
    train_dir = os.path.join(
        config.data['output_dir'],
        f'train_multimodal_{cv_type}_{time.strftime("%Y%m%d_%H%M%S")}'
    )
    os.makedirs(train_dir, exist_ok=True)

    # 创建MONAI转换管道
    train_transforms, val_transforms = create_transforms(config, args)

    # 初始化BERT分词器
    bert_model_name = config.model['tokenizer_path']
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    # 创建数据集
    print("加载多模态数据集...")
    dataset = MultimodalDataset(
        csv_files=config.data['csv_path'],
        transform=None,
        text_tokenizer=tokenizer,
        max_length=config.data.get('max_text_length', 512)
    )

    # 显示数据集基本信息
    print(f"数据集样本总数: {len(dataset.samples)}")

    # ============== ✅ 先初始化 entity_masker，获取 vocab_size ==============
    entity_masker = MedicalEntityMasker(mask_ratio_choices=[0.2, 0.3, 0.4])
    entity_vocab_size = entity_masker.vocab_size
    print(f"✅ 实体词表大小: {entity_vocab_size}")

    # 初始化交叉验证器
    cv = MulCrossValidator(dataset, config)
    device = config.training['device']

    # 存储每折的最佳F1, AUC分数
    fold_scores = []

    # 预加载ViT3D模型配置
    vit3d_config = {
        "num_classes": 2,
        "image_size": config.model["params"]["image_size"],
        "crop_size": config.training["crop_size"],
        "patch_size": config.model["params"]["patch_size"],
        "dim": config.model["params"]["dim"],
        "depth": config.model["params"]["depth"],
        "heads": config.model["params"]["heads"],
        "mlp_dim": config.model["params"]["mlp_dim"],
        "pool": config.model["params"]["pool"],
        'cpt_num': config.model["params"]["cpt_num"],
        'mlp_num': config.model["params"]["mlp_num"],
    }

    best_f1, best_auc = 0.0, 0.0

    # 进行交叉验证
    for fold, train_loader, val_loader, train_counter in cv.get_folds(train_transforms, val_transforms):
        train_loader.dataset._data_counter = train_counter

        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        fold_start_time = time.time()
        fold_type = "聚类折" if config.cross_validation.get('use_clustering', False) else "分层折"
        print(f'\n训练第 {fold + 1} 折 ({fold_type})')

        # 1. 初始化图像编码器（ViT3D）
        vit3d_model = ViT3D(**vit3d_config)

        # 加载ViT3D预训练权重
        if config.training['pretrained_path']:
            print(f"Loading pretrained weights for ViT3D from {config.training['pretrained_path']}...")
            try:
                if os.path.isdir(config.training['pretrained_path']):
                    vit3d_model.load_pretrained_dino(config.training['pretrained_path'])
                else:
                    vit3d_model.load_state_dict(
                        torch.load(config.training['pretrained_path'], map_location=device),
                        strict=False
                    )
                print(f"成功加载ViT3D权重: {config.training['pretrained_path']}")
            except Exception as e:
                print(f"加载ViT3D预训练权重出错: {str(e)}")

        # 2. 构建多模态模型（✅ 现在可以安全使用 entity_vocab_size）
        model = MultimodalMultitaskModel(
            vit_3d_model=vit3d_model,
            image_dim=config.model["params"]["dim"],
            bert_model_name=config.model['bert_model_name'],
            tokenizer=tokenizer,
            text_feature_dim=config.model['text_feature_dim'],
            demographic_dim=config.model['demographic_dim'],
            fusion_dim=config.model['fusion_dim'],
            num_classes=config.data['num_classes'],
            fusion_transformer_heads=config.model['fusion_transformer_heads'],
            fusion_transformer_layers=config.model['fusion_transformer_layers'],
            dropout=config.model['dropout'],
            entity_vocab_size=entity_vocab_size,  # ✅ 使用已初始化的 vocab_size
            # decoder 参数
            use_entity_decoder=True,
            decoder_num_layers=4,
            decoder_num_heads=4,
            decoder_max_seq_len=32,
        )

        model.to(device)

        model.freeze_encoders(freeze_image_encoder=args.frozen1, freeze_text_encoder=args.frozen2)

        # 打印可训练参数数量及名称
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")

        # 训练模型
        f1, auc, f1_model, auc_model = train(
            model=model,
            tokenizer=tokenizer,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            args=args,
            warmup_epochs=config.training["warmup_epochs"],
            fold=fold,
            train_dir=train_dir,
            best_f1=0,
            best_auc=0,
            use_entity_decoder=True,  # ✅ 启用 decoder
            decoder_weight=0.5,  # ✅ decoder loss 权重（可调）
        )

        if best_auc <= auc:
            best_auc = auc
            best_auc_model = auc_model
            torch.save(best_auc_model, f'{train_dir}/best_auc_model_fold_{fold+1}.pth')

        if best_f1 <= f1:
            best_f1 = f1
            best_f1_model = f1_model
            torch.save(best_f1_model, f'{train_dir}/best_f1_model_fold_{fold+1}.pth')

        fold_scores.append((f1, auc))
        fold_time = time.time() - fold_start_time
        print(f'Fold {fold + 1} completed in {fold_time:.2f}s with F1: {f1:.4f} AUC: {auc:.4f}')

        # 释放模型内存
        del model, vit3d_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 输出交叉验证结果
    fold_scores = np.array(fold_scores)
    cv_type_name = "聚类交叉验证" if config.cross_validation.get('use_clustering', False) else "分层交叉验证"
    print(f'\n{cv_type_name}结果:')
    for fold, score in enumerate(fold_scores):
        print(f'Fold {fold + 1}: F1 {score[0]:.4f}   AUC {score[1]:.4f}')
    mean_f1 = np.mean(fold_scores[:, 0])
    std_f1 = np.std(fold_scores[:, 0])
    mean_auc = np.mean(fold_scores[:, 1])
    std_auc = np.std(fold_scores[:, 1])
    print(f'Mean F1: {mean_f1:.4f} ± {std_f1:.4f}\n'
          f'Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}\n')

    cv_results_path = os.path.join(train_dir, 'cv_results.txt')
    with open(cv_results_path, 'w') as f:
        f.write(f"{cv_type_name}结果:\n")
        for fold, score in enumerate(fold_scores):
            f.write(f'Fold {fold + 1}: F1 {score[0]:.4f}   AUC {score[1]:.4f}\n')
        f.write(f'\nMean F1: {mean_f1:.4f} ± {std_f1:.4f}\n')
        f.write(f'Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}\n')
        f.write(f'\n交叉验证类型: {cv_type_name}\n')

    # 打印总运行时间
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n总运行时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")


if __name__ == '__main__':
    main()