import numpy as np
from monai.transforms import (
    Compose, LoadImaged, Orientationd,
    CenterSpatialCropd, SpatialPadd, NormalizeIntensityd, ToTensord,
    RandRotate90d, RandAffined, RandGaussianSmoothd, RandScaleIntensityd,
    RandShiftIntensityd, RandGaussianNoised, RandFlipd, Rand3DElasticd, RandAdjustContrastd, RandCoarseDropoutd, Lambdad
)

def create_transforms(config, args):
    """
    创建 MONAI 转换管线，用于 3D 医学图像的增强、裁剪和归一化.
    此转换管线默认接收一个字典，其中 'image' 为图像路径，'label' 为标签。
    """

    # 获取模型参数中的 3D 图像大小
    image_size = config.model["params"]["image_size"]
    crop_size = config.training["crop_size"]
    # 解析中心裁剪尺寸
    if crop_size:
        try:
            if isinstance(crop_size, int):
                crop_size = (crop_size, crop_size, crop_size)
            else:
                crop_size = crop_size
        except:
            print(f"警告: 无法解析中心裁剪尺寸，使用默认值 ({image_size}x{image_size}x{image_size})")
            crop_size = (image_size, image_size, image_size)
    else:
        crop_size = (image_size, image_size, image_size)

    print(f"使用裁剪尺寸: {crop_size}")

    # 重点: 首先必须加载图像，然后才能应用空间变换
    # 所有的变换都需要先完成图像加载
    base_transforms = [
        # 1. 数据加载 - 必须是第一步
        LoadImaged(keys=["image"], ensure_channel_first=True),
        # # 统一方向
        Orientationd(keys=["image"], axcodes="RAS"),
    ]

    # 训练时数据增强 - 在加载图像后立即应用
    augment_transforms = []
    if config.training['augment']:
        print("启用数据增强...")
        augment_transforms = [
            # 1. 随机翻转：在所有三个方向（上/下, 左/右, 前/后）随机翻转图像
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=[0, 1, 2]),

            # 2. 随机仿射变换：包括旋转、缩放、平移，这是最重要的几何增强
            RandAffined(
                keys=["image"],
                prob=0.3,  # 50%的概率执行
                rotate_range=(np.pi / 12, np.pi / 12, np.pi / 12),  # 在各轴上随机旋转 ±15度
                scale_range=(0.9, 1.1),  # 随机缩放 90% 到 110%
                mode="bilinear",
                padding_mode="border",
            ),
            # 3. 随机调整对比度
            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.8, 1.2)),

            # 4. 随机增加高斯噪声
            RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.1),
        ]

    # 3. 剩余的标准处理 - 所有数据都需要的步骤
    final_transforms = [
        # 强度归一化
        NormalizeIntensityd(keys=["image"]),
        # 中心裁剪
        CenterSpatialCropd(keys=["image"], roi_size=crop_size),
        # 如果尺寸小于目标尺寸，进行填充
        SpatialPadd(keys=["image"], spatial_size=crop_size),
        # 转为Tensor - 放在最后
        ToTensord(keys=["image", "label"])
    ]

    # 组合所有转换
    if config.training['augment']:
        train_transforms = Compose(base_transforms + augment_transforms + final_transforms)
        val_transforms = Compose(base_transforms + final_transforms)  # 验证集不需要增强
    else:
        # 不使用数据增强
        train_transforms = Compose(base_transforms + final_transforms)
        val_transforms = Compose(base_transforms + final_transforms)

    return train_transforms, val_transforms