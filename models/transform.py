import numpy as np
from monai.transforms import (
    Compose, LoadImaged, Orientationd,
    CenterSpatialCropd, SpatialPadd, NormalizeIntensityd, ToTensord,
    RandRotate90d, RandAffined, RandGaussianSmoothd, RandScaleIntensityd,
    RandShiftIntensityd, RandGaussianNoised, RandFlipd, Rand3DElasticd, RandAdjustContrastd, RandCoarseDropoutd
)

def create_transforms(config, args):
    """
    创建 MONAI 转换管线，用于 3D 医学图像的增强、裁剪和归一化.
    此转换管线默认接收一个字典，其中 'image' 为图像路径，'label' 为标签。
    """

    # 获取模型参数中的 3D 图像大小
    image_size = config.model["params"]["image_size"]

    # 解析中心裁剪尺寸
    if args.crop_size:
        try:
            if isinstance(args.crop_size, int):
                crop_size = (args.crop_size, args.crop_size, args.crop_size)
            else:
                crop_size = args.crop_size
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
    if args.augment:
        print("启用数据增强...")
        augment_transforms = [
            RandFlipd(keys = ["image"], prob = 0.5, spatial_axis = [0]),
            RandRotate90d(keys = ["image"], prob = 0.5, spatial_axes = [0, 1]),
            # RandAffined(
            #     keys = ["image"],
            #     prob = 0.3,
            #     rotate_range = (np.pi / 30, np.pi / 30, np.pi / 30),
            #     scale_range = (0.05, 0.05, 0.05),
            #     translate_range = (1, 1, 1),
            #     mode = "bilinear",
            #     padding_mode = "border",
            # ),
            RandAdjustContrastd(keys = ["image"], prob = 0.5, gamma = (0.9, 1.1)),
            # RandGaussianNoised(keys = ["image"], prob = 0.2, mean = 0.0, std = 0.03),
        ]

    # 3. 剩余的标准处理 - 所有数据都需要的步骤
    final_transforms = [
        # # 强度归一化
        # NormalizeIntensityd(keys=["image"]),
        # 中心裁剪
        CenterSpatialCropd(keys=["image"], roi_size=crop_size),
        # 如果尺寸小于目标尺寸，进行填充
        SpatialPadd(keys=["image"], spatial_size=crop_size),
        # 转为Tensor - 放在最后
        ToTensord(keys=["image", "label"])
    ]

    # 组合所有转换
    if args.augment:
        train_transforms = Compose(base_transforms + augment_transforms + final_transforms)
        val_transforms = Compose(base_transforms + final_transforms)  # 验证集不需要增强
    else:
        # 不使用数据增强
        train_transforms = Compose(base_transforms + final_transforms)
        val_transforms = Compose(base_transforms + final_transforms)

    return train_transforms, val_transforms