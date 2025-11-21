import argparse


def parse_args(config):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str, default = 'config/config.yaml',
                        help = 'Path to config file')
    parser.add_argument('--epochs', type = int, default = config.training["num_epochs"],
                        help = 'training epochs')
    parser.add_argument('--output_dir', type = str, nargs = '+', default = config.data["output_dir"],
                        help = 'train output dir')
    parser.add_argument('--class_names', type = str, nargs = '+', default = config.data['class_names'],
                        help = 'class name')
    parser.add_argument('--batch_size', type = int, default = config.training["batch_size"],
                        help = 'Override batch size in config')
    parser.add_argument('--use_amp', type = bool, default = config.training["use_amp"],
                        help = 'Use automatic mixed precision training')
    parser.add_argument('--cache_dataset', action = 'store_true', default = config.training["cache_dataset"],
                        help = 'Cache entire dataset in memory (speeds up training but uses more RAM)')
    parser.add_argument('--cache_dir', type = str, default = None,
                        help = 'Directory for caching dataset on disk (using PersistentDataset)')
    parser.add_argument('--warmup_epochs', type = int, default = config.training["warmup_epochs"],
                        help = 'Number of warmup epochs')
    parser.add_argument('--warmup_type', type = str, default = config.training["warmup_type"],
                        choices = ['linear', 'exponential'],
                        help = 'Type of learning rate warmup')
    parser.add_argument('--max_grad_norm', type = float, default = config.training["max_grad_norm"],
                        help = 'Maximum gradient norm for clipping')
    parser.add_argument('--crop_size', type = int, nargs = '+', default = config.training["crop_size"],
                        help = 'Apply center cropping with size format "DxHxW", e.g. "64x64x64"')
    parser.add_argument('--augment', action = 'store_true', default = config.training["augment"],
                        help = 'Apply medical image augmentation')
    parser.add_argument('--image_size', type = int, nargs = '+', default = config.model['params']["image_size"],
                        help = '3D input image size')
    parser.add_argument('--num_classes', type = int, default = config.data['num_classes'],
                        help = 'num_classes')
    parser.add_argument('--pretrained_path', type = str, default = config.training['pretrained_path'],
                        help = 'pretrained_path')
    parser.add_argument('--lr', type = float, default = config.optimizer['params']['lr'],
                        help = 'learning rate')
    parser.add_argument('--patch_size', type = int, default = config.model['params']["patch_size"],
                        help = 'patch size')
    parser.add_argument('--dim', type = int, default = config.model['params']["dim"],
                        help = 'embed dim')
    parser.add_argument('--depth', type = int, default = config.model['params']["depth"],
                        help = 'VIT encoder layer number')
    parser.add_argument('--heads', type = int, default = config.model['params']["heads"],
                        help = 'VIT encoder head number')
    parser.add_argument('--pool', type = str, default = config.model['params']["pool"],
                        help = 'pool: max min all')
    parser.add_argument('--cpt_num', type=int, default=config.model['params']["cpt_num"],
                        help='Class prototype Token num')
    parser.add_argument('--mlp_num', type=int, default=config.model['params']["mlp_num"],
                        help='MLP 层数')
    parser.add_argument('--frozen1', type=bool, default=config.training['frozen1'],
                        help='是否冻结image_backbone')
    parser.add_argument('--frozen2', type=bool, default=config.training['frozen2'],
                        help='是否冻结text_backbone')

    # 添加损失函数选项
    parser.add_argument("--loss1", type = bool, default = config.losses['CrossEntropyLoss']['enabled'],
                        help = "是否使用CrossEntropyLoss, FocalLoss")
    parser.add_argument("--loss2", type = bool, default = config.losses['BoundaryFlowLoss']['enabled'],
                        help = "是否使用BoundaryFlowLoss")
    parser.add_argument("--loss3", type=bool, default=config.losses['SimilarityLoss']['enabled'],
                        help="是否添加SimilarityLoss")

    parser.add_argument("--loss1_weight", type = float, default = config.losses['CrossEntropyLoss']['weight'], help = "主损失函数权重")
    parser.add_argument("--loss2_weight", type = float, default = config.losses['BoundaryFlowLoss']['weight'], help = "辅助损失函数2权重")

    return parser.parse_args()
