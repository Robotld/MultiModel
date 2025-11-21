import os
import torch
from ruamel.yaml import YAML


class ConfigManager:
    def __init__(self, config_path):
        yaml = YAML()
        with open(config_path, "r", encoding='utf-8') as f:
            self.config = yaml.load(f)

        # 处理设备配置
        self.config['training']["device"] = torch.device(
            "cuda" if torch.cuda.is_available() and self.config['training']["device"] == "cuda"
            else "cpu"
        )

    def __getattr__(self, name):
        return self.config.get(name)

    def get_optimizer_params(self, model):
        return {
            "lr": float(self.optimizer["params"]["lr"]),
            "weight_decay": self.optimizer["params"]["weight_decay"]
        }

    def get_scheduler_params(self):
        return self.scheduler["params"]


def update_config_from_args(config, args):
    """
    将命令行参数的修改更新到配置字典中，并保持原始对象类型

    参数:
        config: 配置字典或配置管理器对象
        args: 解析后的命令行参数
    返回:
        与输入相同类型的更新后配置对象
    """
    import copy

    # 保存原始对象
    original_config = config
    is_config_manager = hasattr(config, 'config')

    # 确保我们有一个配置字典
    if is_config_manager:
        # 如果传入的是配置管理器对象
        config_dict = copy.deepcopy(config.config)
    elif isinstance(config, dict):
        # 如果传入的已经是字典
        config_dict = copy.deepcopy(config)
    else:
        # 尝试转换为字典
        try:
            config_dict = copy.deepcopy(dict(config))
        except (TypeError, ValueError):
            raise TypeError("config 必须是字典类型或有 config 属性的对象")

    # 获取命令行参数字典
    args_dict = vars(args)

    # 映射命令行参数到配置位置
    param_mapping = {
        "data_dir": ["data", "root_dirs"],
        "output_dir": ["data", "output_dir"],
        "batch_size": ["training", "batch_size"],
        "use_amp": ["training", "use_amp"],
        "cache_dataset": ["training", "cache_dataset"],
        "warmup_epochs": ["training", "warmup_epochs"],
        "warmup_type": ["training", "warmup_type"],
        "max_grad_norm": ["training", "max_grad_norm"],
        "crop_size": ["training", "crop_size"],
        "augment": ["training", "augment"],
        "frozen1": ["training", "frozen1"],
        "frozen2": ["training", "frozen2"],
        "image_size": ["model", "params", "image_size"],
        "num_classes": ["data", "num_classes"],
        "pretrained_path": ["training", "pretrained_path"],
        "lr": ["optimizer", "params", "lr"],
        "patch_size": ["model", "params", "patch_size"],
        "dim": ["model", "params", "dim"],
        "depth": ["model", "params", "depth"],
        "heads": ["model", "params", "heads"],
        "pool": ["model", "params", "pool"],
        "cpt_num": ["model", "params", "cpt_num"],
        "mlp_num": ["model", "params", "mlp_num"],
        "loss1": ["losses", 'CrossEntropyLoss', 'enabled'],
        "loss2": ["losses", 'BoundaryFlowLoss', 'enabled'],
        "loss3": ["losses", 'SimilarityLoss', 'enabled'],
        "--loss1_weight": ["losses", 'CrossEntropyLoss', 'weight'],
        "--loss2_weight": ["losses", 'BoundaryFlowLoss', 'weight'],
    }

    # 更新配置
    for arg_name, config_path in param_mapping.items():
        # 检查参数是否在命令行中设置
        if arg_name in args_dict:
            arg_value = args_dict[arg_name]

            # 跳过None值或空列表
            if arg_value is None or (isinstance(arg_value, list) and not arg_value):
                continue
            try:
                # 导航到配置中的正确位置
                config_section = config_dict
                for i, key in enumerate(config_path):
                    if i == len(config_path) - 1:
                        # 设置最终值
                        config_section[key] = arg_value
                    else:
                        config_section = config_section[key]
            except (KeyError, TypeError) as e:
                print(f"警告: 无法设置配置项 {'.'.join(config_path)}: {e}")

    # 根据原始类型返回更新后的配置
    if is_config_manager:
        # 如果原始输入是ConfigManager对象，则更新其内部配置并返回对象本身
        original_config.config = config_dict
        return original_config
    else:
        # 否则返回更新后的字典
        return config_dict


def save_config(config, save_path):
    """保存配置到YAML文件，键不加引号，值保持适当格式"""
    import os
    from copy import deepcopy
    from ruamel.yaml import YAML

    # 提取配置
    config_dict = deepcopy(config.config if hasattr(config, 'config') else config)

    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 处理特殊对象
    config_dict = process_special_objects(config_dict)

    # 配置YAML
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.width = 80

    # 自定义字符串表示，只为值添加引号，键名不加
    yaml.representer.add_representer(str, represent_str)

    # 保存
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f)


def represent_str(representer, data):
    """字符串表示函数，为值添加适当引号"""
    # 判断是否为键名（通过调用堆栈上下文）- ruamel.yaml内部会处理这点
    # 为字符串值添加引号
    if '\\' in data or '\n' in data or ':' in data:
        return representer.represent_scalar('tag:yaml.org,2002:str', data, style='"')
    elif data and (data[0] in '!&*' or data.isspace()):
        return representer.represent_scalar('tag:yaml.org,2002:str', data, style="'")
    return representer.represent_scalar('tag:yaml.org,2002:str', data)


def process_special_objects(obj):
    """递归处理特殊对象，确保可序列化"""
    if isinstance(obj, dict):
        return {k: process_special_objects(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [process_special_objects(i) for i in obj]
    elif not isinstance(obj, (int, float, bool, str, type(None))):
        # 处理非基本类型
        return str(obj)
    return obj
