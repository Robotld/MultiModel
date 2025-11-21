import numpy as np
from collections import Counter
from monai.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from torch.utils.data import Subset, Dataset
from models import BalancedBatchSampler
from torch.utils.data._utils.collate import default_collate


def custom_collate_fn(batch):
    if isinstance(batch[0], dict):
        out = {}
        for k in batch[0].keys():
            try:
                out[k] = default_collate([b[k] for b in batch])
            except Exception as e:
                types = [type(b[k]).__name__ for b in batch]
                raise TypeError(f"[collate] key '{k}' failed. example types: {types[:8]} ...") from e
        return out
    return default_collate(batch)

class MultimodalTransformDataset(Dataset):
    """应用MONAI转换的多模态数据集包装器"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

        if hasattr(dataset, '_data_counter'):
            self._data_counter = dataset._data_counter

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_dict = self.dataset[idx]
        if not self.transform:
            return data_dict
        result_dict = data_dict.copy()
        if isinstance(data_dict["image"], str):
            image_dict = {"image": data_dict["image"], "label": data_dict["label"]}
            transformed = self.transform(image_dict)
            result_dict["image"] = transformed["image"]
        return result_dict

class MulCrossValidator:
    """多模态多任务K折交叉验证管理器（支持聚类分折，支持长短径回归联合任务）"""

    def __init__(self, dataset, config):
        self.dataset = dataset
        self.cfg_cv = config.cross_validation
        self.cfg_train = config.training

        self.all_samples = [dataset[i] for i in range(len(dataset))]
        self.all_labels = np.array([sample['label'] for sample in self.all_samples])
        self.all_ct_numbers = np.array([sample['ct_number'] for sample in self.all_samples])
        self.all_nodule_axes = np.stack([sample['nodule_axis_label'].cpu().numpy() for sample in self.all_samples])

        print("\n------- 数据集类别分布 -------")
        label_counter = Counter(self.all_labels)
        for label, count in sorted(label_counter.items()):
            print(f"  类别 {label}: {count} 个样本 ({(count / len(self.all_labels)) * 100:.2f}%)")
        print("样本总数:", len(self.all_labels))
        print("----------------------------\n")

        base = self.dataset  # MultimodalRecurrenceDataset
        raw = base.samples  # list[dict]，与索引一一对应

        self._stats_arrays = {
            "age": np.array([r["age"] for r in raw], dtype=float),
            "long_axis": np.array([r["long_axis"] for r in raw], dtype=float),  # cm
            "short_axis": np.array([r["short_axis"] for r in raw], dtype=float),  # cm
            "area": np.array([r["area"] for r in raw], dtype=float),  # cm^2（若你想一起报）
            "diff_axis": np.array([r["diff"] for r in raw], dtype=float),  # cm（长-短）
            "gender": np.array([r["gender"] for r in raw], dtype=int),  # 1=男,0=女
            "smoke": np.array([r["smoke"] for r in raw], dtype=int),
            "nodule_loc": np.array([r["nodule_loc"] for r in raw], dtype=int),
            # "surgical": np.array([r["surgical_procedure"] for r in raw], dtype=int),  # 1/2/3
            "label": np.array([r["label"] for r in raw], dtype=int),  # 0/1/2
        }
        # 名称映射（打印更友好）
        self._name_maps = {
            "gender": {1: "Male", 0: "Female"},
            "surgical": {1: "Lobectomy", 2: "Segmentectomy", 3: "Wedge resection"},
            "label": {0: "高风险", 1: "中风险", 2: "低风险"},
            "nodule_loc":{1:"Left upper lobe",3:"Left lower lobe", 4:"Right upper lobe", 5:"Right middle lobe", 6:"Right lower lobe"},
            # nodule_loc/ smoke 你若有明确枚举可在此补映射；否则保留数字
        }
        # 保存每折统计结果
        self.fold_stats = []


        self.use_clustering = self.cfg_cv.get('use_clustering', False)
        if self.use_clustering:
            self.splits = None
        else:
            self.skf = StratifiedKFold(
                n_splits=self.cfg_cv["n_splits"],
                shuffle=self.cfg_cv["shuffle"],
                random_state=self.cfg_train["random_seed"]
            )
            self.splits = list(self.skf.split(np.zeros(len(self.all_labels)), self.all_labels))

    # ---------- 统计辅助 ----------
    @staticmethod
    def _median_iqr(x: np.ndarray):
        """返回 (median, q1, q3)，若为空则返回 (nan, nan, nan)"""
        if x.size == 0:
            return float("nan"), float("nan"), float("nan")
        med = float(np.median(x))
        q1  = float(np.percentile(x, 25))
        q3  = float(np.percentile(x, 75))
        return med, q1, q3

    @staticmethod
    def _counts(x: np.ndarray):
        """返回按键排序的计数字典（键可能是数字）"""
        c = Counter(x.tolist())
        return {k: c[k] for k in sorted(c.keys())}

    def _apply_name_map(self, counts: dict, name_map: dict = None):
        """把数值键映射为中文名；name_map 为空则原样返回"""
        if not name_map:
            return counts
        return {name_map.get(k, k): v for k, v in counts.items()}

    def _compute_stats_for_indices(self, indices: np.ndarray):
        A = self._stats_arrays  # 简写

        # 连续变量：年龄、结节长径/短径（单位 cm）
        age_med, age_q1, age_q3         = self._median_iqr(A["age"][indices])
        long_med, long_q1, long_q3      = self._median_iqr(A["long_axis"][indices])
        short_med, short_q1, short_q3   = self._median_iqr(A["short_axis"][indices])


        # 类别计数
        gender_cnt  = self._apply_name_map(self._counts(A["gender"][indices]),   self._name_maps.get("gender"))
        smoke_cnt   = self._counts(A["smoke"][indices])        # 若有枚举可映射
        loc_cnt     = self._apply_name_map(self._counts(A["nodule_loc"][indices]),   self._name_maps.get("nodule_loc"))
        # surg_cnt    = self._apply_name_map(self._counts(A["surgical"][indices]), self._name_maps.get("surgical"))
        label_cnt   = self._apply_name_map(self._counts(A["label"][indices]),    self._name_maps.get("label"))

        return {
            "N": len(indices),
            "age":   (age_med, age_q1, age_q3),
            "long":  (long_med, long_q1, long_q3),
            "short": (short_med, short_q1, short_q3),
            # "area":  (area_med, area_q1, area_q3),
            # "diff":  (diff_med, diff_q1, diff_q3),
            "gender_counts":  gender_cnt,
            "smoke_counts":   smoke_cnt,
            "loc_counts":     loc_cnt,
            # "surgical_counts":surg_cnt,
            "label_counts":   label_cnt,
        }

    @staticmethod
    def _fmt_med_iqr(med, q1, q3, decimals=2):
        f = lambda v: f"{v:.{decimals}f}" if np.isfinite(v) else "nan"
        return f"{f(med)} ({f(q1)}–{f(q3)})"

    def _print_stats_block(self, title: str, stats: dict):
        print(f"\n—— {title} ——")
        print(f"样本数: {stats['N']}")
        # 年龄 & 结节大小（中位数[IQR]）
        print(f"年龄（岁）: {self._fmt_med_iqr(*stats['age'])}")
        print(f"结节长径（cm）: {self._fmt_med_iqr(*stats['long'])}")
        print(f"结节短径（cm）: {self._fmt_med_iqr(*stats['short'])}")
        # 若需要一起打印面积/长短径差值，解开以下两行
        # print(f"结节面积（cm²）: {self._fmt_med_iqr(*stats['area'])}")
        # print(f"长短径差值（cm）: {self._fmt_med_iqr(*stats['diff'])}")

        # 其它类别计数
        def p(name, d):
            items = [f"{k}:{v}" for k, v in d.items()]
            print(f"{name}: " + (", ".join(items) if items else "（无）"))

        p("性别", stats["gender_counts"])
        p("吸烟史", stats["smoke_counts"])
        p("结节位置", stats["loc_counts"])
        # p("手术方式", stats["surgical_counts"])
        p("标签/类别", stats["label_counts"])
        print("——————————————")


    def get_folds(self, train_transforms=None, val_transforms=None):
        if self.splits is None:
            raise ValueError("尚未初始化splits，请先执行setup_clustering_splits（如需聚类分折）")
        negative_ratio = self.cfg_cv.get('negative_ratio', None)
        for fold, (train_idx, val_idx) in enumerate(self.splits):
            # if negative_ratio is not None:
            #     pos_idx = [i for i in train_idx if self.all_labels[i]==1]
            #     neg_idx = [i for i in train_idx if self.all_labels[i]==0]
            #     target_neg = min(len(neg_idx), int(len(pos_idx)*negative_ratio))
            #     neg_idx_ds = random.sample(neg_idx, target_neg)
            #     train_idx = np.array(pos_idx + neg_idx_ds)
            #     print(f"\n--- Fold {fold+1} 阴性降采样 ---\n阳性:{len(pos_idx)} 阴性:{len(neg_idx)} → 采样后阴性:{len(neg_idx_ds)}\n")

            train_ds = MultimodalTransformDataset(Subset(self.dataset, train_idx), train_transforms)
            val_ds = MultimodalTransformDataset(Subset(self.dataset, val_idx), val_transforms)

            train_labels = [self.all_labels[i] for i in train_idx]

            sampler = BalancedBatchSampler(
                train_ds,
                self.cfg_train["batch_size"],
                labels=train_labels
            )

            print("\n------- Batch采样配置 -------")
            print(f'批次数量: {len(sampler)}')
            print(f"批次大小: {sampler.batch_size}")
            print(f"类别数量: {sampler.n_classes}")
            print(f"每个类别的样本数: {sampler.samples_per_class}")
            if hasattr(sampler, 'class_indices'):
                for label, indices in sampler.class_indices.items():
                    print(f"类别 {label}: 共有 {len(indices)} 个样本")
            print("---------------------------\n")

            train_loader = DataLoader(
                train_ds,
                batch_sampler=sampler,
                num_workers=self.cfg_train["num_workers"],
                pin_memory=True,
                persistent_workers=bool(self.cfg_train["num_workers"] > 0),
                collate_fn=custom_collate_fn
            )

            val_loader = DataLoader(
                val_ds,
                batch_size=self.cfg_train["batch_size"],
                num_workers=self.cfg_train["num_workers"],
                pin_memory=True,
                persistent_workers=bool(self.cfg_train["num_workers"] > 0),
                collate_fn=custom_collate_fn
            )

            # 统计
            train_ct_numbers = set(self.all_ct_numbers[train_idx])
            val_ct_numbers = set(self.all_ct_numbers[val_idx])
            intersection = train_ct_numbers.intersection(val_ct_numbers)
            print(f"\n=== Fold {fold+1} 数据集检查 ===")
            print(f"训练集样本数: {len(train_idx)}, 验证集样本数: {len(val_idx)}")
            print(f"训练CT号数: {len(train_ct_numbers)}, 验证CT号数: {len(val_ct_numbers)}")
            print(f"训练/验证 CT号交集数: {len(intersection)}")
            if intersection:
                print(f"⚠️ 存在 CT号交集（数据泄露风险）: {sorted(intersection)}")
            else:
                print("✅ 无 CT号交集，数据划分无泄露")
            tr_labels = Counter([self.all_labels[i] for i in train_idx])
            va_labels = Counter([self.all_labels[i] for i in val_idx])
            print("训练集标签分布:", tr_labels)
            print("验证集标签分布:", va_labels)

            # 长短径标签统计
            train_axes = self.all_nodule_axes[train_idx]
            val_axes = self.all_nodule_axes[val_idx]
            print("训练集长径均值%.2f±%.2f, 短径%.2f±%.2f" % (
                train_axes[:,0].mean(), train_axes[:,0].std(), train_axes[:,1].mean(), train_axes[:,1].std()))
            print("验证集长径均值%.2f±%.2f, 短径%.2f±%.2f" % (
                val_axes[:,0].mean(), val_axes[:,0].std(), val_axes[:,1].mean(), val_axes[:,1].std()))
            print("======================================\n")

            # ——— 每一折：患者信息统计（年龄/结节大小=中位数[IQR]；其余=类别计数） ———
            train_stats = self._compute_stats_for_indices(train_idx)
            val_stats   = self._compute_stats_for_indices(val_idx)

            self._print_stats_block(f"Fold {fold+1} 训练集患者信息", train_stats)
            self._print_stats_block(f"Fold {fold+1} 验证集患者信息", val_stats)

            # 保存起来，便于后续导出/可视化
            self.fold_stats.append({
                "fold": fold + 1,
                "train": train_stats,
                "val": val_stats
            })


            yield fold, train_loader, val_loader, tr_labels