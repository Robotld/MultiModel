import numpy as np
from torch.utils.data import Subset


class BalancedBatchSampler:
    """改进的批次类别平衡采样器，支持灵活采样比例和适度重采样"""

    def __init__(self, dataset, batch_size, labels=None, indices=None,
                 pos_ratio=0.5, max_batches=None, minority_multiplier=5):
        """
        初始化采样器
        Args:
            dataset: 数据集对象
            batch_size: 批次大小
            labels: 可选的标签数组
            indices: 可选的索引数组
            pos_ratio: 正样本(少数类)在每个批次中的比例，默认0.35
            max_batches: 每个epoch最大批次数，控制过采样程度
            minority_multiplier: 少数类允许的最大重采样倍数，默认3
        """
        self.batch_size = batch_size
        self.pos_ratio = pos_ratio
        self.minority_multiplier = minority_multiplier

        # 处理数据集和索引
        if isinstance(dataset, Subset):
            self.dataset = dataset.dataset
            self.indices = dataset.indices if indices is None else indices
            self.is_subset_input = True
        else:
            self.dataset = dataset
            self.indices = range(len(dataset)) if indices is None else indices
            self.is_subset_input = False

        # 获取标签
        if labels is not None:
            self.labels = np.array(labels)
        else:
            self.labels = np.array([self.dataset[idx][1] for idx in self.indices])

        # 分析类别
        self.unique_labels = np.unique(self.labels)
        self.n_classes = len(self.unique_labels)

        # 计算各类别样本数
        label_counts = {label: np.sum(self.labels == label) for label in self.unique_labels}

        # 假设二分类问题中标签1为少数类(正例)
        if self.n_classes == 2:
            self.minority_label = min(label_counts, key=label_counts.get)
            self.majority_label = max(label_counts, key=label_counts.get)
        else:
            # 多分类情况使用均衡采样
            self.pos_ratio = 1.0 / self.n_classes

        # 按类别分组索引
        self.class_indices = {
            label: np.where(self.labels == label)[0] for label in self.unique_labels
        }

        # 计算每个批次中各类别的样本数
        if self.n_classes == 2:
            # 确保少数类至少有1个样本
            minority_samples = max(1, int(self.batch_size * self.pos_ratio))
            # 确保总数不超过batch_size
            majority_samples = self.batch_size - minority_samples

            self.samples_per_class = {
                self.minority_label: minority_samples,
                self.majority_label: majority_samples
            }
        else:
            # 多分类采样
            self.samples_per_class = {
                label: max(1, self.batch_size // self.n_classes) for label in self.unique_labels
            }
            # 处理除法余数
            total_allocated = sum(self.samples_per_class.values())
            if total_allocated < self.batch_size:
                remainder = self.batch_size - total_allocated
                for i, label in enumerate(self.unique_labels):
                    if i < remainder:
                        self.samples_per_class[label] += 1

        # 改进批次数量计算逻辑
        if self.n_classes == 2:
            # 考虑少数类和多数类的利用率
            minority_size = len(self.class_indices[self.minority_label])
            majority_size = len(self.class_indices[self.majority_label])

            # 少数类允许重采样的批次数
            minority_batches = (minority_size * self.minority_multiplier) // self.samples_per_class[self.minority_label]

            # 多数类不重采样的批次数
            majority_batches = majority_size // self.samples_per_class[self.majority_label]

            # 取较小值，确保两类都能充分利用
            reasonable_batches = min(minority_batches, majority_batches)

        else:
            # 多分类情况：考虑所有类的利用率和重采样
            class_batches = []
            for label in self.unique_labels:
                class_size = len(self.class_indices[label])
                # 对于小类允许更多重采样
                multiplier = self.minority_multiplier if class_size < len(self.indices) / self.n_classes else 1
                possible_batches = (class_size * multiplier) // self.samples_per_class[label]
                class_batches.append(possible_batches)

            reasonable_batches = min(class_batches)

        # 如果指定了最大批次数，使用较小值
        self.batches_per_epoch = reasonable_batches if max_batches is None else min(reasonable_batches, max_batches)

        print(f"每批次包含 {self.samples_per_class} 个样本")
        print(f"每个epoch生成 {self.batches_per_epoch} 个批次")

    def __iter__(self):
        # 为每个类别创建采样器
        indices_by_label = {}
        for label, indices in self.class_indices.items():
            # 复制并打乱索引
            shuffled_indices = indices.copy()
            np.random.shuffle(shuffled_indices)
            indices_by_label[label] = shuffled_indices

        # 生成批次
        for _ in range(self.batches_per_epoch):
            batch = []

            # 从每个类别选择所需数量的样本
            for label, count in self.samples_per_class.items():
                # 如果当前类别的索引用完，重新打乱
                if len(indices_by_label[label]) < count:
                    new_indices = self.class_indices[label].copy()
                    np.random.shuffle(new_indices)
                    indices_by_label[label] = np.concatenate([indices_by_label[label], new_indices])

                # 添加样本到批次
                batch.extend(indices_by_label[label][:count])
                indices_by_label[label] = indices_by_label[label][count:]

            # 打乱批次内的顺序
            np.random.shuffle(batch)

            # 如果需要，转换回原始索引
            if self.is_subset_input:
                batch = [self.indices[i] for i in batch]

            yield batch

    def __len__(self):
        return self.batches_per_epoch