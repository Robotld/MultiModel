import csv
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Union, List


class MultimodalDataset(Dataset):
    """
    多模态数据集：图像 + 文本 + 人口学特征
    新增：返回原始文本以支持遮挡关键词预测
    """

    def __init__(self, csv_files: Union[str, List[str]], transform=None, text_tokenizer=None,
                 max_length=512, return_original_text=True):  # 🔥 新增参数
        self.transform = transform
        self.text_tokenizer = text_tokenizer
        self.max_length = max_length
        self.return_original_text = return_original_text  # 🔥 是否返回原始文本
        self.samples = []

        df_list = []
        for csv_path in csv_files:
            df_list.append(pd.read_csv(csv_path))

        df = pd.concat(df_list, ignore_index=True)

        # 性别编码
        gender_map = {'男': 1, '女': 0}
        # 标签编码
        recurrence_map = {'有': 1, '是': 1, '无': 0, '否': 0, '0': 0, '1': 1}
        class_map = {'IA': 1, 'AIS': 0, 'MIA': 0}

        # 百分数转浮点数函数
        def parse_percentage(val):
            if pd.isnull(val):
                return None
            try:
                s = str(val).strip()
                if s.endswith('%'):
                    return float(s[:-1]) / 100.0
                else:
                    return float(s)
            except Exception:
                return None

        # 结节大小特征提取
        def parse_nodule_size(val):
            if pd.isnull(val):
                return None, None, None, None
            s = str(val).replace('×', '*').replace('X', '*').strip()
            nums = []
            for x in s.split('*'):
                x = x.strip()
                try:
                    nums.append(float(x) / 10.0)  # 转换为 cm
                except Exception:
                    pass
            if len(nums) == 0:
                return None, None, None, None
            if len(nums) == 1:
                long_axis = short_axis = nums[0]
            else:
                long_axis, short_axis = max(nums), min(nums)
            area = long_axis * short_axis
            diff = abs(long_axis - short_axis)
            return long_axis, short_axis, area, diff

        for idx, row in df.iterrows():
            gender = gender_map.get(str(row['性别']).strip(), None)
            age = None
            try:
                age = float(str(row['年龄']).replace('岁', '').strip())
            except Exception:
                pass

            # 吸烟史和结节部位
            smoke = None
            try:
                smoke = int(row['吸烟史'])
            except Exception:
                pass
            nodule_loc = None
            try:
                nodule_loc = int(row['结节位置'])
            except Exception:
                pass

            ct_number = row.get('CT号', '')
            ct_report = row.get('CT报告', '')

            # 拼接报告（你可以根据需要调整）
            report = f"{ct_report}".strip()

            long_axis, short_axis, area, diff = parse_nodule_size(row.get('结节大小', ''))
            img_path = row.get('路径', '')
            label = row['类别'] if '类别' in row else None
            if label in class_map:
                label = class_map[label]

            # 跳过任何关键字段为空的情况
            if any([
                gender is None, age is None, smoke is None,
                nodule_loc is None, long_axis is None, short_axis is None,
                area is None, diff is None,
                not img_path or pd.isnull(img_path),
                label is None,
                not report  # 🔥 确保报告不为空
            ]):
                continue

            self.samples.append({
                "ct_number": ct_number,
                "gender": gender,
                "age": age,
                "smoke": smoke,
                "nodule_loc": nodule_loc,
                "long_axis": long_axis,
                "short_axis": short_axis,
                "area": area,
                "diff": diff,
                "ct_report": str(ct_report),
                "report": report,  # 🔥 保留完整报告文本
                "image_path": str(img_path),
                "label": label
            })

        print(f"数据集加载完毕，有效样本数：{len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 人口学特征
        demographic = torch.tensor([
            sample["gender"],
            sample["age"],
            sample["smoke"],
            sample["nodule_loc"],
        ], dtype=torch.float32)

        # 🔥 文本特征：Tokenize
        text_inputs = {}
        original_text = ""  # 🔥 默认值

        if self.text_tokenizer:
            # 拼接后的报告
            report_text = sample["report"]
            report_encoding = self.text_tokenizer(
                report_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            text_inputs = {k: v.squeeze(0) for k, v in report_encoding.items()}

            # 🔥 保存原始文本
            if self.return_original_text:
                original_text = report_text

        # 回归标签：结节长短径
        nodule_axis_label = torch.tensor(
            [sample["long_axis"], sample["short_axis"]],
            dtype=torch.float32
        )

        # 🔥 返回的数据字典
        data = {
            "ct_number": sample["ct_number"],
            "demographic": demographic,
            "text": text_inputs,  # Tokenized文本
            "original_text": original_text,  # 🔥 新增：原始文本字符串
            "label": sample["label"],  # 分类标签
            "nodule_axis_label": nodule_axis_label,
            "image": sample["image_path"]
        }

        # 图像变换
        if self.transform and data["image"] is not None:
            image_dict = {"image": data["image"], "label": data["label"]}
            transformed = self.transform(image_dict)
            data["image"] = transformed["image"]

        return data