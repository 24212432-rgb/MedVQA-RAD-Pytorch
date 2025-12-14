# src/dataset_advanced.py
import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class VQARADSeqDataset(Dataset):
    def __init__(
            self,
            json_path: str,
            img_dir: str,
            split: str = "train",
            max_q_len: int = 30,
            max_a_len: int = 30,
            tokenizer=None,
            transform=None  # <--- 关键修改：接收数据增强操作
    ):
        self.img_dir = img_dir
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len
        self.transform = transform  # <--- 保存 transform

        # 1. 初始化 Tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        else:
            self.tokenizer = tokenizer

        # 2. 读取数据
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 3. 划分数据集 (根据 JSON 中的 phrase_type 简单区分，或者根据 split 参数)
        # 这里为了确保拿到所有数据，我们假设外部已经做好了 split 逻辑，
        # 或者我们简单地把所有 freeform/para 当作有效数据，具体的训练/测试切分在 main 里通过 Subset 做
        self.data = [item for item in data if item.get("phrase_type") in ("freeform", "para")]

        # 简单过滤：必须有 image_name, question, answer
        self.data = [x for x in self.data if 'image_name' in x and 'question' in x and 'answer' in x]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # --- A. 处理图片 ---
        img_name = item["image_name"]
        img_path = os.path.join(self.img_dir, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except:
            # 容错：全黑图
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        # 关键：应用数据增强 (旋转、翻转等)
        if self.transform:
            image = self.transform(image)

        # --- B. 处理问题 ---
        question = str(item["question"])
        q_enc = self.tokenizer(
            question,
            max_length=self.max_q_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        q_ids = q_enc['input_ids'].squeeze(0)

        # --- C. 处理答案 ---
        answer = str(item["answer"])
        # 手动构建 Seq2Seq 目标: [CLS] answer [SEP]
        a_ids_list = self.tokenizer.encode(answer, add_special_tokens=False)
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id

        # 构建序列
        seq = [cls_id] + a_ids_list + [sep_id]

        # Padding
        if len(seq) < self.max_a_len:
            seq += [pad_id] * (self.max_a_len - len(seq))
        else:
            seq = seq[:self.max_a_len]
            # 确保最后是 SEP (如果被截断)
            seq[-1] = sep_id

        a_ids = torch.tensor(seq, dtype=torch.long)

        return image, q_ids, a_ids