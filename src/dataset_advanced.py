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
            transform=None,
            only_open: bool = False
    ):
        self.img_dir = img_dir
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len
        self.transform = transform 

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        else:
            self.tokenizer = tokenizer

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 1. Basic filtering
        if split.lower() == "train":
            self.data = [item for item in data if item.get("phrase_type") in ("freeform", "para")]
        elif split.lower() == "test":
            self.data = [item for item in data if item.get("phrase_type") in ("freeform", "para")]
        else:
            self.data = data
            
        self.data = [x for x in self.data if 'image_name' in x and 'question' in x and 'answer' in x]

        # 2. Only keep the open questions
        if only_open:
            print(f"⚠️ FILTERING: Keeping ONLY Open-Ended questions for {split}...")
            original_len = len(self.data)
            # Filter out the data whose answers are "yes" or "no".
            self.data = [x for x in self.data if str(x["answer"]).lower().strip() not in ["yes", "no"]]
            print(f"   -> Reduced from {original_len} to {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_name = item["image_name"]
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        question = str(item["question"])
        q_enc = self.tokenizer(
            question,
            max_length=self.max_q_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        q_ids = q_enc['input_ids'].squeeze(0)

        answer = str(item["answer"])
        a_ids_list = self.tokenizer.encode(answer, add_special_tokens=False)
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id

        seq = [cls_id] + a_ids_list + [sep_id]
        if len(seq) < self.max_a_len:
            seq += [pad_id] * (self.max_a_len - len(seq))
        else:
            seq = seq[:self.max_a_len]
            seq[-1] = sep_id
        
        a_ids = torch.tensor(seq, dtype=torch.long)
        return image, q_ids, a_ids