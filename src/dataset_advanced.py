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
            transform=None  # <--- Key modification: Implement data augmentation operation
    ):
        self.img_dir = img_dir
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len
        self.transform = transform  # <--- save transform

        # 1. Initialization Tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        else:
            self.tokenizer = tokenizer

        # 2. Read data
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 3. Divide the dataset (simply distinguish based on the phrase_type in the JSON, or according to the split parameter)
        # Here, in order to ensure that all the data is obtained, we assume that the external system has already implemented the split logic.
        # Or we can simply treat all the freeform/para as valid data, and the specific training/testing split can be done in the main part through Subset.
        self.data = [item for item in data if item.get("phrase_type") in ("freeform", "para")]

        #Simple filtering: Must have image_name, question, answer
        self.data = [x for x in self.data if 'image_name' in x and 'question' in x and 'answer' in x]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # --- A. Process the image ---
        img_name = item["image_name"]
        img_path = os.path.join(self.img_dir, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except:
            # Fault tolerance: Full black image
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        # Key point: Apply data augmentation (rotation, flipping, etc.)
        if self.transform:
            image = self.transform(image)

        # --- B. Deal with the problem ---
        question = str(item["question"])
        q_enc = self.tokenizer(
            question,
            max_length=self.max_q_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        q_ids = q_enc['input_ids'].squeeze(0)

        # --- C. Process the answer ---
        answer = str(item["answer"])
        # Manually construct the Seq2Seq target: [CLS] answer [SEP]
        a_ids_list = self.tokenizer.encode(answer, add_special_tokens=False)
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id

        # Construct a sequence
        seq = [cls_id] + a_ids_list + [sep_id]

        # Padding
        if len(seq) < self.max_a_len:
            seq += [pad_id] * (self.max_a_len - len(seq))
        else:
            seq = seq[:self.max_a_len]
            # Make sure it ends with SEP (if truncated)
            seq[-1] = sep_id

        a_ids = torch.tensor(seq, dtype=torch.long)


        return image, q_ids, a_ids
