import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class VQARADDataset(Dataset):
    def __init__(self, json_path, img_dir, split="train",
                 word2idx=None, answer2idx=None, max_q_len=30):
        """
        Initialize the VQA-RAD dataset.
        Parameters:
            json_path (str): Path of the JSON annotation file.
            img_dir (str): The directory where the image files are located.
            split (str): Data subset selection: "train" or "test".
            word2idx (dict): Optional. Mapping of words to indexes (will be automatically constructed during training if not provided).
            answer2idx (dict): Optional. Mapping from answers to indices (automatically constructed during training if not provided).
            max_q_len (int): The maximum length of the question sequence.
        """
        self.img_dir = img_dir
        self.max_q_len = max_q_len

        # Read the JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Divide the data according to the "split" criterion
        if split.lower() == "train":
            # Training set samples: The "phrase_type" is either "freeform" or "para"
            self.data = [item for item in data
                         if item.get("phrase_type") in ("freeform", "para")]
        elif split.lower() == "test":
            # Test set samples: The "phrase_type" is either "test_freeform" or "test_para"
            self.data = [item for item in data
                         if item.get("phrase_type") in ("test_freeform", "test_para")]
        else:
            self.data = data  # If not specified, use all the data.

        # If no dictionary is provided, it will be constructed based on the training set data.
        if word2idx is None or answer2idx is None:
            train_items = [item for item in data
                           if item.get("phrase_type") in ("freeform", "para")]
            word2idx, answer2idx = self.build_vocab(train_items)
        self.word2idx = word2idx
        self.answer2idx = answer2idx

        # Define image preprocessing transformation
        if split.lower() == "train":
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

        # Record the positions of the "yes/no" answers in the data (distinguish between closed-ended and open-ended questions during evaluation)
        self.closed_indices = [i for i, item in enumerate(self.data)
                               if str(item["answer"]).strip().lower() in ["yes", "no"]]
        self.open_indices = [i for i, item in enumerate(self.data)
                             if str(item["answer"]).strip().lower() not in ["yes", "no"]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Return the image tensor, question tensor and answer label of the index sample."""
        item = self.data[idx]
        # Load the image and perform preprocessing
        img_name = item.get("image_name") or item.get("image")  # The image field in JSON might be called "image_name".
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        # Process the problem text -> Index sequence
        question = str(item["question"]).lower().strip()
        words = question.split()
        # Map each word to an index (unregistered words are mapped to the index of <unk>)
        seq = [self.word2idx.get(w, self.word2idx.get("<unk>")) for w in words]
        # Truncate or pad the problem sequence to a fixed length of max_q_len
        if len(seq) < self.max_q_len:
            seq += [self.word2idx.get("<pad>")] * (self.max_q_len - len(seq))
        else:
            seq = seq[:self.max_q_len]
        import torch
        question_tensor = torch.tensor(seq, dtype=torch.long)
        # Process answer text -> Answer label index
        answer_text = str(item["answer"]).lower().strip()
        # If the answer is not in the mapping (theoretically, it should all be in the training dictionary), then use the index of <unk>
        label_idx = self.answer2idx.get(answer_text, self.answer2idx.get("<unk>"))
        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        return image, question_tensor, label_tensor

    @staticmethod
    def build_vocab(data_items):
        """
        Build a vocabulary list and an answer mapping table.
        Return:
            word2idx: Dictionary of words -> Index of words
            answer2idx: Answer -> Dictionary of Indexes
        """
        # Initialize special markers
        word2idx = {"<pad>": 0, "<unk>": 1}
        answer2idx = {"<unk>": 0}
        word_index = 2  # The next available index (0 and 1 are occupied)
        answer_index = 1  # The answers start from 1 (0 is reserved for the "unknown answer" option)
        for item in data_items:
            # Traverse all the problem texts
            question = str(item["question"]).lower().strip()
            for w in question.split():
                if w not in word2idx:
                    word2idx[w] = word_index
                    word_index += 1
            # Traverse all the answer texts
            answer = str(item["answer"]).lower().strip()
            if answer not in answer2idx:
                answer2idx[answer] = answer_index
                answer_index += 1
        return word2idx, answer2idx
