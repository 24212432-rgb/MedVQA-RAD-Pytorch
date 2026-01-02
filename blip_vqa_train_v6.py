"""
BLIP-VQA Fine-tuning for Medical VQA (V6 - Final Working Version)

This version uses BLIP's built-in loss because the model's output structure
doesn't expose logits in a standard way for custom loss computation.

Anti-overfitting is achieved through:
- Strong weight decay (0.1)
- Increased dropout (0.2)
- Layer freezing (vision=ALL, text=8/12 layers)
- Early stopping (patience=4)
- Data augmentation (50%)
"""

import os
import json
import random
import warnings
import logging
from pathlib import Path
from glob import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')
logging.getLogger('transformers').setLevel(logging.ERROR)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

from transformers import BlipProcessor, BlipForQuestionAnswering


# =============================================================================
# Data Path Discovery
# =============================================================================

def find_data_paths():
    search_roots = [
        "/content/drive/MyDrive",
        "/content/drive/My Drive",
        "/content",
        os.getcwd(),
        os.path.expanduser("~"),
        ".", ".."
    ]
    
    train_path = None
    test_path = None
    image_dir = None
    
    print("=" * 60)
    print("Searching for data files...")
    print("=" * 60)
    
    for root in search_roots:
        if not os.path.exists(root):
            continue
        
        for pattern in ["**/trainset.json", "**/train*.json"]:
            matches = glob(os.path.join(root, pattern), recursive=True)
            if matches:
                train_path = matches[0]
                data_dir = os.path.dirname(train_path)
                print(f"Found train: {train_path}")
                
                for test_name in ["testset.json", "test.json", "val.json"]:
                    candidate = os.path.join(data_dir, test_name)
                    if os.path.exists(candidate):
                        test_path = candidate
                        print(f"Found test:  {test_path}")
                        break
                break
        if train_path:
            break
    
    if train_path:
        data_dir = os.path.dirname(train_path)
        parent_dir = os.path.dirname(data_dir)
        
        image_folder_names = [
            "VQA_RAD Image Folder",
            "VQA_RAD_Image_Folder", 
            "images", "Images", "image", "imgs"
        ]
        
        for search_dir in [data_dir, parent_dir]:
            for folder_name in image_folder_names:
                candidate = os.path.join(search_dir, folder_name)
                if os.path.isdir(candidate):
                    img_files = glob(os.path.join(candidate, "*.jpg")) + \
                               glob(os.path.join(candidate, "*.png"))
                    if img_files:
                        image_dir = candidate
                        print(f"Found images: {image_dir} ({len(img_files)} files)")
                        break
            if image_dir:
                break
    
    print("=" * 60)
    return train_path, test_path, image_dir


def validate_paths(train_path, test_path, image_dir):
    print("\n" + "=" * 60)
    print("DATA PATH VALIDATION")
    print("=" * 60)
    
    valid = True
    
    if train_path and os.path.exists(train_path):
        with open(train_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        print(f"[OK] Train: {train_path} ({len(train_data)} samples)")
    else:
        print(f"[FAIL] Train: NOT FOUND")
        valid = False
    
    if test_path and os.path.exists(test_path):
        with open(test_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        print(f"[OK] Test: {test_path} ({len(test_data)} samples)")
    else:
        print(f"[FAIL] Test: NOT FOUND")
        valid = False
    
    if image_dir and os.path.isdir(image_dir):
        img_count = len(glob(os.path.join(image_dir, "*.*")))
        print(f"[OK] Images: {image_dir} ({img_count} files)")
    else:
        print(f"[FAIL] Images: NOT FOUND")
        valid = False
    
    print("=" * 60)
    return valid


# =============================================================================
# Configuration
# =============================================================================

class Config:
    model_name = "Salesforce/blip-vqa-capfilt-large"
    
    val_ratio = 0.15
    
    epochs = 15
    batch_size = 4
    gradient_accumulation = 4
    learning_rate = 2e-5
    weight_decay = 0.1          # Strong regularization
    max_grad_norm = 1.0
    
    max_question_length = 32
    max_answer_length = 12
    num_beams = 3
    
    freeze_vision_encoder = True
    freeze_text_encoder_layers = 8
    
    dropout_rate = 0.2          # Increased dropout
    augment_prob = 0.5          # Data augmentation
    
    patience = 4
    min_delta = 0.005
    
    seed = 42
    num_workers = 2
    
    @classmethod
    def get_device(cls):
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("\n[WARNING] CUDA not available, using CPU")
            return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Dataset
# =============================================================================

class VQARADDataset(Dataset):
    def __init__(self, data, image_dir, processor, is_train=True):
        self.data = data
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.is_train = is_train
        
        self.close_indices = set()
        self.open_indices = set()
        self._classify_questions()
    
    def _classify_questions(self):
        for i, item in enumerate(self.data):
            answer = self._normalize_answer(item['answer'])
            if answer in ['yes', 'no']:
                self.close_indices.add(i)
            else:
                self.open_indices.add(i)
    
    def _normalize_answer(self, answer):
        if answer is None:
            return ""
        ans = str(answer).lower().strip()
        ans = ' '.join(ans.split())
        return ans
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        img_name = item.get('image_name', item.get('image', item.get('img_name', '')))
        img_path = self.image_dir / img_name
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (384, 384), color='black')
        
        if self.is_train and random.random() < Config.augment_prob:
            image = self._augment(image)
        
        question = item['question']
        answer = self._normalize_answer(item['answer'])
        
        encoding = self.processor(
            images=image,
            text=question,
            padding='max_length',
            truncation=True,
            max_length=Config.max_question_length,
            return_tensors='pt'
        )
        
        answer_encoding = self.processor.tokenizer(
            answer,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=Config.max_answer_length,
            return_tensors='pt'
        )
        
        decoder_input_ids = answer_encoding['input_ids'].squeeze(0)
        decoder_attention_mask = answer_encoding['attention_mask'].squeeze(0)
        
        labels = decoder_input_ids.clone()
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        
        return {
            'pixel_values': encoding['pixel_values'].squeeze(0),
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'decoder_input_ids': decoder_input_ids,
            'decoder_attention_mask': decoder_attention_mask,
            'labels': labels,
            'answer_text': answer,
            'is_close': idx in self.close_indices,
            'question': question
        }
    
    def _augment(self, image):
        arr = np.array(image).astype(np.float32)
        
        if random.random() < 0.5:
            arr = arr * random.uniform(0.8, 1.2)
        
        if random.random() < 0.5:
            mean = arr.mean()
            arr = (arr - mean) * random.uniform(0.8, 1.2) + mean
        
        if random.random() < 0.5:
            arr = np.fliplr(arr).copy()
        
        if random.random() < 0.2:
            arr = arr + np.random.normal(0, 8, arr.shape)
        
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)


# =============================================================================
# Model
# =============================================================================

class BLIPVQAModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        print(f"\n{'='*60}")
        print(f"LOADING MODEL: {Config.model_name}")
        print(f"{'='*60}")
        
        self.model = BlipForQuestionAnswering.from_pretrained(Config.model_name)
        
        self._increase_dropout()
        self._apply_freezing()
        self._print_params()
        print(f"{'='*60}")
    
    def _increase_dropout(self):
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = Config.dropout_rate
    
    def _apply_freezing(self):
        if Config.freeze_vision_encoder:
            for param in self.model.vision_model.parameters():
                param.requires_grad = False
            print("[OK] Vision encoder: FROZEN")
        
        if Config.freeze_text_encoder_layers > 0:
            if hasattr(self.model, 'text_encoder'):
                for i, layer in enumerate(self.model.text_encoder.encoder.layer):
                    if i < Config.freeze_text_encoder_layers:
                        for param in layer.parameters():
                            param.requires_grad = False
                print(f"[OK] Text encoder: first {Config.freeze_text_encoder_layers} layers FROZEN")
    
    def _print_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        
        print(f"Parameters:")
        print(f"  Total:     {total / 1e6:.1f}M")
        print(f"  Trainable: {trainable / 1e6:.1f}M ({100 * trainable / total:.1f}%)")
        print(f"  Frozen:    {frozen / 1e6:.1f}M ({100 * frozen / total:.1f}%)")
    
    def forward(self, pixel_values, input_ids, attention_mask, 
                decoder_input_ids=None, decoder_attention_mask=None, labels=None):
        return self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            return_dict=True
        )
    
    def generate(self, pixel_values, input_ids, attention_mask):
        return self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=Config.max_answer_length,
            num_beams=Config.num_beams,
            early_stopping=True
        )


# =============================================================================
# Trainer
# =============================================================================

class Trainer:
    def __init__(self, model, processor, train_loader, val_loader, test_loader, output_dir, device):
        self.device = device
        self.model = model.to(device)
        self.processor = processor
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.output_dir = output_dir
        
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        self.optimizer = AdamW(
            trainable_params,
            lr=Config.learning_rate,
            weight_decay=Config.weight_decay
        )
        
        total_steps = len(train_loader) * Config.epochs // Config.gradient_accumulation
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=1e-7
        )
        
        self.best_val_acc = 0
        self.best_val_close = 0
        self.best_val_open = 0
        self.patience_counter = 0
        self.history = []
        self.gradient_accumulation = Config.gradient_accumulation
    
    def train(self):
        print("\n" + "=" * 70)
        print("TRAINING STARTED (V6 - Final Working Version)")
        print("=" * 70)
        print(f"Device:           {self.device}")
        print(f"Epochs:           {Config.epochs}")
        print(f"Batch Size:       {Config.batch_size} x {self.gradient_accumulation} = {Config.batch_size * self.gradient_accumulation}")
        print(f"Learning Rate:    {Config.learning_rate}")
        print(f"Weight Decay:     {Config.weight_decay}")
        print(f"Dropout:          {Config.dropout_rate}")
        print(f"Frozen Layers:    Vision=ALL, Text={Config.freeze_text_encoder_layers}/12")
        print("=" * 70)
        
        for epoch in range(1, Config.epochs + 1):
            train_loss = self._train_epoch(epoch)
            val_overall, val_close, val_open = self._evaluate(self.val_loader)
            
            self.history.append({
                'epoch': epoch,
                'train_loss': float(train_loss),
                'val_overall': float(val_overall),
                'val_close': float(val_close),
                'val_open': float(val_open)
            })
            
            self._print_epoch_result(epoch, train_loss, val_overall, val_close, val_open)
            
            if val_overall > self.best_val_acc + Config.min_delta:
                self.best_val_acc = val_overall
                self.best_val_close = val_close
                self.best_val_open = val_open
                self.patience_counter = 0
                self._save_checkpoint(epoch)
                print("   >> NEW BEST MODEL SAVED!")
            else:
                self.patience_counter += 1
                if self.patience_counter >= Config.patience:
                    print(f"\n   >> Early stopping at epoch {epoch}")
                    break
        
        self._final_test_evaluation()
        return self.history
    
    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        self.optimizer.zero_grad()
        total_steps = len(self.train_loader)
        
        for step, batch in enumerate(self.train_loader):
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            decoder_input_ids = batch['decoder_input_ids'].to(self.device)
            decoder_attention_mask = batch['decoder_attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Use model's built-in loss
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss = loss / self.gradient_accumulation
            loss.backward()
            
            if (step + 1) % self.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.gradient_accumulation
            num_batches += 1
            
            pct = 100 * (step + 1) / total_steps
            avg_loss = total_loss / num_batches
            print(f"\r  Epoch {epoch:02d}: [{pct:5.1f}%] Loss: {avg_loss:.4f}  ", end='', flush=True)
        
        print("\r" + " " * 60 + "\r", end='', flush=True)
        return total_loss / num_batches
    
    @torch.no_grad()
    def _evaluate(self, data_loader):
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_is_close = []
        
        for batch in data_loader:
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            generated_ids = self.model.generate(pixel_values, input_ids, attention_mask)
            generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            generated_texts = [self._normalize_text(t) for t in generated_texts]
            
            all_predictions.extend(generated_texts)
            all_targets.extend(batch['answer_text'])
            all_is_close.extend([bool(x) for x in batch['is_close']])
        
        overall_acc, close_acc, open_acc = self._calculate_accuracy(
            all_predictions, all_targets, all_is_close
        )
        return overall_acc, close_acc, open_acc
    
    def _normalize_text(self, text):
        if text is None:
            return ""
        text = str(text).lower().strip()
        text = ' '.join(text.split())
        return text
    
    def _calculate_accuracy(self, predictions, targets, is_close_list):
        correct_all = []
        correct_close = []
        correct_open = []
        
        for pred, target, is_close in zip(predictions, targets, is_close_list):
            is_correct = (pred == target)
            correct_all.append(is_correct)
            
            if is_close:
                correct_close.append(is_correct)
            else:
                correct_open.append(is_correct)
        
        overall_acc = np.mean(correct_all) if correct_all else 0.0
        close_acc = np.mean(correct_close) if correct_close else 0.0
        open_acc = np.mean(correct_open) if correct_open else 0.0
        
        return overall_acc, close_acc, open_acc
    
    def _print_epoch_result(self, epoch, train_loss, val_overall, val_close, val_open):
        print(f"\n+{'-' * 68}+")
        print(f"|  EPOCH {epoch:02d}/{Config.epochs}                                                       |")
        print(f"+{'-' * 68}+")
        print(f"|  TRAIN      |  Loss: {train_loss:.4f}                                     |")
        print(f"+{'-' * 68}+")
        print(f"|  VAL ACC    |  Overall: {100 * val_overall:6.2f}%                               |")
        print(f"|             |  Close:   {100 * val_close:6.2f}%    Open: {100 * val_open:6.2f}%             |")
        print(f"+{'-' * 68}+")
        print(f"|  PATIENCE   |  {self.patience_counter}/{Config.patience}                                                |")
        print(f"+{'-' * 68}+")
    
    def _final_test_evaluation(self):
        print("\n" + "=" * 70)
        print("FINAL EVALUATION ON TEST SET")
        print("=" * 70)
        
        checkpoint_path = os.path.join(self.output_dir, 'best_model.pt')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']}")
        
        test_overall, test_close, test_open = self._evaluate(self.test_loader)
        
        print(f"\n+{'=' * 54}+")
        print(f"|          FINAL TEST RESULTS                        |")
        print(f"+{'=' * 54}+")
        print(f"|  Overall Accuracy:     {100 * test_overall:6.2f}%                     |")
        print(f"|  Close-ended Accuracy: {100 * test_close:6.2f}%                     |")
        print(f"|  Open-ended Accuracy:  {100 * test_open:6.2f}%                     |")
        print(f"+{'=' * 54}+")
        
        print(f"\nComparison with Validation Best:")
        print(f"  Val Overall: {100 * self.best_val_acc:.2f}%")
        print(f"  Val Close:   {100 * self.best_val_close:.2f}%")
        print(f"  Val Open:    {100 * self.best_val_open:.2f}%")
        
        gap = self.best_val_acc - test_overall
        print(f"\n  Val-Test Gap: {100 * gap:.2f}% (lower is better)")
        print("=" * 70)
        
        final_results = {
            'test_overall': float(test_overall),
            'test_close': float(test_close),
            'test_open': float(test_open),
            'val_best_overall': float(self.best_val_acc),
            'val_best_close': float(self.best_val_close),
            'val_best_open': float(self.best_val_open),
            'val_test_gap': float(gap)
        }
        
        results_path = os.path.join(self.output_dir, 'final_results.json')
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        print(f"\nResults saved to: {results_path}")
        
        self._show_prediction_samples()
    
    def _show_prediction_samples(self, num_samples=10):
        print(f"\nPREDICTION SAMPLES:")
        print("-" * 60)
        
        self.model.eval()
        sample_count = 0
        correct_count = 0
        
        for batch in self.test_loader:
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values, input_ids, attention_mask)
                predictions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            predictions = [self._normalize_text(p) for p in predictions]
            
            for pred, target, question, is_close in zip(
                predictions, batch['answer_text'], batch['question'], batch['is_close']
            ):
                if sample_count >= num_samples:
                    break
                
                is_correct = (pred == target)
                if is_correct:
                    correct_count += 1
                    
                status = "[OK]" if is_correct else "[X]"
                q_type = "CLOSE" if is_close else "OPEN"
                q_display = question[:40] + "..." if len(question) > 40 else question
                
                print(f"[{q_type}] Q: {q_display}")
                print(f"  Pred: {pred:<20} | True: {target:<20} {status}")
                print("-" * 60)
                
                sample_count += 1
            
            if sample_count >= num_samples:
                break
        
        print(f"Sample accuracy: {correct_count}/{num_samples}")
    
    def _save_checkpoint(self, epoch):
        os.makedirs(self.output_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': float(self.best_val_acc),
            'config': {
                'learning_rate': Config.learning_rate,
                'weight_decay': Config.weight_decay,
                'dropout_rate': Config.dropout_rate,
                'freeze_text_encoder_layers': Config.freeze_text_encoder_layers
            },
            'history': self.history
        }
        
        torch.save(checkpoint, os.path.join(self.output_dir, 'best_model.pt'))


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n")
    print("=" * 70)
    print("BLIP-VQA FINE-TUNING (V6 - Final Working Version)")
    print("Medical Visual Question Answering")
    print("=" * 70)
    
    set_seed(Config.seed)
    
    device = Config.get_device()
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    train_path, test_path, image_dir = find_data_paths()
    
    if not validate_paths(train_path, test_path, image_dir):
        print("\n[ERROR] Data validation failed.")
        return
    
    output_dir = os.path.join(os.path.dirname(train_path), "outputs_blip_vqa_v6")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    print("\nLoading data...")
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data_full = json.load(f)
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    train_data, val_data = train_test_split(
        train_data_full,
        test_size=Config.val_ratio,
        random_state=Config.seed
    )
    
    print(f"\nDATA SPLIT:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val:   {len(val_data)} samples")
    print(f"  Test:  {len(test_data)} samples (held-out)")
    
    print("\nLoading BLIP processor...")
    processor = BlipProcessor.from_pretrained(Config.model_name)
    
    train_dataset = VQARADDataset(train_data, image_dir, processor, is_train=True)
    val_dataset = VQARADDataset(val_data, image_dir, processor, is_train=False)
    test_dataset = VQARADDataset(test_data, image_dir, processor, is_train=False)
    
    print(f"\nQuestion Types:")
    print(f"  Train - Close: {len(train_dataset.close_indices)}, Open: {len(train_dataset.open_indices)}")
    print(f"  Val   - Close: {len(val_dataset.close_indices)}, Open: {len(val_dataset.open_indices)}")
    print(f"  Test  - Close: {len(test_dataset.close_indices)}, Open: {len(test_dataset.open_indices)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=(device.type == 'cuda')
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=(device.type == 'cuda')
    )
    
    model = BLIPVQAModel()
    
    trainer = Trainer(
        model=model,
        processor=processor,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        output_dir=output_dir,
        device=device
    )
    
    history = trainer.train()
    
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
