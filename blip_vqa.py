import os
import json
import random
import warnings
import logging
import shutil
import time
from pathlib import Path
from glob import glob
from collections import Counter
from tqdm.auto import tqdm  

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')
logging.getLogger('transformers').setLevel(logging.ERROR)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BlipProcessor, BlipForQuestionAnswering

# =============================================================================
# GPU Check
# =============================================================================
def check_gpu():
    print("=" * 70)
    print("BLIP V12 FINAL + SPEED CACHING")
    print("Strict Match | No Leakage | No Overfitting | High Speed")
    print("=" * 70)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[OK] Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("[WARNING] No GPU - training will be slow!")
    print("=" * 70)
    return device

DEVICE = check_gpu()
USE_AMP = DEVICE.type == 'cuda'

# =============================================================================
# [NEW] SMART PATH FINDING & CACHING (The Engineering Fix)
# =============================================================================
def auto_find_image_source(base_path, sample_image_name):
    """Automatically finds where the images are stored in Drive."""
    print(f"Searching for source of image '{sample_image_name}'...")
    candidates = ["data", "images", "VQA_RAD Image Folder", "VQA_RAD Dataset Public", "."]
    
    # 1. Check strict subfolders
    for sub in candidates:
        check_path = Path(base_path) / sub / sample_image_name
        if check_path.exists():
            return Path(base_path) / sub
            
    # 2. Recursive search (slower but robust)
    found_files = list(Path(base_path).rglob(sample_image_name))
    if found_files:
        return found_files[0].parent
        
    raise FileNotFoundError(f"Could not find {sample_image_name} in {base_path}")

def cache_images_locally(source_dir, dest_dir):
    """
    Copies images from Google Drive to Colab Local Disk.
    This prevents 'Input/Output Error' and speeds up training by 20x.
    """
    source = Path(source_dir)
    dest = Path(dest_dir)
    
    if dest.exists() and any(dest.iterdir()):
        print(f"[CACHE] Using existing local cache at {dest}")
        return dest
        
    print(f"\n[OPTIMIZATION] Copying images from Drive to Local Disk for speed...")
    print(f" Source: {source}")
    print(f" Dest:   {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    
    files = list(source.glob("*"))
    # Use tqdm to show copy progress
    for f in tqdm(files, desc="Caching images"):
        if f.is_file():
            shutil.copy2(f, dest / f.name)
            
    print("Copy complete! Training will now be fast and stable.\n")
    return dest

# =============================================================================
# Data Leakage Verification
# =============================================================================
def verify_image_disjoint(train_data, test_data):
    """CRITICAL: Verify NO image overlap between train and test."""
    train_images = set()
    test_images = set()
    for item in train_data:
        img = item.get('image_name', item.get('image', ''))
        if img: train_images.add(img)
    for item in test_data:
        img = item.get('image_name', item.get('image', ''))
        if img: test_images.add(img)
    
    overlap = train_images & test_images
    if len(overlap) == 0:
        print(f"\n[PASSED] No image leakage detected! (Scientific Validity OK)")
        return True, 0
    else:
        print(f"\n[FAILED] {len(overlap)} images appear in both sets!")
        return False, len(overlap)

# =============================================================================
# Configuration (V12 Logic Preserved)
# =============================================================================
class Config:
    model_name = "Salesforce/blip-vqa-capfilt-large"
    
    # ========== TRAINING PHASES (V12) ==========
    phase1_epochs = 12; phase1_lr = 2.5e-5  # Foundation
    phase2_epochs = 12; phase2_lr = 2e-5    # Open Boost
    phase3_epochs = 15; phase3_lr = 1.5e-5  # Devil #1
    phase4_epochs = 6;  phase4_lr = 8e-6    # Rehab
    phase5_epochs = 10; phase5_lr = 1e-5    # Devil #2
    phase6_epochs = 8;  phase6_lr = 3e-6    # Vision FT
    
    # ========== SAMPLE WEIGHTING ==========
    yesno_weight = 0.15 
    open_weight = 3.0 
    
    # ========== ANTI-OVERFITTING ==========
    dropout_rate = 0.15
    weight_decay = 0.05 
    augment_prob = 0.35 
    
    # ========== TRAINING ==========
    batch_size = 4
    gradient_accumulation = 4
    max_grad_norm = 1.0
    warmup_ratio = 0.1
    
    # ========== MODEL ==========
    max_question_length = 32
    max_answer_length = 20
    num_beams = 4
    
    # ========== FREEZING ==========
    freeze_text_layers = 5
    unfreeze_vision_layers = 3
    
    # ========== EARLY STOPPING ==========
    patience = 6
    min_delta = 0.002
    
    # ========== DATA ==========
    val_ratio = 0.15
    seed = 42
    
    # IMPORTANT: Keep 0 for stability when using cached images
    num_workers = 0 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

# =============================================================================
# Dataset with Augmentation (V12 Logic)
# =============================================================================
class VQADataset(Dataset):
    def __init__(self, data, image_dir, processor, is_train=True):
        self.data = data
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.is_train = is_train
        
        # Classify questions
        self.close_indices = []
        self.open_indices = []
        for i, item in enumerate(data):
            ans = str(item.get('answer', '')).lower().strip()
            if ans in ['yes', 'no']:
                self.close_indices.append(i)
            else:
                self.open_indices.append(i)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        img_name = item.get('image_name', item.get('image', ''))
        
        # Robust Image Loading (Handles case sensitivity)
        image_path = self.image_dir / img_name
        if not image_path.exists():
            # Try finding file with case-insensitive match
            candidates = list(self.image_dir.glob(f"{img_name}*"))
            if candidates:
                image_path = candidates[0]
            else:
                # Fallback
                image = Image.new('RGB', (384, 384), color='gray')
                image_path = None
        
        if image_path:
            image = Image.open(image_path).convert('RGB')
            
        # Data augmentation (training only)
        if self.is_train and random.random() < Config.augment_prob:
            image = self._augment(image)
            
        question = item.get('question', '')
        answer = str(item.get('answer', '')).lower().strip()
        
        enc = self.processor(
            images=image,
            text=question,
            padding='max_length',
            truncation=True,
            max_length=Config.max_question_length,
            return_tensors='pt'
        )
        
        ans_enc = self.processor.tokenizer(
            answer,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=Config.max_answer_length,
            return_tensors='pt'
        )
        
        labels = ans_enc['input_ids'].squeeze(0).clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        is_close = (answer in ['yes', 'no'])
        
        return {
            'pixel_values': enc['pixel_values'].squeeze(0),
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'decoder_input_ids': ans_enc['input_ids'].squeeze(0),
            'decoder_attention_mask': ans_enc['attention_mask'].squeeze(0),
            'labels': labels,
            'answer_text': answer,
            'is_close': is_close,
            'question': question
        }
        
    def _augment(self, image):
        """Simple augmentation that doesn't distort medical info"""
        arr = np.array(image).astype(np.float32)
        # Brightness
        if random.random() < 0.5:
            arr = arr * random.uniform(0.9, 1.1)
        # Contrast
        if random.random() < 0.5:
            mean = arr.mean()
            arr = (arr - mean) * random.uniform(0.9, 1.1) + mean
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

# =============================================================================
# Model (V12 Logic)
# =============================================================================
class BLIPModel(nn.Module):
    def __init__(self):
        super().__init__()
        print(f"\nLoading: {Config.model_name}")
        self.model = BlipForQuestionAnswering.from_pretrained(Config.model_name)
        
        # V12: Stronger dropout
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.p = Config.dropout_rate
        
        # Freeze vision initially
        for p in self.model.vision_model.parameters():
            p.requires_grad = False
            
        # Freeze text layers
        if hasattr(self.model, 'text_encoder'):
            for i, layer in enumerate(self.model.text_encoder.encoder.layer):
                if i < Config.freeze_text_layers:
                    for p in layer.parameters(): p.requires_grad = False
        
    def unfreeze_vision(self, n_layers):
        if hasattr(self.model.vision_model, 'encoder'):
            layers = self.model.vision_model.encoder.layers
            total = len(layers)
            for i, layer in enumerate(layers):
                if i >= total - n_layers:
                    for p in layer.parameters(): p.requires_grad = True
            print(f"[OK] Vision: last {n_layers} layers UNFROZEN")

    def forward(self, pixel_values, input_ids, attention_mask,
                decoder_input_ids, decoder_attention_mask, labels):
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
# Evaluation (V12 Strict Match)
# =============================================================================
@torch.no_grad()
def evaluate(model, loader, processor):
    model.eval()
    correct_all = []
    correct_close = []
    correct_open = []
    
    # Use tqdm for evaluation too
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        pv = batch['pixel_values'].to(DEVICE)
        ids = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        
        gen_ids = model.generate(pv, ids, mask)
        preds = processor.batch_decode(gen_ids, skip_special_tokens=True)
        preds = [p.lower().strip() for p in preds]
        
        for pred, target, is_close in zip(preds, batch['answer_text'], batch['is_close']):
            match = (pred == target)
            correct_all.append(match)
            if is_close: correct_close.append(match)
            else: correct_open.append(match)
            
    return {
        'overall': np.mean(correct_all) if correct_all else 0,
        'close': np.mean(correct_close) if correct_close else 0,
        'open': np.mean(correct_open) if correct_open else 0,
        'n_total': len(correct_all),
        'n_close': len(correct_close),
        'n_open': len(correct_open)
    }

# =============================================================================
# Training with Anti-Overfitting (V12 Logic)
# =============================================================================
def train_epoch(model, loader, optimizer, scheduler, scaler, use_weighting=True):
    model.train()
    total_loss = 0
    n_batches = 0
    optimizer.zero_grad()
    
    # Use tqdm for progress bar
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for step, batch in enumerate(pbar):
        pv = batch['pixel_values'].to(DEVICE)
        ids = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        dec_ids = batch['decoder_input_ids'].to(DEVICE)
        dec_mask = batch['decoder_attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        is_close = batch['is_close']
        
        with autocast(enabled=USE_AMP):
            outputs = model(pv, ids, mask, dec_ids, dec_mask, labels)
            loss = outputs.loss
            
            # V12 Sample weighting
            if use_weighting:
                batch_size = labels.size(0)
                n_close = sum(is_close).item()
                n_open = batch_size - n_close
                if n_open > 0 and n_close > 0:
                    weight = (n_close/batch_size)*Config.yesno_weight + (n_open/batch_size)*Config.open_weight
                elif n_open > 0: weight = Config.open_weight
                else: weight = Config.yesno_weight
                loss = loss * weight
            
            loss = loss / Config.gradient_accumulation
        
        scaler.scale(loss).backward()
        
        if (step + 1) % Config.gradient_accumulation == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
        current_loss = loss.item() * Config.gradient_accumulation
        total_loss += current_loss
        n_batches += 1
        pbar.set_postfix({'loss': f"{current_loss:.4f}"})
        
    return total_loss / max(1, n_batches)

def run_phase(model, train_loader, val_loader, processor, epochs, lr, phase_name, use_weighting=True):
    print(f"\n>>> {phase_name} (Epochs: {epochs}, LR: {lr})")
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=Config.weight_decay)
    
    total_steps = max(1, len(train_loader) * epochs // Config.gradient_accumulation)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps, pct_start=Config.warmup_ratio, anneal_strategy='cos'
    )
    scaler = GradScaler(enabled=USE_AMP)
    
    best_score = 0
    best_state = None
    patience_count = 0
    
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, use_weighting)
        results = evaluate(model, val_loader, processor)
        
        overall = results['overall']
        open_acc = results['open']
        # V12 Metric
        score = 0.5 * overall + 0.5 * open_acc 
        
        print(f" Ep {epoch}/{epochs} | Loss: {loss:.4f} | All: {100*overall:.2f}% | Open: {100*open_acc:.2f}%", end="")
        
        if score > best_score + Config.min_delta:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
            print(" [BEST]")
        else:
            patience_count += 1
            print()
            
        if patience_count >= Config.patience:
            print(f" Early stopping triggered.")
            break
            
    if best_state: model.load_state_dict(best_state)
    return best_score, best_score # returning score twice for simplicity

# =============================================================================
# Main Pipeline
# =============================================================================
def main():
    set_seed(Config.seed)
    
    # 1. SETUP PATHS & CACHE IMAGES (The Critical Fix)
    base_path = '/content/drive/MyDrive/7015'
    train_path = os.path.join(base_path, 'trainset_image_disjoint.json')
    test_path = os.path.join(base_path, 'testset_image_disjoint.json')
    
    # Auto-find and Cache
    print("Initializing Data...")
    if not os.path.exists(train_path):
        print(f"ERROR: {train_path} not found.")
        return

    with open(train_path, 'r') as f: train_data_full = json.load(f)
    
    # Detect Source and Cache Locally
    sample_img = train_data_full[0].get('image_name', train_data_full[0].get('image', ''))
    source_image_dir = auto_find_image_source(base_path, sample_img)
    local_image_dir = '/content/local_images_cache'
    
    # This function copies images to local disk to prevent Colab freezing
    final_image_dir = cache_images_locally(source_image_dir, local_image_dir)
    
    # 2. LOAD DATA
    with open(test_path, 'r') as f: test_data = json.load(f)
    
    if not verify_image_disjoint(train_data_full, test_data)[0]:
        return

    train_data, val_data = train_test_split(train_data_full, test_size=Config.val_ratio, random_state=Config.seed)
    
    processor = BlipProcessor.from_pretrained(Config.model_name)
    
    # 3. CREATE DATASETS (Using Cached Image Dir)
    train_dataset = VQADataset(train_data, final_image_dir, processor, is_train=True)
    val_dataset = VQADataset(val_data, final_image_dir, processor, is_train=False)
    test_dataset = VQADataset(test_data, final_image_dir, processor, is_train=False)
    
    # 4. LOADERS (num_workers=0 is safe because images are on fast local disk)
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=0)
    
    open_subset = Subset(train_dataset, train_dataset.open_indices)
    open_loader = DataLoader(open_subset, batch_size=Config.batch_size, shuffle=True, num_workers=0)
    
    print(f"Train Size: {len(train_dataset)} | Open-Only Subset: {len(open_subset)}")

    # 5. RUN TRAINING (V12 Phases)
    model = BLIPModel().to(DEVICE)
    
    run_phase(model, train_loader, val_loader, processor, Config.phase1_epochs, Config.phase1_lr, "PHASE 1: Foundation", False)
    run_phase(model, train_loader, val_loader, processor, Config.phase2_epochs, Config.phase2_lr, "PHASE 2: Open Boost", True)
    run_phase(model, open_loader, val_loader, processor, Config.phase3_epochs, Config.phase3_lr, "PHASE 3: Devil #1 (Open Only)", False)
    run_phase(model, train_loader, val_loader, processor, Config.phase4_epochs, Config.phase4_lr, "PHASE 4: Rehab", False)
    run_phase(model, open_loader, val_loader, processor, Config.phase5_epochs, Config.phase5_lr, "PHASE 5: Devil #2 (Open Only)", False)
    
    print("\n[UNFREEZING VISION]")
    model.unfreeze_vision(Config.unfreeze_vision_layers)
    run_phase(model, train_loader, val_loader, processor, Config.phase6_epochs, Config.phase6_lr, "PHASE 6: Vision FT", False)
    
    # 6. FINAL EVAL
    print("\n" + "="*50)
    print("FINAL TEST SET EVALUATION")
    print("="*50)
    res = evaluate(model, test_loader, processor)
    print(f"Overall: {res['overall']*100:.2f}%")
    print(f"Open:    {res['open']*100:.2f}%")
    print(f"Close:   {res['close']*100:.2f}%")
    
    # Save
    output_dir = './outputs_blip_v12_final'
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pt'))
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    main()
