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

# Environment Setup
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
# 1. GPU Configuration
# =============================================================================
def check_gpu():
    print("=" * 70)
    print("BLIP TRAINING: V12 (Smart Augmentation + Phased Learning)")
    print("=" * 70)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("[WARNING] No GPU found. Training will be extremely slow.")
    print("=" * 70)
    return device

DEVICE = check_gpu()
USE_AMP = DEVICE.type == 'cuda'

# =============================================================================
# 2. Configuration Class
# =============================================================================
class Config:
    model_name = "Salesforce/blip-vqa-capfilt-large"
    
    # --- Training Phases (The "Secret Sauce" for high accuracy) ---
    phase1_epochs = 12; phase1_lr = 2.5e-5  # Phase 1: Foundation
    phase2_epochs = 12; phase2_lr = 2e-5    # Phase 2: Open-Ended Boost
    phase3_epochs = 15; phase3_lr = 1.5e-5  # Phase 3: Hard Samples Only (Devil #1)
    phase4_epochs = 6;  phase4_lr = 8e-6    # Phase 4: Recovery (Rehab)
    phase5_epochs = 10; phase5_lr = 1e-5    # Phase 5: Hard Samples Refinement (Devil #2)
    phase6_epochs = 8;  phase6_lr = 3e-6    # Phase 6: Visual Fine-tuning
    
    # --- Weighting & Augmentation ---
    yesno_weight = 0.15   # Down-weight easy Yes/No questions
    open_weight = 3.0     # Up-weight hard Open questions
    augment_prob = 0.40   # 40% chance to augment image (Make it smarter)
    
    # --- System Params ---
    batch_size = 4        # Low batch size for stability
    gradient_accumulation = 4
    max_grad_norm = 1.0
    warmup_ratio = 0.1
    
    # --- Model Params ---
    max_question_length = 32
    max_answer_length = 20
    
    # --- Freezing Strategy ---
    freeze_text_layers = 5
    unfreeze_vision_layers = 3
    
    # --- Early Stopping ---
    patience = 6
    min_delta = 0.002
    
    # --- Paths ---
    val_ratio = 0.15
    seed = 42
    num_workers = 0 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

# =============================================================================
# 3. Smart Path Finding & Caching
# =============================================================================
def auto_find_image_source(base_path, sample_image_name):
    """Automatically locates the image folder."""
    print(f"[INFO] Searching for image source: {sample_image_name}...")
    candidates = ["data", "images", "VQA_RAD Image Folder", "VQA_RAD Dataset Public", "."]
    
    for sub in candidates:
        check_path = Path(base_path) / sub / sample_image_name
        if check_path.exists():
            return Path(base_path) / sub
            
    found_files = list(Path(base_path).rglob(sample_image_name))
    if found_files:
        return found_files[0].parent
        
    raise FileNotFoundError(f"Could not find {sample_image_name} in {base_path}")

def cache_images_locally(source_dir, dest_dir):
    """Copies images to local Colab disk for 20x speedup."""
    source = Path(source_dir)
    dest = Path(dest_dir)
    
    if dest.exists() and any(dest.iterdir()):
        print(f"[INFO] Using existing local cache at {dest}")
        return dest
        
    print(f"\n[INFO] Caching images to local disk (Speed Optimization)...")
    dest.mkdir(parents=True, exist_ok=True)
    
    files = list(source.glob("*"))
    for f in tqdm(files, desc="Caching"):
        if f.is_file():
            shutil.copy2(f, dest / f.name)
            
    print("[INFO] Cache complete.\n")
    return dest

# =============================================================================
# 4. Dataset with Smart Augmentation
# =============================================================================
class VQADataset(Dataset):
    def __init__(self, data, image_dir, processor, is_train=True):
        self.data = data
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.is_train = is_train
        
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
        
        # Robust Image Loading
        image_path = self.image_dir / img_name
        if not image_path.exists():
            candidates = list(self.image_dir.glob(f"{img_name}*"))
            if candidates:
                image_path = candidates[0]
            else:
                image = Image.new('RGB', (384, 384), color='gray')
                image_path = None
        
        if image_path:
            image = Image.open(image_path).convert('RGB')
            
        # === SMART AUGMENTATION ===
        # Randomly adjust brightness/contrast to make model robust
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
            padding='max_length',
            truncation=True,
            max_length=Config.max_answer_length,
            return_tensors='pt'
        )
        
        labels = ans_enc['input_ids'].squeeze(0).clone()
        # Ignore padding in loss calculation
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
            'is_close': is_close
        }
        
    def _augment(self, image):
        """Applies medical-safe augmentations."""
        arr = np.array(image).astype(np.float32)
        # Random Brightness
        if random.random() < 0.5:
            arr = arr * random.uniform(0.8, 1.2)
        # Random Contrast
        if random.random() < 0.5:
            mean = arr.mean()
            arr = (arr - mean) * random.uniform(0.8, 1.2) + mean
        
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

# =============================================================================
# 5. Model Definition
# =============================================================================
class BLIPModel(nn.Module):
    def __init__(self):
        super().__init__()
        print(f"[INFO] Loading Pretrained Model: {Config.model_name}")
        self.model = BlipForQuestionAnswering.from_pretrained(Config.model_name)
        
        # Freeze Vision Encoder initially
        for p in self.model.vision_model.parameters():
            p.requires_grad = False
            
        # Freeze lower text layers
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
            print(f"[INFO] Unfrozen last {n_layers} vision layers for Fine-Tuning.")

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
            max_length=Config.max_answer_length
        )

# =============================================================================
# 6. Training Engine
# =============================================================================
@torch.no_grad()
def evaluate(model, loader, processor):
    model.eval()
    correct_all = []
    correct_open = []
    
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        pv = batch['pixel_values'].to(DEVICE)
        ids = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        
        gen_ids = model.generate(pv, ids, mask)
        preds = processor.batch_decode(gen_ids, skip_special_tokens=True)
        
        for pred, target, is_close in zip(preds, batch['answer_text'], batch['is_close']):
            match = (pred.lower().strip() == target)
            correct_all.append(match)
            if not is_close: correct_open.append(match)
            
    return {
        'overall': np.mean(correct_all) if correct_all else 0,
        'open': np.mean(correct_open) if correct_open else 0
    }

def train_epoch(model, loader, optimizer, scheduler, scaler):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for step, batch in enumerate(pbar):
        optimizer.zero_grad()
        
        pv = batch['pixel_values'].to(DEVICE)
        ids = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        
        # Weighted Loss Calculation
        is_close = batch['is_close']
        
        with autocast(enabled=USE_AMP):
            outputs = model(
                pixel_values=pv, input_ids=ids, attention_mask=mask,
                decoder_input_ids=batch['decoder_input_ids'].to(DEVICE),
                decoder_attention_mask=batch['decoder_attention_mask'].to(DEVICE),
                labels=labels
            )
            loss = outputs.loss
            
            # Dynamic Reweighting (Focus on Hard Questions)
            batch_size = labels.size(0)
            n_close = sum(is_close).item()
            n_open = batch_size - n_close
            
            if n_open > 0:
                weight = Config.open_weight
            else:
                weight = Config.yesno_weight
                
            loss = loss * weight
            loss = loss / Config.gradient_accumulation
        
        scaler.scale(loss).backward()
        
        if (step + 1) % Config.gradient_accumulation == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        
        total_loss += loss.item() * Config.gradient_accumulation
        pbar.set_postfix({'loss': f"{loss.item() * Config.gradient_accumulation:.4f}"})
        
    return total_loss / len(loader)

def run_phase(model, train_loader, val_loader, processor, epochs, lr, phase_name):
    print(f"\n>>> STARTING {phase_name} (LR: {lr})")
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    total_steps = len(train_loader) * epochs // Config.gradient_accumulation
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps, pct_start=Config.warmup_ratio
    )
    scaler = GradScaler(enabled=USE_AMP)
    
    best_score = 0
    
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler, scaler)
        res = evaluate(model, val_loader, processor)
        
        score = 0.5 * res['overall'] + 0.5 * res['open']
        print(f" Ep {epoch}/{epochs} | Loss: {loss:.4f} | Acc: {res['overall']*100:.1f}% | Open: {res['open']*100:.1f}%", end="")
        
        if score > best_score:
            best_score = score
            print(" [BEST]")
            # Save intermediate best
            torch.save(model.state_dict(), './outputs_blip_v12_final/temp_best.pt')
        else:
            print()

# =============================================================================
# 7. Main Execution
# =============================================================================
def main():
    set_seed(Config.seed)
    
    # 1. Paths
    base_path = '/content/drive/MyDrive/7015'
    train_path = os.path.join(base_path, 'trainset_image_disjoint.json')
    test_path = os.path.join(base_path, 'testset_image_disjoint.json')
    
    # 2. Data Initialization
    print("[INFO] Loading Datasets...")
    with open(train_path, 'r') as f: train_data_full = json.load(f)
    with open(test_path, 'r') as f: test_data = json.load(f)
    
    # Cache Images
    sample_img = train_data_full[0].get('image_name', train_data_full[0].get('image', ''))
    source_dir = auto_find_image_source(base_path, sample_img)
    image_dir = cache_images_locally(source_dir, '/content/local_images_cache')
    
    # 3. Create Processors & Datasets
    processor = BlipProcessor.from_pretrained(Config.model_name)
    train_data, val_data = train_test_split(train_data_full, test_size=Config.val_ratio, random_state=Config.seed)
    
    train_ds = VQADataset(train_data, image_dir, processor, is_train=True)
    val_ds = VQADataset(val_data, image_dir, processor, is_train=False)
    test_ds = VQADataset(test_data, image_dir, processor, is_train=False)
    
    # Subsets for Phased Training
    open_subset = Subset(train_ds, train_ds.open_indices)
    
    # Loaders
    train_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True, num_workers=0)
    open_loader = DataLoader(open_subset, batch_size=Config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=Config.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=Config.batch_size, shuffle=False)
    
    # 4. Model Setup
    model = BLIPModel().to(DEVICE)
    
    # 5. Phased Training Schedule (FULL 6 PHASES)
    # Phase 1: Foundation (All Data)
    run_phase(model, train_loader, val_loader, processor, Config.phase1_epochs, Config.phase1_lr, "PHASE 1: Foundation")
    
    # Phase 2: Open Boost (All Data, Higher LR)
    run_phase(model, train_loader, val_loader, processor, Config.phase2_epochs, Config.phase2_lr, "PHASE 2: Open Boost")
    
    # Phase 3: Devil #1 (Hard Data ONLY)
    run_phase(model, open_loader, val_loader, processor, Config.phase3_epochs, Config.phase3_lr, "PHASE 3: Devil #1 (Open Only)")
    
    # Phase 4: Rehab (All Data - Fix for 'forgetting')
    run_phase(model, train_loader, val_loader, processor, Config.phase4_epochs, Config.phase4_lr, "PHASE 4: Rehab")
    
    # Phase 5: Devil #2 (Hard Data ONLY - Final polish before vision)
    run_phase(model, open_loader, val_loader, processor, Config.phase5_epochs, Config.phase5_lr, "PHASE 5: Devil #2 (Open Only)")
    
    # Phase 6: Vision Fine-Tuning (Unfreeze Vision + All Data)
    print("\n[INFO] Unfreezing Vision Layers for Final Fine-Tuning...")
    model.unfreeze_vision(Config.unfreeze_vision_layers)
    run_phase(model, train_loader, val_loader, processor, Config.phase6_epochs, Config.phase6_lr, "PHASE 6: Vision Fine-Tuning")
    
    # 6. Final Save
    output_dir = './outputs_blip_v12_final'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'final_model.pt')
    torch.save(model.state_dict(), save_path)
    print(f"\n[SUCCESS] Training Complete. Model saved to: {save_path}")
    
    # 7. Final Verification
    print("\n[INFO] Running Final Test Evaluation...")
    res = evaluate(model, test_loader, processor)
    print(f"Final Test Accuracy: {res['overall']*100:.2f}%")

if __name__ == "__main__":
    main()
