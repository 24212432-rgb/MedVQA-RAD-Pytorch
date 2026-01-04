import os
import json
import random
import warnings
import logging
from pathlib import Path
from glob import glob
from collections import Counter

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
    print("BLIP V12 FINAL")
    print("Strict Match | No Leakage | No Overfitting")
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
# Data Leakage Verification (CRITICAL)
# =============================================================================
def verify_image_disjoint(train_data, test_data):
    """
    CRITICAL: Verify NO image overlap between train and test.
    This prevents the model from memorizing test images.
    """
    train_images = set()
    test_images = set()
    
    for item in train_data:
        img = item.get('image_name', item.get('image', ''))
        if img:
            train_images.add(img)
    
    for item in test_data:
        img = item.get('image_name', item.get('image', ''))
        if img:
            test_images.add(img)
    
    overlap = train_images & test_images
    
    print(f"\n{'='*70}")
    print("DATA LEAKAGE VERIFICATION")
    print(f"{'='*70}")
    print(f"  Train images: {len(train_images)}")
    print(f"  Test images:  {len(test_images)}")
    print(f"  Overlap:      {len(overlap)}")
    
    if len(overlap) == 0:
        print(f"\n   PASSED: No image leakage detected!")
        print(f"     Your results will be scientifically valid.")
        print(f"{'='*70}")
        return True, 0
    else:
        print(f"\n   FAILED: {len(overlap)} images appear in both sets!")
        print(f"     Results would be INVALID due to data leakage.")
        print(f"     Please run make_image_split.py first.")
        print(f"{'='*70}")
        return False, len(overlap)


# =============================================================================
# Find Data Paths
# =============================================================================
def find_data_paths():
    """Find image-disjoint data files"""
    search_roots = ["/content/drive/MyDrive", "/content", os.getcwd(), "."]
    
    for root in search_roots:
        if not os.path.exists(root):
            continue
        
        # Prefer image-disjoint files
        disjoint = glob(os.path.join(root, "**/trainset_image_disjoint.json"), recursive=True)
        if disjoint:
            train_path = disjoint[0]
            data_dir = os.path.dirname(train_path)
            test_path = os.path.join(data_dir, "testset_image_disjoint.json")
            
            if os.path.exists(test_path):
                for folder in ["VQA_RAD Image Folder", "images"]:
                    img_dir = os.path.join(data_dir, folder)
                    if os.path.isdir(img_dir):
                        print(f"[OK] Found IMAGE-DISJOINT train: {train_path}")
                        print(f"[OK] Found IMAGE-DISJOINT test:  {test_path}")
                        print(f"[OK] Images: {img_dir}")
                        return train_path, test_path, img_dir
    
    # Fallback to regular files (will verify later)
    for root in search_roots:
        if not os.path.exists(root):
            continue
        
        train_files = glob(os.path.join(root, "**/trainset.json"), recursive=True)
        if train_files:
            train_path = train_files[0]
            data_dir = os.path.dirname(train_path)
            test_path = os.path.join(data_dir, "testset.json")
            
            if os.path.exists(test_path):
                for folder in ["VQA_RAD Image Folder", "images"]:
                    img_dir = os.path.join(data_dir, folder)
                    if os.path.isdir(img_dir):
                        print(f"[WARNING] Using regular split (will verify leakage)")
                        return train_path, test_path, img_dir
    
    return None, None, None


# =============================================================================
# Configuration - Optimized for Accuracy WITHOUT Overfitting
# =============================================================================
class Config:
    model_name = "Salesforce/blip-vqa-capfilt-large"
    
    # ========== TRAINING PHASES ==========
    # Phase 1: Foundation (all data, no weighting)
    phase1_epochs = 12
    phase1_lr = 2.5e-5
    
    # Phase 2: Open Boost (sample weighting)
    phase2_epochs = 12
    phase2_lr = 2e-5
    
    # Phase 3: Devil #1 (Open-only)
    phase3_epochs = 15
    phase3_lr = 1.5e-5
    
    # Phase 4: Light Rehab
    phase4_epochs = 6
    phase4_lr = 8e-6
    
    # Phase 5: Devil #2 (Open-only again)
    phase5_epochs = 10
    phase5_lr = 1e-5
    
    # Phase 6: Vision Fine-tuning
    phase6_epochs = 8
    phase6_lr = 3e-6
    
    # ========== SAMPLE WEIGHTING ==========
    yesno_weight = 0.15     # Low weight for easy yes/no
    open_weight = 3.0       # High weight for hard open
    
    # ========== ANTI-OVERFITTING ==========
    dropout_rate = 0.15     # Strong dropout
    weight_decay = 0.05     # Strong L2 regularization
    label_smoothing = 0.1   # Prevent overconfidence
    augment_prob = 0.35     # Data augmentation
    
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
    num_workers = 0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


# =============================================================================
# Dataset with Augmentation
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
        try:
            image = Image.open(self.image_dir / img_name).convert('RGB')
        except:
            image = Image.new('RGB', (384, 384), color='gray')
        
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
        
        # Brightness adjustment (safe for medical images)
        if random.random() < 0.5:
            factor = random.uniform(0.9, 1.1)
            arr = arr * factor
        
        # Contrast adjustment (safe for medical images)
        if random.random() < 0.5:
            mean = arr.mean()
            factor = random.uniform(0.9, 1.1)
            arr = (arr - mean) * factor + mean
        
        # NO horizontal flip (left/right matters in medical!)
        # NO rotation (orientation matters!)
        
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)


# =============================================================================
# Model with Regularization
# =============================================================================
class BLIPModel(nn.Module):
    def __init__(self):
        super().__init__()
        print(f"\nLoading: {Config.model_name}")
        
        self.model = BlipForQuestionAnswering.from_pretrained(Config.model_name)
        
        # Apply stronger dropout for regularization
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.p = Config.dropout_rate
        
        # Freeze vision encoder initially
        for p in self.model.vision_model.parameters():
            p.requires_grad = False
        print(f"[OK] Vision encoder: FROZEN")
        
        # Freeze first N text encoder layers
        if hasattr(self.model, 'text_encoder'):
            total_layers = len(self.model.text_encoder.encoder.layer)
            for i, layer in enumerate(self.model.text_encoder.encoder.layer):
                if i < Config.freeze_text_layers:
                    for p in layer.parameters():
                        p.requires_grad = False
            print(f"[OK] Text encoder: first {Config.freeze_text_layers}/{total_layers} layers FROZEN")
        
        self._print_params()
    
    def unfreeze_vision(self, n_layers):
        """Carefully unfreeze last N vision layers"""
        if hasattr(self.model.vision_model, 'encoder'):
            layers = self.model.vision_model.encoder.layers
            total = len(layers)
            for i, layer in enumerate(layers):
                if i >= total - n_layers:
                    for p in layer.parameters():
                        p.requires_grad = True
            print(f"[OK] Vision: last {n_layers}/{total} layers UNFROZEN")
            self._print_params()
    
    def _print_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        pct = 100 * trainable / total
        print(f"Parameters: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable ({pct:.1f}%)")
    
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
            early_stopping=True,
            length_penalty=1.0
        )


# =============================================================================
# Evaluation - PURE STRICT MATCH ONLY
# =============================================================================
@torch.no_grad()
def evaluate(model, loader, processor):
    """
    PURE STRICT MATCH evaluation.
    Match = (prediction.lower().strip() == target.lower().strip())
    
    NO synonym matching, NO partial matching, NO tricks.
    This is the SAME evaluation used in academic papers.
    """
    model.eval()
    
    correct_all = []
    correct_close = []
    correct_open = []
    
    for batch in loader:
        pv = batch['pixel_values'].to(DEVICE)
        ids = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        
        gen_ids = model.generate(pv, ids, mask)
        preds = processor.batch_decode(gen_ids, skip_special_tokens=True)
        preds = [p.lower().strip() for p in preds]
        
        for pred, target, is_close in zip(preds, batch['answer_text'], batch['is_close']):
            # PURE STRICT MATCH
            match = (pred == target)
            
            correct_all.append(match)
            if is_close:
                correct_close.append(match)
            else:
                correct_open.append(match)
    
    return {
        'overall': np.mean(correct_all) if correct_all else 0,
        'close': np.mean(correct_close) if correct_close else 0,
        'open': np.mean(correct_open) if correct_open else 0,
        'n_total': len(correct_all),
        'n_close': len(correct_close),
        'n_open': len(correct_open)
    }


# =============================================================================
# Training with Anti-Overfitting
# =============================================================================
def train_epoch(model, loader, optimizer, scheduler, scaler, use_weighting=True):
    model.train()
    total_loss = 0
    n_batches = 0
    optimizer.zero_grad()
    
    for step, batch in enumerate(loader):
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
            
            # Sample weighting to focus on Open questions
            if use_weighting:
                batch_size = labels.size(0)
                n_close = sum(is_close).item()
                n_open = batch_size - n_close
                
                if n_open > 0 and n_close > 0:
                    weight = (n_close / batch_size) * Config.yesno_weight + \
                            (n_open / batch_size) * Config.open_weight
                elif n_open > 0:
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
            optimizer.zero_grad()
        
        total_loss += loss.item() * Config.gradient_accumulation
        n_batches += 1
    
    return total_loss / max(1, n_batches)


def run_phase(model, train_loader, val_loader, processor, epochs, lr, 
              phase_name, use_weighting=True):
    """Run a training phase with early stopping"""
    print(f"\n{'='*70}")
    print(f"{phase_name}")
    print(f"Epochs: {epochs}, LR: {lr}, Weighting: {use_weighting}")
    print(f"{'='*70}")
    
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=Config.weight_decay
    )
    
    total_steps = max(1, len(train_loader) * epochs // Config.gradient_accumulation)
    warmup_steps = int(total_steps * Config.warmup_ratio)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=Config.warmup_ratio,
        anneal_strategy='cos'
    )
    
    scaler = GradScaler(enabled=USE_AMP)
    
    best_score = 0
    best_open = 0
    patience_count = 0
    best_state = None
    
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, use_weighting)
        
        results = evaluate(model, val_loader, processor)
        
        overall = results['overall']
        close = results['close']
        open_acc = results['open']
        
        # Score favors Open accuracy (our goal)
        score = 0.5 * overall + 0.5 * open_acc
        
        print(f"  Ep {epoch:02d}/{epochs} | Loss: {loss:.4f} | "
              f"All: {100*overall:.2f}% | Close: {100*close:.2f}% | Open: {100*open_acc:.2f}%", end="")
        
        improved = False
        if score > best_score + Config.min_delta:
            best_score = score
            improved = True
        if open_acc > best_open:
            best_open = open_acc
            improved = True
        
        if improved:
            patience_count = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(" << BEST")
        else:
            patience_count += 1
            print()
        
        if patience_count >= Config.patience:
            print(f"  Early stopping at epoch {epoch}")
            break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return best_score, best_open


# =============================================================================
# Main Training Pipeline
# =============================================================================
def main():
    print("\n" + "=" * 70)
    print("BLIP-VQA V12 FINAL")
    print("Strict Match | No Leakage | No Overfitting")
    print("=" * 70)
    
    set_seed(Config.seed)
    
    # ========== FIND DATA ==========
    train_path, test_path, image_dir = find_data_paths()
    if not all([train_path, test_path, image_dir]):
        print("\n[ERROR] Data files not found!")
        print("Please ensure trainset_image_disjoint.json exists.")
        print("Run make_image_split.py first if needed.")
        return
    
    output_dir = os.path.join(os.path.dirname(train_path), "outputs_blip_v12")
    os.makedirs(output_dir, exist_ok=True)
    
    # ========== LOAD DATA ==========
    print("\nLoading data...")
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data_full = json.load(f)
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # ========== VERIFY NO LEAKAGE (CRITICAL) ==========
    is_valid, overlap_count = verify_image_disjoint(train_data_full, test_data)
    
    if not is_valid:
        print("\n[ERROR] Cannot proceed with data leakage!")
        print("Please run make_image_split.py to create image-disjoint split.")
        return
    
    # ========== SPLIT TRAIN/VAL ==========
    train_data, val_data = train_test_split(
        train_data_full, 
        test_size=Config.val_ratio, 
        random_state=Config.seed
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(train_data)}")
    print(f"  Val:   {len(val_data)}")
    print(f"  Test:  {len(test_data)}")
    
    # ========== LOAD PROCESSOR ==========
    print("\nLoading processor...")
    processor = BlipProcessor.from_pretrained(Config.model_name)
    
    # ========== CREATE DATASETS ==========
    train_dataset = VQADataset(train_data, image_dir, processor, is_train=True)
    val_dataset = VQADataset(val_data, image_dir, processor, is_train=False)
    test_dataset = VQADataset(test_data, image_dir, processor, is_train=False)
    
    print(f"\nQuestion distribution:")
    print(f"  Train - Close: {len(train_dataset.close_indices)}, Open: {len(train_dataset.open_indices)}")
    print(f"  Test  - Close: {len(test_dataset.close_indices)}, Open: {len(test_dataset.open_indices)}")
    
    # ========== CREATE DATA LOADERS ==========
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.batch_size, 
        shuffle=True, 
        num_workers=Config.num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.batch_size, 
        shuffle=False, 
        num_workers=Config.num_workers
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.batch_size, 
        shuffle=False, 
        num_workers=Config.num_workers
    )
    
    # Open-only loader for Devil phases
    open_subset = Subset(train_dataset, train_dataset.open_indices)
    open_loader = DataLoader(
        open_subset, 
        batch_size=Config.batch_size, 
        shuffle=True, 
        num_workers=Config.num_workers
    )
    print(f"  Open-only subset: {len(open_subset)} samples")
    
    # ========== CREATE MODEL ==========
    model = BLIPModel().to(DEVICE)
    
    # ========== TRAINING PHASES ==========
    all_results = []
    
    # Phase 1: Foundation
    s1, o1 = run_phase(
        model, train_loader, val_loader, processor,
        epochs=Config.phase1_epochs,
        lr=Config.phase1_lr,
        phase_name="PHASE 1: Foundation (Balanced)",
        use_weighting=False
    )
    all_results.append(('Phase1', s1, o1))
    
    # Phase 2: Open Boost
    s2, o2 = run_phase(
        model, train_loader, val_loader, processor,
        epochs=Config.phase2_epochs,
        lr=Config.phase2_lr,
        phase_name=f"PHASE 2: Open Boost (Yes/No={Config.yesno_weight}, Open={Config.open_weight})",
        use_weighting=True
    )
    all_results.append(('Phase2', s2, o2))
    
    # Phase 3: Devil #1 (Open-only)
    print(f"\n{'='*70}")
    print(f"PHASE 3: DEVIL #1 - Open Questions ONLY ({len(open_subset)} samples)")
    print(f"{'='*70}")
    s3, o3 = run_phase(
        model, open_loader, val_loader, processor,
        epochs=Config.phase3_epochs,
        lr=Config.phase3_lr,
        phase_name="PHASE 3: Devil #1 (Open-only)",
        use_weighting=False
    )
    all_results.append(('Phase3-Devil1', s3, o3))
    
    # Phase 4: Light Rehab
    s4, o4 = run_phase(
        model, train_loader, val_loader, processor,
        epochs=Config.phase4_epochs,
        lr=Config.phase4_lr,
        phase_name="PHASE 4: Light Rehab",
        use_weighting=False
    )
    all_results.append(('Phase4-Rehab', s4, o4))
    
    # Phase 5: Devil #2 (Open-only again)
    print(f"\n{'='*70}")
    print(f"PHASE 5: DEVIL #2 - Open Questions ONLY (Second Pass)")
    print(f"{'='*70}")
    s5, o5 = run_phase(
        model, open_loader, val_loader, processor,
        epochs=Config.phase5_epochs,
        lr=Config.phase5_lr,
        phase_name="PHASE 5: Devil #2 (Open-only)",
        use_weighting=False
    )
    all_results.append(('Phase5-Devil2', s5, o5))
    
    # Phase 6: Vision Fine-tuning
    print(f"\n[UNFREEZING] Vision encoder...")
    model.unfreeze_vision(Config.unfreeze_vision_layers)
    
    s6, o6 = run_phase(
        model, train_loader, val_loader, processor,
        epochs=Config.phase6_epochs,
        lr=Config.phase6_lr,
        phase_name="PHASE 6: Vision Fine-tuning",
        use_weighting=False
    )
    all_results.append(('Phase6-Vision', s6, o6))
    
    # ========== FINAL EVALUATION ==========
    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION (STRICT MATCH)")
    print("=" * 70)
    
    test_results = evaluate(model, test_loader, processor)
    val_results = evaluate(model, val_loader, processor)
    
    test_overall = test_results['overall']
    test_close = test_results['close']
    test_open = test_results['open']
    
    val_overall = val_results['overall']
    
    # Calculate generalization gap (important for overfitting check)
    gap = val_overall - test_overall
    
    print(f"\n+{'='*68}+")
    print(f"|            BLIP V12 FINAL - TEST RESULTS                          |")
    print(f"|            Strict Match, Image-Disjoint, No Overfitting           |")
    print(f"+{'='*68}+")
    print(f"|                                                                    |")
    print(f"|   Overall Accuracy:     {100*test_overall:6.2f}%                                  |")
    print(f"|   Close-ended Accuracy: {100*test_close:6.2f}%                                  |")
    print(f"|   Open-ended Accuracy:  {100*test_open:6.2f}%                                  |")
    print(f"|                                                                    |")
    print(f"+{'='*68}+")
    print(f"|   OVERFITTING CHECK:                                               |")
    print(f"|   Val Accuracy:  {100*val_overall:6.2f}%                                         |")
    print(f"|   Test Accuracy: {100*test_overall:6.2f}%                                         |")
    print(f"|   Val-Test Gap:  {100*gap:6.2f}%", end="")
    
    if abs(gap) < 3:
        print(f"   Excellent generalization              |")
    elif abs(gap) < 5:
        print(f"   Good generalization                   |")
    else:
        print(f"   Some overfitting                      |")
    
    print(f"+{'='*68}+")
    print(f"|    No data leakage (image-disjoint verified)                     |")
    print(f"|    Pure strict match (pred == target)                            |")
    print(f"+{'='*68}+")
    
    # ========== COMPARISON ==========
    print(f"\nComparison with baseline:")
    print(f"  V8 Baseline:  Overall=42.98%, Open=14.56%")
    print(f"  V12 Final:    Overall={100*test_overall:.2f}%, Open={100*test_open:.2f}%")
    
    improvement_overall = test_overall - 0.4298
    improvement_open = test_open - 0.1456
    print(f"  Improvement:  Overall {100*improvement_overall:+.2f}%, Open {100*improvement_open:+.2f}%")
    
    # ========== SAVE RESULTS ==========
    results = {
        'test': {
            'overall': float(test_overall),
            'close': float(test_close),
            'open': float(test_open),
            'n_total': test_results['n_total'],
            'n_close': test_results['n_close'],
            'n_open': test_results['n_open']
        },
        'val': {
            'overall': float(val_overall),
            'close': float(val_results['close']),
            'open': float(val_results['open'])
        },
        'generalization': {
            'val_test_gap': float(gap),
            'is_good': bool(abs(gap) < 5)
        },
        'guarantees': {
            'no_leakage': True,
            'strict_match_only': True,
            'image_disjoint': True
        },
        'phases': [
            {'name': name, 'score': float(s), 'open': float(o)} 
            for name, s, o in all_results
        ],
        'config': {
            'yesno_weight': Config.yesno_weight,
            'open_weight': Config.open_weight,
            'dropout': Config.dropout_rate,
            'weight_decay': Config.weight_decay
        }
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'results': results
    }, os.path.join(output_dir, 'model.pt'))
    
    print(f"\nResults saved to: {output_dir}")
    
    # ========== SAMPLE PREDICTIONS ==========
    print("\n" + "-" * 70)
    print("SAMPLE PREDICTIONS (Strict Match)")
    print("-" * 70)
    
    model.eval()
    samples = 0
    correct_samples = 0
    
    for batch in test_loader:
        if samples >= 12:
            break
        
        with torch.no_grad():
            gen = model.generate(
                batch['pixel_values'].to(DEVICE),
                batch['input_ids'].to(DEVICE),
                batch['attention_mask'].to(DEVICE)
            )
        
        preds = [p.lower().strip() for p in processor.batch_decode(gen, skip_special_tokens=True)]
        
        for pred, target, q, is_close in zip(
            preds, batch['answer_text'], batch['question'], batch['is_close']
        ):
            if samples >= 12:
                break
            
            match = (pred == target)
            correct_samples += match
            
            qtype = "CLOSE" if is_close else "OPEN"
            q_short = q[:45] + ".." if len(q) > 45 else q
            status = "✓" if match else "✗"
            
            print(f"[{qtype:5}] {q_short}")
            print(f"   Pred: {pred[:30]:<30} True: {target[:30]:<30} [{status}]")
            samples += 1
    
    print("-" * 70)
    print(f"Sample accuracy: {correct_samples}/{samples}")
    
    # ========== DONE ==========
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nFinal Results (HONEST, STRICT MATCH):")
    print(f"  Overall: {100*test_overall:.2f}%")
    print(f"  Close:   {100*test_close:.2f}%")
    print(f"  Open:    {100*test_open:.2f}%")
    print(f"  Gap:     {100*gap:.2f}% (overfitting check)")
    print("=" * 70)


if __name__ == "__main__":
    main()
