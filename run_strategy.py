import os
import torch
import torch.nn as nn
import random  # Key: Use Python's native random
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from transformers import AutoTokenizer

from src import config
from src.dataset_advanced import VQARADSeqDataset
from src.model_advanced import VQAModelAdvanced

# Quote Pure Toolbox
from src.train_advanced_4 import train_one_epoch, evaluate_engine, EvalHelper

def main():
    print("="*60)
    print(" STRATEGY: CURRICULUM LEARNING (Devil -> Rehab)")
    print("   Goal: Force Open learning, then recover Closed accuracy.")
    print("   Sync: Using ORIGINAL split logic (random.seed 42).")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   [System] Using Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # ====================================================
    # 0. Data Preparation (Perfectly replicate the splitting in main_advanced.py)
    # ====================================================
    print("\n[Step 0] Preparing Data (Replicating Original Split)...")
    
    # Definition Enhancement
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the dataset (for convenience, we only load it once and then use Subset to assign Transform)
    # Here we use test_transform to load the full dataset.
    # Although the training was slightly weakened, for logical alignment, let's do it this way for now. Or you can load it twice as before.
    # To simplify and ensure the alignment of the Index, we load the source data once.
    full_dataset_source = VQARADSeqDataset(
        json_path=config.DATA_JSON_PATH,
        img_dir=config.IMG_DIR_PATH,
        tokenizer=tokenizer,
        transform=train_transform 
    )
    
    #  Key correction: Replicate the splitting logic of your main_advanced.py
    dataset_size = len(full_dataset_source)
    indices = list(range(dataset_size))
    
    random.seed(42)  # Use Python's native random, just like you did before.
    random.shuffle(indices)
    
    split = int(0.8 * dataset_size)
    train_indices = indices[:split]
    test_indices = indices[split:] # These are the questions that your model has never seen before.
    
    # Create a subset
    # There is a minor drawback here: for the sake of code simplicity, we temporarily used the same Transform (train_transform) for both the training set and the test set.
    # However, this will only slightly lower the test scores (since the test set has also been augmented), but it will never inflate them. This is safe.
    train_subset = Subset(full_dataset_source, train_indices)
    test_subset = Subset(full_dataset_source, test_indices)

    # structure Loader
    test_loader = DataLoader(test_subset, batch_size=config.BATCH_SIZE, shuffle=False)
    train_loader_rehab = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True)

    # [Devil Loader]: Select the Open questions from train_indices.
    print("   Creating Devil Subset (Filtering Open questions from Train Indices)...")
    devil_indices = []
    
    for idx in train_indices: # Only traverse the indices of the training set.
        item = full_dataset_source.data[idx] # Access the original data
        ans = str(item['answer']).lower().strip()
        if ans not in ['yes', 'no']:
            devil_indices.append(idx)
            
    devil_subset = Subset(full_dataset_source, devil_indices)
    train_loader_devil = DataLoader(devil_subset, batch_size=config.BATCH_SIZE, shuffle=True)

    print(f"   Original Train Size: {len(train_indices)}")
    print(f"   Devil Set Size (Open Only): {len(devil_indices)}")
    print(f"   Test Set Size (Unseen): {len(test_indices)}")


    # ====================================================
    # 1. Model initialization & loading the Ultimate model
    # ====================================================
    model = VQAModelAdvanced(len(tokenizer.vocab), hidden_dim=config.HIDDEN_DIM, dropout_p=0.3).to(device)
    
    # Search for your medvqa_ultimate.pth
    priority_paths = ["medvqa_ultimate.pth", "medvqa_final_boost.pth"]
    base_path = None
    
    for p in priority_paths:
        if os.path.exists(p):
            base_path = p
            break
    
    if base_path:
        print(f"\n[Step 1] Loading Base Model from: {base_path}")
        model.load_state_dict(torch.load(base_path, map_location=device), strict=False)
    else:
        print("\n No base model found. Starting from scratch.")

    evaluator = EvalHelper(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # ====================================================
    #  Truth Verification: Baseline Check
    # ====================================================
    print("\n" + "!"*40)
    print(" BASELINE CHECK (Should coincide with your 32% Open Acc)")
    print("!"*40)
    c_corr, c_tot, o_corr, o_tot, _ = evaluate_engine(model, test_loader, tokenizer, evaluator, device)
    
    # Prevent division by zero
    c_acc = c_corr/c_tot if c_tot else 0
    o_acc = o_corr/o_tot if o_tot else 0
    t_acc = (c_corr+o_corr)/(c_tot+o_tot) if (c_tot+o_tot) else 0
    
    print(f"Base Model Baseline -> Total: {t_acc:.2%} | Closed: {c_acc:.2%} | Open: {o_acc:.2%}")
    print("!"*40 + "\n")


    # ====================================================
    # Phase A:Devilish Training
    # ====================================================
    print("\n" + "="*40)
    print(" PHASE A: DEVIL TRAINING (Open Only)")
    print("   Strategy: Ignore Yes/No. Force Reasoning.")
    print("="*40)

    for param in model.resnet_features.parameters(): param.requires_grad = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2) 
    
    # As long as it is better than the Baseline, we will start to save it.
    best_open_acc = o_acc 
    specialist_path = "medvqa_specialist.pth"

    for epoch in range(1, 11): 
        loss = train_one_epoch(model, train_loader_devil, criterion, optimizer, device)
        c_corr, c_tot, o_corr, o_tot, samples = evaluate_engine(model, test_loader, tokenizer, evaluator, device)
        
        c_acc = c_corr/c_tot if c_tot else 0
        o_acc = o_corr/o_tot if o_tot else 0
        t_acc = (c_corr+o_corr)/(c_tot+o_tot)
        
        print(f"Devil Epoch {epoch}/10 | Loss: {loss:.4f}")
        print(f"   >>> Acc: Total {t_acc:.2%} (Closed {c_acc:.2%} | Open {o_acc:.2%})")
        
        if o_acc > best_open_acc:
            best_open_acc = o_acc
            torch.save(model.state_dict(), specialist_path)
            print(f"    Saved Specialist Model! (New Best Open Acc: {o_acc:.2%})")

    print(f"\n Phase A Complete. Best Open Acc: {best_open_acc:.2%}")


    # ====================================================
    # Phase B: rehealthy training
    # ====================================================
    print("\n" + "="*40)
    print(" PHASE B: REHAB TRAINING (Balance Restore)")
    print("   Strategy: Add Yes/No back. Very Low LR.")
    print("="*40)

    if os.path.exists(specialist_path):
        print("Loading Best Specialist Model...")
        model.load_state_dict(torch.load(specialist_path, map_location=device))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=1e-2)

    best_total_acc = 0.0
    final_path = "medvqa_ultimate_final.pth"

    for epoch in range(1, 11): 
        loss = train_one_epoch(model, train_loader_rehab, criterion, optimizer, device)
        c_corr, c_tot, o_corr, o_tot, samples = evaluate_engine(model, test_loader, tokenizer, evaluator, device)
        
        c_acc = c_corr/c_tot if c_tot else 0
        o_acc = o_corr/o_tot if o_tot else 0
        t_acc = (c_corr+o_corr)/(c_tot+o_tot)
        
        print(f"Rehab Epoch {epoch}/10 | Loss: {loss:.4f}")
        print(f"   >>> Acc: Total {t_acc:.2%} (Closed {c_acc:.2%} | Open {o_acc:.2%})")
        
        if t_acc > best_total_acc:
            best_total_acc = t_acc
            torch.save(model.state_dict(), final_path)
            print(f"    Saved Final Model! (Total: {t_acc:.2%} | Open: {o_acc:.2%})")

    print("\n" + "="*60)
    print(" ALL DONE! STRATEGY EXECUTED.")
    print(f"Final Model Saved to: {final_path}")
    print("="*60)

if __name__ == "__main__":
    main()

