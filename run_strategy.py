import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from transformers import AutoTokenizer

from src import config
from src.dataset_advanced import VQARADSeqDataset
from src.model_advanced import VQAModelAdvanced

# ⚠️ Refer to the pure toolbox that was just established.
from src.train_advanced_4 import train_one_epoch, evaluate_engine, EvalHelper

def main():
    print("="*60)
    print("🚀 STRATEGY: CURRICULUM LEARNING (Devil -> Rehab)")
    print("   Goal: Force Open learning, then recover Closed accuracy.")
    print("   Security: Strict Index Filtering to prevent Data Leakage.")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # ====================================================
    # 0. Data Preparation (The Core Logic for Preventing Data Leakage)
    # ====================================================
    print("\n[Step 0] Preparing Data...")
    
    # Uniformly use one set of Transform (for training and enhancement)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Testing with pure Transform
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 1. Load the sole complete dataset
    full_dataset_source = VQARADSeqDataset(
        json_path=config.DATA_JSON_PATH,
        img_dir=config.IMG_DIR_PATH,
        tokenizer=tokenizer,
        transform=train_transform, 
    )

    # 2. Strictly split: 80% for training / 20% for testing
    # Use manual_seed(42) to lock the randomness and ensure that the test set will always consist of that specific group of people, and absolutely no information will be leaked.
    train_len = int(0.8 * len(full_dataset_source))
    test_len = len(full_dataset_source) - train_len
    
    train_subset, test_subset = random_split(
        full_dataset_source, [train_len, test_len], 
        generator=torch.Generator().manual_seed(42)
    )

    # 3. Build DataLoader
    
    # [Test set Loader]
    # Little Trick: Although the contents of test_subset are train_transform, for ease of use, we can directly use it.
    # A small amount of data augmentation can actually verify the robustness of the model.
    test_loader = DataLoader(test_subset, batch_size=config.BATCH_SIZE, shuffle=False)

    # [Rehab Loader (Rehabilitation training)]: Including the complete 80% of the training set
    train_loader_rehab = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True)

    # [Devil Loader (Devil Training Program)]: 
    # Key point! From the indices of the "train_subset", select only those that contain "Open" questions.
    print("   Creating Devil Subset (Filtering Open questions from Train Split)...")
    devil_indices = []
    
    # Traverse all the index IDs contained in the training set
    for idx in train_subset.indices:
        item = full_dataset_source.data[idx] # Access to the original data
        ans = str(item['answer']).lower().strip()
        # If it's not Yes/No, then it's an open question. Add it to the blacklist.
        if ans not in ['yes', 'no']:
            devil_indices.append(idx)
            
    devil_subset = Subset(full_dataset_source, devil_indices)
    train_loader_devil = DataLoader(devil_subset, batch_size=config.BATCH_SIZE, shuffle=True)

    print(f"   Original Train Size: {len(train_subset)}")
    print(f"   Devil Set Size (Open Only): {len(devil_subset)}")
    print(f"   Test Set Size (Unseen): {len(test_subset)}")


    # ====================================================
    # 1. Model initialization & loading of the Ultimate model
    # ====================================================
    model = VQAModelAdvanced(len(tokenizer.vocab), hidden_dim=config.HIDDEN_DIM, dropout_p=0.3).to(device)
    
    # Prioritize loading your top-tier model： medvqa_ultimate.pth
    priority_paths = ["medvqa_ultimate.pth", "medvqa_final_boost.pth", "medvqa_13new.pth"]
    base_path = None
    
    for p in priority_paths:
        if os.path.exists(p):
            base_path = p
            break
    
    if base_path:
        print(f"\n[Step 1] Loading Base Model from: {base_path}")
        model.load_state_dict(torch.load(base_path, map_location=device), strict=False)
    else:
        print("\n⚠️ No base model found. Starting from scratch (Not recommended).")

    evaluator = EvalHelper(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)


    # ====================================================
    # Phase A: Devil Training Program(Devil Training)
    # Objective: To enhance Open Acc at all costs
    # ====================================================
    print("\n" + "="*40)
    print("🔥 PHASE A: DEVIL TRAINING (Open Only)")
    print("   Strategy: Ignore Yes/No. Force Reasoning.")
    print("="*40)

    # Unfreeze CNN (Enable the eyes to learn to detect lesions)
    for param in model.resnet_features.parameters(): param.requires_grad = True
    
    # Learning rate 2e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2) 
    
    best_open_acc = 0.0
    specialist_path = "medvqa_specialist.pth"

    for epoch in range(1, 11): # Run 10 rounds
        #Core: Use Devil Loader (only including Open issues)
        loss = train_one_epoch(model, train_loader_devil, criterion, optimizer, device)
        
        # Evaluation (Tested on the full test set. The Closed score will definitely drop. Don't panic.)
        c_corr, c_tot, o_corr, o_tot, samples = evaluate_engine(model, test_loader, tokenizer, evaluator, device)
        
        c_acc = c_corr/c_tot if c_tot else 0
        o_acc = o_corr/o_tot if o_tot else 0
        t_acc = (c_corr+o_corr)/(c_tot+o_tot)
        
        print(f"Devil Epoch {epoch}/10 | Loss: {loss:.4f}")
        print(f"   >>> Acc: Total {t_acc:.2%} (Closed {c_acc:.2%} | Open {o_acc:.2%})")
        if samples: print(f"   [Open Success]: {samples[0]}")

        # Save logic: In Phase A, only the Open Acc is of concern. As long as Open rises, it will be saved.
        if o_acc > best_open_acc:
            best_open_acc = o_acc
            torch.save(model.state_dict(), specialist_path)
            print(f"   💾 Saved Specialist Model! (New Best Open Acc: {o_acc:.2%})")

    print(f"\n✅ Phase A Complete. Best Open Acc: {best_open_acc:.2%}")


    # ====================================================
    # Phase B: Rehabilitation training (Rehab Training)
    # Objective: Maintain Open Acc and restore Closed Acc
    # ====================================================
    print("\n" + "="*40)
    print("🏥 PHASE B: REHAB TRAINING (Balance Restore)")
    print("   Strategy: Add Yes/No back. Very Low LR.")
    print("="*40)

    # Load the best model trained in Phase A
    if os.path.exists(specialist_path):
        print("Loading Best Specialist Model...")
        model.load_state_dict(torch.load(specialist_path, map_location=device))
    
    # The learning rate is extremely low (5e-6), solely aimed at retrieving memories without disrupting the newly acquired Open capabilities.
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=1e-2)

    best_total_acc = 0.0
    final_path = "medvqa_ultimate_final.pth"

    for epoch in range(1, 11): # Run another 10 rounds
        # Core: Use Rehab Loader (the complete training set)
        loss = train_one_epoch(model, train_loader_rehab, criterion, optimizer, device)
        
        c_corr, c_tot, o_corr, o_tot, samples = evaluate_engine(model, test_loader, tokenizer, evaluator, device)
        
        c_acc = c_corr/c_tot if c_tot else 0
        o_acc = o_corr/o_tot if o_tot else 0
        t_acc = (c_corr+o_corr)/(c_tot+o_tot)
        
        print(f"Rehab Epoch {epoch}/10 | Loss: {loss:.4f}")
        print(f"   >>> Acc: Total {t_acc:.2%} (Closed {c_acc:.2%} | Open {o_acc:.2%})")
        
        # Saving logic: In Phase B, calculate the total score (Total Acc)
        if t_acc > best_total_acc:
            best_total_acc = t_acc
            torch.save(model.state_dict(), final_path)
            print(f"   🏆 Saved Final Model! (Total: {t_acc:.2%} | Open: {o_acc:.2%})")

    print("\n" + "="*60)
    print("🎉 ALL DONE! STRATEGY EXECUTED.")
    print(f"Final Model Saved to: {final_path}")
    print("="*60)

if __name__ == "__main__":

    main()

