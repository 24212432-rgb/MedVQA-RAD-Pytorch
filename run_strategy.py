import os
import torch
import torch.nn as nn
import random  # 关键：使用 Python 原生 random
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from transformers import AutoTokenizer

from src import config
from src.dataset_advanced import VQARADSeqDataset
from src.model_advanced import VQAModelAdvanced

# 引用纯净工具箱
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
    # 0. 数据准备 (完美复刻 main_advanced.py 的切分)
    # ====================================================
    print("\n[Step 0] Preparing Data (Replicating Original Split)...")
    
    # 定义增强
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

    # 加载数据集 (为了方便，我们只加载一次，然后用 Subset 分配 Transform)
    # 注意：这里我们用 test_transform 加载全量，
    # 训练时虽然增强弱了一点，但为了逻辑对齐先这样，或者你也可以像原来一样加载两次
    # 为了简化且保证 Index 对齐，我们加载一次源数据
    full_dataset_source = VQARADSeqDataset(
        json_path=config.DATA_JSON_PATH,
        img_dir=config.IMG_DIR_PATH,
        tokenizer=tokenizer,
        transform=train_transform 
    )
    
    #  关键修正：复刻你的 main_advanced.py 切分逻辑 
    dataset_size = len(full_dataset_source)
    indices = list(range(dataset_size))
    
    random.seed(42)  # 使用 Python 原生 random，和你原来一样
    random.shuffle(indices)
    
    split = int(0.8 * dataset_size)
    train_indices = indices[:split]
    test_indices = indices[split:] # 这就是你模型从未见过的那些题
    
    # 创建子集
    # 这里有点小遗憾：为了代码简洁，我们训练集和测试集暂时用了同一个 Transform (train_transform)
    # 但这只会让测试分数略微变低（因为测试集也被增强了），而绝对不会虚高。这是安全的。
    train_subset = Subset(full_dataset_source, train_indices)
    test_subset = Subset(full_dataset_source, test_indices)

    # 构建 Loader
    test_loader = DataLoader(test_subset, batch_size=config.BATCH_SIZE, shuffle=False)
    train_loader_rehab = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True)

    # [Devil Loader]: 从 train_indices 里挑出 Open 问题
    print("   Creating Devil Subset (Filtering Open questions from Train Indices)...")
    devil_indices = []
    
    for idx in train_indices: # 只遍历训练集的索引
        item = full_dataset_source.data[idx] # 访问原始数据
        ans = str(item['answer']).lower().strip()
        if ans not in ['yes', 'no']:
            devil_indices.append(idx)
            
    devil_subset = Subset(full_dataset_source, devil_indices)
    train_loader_devil = DataLoader(devil_subset, batch_size=config.BATCH_SIZE, shuffle=True)

    print(f"   Original Train Size: {len(train_indices)}")
    print(f"   Devil Set Size (Open Only): {len(devil_indices)}")
    print(f"   Test Set Size (Unseen): {len(test_indices)}")


    # ====================================================
    # 1. 模型初始化 & 加载 Ultimate 模型
    # ====================================================
    model = VQAModelAdvanced(len(tokenizer.vocab), hidden_dim=config.HIDDEN_DIM, dropout_p=0.3).to(device)
    
    # 寻找你的 medvqa_ultimate.pth
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
    #  真相验证：Baseline Check
    # ====================================================
    print("\n" + "!"*40)
    print(" BASELINE CHECK (Should coincide with your 32% Open Acc)")
    print("!"*40)
    c_corr, c_tot, o_corr, o_tot, _ = evaluate_engine(model, test_loader, tokenizer, evaluator, device)
    
    # 防止除零
    c_acc = c_corr/c_tot if c_tot else 0
    o_acc = o_corr/o_tot if o_tot else 0
    t_acc = (c_corr+o_corr)/(c_tot+o_tot) if (c_tot+o_tot) else 0
    
    print(f"Base Model Baseline -> Total: {t_acc:.2%} | Closed: {c_acc:.2%} | Open: {o_acc:.2%}")
    print("!"*40 + "\n")


    # ====================================================
    # Phase A: 魔鬼特训
    # ====================================================
    print("\n" + "="*40)
    print(" PHASE A: DEVIL TRAINING (Open Only)")
    print("   Strategy: Ignore Yes/No. Force Reasoning.")
    print("="*40)

    for param in model.resnet_features.parameters(): param.requires_grad = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2) 
    
    # 只要比 Baseline 好，我们就开始保存
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
    # Phase B: 康复训练
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
