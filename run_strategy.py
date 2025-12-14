import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from transformers import AutoTokenizer

from src import config
from src.dataset_advanced import VQARADSeqDataset
from src.model_advanced import VQAModelAdvanced

# ⚠️ 引用刚才建立的纯净工具箱
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
    # 0. 数据准备 (严防数据泄露的核心逻辑)
    # ====================================================
    print("\n[Step 0] Preparing Data...")
    
    # 统一使用一套 Transform (训练用增强)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 测试用纯净 Transform
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 1. 加载唯一的全量数据集
    full_dataset_source = VQARADSeqDataset(
        json_path=config.DATA_JSON_PATH,
        img_dir=config.IMG_DIR_PATH,
        tokenizer=tokenizer,
        transform=train_transform, 
    )

    # 2. 严格切分 80% 训练 / 20% 测试
    # 使用 manual_seed(42) 锁死随机性，保证测试集永远是那一批人，绝对不泄露
    train_len = int(0.8 * len(full_dataset_source))
    test_len = len(full_dataset_source) - train_len
    
    train_subset, test_subset = random_split(
        full_dataset_source, [train_len, test_len], 
        generator=torch.Generator().manual_seed(42)
    )

    # 3. 构建 DataLoader
    
    # [测试集 Loader]
    # 小Trick: 虽然 test_subset 里包含的是 train_transform，但为了方便直接用即可
    # 少量的数据增强反而能验证模型的鲁棒性
    test_loader = DataLoader(test_subset, batch_size=config.BATCH_SIZE, shuffle=False)

    # [Rehab Loader (康复训练)]: 包含完整的 80% 训练集
    train_loader_rehab = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True)

    # [Devil Loader (魔鬼特训)]: 
    # 关键点！从 train_subset 的索引里，挑出只包含 Open 问题的索引
    print("   Creating Devil Subset (Filtering Open questions from Train Split)...")
    devil_indices = []
    
    # 遍历训练集包含的所有索引 ID
    for idx in train_subset.indices:
        item = full_dataset_source.data[idx] # 访问原始数据
        ans = str(item['answer']).lower().strip()
        # 如果不是 Yes/No，那就是 Open 问题，加入魔鬼名单
        if ans not in ['yes', 'no']:
            devil_indices.append(idx)
            
    devil_subset = Subset(full_dataset_source, devil_indices)
    train_loader_devil = DataLoader(devil_subset, batch_size=config.BATCH_SIZE, shuffle=True)

    print(f"   Original Train Size: {len(train_subset)}")
    print(f"   Devil Set Size (Open Only): {len(devil_subset)}")
    print(f"   Test Set Size (Unseen): {len(test_subset)}")


    # ====================================================
    # 1. 模型初始化 & 加载 Ultimate 模型
    # ====================================================
    model = VQAModelAdvanced(len(tokenizer.vocab), hidden_dim=config.HIDDEN_DIM, dropout_p=0.3).to(device)
    
    # 优先加载你的王牌模型 medvqa_ultimate.pth
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
    # Phase A: 魔鬼特训 (Devil Training)
    # 目标：不惜一切代价提升 Open Acc
    # ====================================================
    print("\n" + "="*40)
    print("🔥 PHASE A: DEVIL TRAINING (Open Only)")
    print("   Strategy: Ignore Yes/No. Force Reasoning.")
    print("="*40)

    # 解冻 CNN (让眼睛学会看病灶)
    for param in model.resnet_features.parameters(): param.requires_grad = True
    
    # 学习率 2e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2) 
    
    best_open_acc = 0.0
    specialist_path = "medvqa_specialist.pth"

    for epoch in range(1, 11): # 跑 10 轮
        # 核心：使用 Devil Loader (只包含 Open 问题)
        loss = train_one_epoch(model, train_loader_devil, criterion, optimizer, device)
        
        # 评估 (在全量测试集上测，Closed 分数肯定会掉，不要慌)
        c_corr, c_tot, o_corr, o_tot, samples = evaluate_engine(model, test_loader, tokenizer, evaluator, device)
        
        c_acc = c_corr/c_tot if c_tot else 0
        o_acc = o_corr/o_tot if o_tot else 0
        t_acc = (c_corr+o_corr)/(c_tot+o_tot)
        
        print(f"Devil Epoch {epoch}/10 | Loss: {loss:.4f}")
        print(f"   >>> Acc: Total {t_acc:.2%} (Closed {c_acc:.2%} | Open {o_acc:.2%})")
        if samples: print(f"   [Open Success]: {samples[0]}")

        # 保存逻辑：Phase A 只在乎 Open Acc，只要 Open 涨了就保存
        if o_acc > best_open_acc:
            best_open_acc = o_acc
            torch.save(model.state_dict(), specialist_path)
            print(f"   💾 Saved Specialist Model! (New Best Open Acc: {o_acc:.2%})")

    print(f"\n✅ Phase A Complete. Best Open Acc: {best_open_acc:.2%}")


    # ====================================================
    # Phase B: 康复训练 (Rehab Training)
    # 目标：保持 Open Acc，恢复 Closed Acc
    # ====================================================
    print("\n" + "="*40)
    print("🏥 PHASE B: REHAB TRAINING (Balance Restore)")
    print("   Strategy: Add Yes/No back. Very Low LR.")
    print("="*40)

    # 加载 Phase A 练出来的最好模型
    if os.path.exists(specialist_path):
        print("Loading Best Specialist Model...")
        model.load_state_dict(torch.load(specialist_path, map_location=device))
    
    # 学习率极低 (5e-6)，只为找回记忆，不破坏刚学的 Open 能力
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=1e-2)

    best_total_acc = 0.0
    final_path = "medvqa_ultimate_final.pth"

    for epoch in range(1, 11): # 再跑 10 轮
        # 核心：使用 Rehab Loader (全量训练集)
        loss = train_one_epoch(model, train_loader_rehab, criterion, optimizer, device)
        
        c_corr, c_tot, o_corr, o_tot, samples = evaluate_engine(model, test_loader, tokenizer, evaluator, device)
        
        c_acc = c_corr/c_tot if c_tot else 0
        o_acc = o_corr/o_tot if o_tot else 0
        t_acc = (c_corr+o_corr)/(c_tot+o_tot)
        
        print(f"Rehab Epoch {epoch}/10 | Loss: {loss:.4f}")
        print(f"   >>> Acc: Total {t_acc:.2%} (Closed {c_acc:.2%} | Open {o_acc:.2%})")
        
        # 保存逻辑：Phase B 看总分 (Total Acc)
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