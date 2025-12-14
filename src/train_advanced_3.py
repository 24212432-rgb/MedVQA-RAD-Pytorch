# src/train_advanced.py
import torch
import torch.nn as nn
import re
import string
import os
import time

# --- 高级语义评估模块 ---
try:
    from sentence_transformers import SentenceTransformer, util
    HAS_SBERT = True
    print(">> [Info] SBERT enabled.")
except ImportError:
    HAS_SBERT = False

class EvalHelper:
    def __init__(self, device, threshold=0.85):
        self.device = device
        self.threshold = threshold
        self.sbert_model = None
        if HAS_SBERT:
            try:
                self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            except:
                self.sbert_model = None

    def normalize_text(self, text):
        if not isinstance(text, str): return str(text)
        text = text.lower()
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def is_match(self, pred, gt):
        norm_pred = self.normalize_text(pred)
        norm_gt = self.normalize_text(gt)
        if norm_pred == norm_gt: return True
        if len(norm_gt) > 3 and norm_gt in norm_pred: return True
        if self.sbert_model is not None:
            emb1 = self.sbert_model.encode(norm_pred, convert_to_tensor=True)
            emb2 = self.sbert_model.encode(norm_gt, convert_to_tensor=True)
            similarity = util.cos_sim(emb1, emb2).item()
            if similarity > self.threshold: return True
        return False

def train_model_seq(model, train_loader, test_loader, config, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"\n{'='*60}")
    print(f"🚀 PHASE 3: UNFREEZE THE EYES (Fine-tuning CNN)")
    print(f"   Strategy: Load 'medvqa_final_boost.pth' -> Unfreeze ResNet -> Ultra Low LR")
    print(f"{'='*60}\n")

    # 1. 加载准确率的最新模型
    # ---------------------------------------------------------
    pretrained_path = "medvqa_final_boost.pth" 
    # ---------------------------------------------------------
    
    if os.path.exists(pretrained_path):
        print(f"🔍 Loading your best model: {pretrained_path}")
        try:
            state_dict = torch.load(pretrained_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            print("✅ Loaded successfully!")
        except Exception as e:
            print(f"⚠️ Load failed: {e}. Cannot proceed with fine-tuning.")
            return
    else:
        print(f"❌ Error: File '{pretrained_path}' not found! Did you delete it?")
        return

    # 2. 关键步骤：解冻 CNN (ResNet)
    print("🔓 Unfreezing ResNet parameters for medical fine-tuning...")
    for param in model.resnet_features.parameters():
        param.requires_grad = True
    
    # 3. 学习率：必须非常非常小 (1e-5)，否则会破坏刚学好的知识
    lr = 1e-5 
    print(f"🔧 Learning Rate set to ULTRA LOW: {lr} (Safety Mode)")
    
    # 注意：这里必须把 model.parameters() 传进去，包含刚解冻的 CNN 参数
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    # 4. 再训练 20 轮 (精细打磨)
    num_epochs = 20
    best_test_acc = 0.0 
    evaluator = EvalHelper(device=device)
    
    yes_ids = tokenizer.encode("yes", add_special_tokens=False) 
    no_ids = tokenizer.encode("no", add_special_tokens=False)
    
    bos_idx = tokenizer.cls_token_id
    eos_idx = tokenizer.sep_token_id

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        # 保持对 Yes/No 的适度惩罚 (0.2)，让它不要忘记 Open
        loss_weights = torch.ones(tokenizer.vocab_size).to(device)
        strategy = "👁️ CNN Fine-Tuning (LR=1e-5)"
        w = 0.2 
            
        for idx in yes_ids: loss_weights[idx] = w
        for idx in no_ids: loss_weights[idx] = w
        
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, weight=loss_weights)

        # --- Train ---
        model.train()
        total_loss = 0.0
        for images, questions, answers_seq in train_loader:
            images = images.to(device)
            questions = questions.to(device)
            answers_seq = answers_seq.to(device)
            
            decoder_input = answers_seq[:, :-1]
            targets = answers_seq[:, 1:]
            
            optimizer.zero_grad()
            scores = model(images, questions, decoder_input)
            loss = criterion(scores.reshape(-1, scores.size(-1)), targets.reshape(-1))
            loss.backward()
            
            # 梯度裁剪：保护 CNN 不被梯度爆炸摧毁
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # --- Evaluate ---
        model.eval()
        closed_correct = 0; closed_total = 0
        open_correct = 0; open_total = 0
        
        debug_corrects = []

        with torch.no_grad():
            for images, questions, answers_seq in test_loader:
                images = images.to(device)
                questions = questions.to(device)
                gen_ids = model.generate_answer(images, questions, bos_idx, eos_idx)
                
                pred_str = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                gt_str = tokenizer.decode(answers_seq[0], skip_special_tokens=True)
                
                is_closed = evaluator.normalize_text(gt_str) in ["yes", "no"]
                is_correct = evaluator.is_match(pred_str, gt_str)
                
                if is_closed:
                    closed_total += 1
                    if is_correct: closed_correct += 1
                else:
                    open_total += 1
                    if is_correct: open_correct += 1
                
                if is_correct and not is_closed and len(debug_corrects) < 5:
                    debug_corrects.append(f"GT: {gt_str} | Pred: {pred_str}")

        closed_acc = closed_correct / closed_total if closed_total else 0.0
        open_acc = open_correct / open_total if open_total else 0.0
        total_acc = (closed_correct + open_correct) / (closed_total + open_total) if (closed_total + open_total) else 0.0
        
        epoch_time = time.time() - start_time
        
        print(f"FineTune Epoch {epoch}/{num_epochs} [{strategy}] | Loss: {avg_loss:.4f}")
        print(f"   >>> Total: {total_acc:.2%} (Closed: {closed_acc:.2%} | Open: {open_acc:.2%})")

        if len(debug_corrects) > 0:
            print("   [✨ Open Success]:")
            for s in debug_corrects: print(f"    -> {s}")

        # 保存策略：只要有进步就保存
        # 保存为最终的 Ultimate 版本
        if total_acc >= best_test_acc or open_acc > 0.30:
            if total_acc > best_test_acc: best_test_acc = total_acc
            
            save_name = "medvqa_ultimate.pth"
            torch.save(model.state_dict(), save_name)
            print(f"   🏆 Saved Ultimate Model: {save_name}")

    print(f"\n✅ All Training Finished.")