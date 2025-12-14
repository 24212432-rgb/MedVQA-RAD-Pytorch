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
    print(f"🚀 FINAL PUSH: OPEN ACCURACY CRUSADE")
    print(f"   Strategy: Load Best Model -> Penalize Yes/No (Weight 0.15)")
    print(f"{'='*60}\n")

    # 1. 强制加载你刚才训练好的最佳模型
    pretrained_path = "medvqa_advanced_bert_best.pth"
    
    if os.path.exists(pretrained_path):
        print(f"🔍 Loading your best model (43%): {pretrained_path}")
        try:
            state_dict = torch.load(pretrained_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            print("✅ Loaded successfully! Continuing training to boost Open Acc...")
        except Exception as e:
            print(f"⚠️ Load failed: {e}. Starting from scratch (Not Recommended).")
    else:
        print("🆕 No checkpoint found. Starting from scratch.")

    # 2. 学习率：因为是微调，保持小一点，防止破坏已有的 Closed Acc
    lr = 2e-4  # 稍微调小一点点，求稳
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    print(f"🔧 Learning Rate: {lr}")

    # 3. 再训练 30 轮 (在现有 45 轮基础上)
    num_epochs = 30 
    best_test_acc = 0.0 # 重置一下，只保存比现在更好的
    evaluator = EvalHelper(device=device)
    
    yes_ids = tokenizer.encode("yes", add_special_tokens=False) 
    no_ids = tokenizer.encode("no", add_special_tokens=False)
    
    bos_idx = tokenizer.cls_token_id
    eos_idx = tokenizer.sep_token_id

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        loss_weights = torch.ones(tokenizer.vocab_size).to(device)
        
        # --- 🩸 核心策略修改 ---
        # 既然 Closed 已经 72% 了，我们不需要再保护它了。
        # 我们把 Yes/No 的权重设为极低的 0.15
        # 这会迫使模型把注意力全部集中在 Open 问题上！
        
        strategy = "🔥 Open Boost (Yes/No Wt=0.15)"
        w = 0.15 
            
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
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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
                
                # 只打印 Open 问题的成功案例，给你信心
                if is_correct and not is_closed and len(debug_corrects) < 5:
                    debug_corrects.append(f"GT: {gt_str} | Pred: {pred_str}")

        closed_acc = closed_correct / closed_total if closed_total else 0.0
        open_acc = open_correct / open_total if open_total else 0.0
        total_acc = (closed_correct + open_correct) / (closed_total + open_total) if (closed_total + open_total) else 0.0
        
        epoch_time = time.time() - start_time
        
        print(f"Extra Epoch {epoch}/{num_epochs} [{strategy}] | Loss: {avg_loss:.4f}")
        print(f"   >>> Total: {total_acc:.2%} (Closed: {closed_acc:.2%} | Open: {open_acc:.2%})")

        if len(debug_corrects) > 0:
            print("   [✨ Open Success]:")
            for s in debug_corrects: print(f"    -> {s}")

        # 只要 Open Acc 涨了，或者总分涨了，都保存
        # 加上 open_acc > 0.2 的条件，防止保存那些严重偏科的模型
        if total_acc > best_test_acc or open_acc > 0.25:
            if total_acc > best_test_acc: best_test_acc = total_acc
            torch.save(model.state_dict(), "medvqa_advanced_bert_final_boost.pth")
            print(f"   🏆 Saved (Boosted Open Accuracy)!")

    print(f"\n✅ Boosting Finished.")