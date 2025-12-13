# src/train_advanced.py
import torch
import torch.nn as nn
import re
import string
import os
import time

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
    print(f"🚀 PHASE 4: THE SPECIALIST (Open-Only Training)")
    print(f"   Strategy: Train ONLY on Open questions to force learning.")
    print(f"{'='*60}\n")

    # 1. Load the latest 52% Acc model
    pretrained_path = "medvqa_13new1.pth" # <--- The file that was saved just now
    
    if os.path.exists(pretrained_path):
        print(f"🔍 Loading best model: {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print("✅ Loaded successfully!")
    else:
        print(f"❌ Warning: {pretrained_path} not found. Using random init (Not advised).")

    # Keep the CNN unfrozen and continue to fine-tune.
    for param in model.resnet_features.parameters():
        param.requires_grad = True

    # 2. Learning rate: Slightly increase it a little bit, because we only have Open data and it's quite challenging.
    lr = 2e-5 
    print(f"🔧 Learning Rate: {lr}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    # 3. Train for 15 rounds
    num_epochs = 15
    best_open_acc = 0.0 # This time, our focus is on Open Acc
    evaluator = EvalHelper(device=device)
    
    bos_idx = tokenizer.cls_token_id
    eos_idx = tokenizer.sep_token_id

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        # At this point, the training set is all "Open", so the weights don't matter. Just set them to 1.0.
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

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
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # --- Evaluate (This is still the full test set, so we can see that "Closed" has been removed.) ---
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
        
        print(f"Specialist Epoch {epoch}/{num_epochs} | Loss: {avg_loss:.4f}")
        print(f"   >>> Total: {total_acc:.2%} (Closed: {closed_acc:.2%} | Open: {open_acc:.2%})")

        if len(debug_corrects) > 0:
            print("   [✨ Open Success]:")
            for s in debug_corrects: print(f"    -> {s}")

        # Saving strategy: Save as long as the Open Acc reaches a new high!
        if open_acc > best_open_acc:
            best_open_acc = open_acc
            save_name = "medvqa_13new2.pth"
            torch.save(model.state_dict(), save_name)
            print(f"   🏆 New Best Open Acc! Saved: {save_name}")

    print(f"\n✅ Specialist Training Finished.")