import torch
import torch.nn as nn
import re
import string
import os
import time

# --- Advanced Semantic Evaluation Module ---
try:
    from sentence_transformers import SentenceTransformer, util

    HAS_SBERT = True
    print(">> [Info] SBERT enabled for semantic evaluation.")
except ImportError:
    HAS_SBERT = False
    print(">> [Warning] SBERT not found. Using strict matching.")


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


# --- Core training function ---
def train_model_seq(model, train_loader, test_loader, config, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"\n{'=' * 60}")
    print(f"🚀 CORRECTED STRATEGY: Stability First, Then Boost")
    print(f"   Goal: Build Foundation (Ep 1-15) -> Augment & Boost (Ep 16-45)")
    print(f"{'=' * 60}\n")

    # 1. Intelligent weight loading
    pretrained_path = "medvqa_advanced_bert_best.pth"  # Prioritize loading the BERT weights
    loaded_flag = False

    if os.path.exists(pretrained_path):
        print(f"🔍 Checking checkpoint: {pretrained_path}")
        try:
            state_dict = torch.load(pretrained_path, map_location=device)
            # Check the dimensions of the vocabulary list
            if 'embedding.weight' in state_dict:
                ckpt_vocab = state_dict['embedding.weight'].shape[0]
                model_vocab = tokenizer.vocab_size
                if ckpt_vocab != model_vocab:
                    print(f"⚠️  Shape Mismatch! Restarting from SCRATCH.")
                    loaded_flag = False
                else:
                    model.load_state_dict(state_dict, strict=False)
                    print("✅ Checkpoint loaded successfully! Continuing training...")
                    loaded_flag = True
            else:
                model.load_state_dict(state_dict, strict=False)
                loaded_flag = True
        except Exception as e:
            print(f"⚠️  Load error: {e}. Training from SCRATCH.")
            loaded_flag = False
    else:
        print("🆕 No BERT checkpoint found. Starting from SCRATCH.")

    # 2. Optimizer configuration
    # The BERT vocabulary is quite large. For training from scratch, it is recommended to use 5e-4 or 1e-3. If there is oscillation, then reduce the value.
    lr = 5e-4 if not loaded_flag else 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    print(f"🔧 Learning Rate: {lr}")

    # 3. Training cycle (45 rounds)
    num_epochs = 45
    best_test_acc = 0.0
    evaluator = EvalHelper(device=device)

    yes_ids = tokenizer.encode("yes", add_special_tokens=False)
    no_ids = tokenizer.encode("no", add_special_tokens=False)

    bos_idx = tokenizer.cls_token_id
    eos_idx = tokenizer.sep_token_id

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # --- Key modification: Dynamic weighting strategy ---
        loss_weights = torch.ones(tokenizer.vocab_size).to(device)

        # 🟢 Phase 1 (Epoch 1-15): Basal period
        # The weight is set to 1.0 (balanced).
        # Objective: Let the model first learn "Yes/No", build confidence, and reduce the Loss.
        if epoch <= 15 and not loaded_flag:
            strategy = "Phase 1: Foundation (Balanced)"
            w = 1.0
            # 🟢 Phase 2 (Epoch 16-45): Growth period
        # The weight is set to 0.5.
        # Objective: Once the foundation is laid, start applying moderate penalties (Yes/No) to force the model to learn difficult words.
        else:
            strategy = "Phase 2: Hybrid Boosting"
            w = 0.5

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
        closed_correct = 0;
        closed_total = 0
        open_correct = 0;
        open_total = 0

        debug_corrects = []  # Record the correct Open answer

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

                if is_correct and not is_closed and len(debug_corrects) < 3:
                    debug_corrects.append(f"GT: {gt_str} | Pred: {pred_str}")

        # Statistics
        closed_acc = closed_correct / closed_total if closed_total else 0.0
        open_acc = open_correct / open_total if open_total else 0.0
        total_acc = (closed_correct + open_correct) / (closed_total + open_total) if (
                    closed_total + open_total) else 0.0

        epoch_time = time.time() - start_time

        print(f"Epoch {epoch}/{num_epochs} [{strategy}] | Loss: {avg_loss:.4f} | Time: {epoch_time:.0f}s")
        print(f"   >>> Total: {total_acc:.2%} (Closed: {closed_acc:.2%} | Open: {open_acc:.2%})")

        if len(debug_corrects) > 0:
            print("   [✨ Open Success]:")
            for s in debug_corrects: print(f"    -> {s}")

        # Save
        if total_acc > best_test_acc:
            best_test_acc = total_acc
            torch.save(model.state_dict(), "medvqa_13best.pth")
            print(f"   🏆 New Best Saved! ({best_test_acc:.2%})")


    print(f"\n✅ Training Finished. Best Accuracy: {best_test_acc:.2%}")
