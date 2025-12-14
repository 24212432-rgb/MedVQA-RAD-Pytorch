import torch
import torch.nn as nn
import re
import string
import time
import os

try:
    from sentence_transformers import SentenceTransformer, util
    HAS_SBERT = True
    print(">> [Info] SBERT enabled for semantic evaluation.")
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

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """单轮训练函数"""
    model.train()
    total_loss = 0.0
    
    for images, questions, answers_seq in train_loader:
        images, questions, answers_seq = images.to(device), questions.to(device), answers_seq.to(device)
        
        decoder_input = answers_seq[:, :-1]
        targets = answers_seq[:, 1:]
        
        optimizer.zero_grad()
        scores = model(images, questions, decoder_input)
        loss = criterion(scores.reshape(-1, scores.size(-1)), targets.reshape(-1))
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def evaluate_engine(model, test_loader, tokenizer, evaluator, device):
    """纯粹的评估引擎，不含任何训练逻辑"""
    model.eval()
    closed_correct = 0; closed_total = 0
    open_correct = 0; open_total = 0
    debug_samples = []
    
    bos_idx = tokenizer.cls_token_id
    eos_idx = tokenizer.sep_token_id

    with torch.no_grad():
        for images, questions, answers_seq in test_loader:
            images, questions = images.to(device), questions.to(device)
            
            # 使用生成模式
            gen_ids = model.generate_answer(images, questions, bos_idx, eos_idx)
            
            # 批次解码
            for j in range(len(gen_ids)):
                pred_str = tokenizer.decode(gen_ids[j], skip_special_tokens=True)
                gt_str = tokenizer.decode(answers_seq[j], skip_special_tokens=True)
                
                is_correct = evaluator.is_match(pred_str, gt_str)
                is_closed = evaluator.normalize_text(gt_str) in ["yes", "no"]
                
                if is_closed:
                    closed_total += 1
                    if is_correct: closed_correct += 1
                else:
                    open_total += 1
                    if is_correct: open_correct += 1
                
                # 收集少量 Open 题的正确案例用于展示
                if not is_closed and is_correct and len(debug_samples) < 3:
                    debug_samples.append(f"GT: {gt_str} | Pred: {pred_str}")

    return closed_correct, closed_total, open_correct, open_total, debug_samples