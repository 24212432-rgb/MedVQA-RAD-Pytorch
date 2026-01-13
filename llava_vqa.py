import os
import json
import random
import re
import gc
import math
import warnings
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoProcessor, LlavaForConditionalGeneration, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel

# ==========================================
# 0. System Setup
# ==========================================
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

gc.collect()
torch.cuda.empty_cache()

# ==========================================
# 1. Config
# ==========================================
class Config:
    DATA_DIR = "/content/drive/MyDrive/7015/data"
    TRAIN_JSON = f"{DATA_DIR}/trainset_image_disjoint.json"
    TEST_JSON  = f"{DATA_DIR}/testset_image_disjoint.json"
    IMG_DIR    = f"{DATA_DIR}/VQA_RAD Image Folder"
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    SAVE_DIR = "saved_models/llava_a100_v_final_mix"
    MODEL_ID = "llava-hf/llava-1.5-7b-hf"

    SEED = 42
    VAL_SPLIT_RATIO = 0.1

    BATCH_SIZE = 8
    GRAD_ACCUM = 4
    NUM_WORKERS = 4
    NUM_EPOCHS = 20

   
    LR = 8e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.05
    MAX_GRAD_NORM = 1.0

    LORA_R = 64
    LORA_ALPHA = 128
    LORA_DROPOUT = 0.05

    PATIENCE = 6


    MAX_LENGTH = 1024


    OPEN_WEIGHT = 3.0
    CLOSED_WEIGHT = 1.0

    PRINT_TYPE_STATS = True
    PRINT_SANITY_EVAL = True
    RUN_MASK_PREFLIGHT = True

 
    OPEN_MAX_NEW_TOKENS_VAL  = 12
    OPEN_MAX_NEW_TOKENS_TEST = 16


    USE_GRAD_CHECKPOINTING = False

   
    TRAIN_MM_PROJECTOR = True


    OPEN_TOPK_CANDIDATES = 512           
    OPEN_USE_CONSTRAINED_DECODE = True   
    OPEN_CONSTRAINED_NUM_BEAMS_VAL  = 5
    OPEN_CONSTRAINED_NUM_BEAMS_TEST = 5
    OPEN_CONSTRAINED_LENGTH_PENALTY = 0.6  
   
    OPEN_FALLBACK_AVG_LOGPROB = -2.2

    @classmethod
    def save_path(cls):
        os.makedirs(cls.SAVE_DIR, exist_ok=True)
        return cls.SAVE_DIR


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ==========================================
# 2. Integrity Checks
# ==========================================
def check_image_integrity_unique(data, img_dir):
    print("Deep checking images (unique mode)...")
    dir_files = {
        f.name.lower(): f
        for f in Path(img_dir).rglob("*")
        if f.is_file() and f.suffix.lower() in Config.IMAGE_EXTS
    }
    unique_imgs = set(str(x["image_name"]).lower() for x in data)
    missing, corrupt = 0, 0
    for img_key in tqdm(unique_imgs, desc="Verifying images", leave=False):
        if img_key not in dir_files:
            missing += 1
            continue
        path = dir_files[img_key]
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception:
            corrupt += 1
    if missing > 0 or corrupt > 0:
        raise FileNotFoundError(f"Strict integrity failed: missing={missing}, corrupt={corrupt}")
    print(f"Verified {len(unique_imgs)} unique images.")


def verify_strict_disjointness(train_data, test_data):
    train_imgs = set(str(x["image_name"]).lower() for x in train_data)
    test_imgs = set(str(x["image_name"]).lower() for x in test_data)
    overlap = train_imgs.intersection(test_imgs)
    if overlap:
        raise ValueError(f"Data leakage: {len(overlap)} images appear in both train and test.")
    print("Disjoint check passed.")

# ==========================================
# 3. Prompt + Open/Closed
# ==========================================
def build_prompt(question, is_closed):
    if is_closed:
        instruction = "Answer only 'yes' or 'no'."
    else:
        instruction = (
            "Answer with ONLY the final answer (1-3 words). "
            "No explanation. No 'and/or'."
        )
    content = f"<image>\n{question}\n{instruction}"
    return f"USER: {content}\nASSISTANT:"


def is_closed_question(item):
    if "question_type" in item:
        q_type = str(item["question_type"]).upper()
        if "OPEN" in q_type:
            return False
        if "CLOSE" in q_type or "YES" in q_type:
            return True

    ans = str(item.get("answer", "")).strip().lower()
    if ans in ("yes", "no") or ans.startswith("y") or ans.startswith("n"):
        return True
    if ans != "":
        return False

    q = str(item.get("question", "")).strip().lower()
    starters = ["is","are","was","were","do","does","did","can","could","should","has","have","will","would"]
    return (q.split(" ")[0] in starters) if q else False

# ==========================================
# 4. Text Normalization 
# ==========================================
_ARTICLES = ("the ", "a ", "an ")

def strip_articles(s: str) -> str:
    s = s.strip()
    s_low = s.lower()
    for a in _ARTICLES:
        if s_low.startswith(a):
            return s[len(a):].strip()
    return s

def normalize_open_answer(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(".,;!?")
    s = strip_articles(s)
    return s

def normalize_text_for_eval(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = strip_articles(s)
    s = re.sub(r"[^\w\s]", "", s)
    return s.strip()

# ==========================================
# 5. Helpers
# ==========================================
def build_image_cache(img_dir):
    cache = {}
    collisions = []
    for f in Path(img_dir).rglob("*"):
        if f.is_file() and f.suffix.lower() in Config.IMAGE_EXTS:
            k = f.name.lower()
            if k in cache:
                collisions.append((k, str(cache[k]), str(f)))
            else:
                cache[k] = f
    if collisions:
        raise ValueError(f"Filename collisions found. Example: {collisions[:3]}")
    return cache


def load_image_from_cache(img_cache, image_name):
    p = img_cache.get(str(image_name).lower())
    if p is None:
        return Image.new("RGB", (336, 336), (0, 0, 0))
    try:
        with Image.open(p) as im:
            return im.convert("RGB")
    except Exception:
        return Image.new("RGB", (336, 336), (0, 0, 0))


def _normalize_answer_for_len(a: str, is_closed: bool) -> str:
    a = str(a).strip()
    if is_closed:
        a = a.lower()
        if a.startswith("y"): return "yes"
        if a.startswith("n"): return "no"
        return a
    return normalize_open_answer(a)


def estimate_required_max_length_textonly(all_items, tokenizer, image_seq_len: int, safety_pad: int = 8) -> int:
 
    eos = tokenizer.eos_token or ""
    image_id = None
    try:
        image_id = tokenizer.convert_tokens_to_ids("<image>")
        if image_id is None or image_id < 0:
            image_id = None
    except Exception:
        image_id = None

    max_len = 0
    for it in tqdm(all_items, desc="Estimating MAX_LENGTH (text-only)", leave=False):
        if str(it.get("answer", "")).strip() == "":
            continue
        closed = is_closed_question(it)
        a = _normalize_answer_for_len(it.get("answer", ""), closed)
        text = build_prompt(it.get("question", ""), closed) + " " + a + eos

        ids = tokenizer.encode(text, add_special_tokens=False)
        if image_id is None or image_seq_len is None:
            L = len(ids)
        else:
            cnt = sum(1 for x in ids if x == image_id)
            L = len(ids) - cnt + cnt * int(image_seq_len)

        max_len = max(max_len, L)

    return int(max_len + safety_pad)


def find_all_subsequence_positions(hay, needle):
    if not needle or len(needle) > len(hay):
        return []
    out = []
    for s in range(0, len(hay) - len(needle) + 1):
        if hay[s : s + len(needle)] == needle:
            out.append(s)
    return out


def locate_answer_span_in_full_seq(full_seq, tokenizer, answer_text):
    eos_id = tokenizer.eos_token_id

    image_id = None
    try:
        image_id = tokenizer.convert_tokens_to_ids("<image>")
        if image_id is None or image_id < 0:
            image_id = None
    except Exception:
        image_id = None

    end = len(full_seq)

    if image_id is not None:
        while end > 0 and full_seq[end - 1] == image_id:
            end -= 1

    eos_pos = None
    if eos_id is not None and end > 0 and full_seq[end - 1] == eos_id:
        eos_pos = end - 1
        end = eos_pos

    prefix = full_seq[:end]

    cands = [
        tokenizer.encode(" " + answer_text, add_special_tokens=False),
        tokenizer.encode(answer_text, add_special_tokens=False),
    ]
    cands = [c for c in cands if len(c) > 0]

    for cand in cands:
        if len(prefix) >= len(cand) and prefix[-len(cand):] == cand:
            s = len(prefix) - len(cand)
            return s, len(prefix), eos_pos

    best_s = -1
    best_len = 0
    for cand in cands:
        pos = find_all_subsequence_positions(prefix, cand)
        if pos:
            s = max(pos)
            if s > best_s:
                best_s = s
                best_len = len(cand)

    if best_s >= 0:
        return best_s, best_s + best_len, eos_pos

    tail_ids = full_seq[max(0, len(full_seq) - 160):]
    tail_txt = tokenizer.decode(tail_ids, skip_special_tokens=False)
    raise RuntimeError(f"Failed to locate answer tokens. Answer='{answer_text}'. TailDecoded='{tail_txt}'")


def left_pad_batch(input_ids, attention_mask, pad_id):
    B, L = input_ids.shape
    lens = attention_mask.sum(dim=1).to(torch.long)

    out_ids  = torch.full_like(input_ids, pad_id)
    out_mask = torch.zeros_like(attention_mask)

    for i in range(B):
        li = int(lens[i].item())
        if li <= 0:
            continue
        out_ids[i, L-li:L]  = input_ids[i, :li]
        out_mask[i, L-li:L] = 1
    return out_ids, out_mask, lens.tolist(), L

# ==========================================
# 6. Dataset
# ==========================================
class FinalDataset(Dataset):
    def __init__(self, data, img_dir, processor, train=True):
        self.data = [x for x in data if str(x.get("answer", "")).strip() != ""]
        self.processor = processor
        self.train = train

        self.img_cache = build_image_cache(img_dir)
        self.closed_flags = [is_closed_question(x) for x in self.data]

    def __len__(self):
        return len(self.data)

    def normalize_answer(self, a, is_closed):
        a = str(a).strip()
        if is_closed:
            a = a.lower()
            if a.startswith("y"): return "yes"
            if a.startswith("n"): return "no"
            return a
    
        return normalize_open_answer(a)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = load_image_from_cache(self.img_cache, item["image_name"])

        q = item["question"]
        is_closed = self.closed_flags[idx]
        a = self.normalize_answer(item["answer"], is_closed)
        prompt_text = build_prompt(q, is_closed)

        tok = self.processor.tokenizer
        eos = tok.eos_token or ""

        if self.train:
            full_text = prompt_text + " " + a + eos

            inputs = self.processor(
                text=full_text,
                images=image,
                padding="max_length",
                truncation=True,
                max_length=Config.MAX_LENGTH,
                return_tensors="pt",
            )
            input_ids = inputs["input_ids"].squeeze(0)
            mask = inputs["attention_mask"].squeeze(0)
            pixel_values = inputs["pixel_values"].squeeze(0)

            nz = torch.nonzero(mask, as_tuple=False)
            nonpad = int(nz[-1].item()) + 1 if nz.numel() > 0 else 0
            full_seq = input_ids[:nonpad].tolist()

            ans_start, ans_end, eos_pos = locate_answer_span_in_full_seq(full_seq, tok, a)

            labels = torch.full_like(input_ids, -100)
            labels[mask == 0] = -100

            labels[ans_start:ans_end] = input_ids[ans_start:ans_end]

            if eos_pos is not None and ans_end == eos_pos:
                labels[eos_pos] = input_ids[eos_pos]

        else:
            eval_prompt = prompt_text + " "
            inputs = self.processor(
                text=eval_prompt,
                images=image,
                padding="max_length",
                truncation=True,
                max_length=Config.MAX_LENGTH,
                return_tensors="pt",
            )
            input_ids = inputs["input_ids"].squeeze(0)
            mask = inputs["attention_mask"].squeeze(0)
            pixel_values = inputs["pixel_values"].squeeze(0)
            labels = torch.full((Config.MAX_LENGTH,), -100, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": mask,
            "pixel_values": pixel_values,
            "labels": labels,
            "answer": a,
            "closed": is_closed,
        }


def collate(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
        "answers": [x["answer"] for x in batch],
        "closed": [x["closed"] for x in batch],
    }

# ==========================================
# 7. Open Candidate Trie 
# ==========================================
class TrieNode:
    __slots__ = ("children",)
    def __init__(self):
        self.children: Dict[int, "TrieNode"] = {}

class TokenTrie:
    def __init__(self, sequences: List[List[int]]):
        self.root = TrieNode()
        for seq in sequences:
            node = self.root
            for t in seq:
                if t not in node.children:
                    node.children[t] = TrieNode()
                node = node.children[t]

    def allowed_next(self, prefix: List[int]) -> List[int]:
        node = self.root
        for t in prefix:
            if t not in node.children:
                return []  # prefix not found
            node = node.children[t]
        return list(node.children.keys())


def build_open_candidates_and_trie(train_list, tokenizer, top_k: int) -> Tuple[List[str], Optional[TokenTrie], int]:

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise RuntimeError("Tokenizer has no eos_token_id.")

    counter = Counter()
    for it in train_list:
        if str(it.get("answer", "")).strip() == "":
            continue
        if is_closed_question(it):
            continue
        a = normalize_open_answer(it.get("answer", ""))
        if not a:
            continue

        a = " ".join(a.split()[:3])
        if not a:
            continue
        counter[a] += 1

    most = counter.most_common(top_k)
    candidates = [a for a, _ in most]

  
    total_open = sum(counter.values())
    covered = sum(cnt for _, cnt in most)
    cov = covered / max(total_open, 1)
    print(f"[OpenCand] unique_open={len(counter)}  topK={len(candidates)}  coverage_in_train_open={cov:.2%}")

 
    seqs = []
    max_new = 1
    for c in candidates:
        ids = tokenizer.encode(" " + c, add_special_tokens=False)
        ids = ids + [eos_id]
        seqs.append(ids)
        max_new = max(max_new, len(ids))

    trie = TokenTrie(seqs) if seqs else None
    return candidates, trie, int(max_new)

# ==========================================
# 8. Evaluation Helpers
# ==========================================
def clean_open_pred(t: str) -> str:
    t = t.strip().lower()
    if "assistant:" in t:
        t = t.split("assistant:")[-1].strip()
    if "user:" in t:
        t = t.split("user:")[-1].strip()

    t = re.split(r"[\n\.\?!]", t)[0].strip()

    toks = t.split()
    out = []
    for w in toks:
        if w in {"and", "or"}:
            break
        out.append(w)
        if len(out) >= 3:
            break

    t2 = " ".join(out).strip()
    t2 = strip_articles(t2)
    return t2


def _compute_avg_logprob_from_generate(model, gen_out, prompt_len: int, eos_id: int, pad_id: int) -> torch.Tensor:

    seq = gen_out.sequences  # [B, prompt_len + new]
    scores = getattr(gen_out, "scores", None)
    if scores is None:
        return torch.full((seq.size(0),), -999.0, device=seq.device)

    beam_indices = getattr(gen_out, "beam_indices", None)

    try:
        if beam_indices is not None:
            trans = model.compute_transition_scores(seq, scores, beam_indices, normalize_logits=True)
        else:
            trans = model.compute_transition_scores(seq, scores, normalize_logits=True)
        # trans shape: [B, max_new_tokens]
        gen_tokens = seq[:, prompt_len:]
        B = gen_tokens.size(0)
        lengths = []
        for i in range(B):
            toks = gen_tokens[i].tolist()
            L = 0
            for x in toks:
                if x == pad_id:
                    break
                L += 1
                if x == eos_id:
                    break
            lengths.append(max(L, 1))
        lengths_t = torch.tensor(lengths, device=seq.device, dtype=torch.long)
        # sum first L scores
        s = torch.zeros((B,), device=seq.device, dtype=trans.dtype)
        for i in range(B):
            s[i] = trans[i, : lengths[i]].sum()
        return s / lengths_t.to(s.dtype)
    except Exception:
        return torch.full((seq.size(0),), -999.0, device=seq.device)


@torch.inference_mode()
def predict_yesno_by_logprob(model, processor, ids, mask, pix):

    tok = processor.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id

    yes_seq = tok.encode(" yes", add_special_tokens=False)
    no_seq  = tok.encode(" no",  add_special_tokens=False)
    if len(yes_seq) == 0 or len(no_seq) == 0:
        raise RuntimeError("Tokenizer produced empty yes/no tokens.")

    prompt_lens = (ids != pad_id).sum(dim=1).to(torch.long)  # [bs]
    last_idx = (prompt_lens - 1).clamp(min=0)

    out1 = model(input_ids=ids, attention_mask=mask, pixel_values=pix, use_cache=False)
    logp1 = torch.log_softmax(out1.logits, dim=-1)

    bs = ids.size(0)
    b = torch.arange(bs, device=ids.device)

    s_yes = logp1[b, last_idx, torch.tensor(yes_seq[0], device=ids.device)]
    s_no  = logp1[b, last_idx, torch.tensor(no_seq[0],  device=ids.device)]

    if len(yes_seq) == 1 and len(no_seq) == 1:
        return ["yes" if s_yes[i] > s_no[i] else "no" for i in range(bs)]

    def forward_append_first(first_token_id: int):
        seqs = []
        lens = []
        for i in range(bs):
            pl = int(prompt_lens[i].item())
            prompt = ids[i, :pl]
            seq = torch.cat(
                [prompt, torch.tensor([first_token_id], device=ids.device, dtype=ids.dtype)],
                dim=0
            )
            seqs.append(seq)
            lens.append(seq.numel())

        max_len = max(lens)
        batch_ids  = torch.full((bs, max_len), pad_id, device=ids.device, dtype=ids.dtype)
        batch_mask = torch.zeros((bs, max_len), device=ids.device, dtype=mask.dtype)

        for i, seq in enumerate(seqs):
            L = seq.numel()
            batch_ids[i, :L] = seq
            batch_mask[i, :L] = 1

        out = model(input_ids=batch_ids, attention_mask=batch_mask, pixel_values=pix, use_cache=False)
        return out, lens

    if len(yes_seq) >= 2:
        out2, lens2 = forward_append_first(yes_seq[0])
        logp2 = torch.log_softmax(out2.logits, dim=-1)
        pos2 = torch.tensor([L - 1 for L in lens2], device=ids.device)
        s_yes = s_yes + logp2[b, pos2, torch.tensor(yes_seq[1], device=ids.device)]

    if len(no_seq) >= 2:
        out3, lens3 = forward_append_first(no_seq[0])
        logp3 = torch.log_softmax(out3.logits, dim=-1)
        pos3 = torch.tensor([L - 1 for L in lens3], device=ids.device)
        s_no = s_no + logp3[b, pos3, torch.tensor(no_seq[1], device=ids.device)]

    if len(yes_seq) > 2 or len(no_seq) > 2:
        raise RuntimeError(f"yes/no tokenized to >2 tokens: yes={yes_seq}, no={no_seq}")

    return ["yes" if s_yes[i] > s_no[i] else "no" for i in range(bs)]


def _make_prefix_allowed_fn(trie: TokenTrie, base_len: int, eos_id: int):
  
    def fn(batch_id: int, input_ids: torch.Tensor):
       
        prefix = input_ids[base_len:].tolist()
        allowed = trie.allowed_next(prefix)
        if not allowed:
         
            return [eos_id]
        return allowed
    return fn


@torch.inference_mode()
def generate_open_constrained(
    model,
    tokenizer,
    ids_lp,
    mask_lp,
    pix,
    trie: TokenTrie,
    max_new_tokens: int,
    num_beams: int,
    length_penalty: float,
    pad_id: int,
):
   
    base_len = ids_lp.size(1)
    eos_id = tokenizer.eos_token_id
    prefix_fn = _make_prefix_allowed_fn(trie, base_len, eos_id)

    out = model.generate(
        input_ids=ids_lp,
        attention_mask=mask_lp,
        pixel_values=pix,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=num_beams,
        length_penalty=length_penalty,
        early_stopping=True,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        use_cache=False,
        prefix_allowed_tokens_fn=prefix_fn,
        output_scores=True,
        return_dict_in_generate=True,
    )

    avg_lp = _compute_avg_logprob_from_generate(model, out, base_len, eos_id, pad_id)
    gen_only = out.sequences[:, base_len:]
    decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
    preds = [clean_open_pred(x) for x in decoded]
    return preds, avg_lp


@torch.inference_mode()
def generate_open_free(
    model,
    processor,
    ids_lp,
    mask_lp,
    pix,
    max_new_tokens: int,
    num_beams: int,
    pad_id: int,
    repetition_penalty: float = 1.10,
):
    tok = processor.tokenizer
    base_len = ids_lp.size(1)

    out = model.generate(
        input_ids=ids_lp,
        attention_mask=mask_lp,
        pixel_values=pix,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=num_beams,
        early_stopping=True if num_beams > 1 else False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=pad_id,
        use_cache=False,
        no_repeat_ngram_size=3,
        repetition_penalty=repetition_penalty,
        output_scores=True,
        return_dict_in_generate=True,
    )

    avg_lp = _compute_avg_logprob_from_generate(model, out, base_len, tok.eos_token_id, pad_id)
    gen_only = out.sequences[:, base_len:]
    decoded = processor.batch_decode(gen_only, skip_special_tokens=True)
    preds = [clean_open_pred(x) for x in decoded]
    return preds, avg_lp


@torch.inference_mode()
def evaluate(model, loader, processor, desc="Eval", mode="val",
             open_trie: Optional[TokenTrie] = None,
             open_cand_max_new: int = 8):
    model.eval()
    tok = processor.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id

    stats = defaultdict(float)
    counts = defaultdict(int)

    sanity_printed = False
    is_test = (mode == "test")

   
    free_beams = 3 if is_test else 3
    free_max_tokens = Config.OPEN_MAX_NEW_TOKENS_TEST if is_test else Config.OPEN_MAX_NEW_TOKENS_VAL

    cons_beams = Config.OPEN_CONSTRAINED_NUM_BEAMS_TEST if is_test else Config.OPEN_CONSTRAINED_NUM_BEAMS_VAL
    cons_lenpen = Config.OPEN_CONSTRAINED_LENGTH_PENALTY
    cons_max_tokens = max(open_cand_max_new, 6)  # 候选最长 tokens（含 eos）

    for batch in tqdm(loader, desc=f"{desc} (mode={mode})", leave=False):
        ids  = batch["input_ids"].to(model.device)
        mask = batch["attention_mask"].to(model.device)
        pix  = batch["pixel_values"].to(model.device).to(torch.bfloat16)

        bs = ids.shape[0]
        final_preds = [""] * bs

        closed_idxs = [i for i, c in enumerate(batch["closed"]) if c]
        open_idxs   = [i for i, c in enumerate(batch["closed"]) if not c]

        # ---- closed: yes/no logprob ----
        if closed_idxs:
            c_preds = predict_yesno_by_logprob(
                model, processor, ids[closed_idxs], mask[closed_idxs], pix[closed_idxs]
            )
            for idx, p in zip(closed_idxs, c_preds):
                final_preds[idx] = p

    
        if open_idxs:
            ids_o  = ids[open_idxs]
            mask_o = mask[open_idxs]
            ids_lp, mask_lp, _, _ = left_pad_batch(ids_o, mask_o, pad_id)

        
            cons_preds = None
            cons_lp = None
            if Config.OPEN_USE_CONSTRAINED_DECODE and open_trie is not None:
                cons_preds, cons_lp = generate_open_constrained(
                    model=model,
                    tokenizer=tok,
                    ids_lp=ids_lp,
                    mask_lp=mask_lp,
                    pix=pix[open_idxs],
                    trie=open_trie,
                    max_new_tokens=cons_max_tokens,
                    num_beams=cons_beams,
                    length_penalty=cons_lenpen,
                    pad_id=pad_id,
                )
            else:
               
                cons_preds, cons_lp = None, None

           
            use_free = [True] * len(open_idxs)
            chosen = [""] * len(open_idxs)

            if cons_preds is not None and cons_lp is not None:
                for j in range(len(open_idxs)):
                    # 置信度够 & 结果非空 => 用 constrained
                    if cons_preds[j] and (float(cons_lp[j].item()) >= Config.OPEN_FALLBACK_AVG_LOGPROB):
                        use_free[j] = False
                        chosen[j] = cons_preds[j]

            
            if any(use_free):
                free_sel = [j for j, f in enumerate(use_free) if f]
                ids_lp_f = ids_lp[free_sel]
                mask_lp_f = mask_lp[free_sel]
                pix_f = pix[open_idxs][free_sel]

                free_preds, _ = generate_open_free(
                    model=model,
                    processor=processor,
                    ids_lp=ids_lp_f,
                    mask_lp=mask_lp_f,
                    pix=pix_f,
                    max_new_tokens=free_max_tokens,
                    num_beams=free_beams,
                    pad_id=pad_id,
                )
                for k, j in enumerate(free_sel):
                    chosen[j] = free_preds[k]

            # 写回最终
            for idx, pred in zip(open_idxs, chosen):
                final_preds[idx] = pred

        # ---- sanity print ----
        if Config.PRINT_SANITY_EVAL and (not sanity_printed) and open_idxs:
            j = open_idxs[0]
            print("\nSanity check (open sample)")
            print(f"Truth: {batch['answers'][j]}")
            print(f"Pred:  {final_preds[j]}")
            sanity_printed = True

        # ---- metrics ----
        for p, t, c in zip(final_preds, batch["answers"], batch["closed"]):
            p_norm = normalize_text_for_eval(p)
            t_norm = normalize_text_for_eval(t)

            em = 1.0 if p_norm == t_norm else 0.0
            acc = 1.0 if (p_norm == t_norm) or (t_norm in p_norm and len(p_norm) <= len(t_norm) + 5) else 0.0

            p_toks, t_toks = p_norm.split(), t_norm.split()
            if not p_toks or not t_toks:
                f1 = 1.0 if p_toks == t_toks else 0.0
            else:
                common = Counter(p_toks) & Counter(t_toks)
                num = sum(common.values())
                if num == 0:
                    f1 = 0.0
                else:
                    prec = num / len(p_toks)
                    rec  = num / len(t_toks)
                    f1 = 2 * prec * rec / (prec + rec)

            stats["acc"] += acc
            stats["em"]  += em
            stats["f1"]  += f1
            counts["total"] += 1

            k = "closed" if c else "open"
            stats[f"{k}_acc"] += acc
            stats[f"{k}_em"]  += em
            stats[f"{k}_f1"]  += f1
            counts[k] += 1

    def score(k, c):
        return stats[k] / max(counts[c], 1)

    return {
        "Overall Accuracy": score("acc", "total"),
        "Overall EM":       score("em", "total"),
        "Overall F1":       score("f1", "total"),
        "Closed Accuracy":  score("closed_acc", "closed"),
        "Open Accuracy":    score("open_acc", "open"),
        "Open F1":          score("open_f1", "open"),
    }

# ==========================================
# 9. Preflight
# ==========================================
def print_type_stats(train_list, val_list, test_list):
    def _count_types(lst):
        c = sum(1 for x in lst if is_closed_question(x))
        o = len(lst) - c
        return c, o

    c_tr, o_tr = _count_types(train_list)
    c_va, o_va = _count_types(val_list)
    c_te, o_te = _count_types(test_list)
    print(f"TypeStats train closed/open: {c_tr}/{o_tr}")
    print(f"TypeStats val   closed/open: {c_va}/{o_va}")
    print(f"TypeStats test  closed/open: {c_te}/{o_te}")

    bad = 0
    for x in train_list:
        if is_closed_question(x):
            ans = str(x.get("answer", "")).strip().lower()
            if not (ans in ("yes", "no") or ans.startswith("y") or ans.startswith("n")):
                bad += 1
    print(f"TypeStats closed==True but answer not yes/no (train): {bad}")


def mask_preflight_check(train_ds, processor):
    print("Running mask preflight check...")
    item = train_ds[0]
    labels = item["labels"]
    ids = item["input_ids"]
    supervised = (labels != -100)
    n_sup = int(supervised.sum().item())
    sup_ids = ids[supervised].tolist()
    sup_text = processor.tokenizer.decode(sup_ids, skip_special_tokens=False)
    print(f"MaskPreflight sample_answer='{item['answer']}', closed={item['closed']}")
    print(f"MaskPreflight supervised_tok={n_sup}")
    print(f"MaskPreflight supervised_text='{sup_text}'")

# ==========================================
# 10. Main
# ==========================================
def main():
    set_seed(Config.SEED)
    print("V-MIX (TopK constrained open + fallback free + strip articles + MAXLEN safe)")

    try:
        processor = AutoProcessor.from_pretrained(Config.MODEL_ID, use_fast=False)
    except TypeError:
        processor = AutoProcessor.from_pretrained(Config.MODEL_ID)

    tok = processor.tokenizer
    tok.padding_side = "right"
    tok.truncation_side = "right"  
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    yes_ids = tok.encode(" yes", add_special_tokens=False)
    no_ids  = tok.encode(" no",  add_special_tokens=False)
    print(f"Token check: ' yes'->{yes_ids}, ' no'->{no_ids}")

    # 加载模型
    model = LlavaForConditionalGeneration.from_pretrained(
        Config.MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.config.use_cache = False

    if Config.USE_GRAD_CHECKPOINTING:
        model.gradient_checkpointing_enable()

    # LoRA
    peft_config = LoraConfig(
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        lora_dropout=Config.LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

   
    if Config.TRAIN_MM_PROJECTOR:
        unfrozen = 0
        for name, p in model.named_parameters():
            if ("multi_modal_projector" in name) or ("mm_projector" in name):
                p.requires_grad = True
                unfrozen += p.numel()
        print(f"[MMProjector] trainable_params_added={unfrozen}")

    model.print_trainable_parameters()

   
    with open(Config.TRAIN_JSON) as f:
        full_train = json.load(f)
    with open(Config.TEST_JSON) as f:
        test_list = json.load(f)

    check_image_integrity_unique(full_train, Config.IMG_DIR)
    check_image_integrity_unique(test_list, Config.IMG_DIR)
    verify_strict_disjointness(full_train, test_list)


    def get_split(data, val_ratio=0.1):
        img_map = defaultdict(list)
        for x in data:
            img_map[str(x["image_name"]).lower()].append(x)
        imgs = sorted(list(img_map.keys()))
        random.shuffle(imgs)
        split = int(len(imgs) * val_ratio)
        v_imgs = set(imgs[:split])

        train_items, val_items = [], []
        for k in imgs:
            items = img_map[k]
            if k in v_imgs:
                val_items.extend(items)
            else:
                train_items.extend(items)
        return train_items, val_items

    train_list, val_list = get_split(full_train, Config.VAL_SPLIT_RATIO)

    if Config.PRINT_TYPE_STATS:
        print_type_stats(train_list, val_list, test_list)


    image_seq_len = getattr(getattr(model, "config", None), "image_seq_length", None)
    if image_seq_len is None:
        image_seq_len = 576

    ctx_limit = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if ctx_limit is None or ctx_limit > 100000:
        ctx_limit = 4096

    all_items_for_len = train_list + val_list + test_list
    required_len = estimate_required_max_length_textonly(all_items_for_len, tok, image_seq_len, safety_pad=8)
    if required_len > ctx_limit:
        raise RuntimeError(f"Required sequence length {required_len} exceeds context limit {ctx_limit}.")
    Config.MAX_LENGTH = int(required_len)
    print(f"Auto MAX_LENGTH set to {Config.MAX_LENGTH} (image_seq_len={image_seq_len}, ctx_limit={ctx_limit})")

    open_candidates, open_trie, open_cand_max_new = build_open_candidates_and_trie(
        train_list=train_list,
        tokenizer=tok,
        top_k=Config.OPEN_TOPK_CANDIDATES,
    )

    # Dataset/Loader
    train_ds = FinalDataset(train_list, Config.IMG_DIR, processor, train=True)
    val_ds   = FinalDataset(val_list,   Config.IMG_DIR, processor, train=False)
    test_ds  = FinalDataset(test_list,  Config.IMG_DIR, processor, train=False)

    if Config.RUN_MASK_PREFLIGHT:
        mask_preflight_check(train_ds, processor)

    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False,
        collate_fn=collate, num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=Config.BATCH_SIZE, shuffle=False,
        collate_fn=collate, num_workers=Config.NUM_WORKERS, pin_memory=True
    )

    print("Building sampler...")
    sample_weights = [(Config.CLOSED_WEIGHT if c else Config.OPEN_WEIGHT) for c in train_ds.closed_flags]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,
        collate_fn=collate,
        num_workers=Config.NUM_WORKERS,
        worker_init_fn=seed_worker,
        pin_memory=True,
        persistent_workers=(Config.NUM_WORKERS > 0),
        prefetch_factor=2 if Config.NUM_WORKERS > 0 else None,
    )

    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY, fused=True)
        print("Using fused AdamW.")
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)

    total_steps = math.ceil(len(train_loader) / Config.GRAD_ACCUM) * Config.NUM_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * Config.WARMUP_RATIO), total_steps
    )

    print("Start training...")
    best_score = 0.0
    patience = 0

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        accum_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for _, batch in enumerate(pbar):
            accum_steps += 1

            inputs = {k: v.to(model.device, non_blocking=True) for k, v in batch.items()
                      if k in ["input_ids", "attention_mask", "labels"]}
            inputs["pixel_values"] = batch["pixel_values"].to(model.device, non_blocking=True).to(torch.bfloat16)

            loss = model(**inputs, use_cache=False).loss / Config.GRAD_ACCUM
            loss.backward()
            epoch_loss += loss.item() * Config.GRAD_ACCUM

            if accum_steps % Config.GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accum_steps = 0

                lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({"loss": f"{loss.item()*Config.GRAD_ACCUM:.3f}", "lr": f"{lr:.2e}"})

        if accum_steps > 0:
            scale = Config.GRAD_ACCUM / accum_steps
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        val_res = evaluate(
            model, val_loader, processor, desc="Eval", mode="val",
            open_trie=open_trie, open_cand_max_new=open_cand_max_new
        )
        curr_score = val_res["Overall Accuracy"] + val_res["Open F1"]

        print(f"Epoch {epoch} | Loss: {epoch_loss/len(train_loader):.4f}")
        print(
            f"  Acc: {val_res['Overall Accuracy']:.2%} | "
            f"Open Acc: {val_res['Open Accuracy']:.2%} | "
            f"Closed Acc: {val_res['Closed Accuracy']:.2%} | "
            f"Open F1: {val_res['Open F1']:.2%}"
        )

        if curr_score > best_score:
            best_score = curr_score
            patience = 0
            model.save_pretrained(Config.save_path())
            processor.save_pretrained(Config.save_path())
            print("Best model saved.")
        else:
            patience += 1
            if patience >= Config.PATIENCE:
                print("Early stopping.")
                break

    print("\nFinal test...")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    base = LlavaForConditionalGeneration.from_pretrained(
        Config.MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, Config.save_path())

    res = evaluate(
        model, test_loader, processor, desc="Testing", mode="test",
        open_trie=open_trie, open_cand_max_new=open_cand_max_new
    )

    print("\nFinal results")
    for k, v in res.items():
        print(f"{k:<25} | {v:.2%}")

    with open(f"{Config.SAVE_DIR}/final_results.json", "w") as f:
        json.dump(res, f, indent=4)


if __name__ == "__main__":
    main()
