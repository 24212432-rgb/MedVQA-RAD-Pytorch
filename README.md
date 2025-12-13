# MedVQA-RAD — Baseline vs Specialist Seq2Seq w/ Attention (PyTorch)

> **Medical Visual Question Answering (Med-VQA)** on **VQA-RAD**  

## 🧭 What is this?

This repo builds a **Medical Visual Question Answering** system: given a **radiology image** (X-ray / CT / MRI) and a **natural-language question**, the model outputs an answer.

I implement and compare:

- **Baseline (classification):** ResNet50 + LSTM classifier (optional GloVe)
- **Advanced (generation):** ResNet50 spatial features + **Attention** + **Seq2Seq** decoder (tokenized with `bert-base-uncased`)

---

## ✨ Key contributions

- **Closed vs Open breakdown**: report accuracy separately for **Yes/No** and **descriptive** questions.
- **Specialist Curriculum Learning (Phase 4)**: fine-tune on **open-ended** questions to reduce the “always answer yes” shortcut.
- **Semantic-match evaluation** for generated answers: optional **SBERT similarity** check to tolerate small wording variations.

---

## 🏆 Results (Highlights)

**Best reported performance** (from my final experiments):

| Model | Architecture | Total Accuracy | Closed Accuracy (Yes/No) | Open Accuracy (Descriptive) |
|---|---|---:|---:|---:|
| Baseline | ResNet50 + LSTM + (optional) GloVe | ~50.00% | 65.10% | 15.20% |
| **Advanced (Ours)** | **ResNet50 (spatial) + Seq2Seq + Attention** *(BERT tokenizer IDs)* | **56.39%** | **72.63%** | **38.24%** |

**Key takeaway:** the largest improvement is on **open-ended questions** (≈ **2.5×**), enabled by the **specialist open-only fine-tuning** stage.

> ⚠️ Note on wording: this repo uses **`bert-base-uncased` tokenizer** to convert text to IDs, but the **text encoder is an Embedding + LSTM** (not a full BERT Transformer). See `src/model_advanced.py`.

---

## 0) What you need (super explicit)

### Hardware
- GPU recommended (training is much faster), but CPU works with smaller batch sizes.

### Software
- Python 3.9+ (recommended: **3.10**)
- Install dependencies via `requirements.txt`

### Data (not included)
You must download and place the VQA-RAD files locally (see Section 2).

### Optional: Pretrained checkpoints
- Baseline saves:
  - `medvqa_baseline_best.pth`
  - `medvqa_baseline_last.pth`
- Advanced:
  - loads `medvqa_13new1.pth`
  - saves `medvqa_13new2.pth` (best Open Acc during Phase 4)

---

## 1) Environment setup

### Option A — pip (simple)
```bash
pip install -r requirements.txt
```

### Option B — create a clean environment (recommended)
```bash
conda create -n medvqa python=3.10 -y
conda activate medvqa
pip install -r requirements.txt
```

---

## 2) Data preparation (required)

This project reads paths from `src/config.py`, so your folder names should match exactly:

```text
data/
├── VQA_RAD Dataset Public.json
├── VQA_RAD Image Folder/
│   ├── synpic12345.jpg
│   ├── ...
└── glove.840B.300d.txt          # (optional, Baseline only)
```

### Where to download
VQA-RAD paper (and official data citation/OSF entry):

```text
https://www.nature.com/articles/sdata2018251
https://doi.org/10.17605/OSF.IO/89KPS
```

GloVe embeddings (only if you want baseline with pre-trained word vectors):

```text
https://nlp.stanford.edu/projects/glove/
```

---

## 3) Quickstart (run commands)

### 3.1 Train Baseline (ResNet50 + LSTM + optional GloVe)
```bash
python main_baseline.py
```

Expected outputs (saved in repo root):
- `medvqa_baseline_best.pth`
- `medvqa_baseline_last.pth`

Notes:
- If `data/glove.840B.300d.txt` does **not** exist, the baseline will automatically fall back to **random embeddings** (so it still runs).

---

### 3.2 Train / Fine-tune Advanced Specialist model (Phase 4)
```bash
python main_advanced.py
```

Expected outputs:
- `medvqa_13new2.pth` (saved whenever Open Accuracy reaches a new best)

How it works:
- The script will try to load **`medvqa_13new1.pth`** (warm-start checkpoint).
- Then it runs **Phase 4 “Specialist” training** and evaluates after each epoch (prints Total / Closed / Open).

✅ **To enable true open-only training**, set `only_open=True` inside `main_advanced.py` when building the training dataset.

---

## 4) Project structure

```text
.
├── main_baseline.py             # Baseline training entry
├── main_advanced.py             # Advanced “specialist” training entry
├── requirements.txt
├── assets/
│   └── performance_table.png    # (optional) results figure shown in README
└── src/
    ├── config.py                # paths + hyperparameters
    ├── dataset.py               # baseline dataset (uses phrase_type for train/test)
    ├── dataset_advanced.py       # seq2seq dataset + optional open-only filter
    ├── glove_utils.py           # load glove.840B.300d.txt
    ├── model_baseline.py        # ResNet50 + LSTM classifier
    ├── model_advanced.py        # ResNet50 spatial + Attention + seq2seq decoder
    ├── train_baseline.py        # baseline training + closed/open breakdown
    └── train_advanced.py        # Phase-4 specialist training + semantic-match eval
```

---

## 5) Method summary

### Baseline (classification)
- **Image encoder:** ResNet-50 (ImageNet pretrained)
- **Question encoder:** word embedding + LSTM
- **Answer head:** MLP classifier over answer vocabulary

### Advanced (generation)
- **Image encoder:** ResNet-50 *spatial* feature map (49 regions)
- **Question encoder:** tokenizer IDs → embedding → LSTM encoder
- **Fusion:** **Attention** over image regions conditioned on question/decoder state
- **Decoder:** LSTMCell autoregressively generates answer tokens

### Specialist Curriculum Learning (Phase 4)
To address the strong imbalance of Yes/No questions in Med-VQA datasets:
- Phase 4 fine-tunes on **open-ended questions only** to force descriptive reasoning.

---

## 6) Evaluation details (Closed vs Open)

A sample is treated as **Closed-ended** if the ground-truth answer is exactly:
- `yes` or `no`

Otherwise it is counted as **Open-ended**.

For the advanced model, evaluation includes:
- exact match / substring match
- optional **SBERT semantic similarity** (threshold 0.85) if `sentence-transformers` is installed

---

## 7) Repro tips (for stable results)

Deep learning results can vary slightly due to randomness. If you want deterministic runs, you can set seeds at the top of `main_baseline.py` and `main_advanced.py`:

```python
import random, numpy as np, torch
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## 8) Troubleshooting

**(1) `FileNotFoundError: VQA_RAD Dataset Public.json`**
- Ensure the file name and folder structure match Section 2.

**(2) GloVe missing**
- Baseline still runs (random embeddings), but results may differ from the reported GloVe baseline.

**(3) `ModuleNotFoundError: No module named 'src'`**
- Run scripts from the repository root (same level as `main_baseline.py`).

---

## 9) Citation

If you use this repo, please cite:

- Lau, J. J., et al. (2018). *VQA-RAD: A dataset for medical visual question answering*. Scientific Data.

```bibtex
@article{lau2018vqarad,
  title={A dataset of clinically generated visual questions and answers about radiology images},
  author={Lau, Jason J and Gayen, Soumya and Ben Abacha, Asma and Demner-Fushman, Dina},
  journal={Scientific Data},
  year={2018}
}
```

---

## Disclaimer
This project is for **research / educational purposes only** and is **not** a medical device.
Do not use it for clinical decision making.
