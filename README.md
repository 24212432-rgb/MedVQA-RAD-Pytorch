# MedVQA-Curriculum

**Curriculum Learning for Medical Visual Question Answering (VQA-RAD)**  
**Baseline CNN-LSTM · Attention Seq2Seq · LLaVA-VQA Fine-tuning**  
**Devil-to-Rehab Strategy · Image-Disjoint Split · Strict Match Evaluation**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Transformers](https://img.shields.io/badge/Transformers-%E2%89%A54.30-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

> End-to-end Medical VQA implementation on **VQA-RAD**.
> **Key Achievement:** Our **LLaVA-VQA** fine-tuning pipeline (image-disjoint) achieved **45.88% Strict Accuracy** and **48.82% Token-F1** on the **image-disjoint** test set under a strict anti-leakage protocol.
>
> This repository contains **three model families**:
> 1) **Baseline CNN-LSTM** (classification-style VQA)  
> 2) **Attention Seq2Seq** (generative VQA) + multi-stage training + Devil→Rehab curriculum  
> 3) **LLaVA-VQA** fine-tuning (pretrained VLM baseline, LoRA)

---

## Table of Contents

- [Overview](#overview)
- [What This Repo Reproduces](#what-this-repo-reproduces)
- [Dataset](#dataset)
- [Data Split & Anti-Leakage](#data-split--anti-leakage)
- [Methods](#methods)
  - [Model 1 — Baseline CNN-LSTM (Classification)](#model-1--baseline-cnn-lstm-classification)
  - [Model 2 — Attention Seq2Seq (Generation)](#model-2--attention-seq2seq-generation)
  - [Curriculum — Devil → Rehab](#curriculum--devil--rehab)
  - [Model 3 — LLaVA-VQA Fine-tuning](#model-3--llava-vqa-fine-tuning)
- [Evaluation](#evaluation)
- [Results](#results)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [How to Run](#how-to-run)
- [Outputs](#outputs)
- [Reproducibility](#reproducibility)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Notes & Limitations](#notes--limitations)
- [References](#references)
- [Acknowledgments](#acknowledgments)

---

## Overview

Medical Visual Question Answering (MedVQA) requires answering clinical questions from radiology images.

A common failure mode is **Yes/No bias**: models overfit to frequent short answers (`yes`, `no`) and under-learn image-grounded reasoning needed for **open-ended** answers.

This repository provides:
- A baseline **classification** model (CNN-LSTM)
- A stronger **generative Seq2Seq** pipeline with multi-stage training
- A curriculum strategy (**Devil → Rehab**) to boost open-ended performance
- A pretrained VLM baseline (**LLaVA-VQA**) fine-tuned on VQA-RAD

---

## What This Repo Reproduces

This repo supports reproduction and comparison of multiple MedVQA pipelines on VQA-RAD:

- **Baseline CNN-LSTM**: stable baseline, but limited open-ended generation
- **Attention Seq2Seq**: better open-ended answers via attention-based decoding
- **Devil→Rehab curriculum**: reduce Yes/No bias, improve open-ended reasoning
- **LLaVA-VQA fine-tuning**: modern pretrained VLM baseline for comparison (LoRA)

---

## Dataset

This project uses the **VQA-RAD** dataset.

> Due to licensing/copyright, **raw images are not included** in this repo.

### Official download
- OSF (official): https://osf.io/89kps/
- HuggingFace mirror: https://huggingface.co/datasets/flaviagiammarino/vqa-rad

---

## Data Split & Anti-Leakage

### Important: Image-Disjoint Split (Recommended)

The VQA-RAD release provides a single JSON file (`VQA_RAD Dataset Public.json`) containing **multiple QA pairs per image**.  
If you split the dataset by QA pairs (question-level split), the **same images can appear in both train and test**, which may inflate performance and does not reflect true generalization to unseen images.

To avoid this, we provide `make_image_split.py` to create an **image-disjoint split**:

- **0 images overlap** between train and test (image-level)
- Better reflects generalization to **unseen images**
- More suitable for strict academic reporting

### Create the image-disjoint split

```bash
python make_image_split.py \
  --input "data/VQA_RAD Dataset Public.json" \
  --output_dir "data/"
```

This generates:
- `trainset_image_disjoint.json` (1799 samples, 252 images)
- `testset_image_disjoint.json` (449 samples, 62 images)

The script prints a verification summary such as:

```text
VERIFICATION PASSED: NO IMAGE LEAKAGE!
Train images: 252
Test images:  62
Overlap:      0
```

### Comparison of splits

| Split Type | Train Images | Test Images | Overlap | Validity |
|------------|--------------|-------------|---------|----------|
| **Image-Disjoint** | 252 | 62 | **0 (0%)** | ✅ Valid |

> **Note**: VQA-RAD contains multiple QA pairs per image. Any question-level (QA-pair) split may place the same image into both train and test, causing image-level leakage. Our split avoids this by design.

---

## Methods

### Model 1 — Baseline CNN-LSTM (Classification)

**Goal:** simple baseline for MedVQA.

**High-level design**
- **Vision:** ImageNet-pretrained **ResNet-50** encoder
- **Text:** **LSTM** question encoder (optionally initialized with **GloVe**)
- **Fusion:** image + question features
- **Output:** classifier over a fixed answer vocabulary

**Pros:** stable, fast, easy to train  
**Cons:** open-ended answers are restricted by the answer vocabulary.

---

### Model 2 — Attention Seq2Seq (Generation)

**Goal:** generative VQA for open-ended answers.

**High-level design**
- **Vision:** ResNet visual feature map (spatial grid)
- **Tokenizer:** **BERT tokenizer** (`bert-base-uncased`) for robust tokenization
- **Question Encoder:** Embedding + LSTM encoder
- **Answer Decoder:** Attention-based LSTM decoder + greedy decoding

**Multi-stage training (Step 1 → Step 3)**
- **Step 1**: stability-first foundation training  
- **Step 2**: open-ended boost by penalizing `yes/no` dominance  
- **Step 3**: unfreeze CNN backbone and fine-tune with ultra-low LR  

---

### Curriculum — Devil → Rehab

**Motivation:** improve open-ended reasoning while reducing Yes/No bias.

**Two phases**
- **Devil (Open-only):** train only on open-ended samples (filter out exact answers `yes`/`no`)
- **Rehab (Mixed):** reintroduce all samples with a very low LR to recover closed-ended performance

---

### Model 3 — LLaVA-VQA Fine-tuning

**Goal:** use a strong pretrained Vision-Language Model baseline and fine-tune on VQA-RAD.

**Backbone:** `llava-hf/llava-1.5-7b-hf`

**Training strategy (LLaVA-VQA - Final)**

| Component | Configuration |
|---|---|
| Fine-tuning | LoRA on LLaMA attention + MLP blocks |
| Optional | Unfreeze `mm_projector` for better image-text alignment |
| Mixed QA | Open + Closed (with weighted sampling to balance) |
| Prompting | Closed: force `yes/no`; Open: 1–3 words only |
| Anti-leakage | Strict image-disjoint split |
| Decoding | Open: **Top-K constrained** (trie) + **fallback free** for unseen answers |
| Answer normalization | Strip articles `the/a/an` for training + evaluation consistency |

#### Why “Constrained + Fallback”?

- **Constrained decode** improves strict metrics by forcing open-ended answers to be from a learned candidate set (Top-K from training open answers).
- **Fallback free decode** keeps the model **usable in real scenarios**: when the constrained result is low-confidence, we allow free generation so it can answer **unseen** terms.

This matches your goal: **higher metrics + real usability**.

---

## Evaluation

We report **Strict Match Accuracy** (exact string match after lowercase + strip):

```python
match = (prediction.lower().strip() == target.lower().strip())
```

In this repo, we additionally report **Token-F1** (soft score):
- Token overlap F1 between prediction and target (after normalization)
- Rewards partial keyword matches (e.g., “ascending aorta” vs “aorta”)

**Split by answer type**
- **Close-ended**: answer is exactly `yes` or `no`
- **Open-ended**: all other answers
- **Overall**: all answers

---

## Results

### LLaVA-VQA (Image-Disjoint Split)

```
+===========================================================+
|         LLaVA-VQA - TEST RESULTS (IMAGE-DISJOINT)         |
+===========================================================+
|   Overall Accuracy (Strict):  45.88%                      |
|   Overall Token-F1 (Soft):    48.82%                      |
|   Close-ended Accuracy:       65.59%                      |
|   Open-ended Accuracy:        21.78%                      |
|   Open-ended Token-F1:        28.31%                      |
+===========================================================+
```

> Note: VQA-RAD is small (~2200 QA pairs). Open-ended strict match is very challenging under an image-disjoint split. Soft metrics (Token-F1) better reflect partial clinical correctness.

---

## Installation

```bash
pip install -r requirements.txt
```

**requirements.txt (suggested):**

```text
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
peft>=0.7.0
accelerate>=0.21.0
pillow
numpy
tqdm
scikit-learn
```

---

## Data Preparation

Expected layout:

```text
data/
├── VQA_RAD Dataset Public.json           # original full dataset
├── trainset_image_disjoint.json          # generated by make_image_split.py
├── testset_image_disjoint.json           # generated by make_image_split.py
├── VQA_RAD Image Folder/
│   └── (all image files...)
└── glove.840B.300d.txt                   # optional (baseline embedding)
```

### Step 1: Create image-disjoint split

```bash
python make_image_split.py \
  --input "data/VQA_RAD Dataset Public.json" \
  --output_dir "data/"
```

---

## How to Run

### A) Baseline CNN-LSTM

```bash
python main_baseline.py
```

### B) Advanced Seq2Seq

```bash
python main_advanced_1.py
python main_advanced_2.py
python main_advanced_3.py
```

### C) Curriculum (Devil → Rehab) for Seq2Seq

```bash
python run_strategy.py
```

### D) LLaVA-VQA Fine-tuning (Final)

> You said you renamed the script to **`llava_vqa.py`**. Use this:

```bash
# 1) Create image-disjoint split (if not done)
python make_image_split.py --input "data/VQA_RAD Dataset Public.json" --output_dir "data/"

# 2) Train + validate + test
python llava_vqa.py
```

**If you are on Google Colab**
```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/MedVQA-Curriculum
!pip install -r requirements.txt
!python llava_vqa.py
```

---

## Outputs

| Model | Output Location |
|---|---|
| Seq2Seq | `*.pth` in repo root |
| LLaVA-VQA | `saved_models/llava_a100_v_final_mix/` (configurable in script) |
| Results | `saved_models/.../final_results.json` |

---

## Reproducibility

- Default seed: **42**
- Image-disjoint split for valid evaluation
- Strict leakage check (no overlapping images)
- Deterministic operations where possible
- Keep the same JSON files for fair comparison

---

## Project Structure

```text
.
├── main_baseline.py                      # Baseline CNN-LSTM（model 1）
├── main_advanced_1.py                    # Seq2Seq Step 1（model 2）
├── main_advanced_2.py                    # Seq2Seq Step 2（model 2）
├── main_advanced_3.py                    # Seq2Seq Step 3（model 2）
├── run_strategy.py                       # Devil→Rehab curriculum（model 2）
├── make_image_split.py                   # Create image-disjoint split
├── llava_vqa.py                          # LLaVA-VQA fine-tuning (final)
├── data/
│   ├── VQA_RAD Dataset Public.json
│   ├── trainset_image_disjoint.json
│   ├── testset_image_disjoint.json
│   └── VQA_RAD Image Folder/
└── src/
    ├── config.py
    ├── dataset.py
    ├── model_baseline.py
    ├── model_advanced.py
    └── ...
```

---

## Troubleshooting

| Issue | Solution |
|---|---|
| CUDA Out of Memory | Reduce `batch_size` or gradient accumulation |
| Dataset Not Found | Check `data/` layout, run `make_image_split.py` |
| Leakage verification fails | Re-run `make_image_split.py` |
| cuDNN / oneDNN warnings | Usually harmless, ignore |
| Training very slow | Reduce beams in validation/test, reduce num_workers, disable constrained decode during training eval, or lower `OPEN_CONSTRAINED_NUM_BEAMS_*` |

---

## Notes & Limitations

- **Research/Education use only. Not for clinical deployment.**
- VQA-RAD is small; results can vary with random seed.
- Open-ended strict match may underestimate clinically correct synonyms/paraphrases.
- Image-disjoint split gives lower but more honest results.
- For real-world deployment, consider medical-specific VLMs and synonym-aware evaluation.

---

## References

- Lau, J. J., et al. "A dataset for visual question answering in radiology." *Scientific Data* (2018)
- Li, J., et al. "BLIP: Bootstrapping Language-Image Pre-training." *ICML* (2022)
- LLaVA: Large Language and Vision Assistant (original paper / ecosystem)

---

## Acknowledgments

* Dataset: **VQA-RAD** (OSF: https://osf.io/89kps/)
* Libraries: PyTorch, HuggingFace Transformers, PEFT
* Thanks to the research community for MedVQA baselines.
