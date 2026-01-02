# 🏥 MedVQA-Curriculum

**Curriculum Learning for Medical Visual Question Answering (VQA-RAD)**  
**Baseline CNN-LSTM · Attention Seq2Seq · BLIP-VQA Fine-tuning**  
**Devil-to-Rehab Strategy · Deterministic Split (Seed=42) · Semantic Eval (SBERT Optional)**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Transformers](https://img.shields.io/badge/Transformers-%E2%89%A54.30-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

> End-to-end Medical VQA implementation on **VQA-RAD** (course/research project).  
> This repository contains **three models** and a curriculum strategy designed to improve **open-ended** MedVQA:
>
> 1) **Baseline CNN-LSTM** (classification-style VQA)  
> 2) **Advanced Attention Seq2Seq** (generative VQA) + multi-stage training + Devil→Rehab curriculum  
> 3) **BLIP-VQA** fine-tuning (pretrained VLM baseline)

---

## Table of Contents

- [Overview](#overview)
- [What This Repo Reproduces](#what-this-repo-reproduces)
- [Dataset](#dataset)
- [Methods](#methods)
  - [Model 1 — Baseline CNN-LSTM (Classification)](#model-1--baseline-cnn-lstm-classification)
  - [Model 2 — Attention Seq2Seq (Generation)](#model-2--attention-seq2seq-generation)
  - [Curriculum — Devil → Rehab](#curriculum--devil--rehab)
  - [Model 3 — BLIP-VQA Fine-tuning](#model-3--blip-vqa-fine-tuning)
- [Evaluation](#evaluation)
- [Results](#results)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [How to Run](#how-to-run)
  - [A) Baseline CNN-LSTM](#a-baseline-cnn-lstm)
  - [B) Advanced Seq2Seq (Step 1 → Step 3)](#b-advanced-seq2seq-step-1--step-3)
  - [C) Curriculum (Optional Step 4)](#c-curriculum-optional-step-4)
  - [D) BLIP-VQA Fine-tuning (Model 3)](#d-blip-vqa-fine-tuning-model-3)
- [Outputs](#outputs)
- [Reproducibility](#reproducibility)
- [Project Structure](#project-structure)
- [Suggested .gitignore](#suggested-gitignore)
- [Troubleshooting](#troubleshooting)
- [Notes & Limitations](#notes--limitations)
- [Acknowledgments](#acknowledgments)
- [References](#references)

---

## Overview

Medical Visual Question Answering (MedVQA) requires answering clinical questions from radiology images.

A common failure mode is **Yes/No bias**:
models overfit to frequent short answers (`yes`, `no`) and under-learn image-grounded reasoning needed for **open-ended** answers.

This repository provides:
- A baseline **classification** model (CNN-LSTM)
- A stronger **generative Seq2Seq** pipeline with multi-stage training
- A curriculum strategy (**Devil → Rehab**) to boost open-ended performance
- A pretrained VLM baseline (**BLIP-VQA**) fine-tuned on VQA-RAD

---

## What This Repo Reproduces

This repo supports reproduction and comparison of multiple MedVQA pipelines on VQA-RAD:

- **Baseline CNN-LSTM**: strong baseline, but limited open-ended generation
- **Attention Seq2Seq**: better open-ended answers via attention-based decoding
- **Devil→Rehab curriculum**: reduce Yes/No bias, improve open-ended reasoning
- **BLIP-VQA fine-tuning**: modern pretrained VLM baseline for comparison

---

## Dataset

This project uses the **VQA-RAD** dataset.

> ⚠️ Due to licensing/copyright, **raw images are not included** in this repo.

### Official download
- OSF (official): https://osf.io/89k6j/
- HuggingFace mirror: https://huggingface.co/datasets/flaviagiammarino/vqa-rad

---

## Methods

## Model 1 — Baseline CNN-LSTM (Classification)

**Goal:** simple baseline for MedVQA.

**High-level design**
- **Vision:** ImageNet-pretrained **ResNet-50** encoder
- **Text:** **LSTM** question encoder (optionally initialized with **GloVe**)
- **Fusion:** image + question features
- **Output:** classifier over a fixed answer vocabulary

**Pros:** stable, fast, easy to train  
**Cons:** open-ended answers are restricted by the answer vocabulary.

---

## Model 2 — Attention Seq2Seq (Generation)

**Goal:** generative VQA for open-ended answers.

**High-level design**
- **Vision:** ResNet visual feature map (spatial grid)
- **Tokenizer:** **BERT tokenizer** (`bert-base-uncased`) for robust tokenization  
  *(Note: tokenizer is used for tokenization; the question encoder is still LSTM-based in this project.)*
- **Question Encoder:** Embedding + LSTM encoder
- **Answer Decoder:** Attention-based LSTM decoder + greedy decoding

### Multi-stage training (Step 1 → Step 3)
- **Step 1 (`main_advanced_1.py`)**: stability-first foundation training  
- **Step 2 (`main_advanced_2.py`)**: open-ended boost by penalizing `yes/no` dominance  
- **Step 3 (`main_advanced_3.py`)**: unfreeze CNN backbone and fine-tune with ultra-low LR  

---

## Curriculum — Devil → Rehab

**Motivation:** improve open-ended reasoning while reducing Yes/No bias.

**Two phases**
- **Devil (Open-only):** train only on open-ended samples (filter out exact answers `yes`/`no`)
- **Rehab (Mixed):** reintroduce all samples with a very low LR to recover closed-ended performance

Run via: `run_strategy.py`

---

## Model 3 — BLIP-VQA Fine-tuning

**Goal:** use a strong pretrained Vision-Language Model baseline.

**Model**
- `Salesforce/blip-vqa-capfilt-large`

**Training strategy (as in `blip_vqa_train_v6.py`)**
- Freeze **vision encoder**
- Freeze first **8** layers of the text encoder
- Fine-tune remaining layers with small LR
- Generate answers using `model.generate`

---

## Evaluation

We report **Exact-Match Accuracy** (string match after normalization) on:
- **Close-ended**: answer is exactly `yes` or `no`
- **Open-ended**: all other answers
- **Overall**

### Optional semantic evaluation (SBERT)
Open-ended answers can be correct with different wording.  
Seq2Seq pipeline supports optional SBERT similarity matching if `sentence-transformers` is installed.

---

## Results

**Split:** deterministic split (Seed = 42).  
**Note:** open-ended exact match is strict (semantic correctness may be underestimated).

| Model / Stage | Overall | Closed | Open |
|---|---:|---:|---:|
| Baseline CNN-LSTM | 33.70% | 56.18% | 5.50% |
| Seq2Seq (Step 1) | 46.67% | 67.89% | 22.94% |
| Seq2Seq (Step 3) | 54.72% | 70.00% | 33.65% |
| Curriculum (Devil→Rehab, Step 4) | **57.78%** | **73.16%** | **40.59%** |
| BLIP-VQA Fine-tune (V6) | 45.01% | 66.93% | 17.50% |

**BLIP training setup (V6):**
- Epochs: 15  
- Batch: 4 with grad-accum=4 (effective 16)  
- LR: 2e-5, Weight Decay: 0.1, Dropout: 0.2  
- Early stopping: patience=4  
- GPU: Tesla T4 (Colab)

---

## Installation

```bash
pip install -r requirements.txt
```

Recommended `requirements.txt`:

```text
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
sentence-transformers>=2.2.2   # optional (semantic eval)
pillow
numpy
tqdm
```

---

## Data Preparation

Expected layout:

```text
data/
├── VQA_RAD Dataset Public.json           # optional (full dataset)
├── trainset.json                         # used by BLIP script
├── testset.json                          # used by BLIP script
├── VQA_RAD Image Folder/
│   ├── (all image files...)
└── glove.840B.300d.txt                   # optional (baseline embedding init)
```

If you rename files/folders, update paths in `src/config.py` or in the training scripts.

---

## How to Run

### Option A: Google Colab (Recommended)

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/MedVQA-Curriculum
!pip install -r requirements.txt
!nvidia-smi
```

---

## A) Baseline CNN-LSTM

```bash
python main_baseline.py
```

---

## B) Advanced Seq2Seq (Step 1 → Step 3)

Run in order:

```bash
python main_advanced_1.py
python main_advanced_2.py
python main_advanced_3.py
```

Checkpoint alignment (if needed):

```bash
cp medvqa_13best.pth medvqa_advanced_bert_best.pth
cp medvqa_advanced_bert_final_boost.pth medvqa_final_boost.pth
```

---

## C) Curriculum (Optional Step 4)

```bash
python run_strategy.py
```

Outputs (typical):
- `medvqa_specialist.pth` (best Devil-phase checkpoint)
- `medvqa_ultimate_final.pth` (final Rehab checkpoint)

---

## D) BLIP-VQA Fine-tuning (Model 3)

```bash
python blip_vqa_train_v6.py
```

This script automatically searches for:
- `data/trainset.json`
- `data/testset.json`
- `data/VQA_RAD Image Folder/`

Outputs are saved to:
- `data/outputs_blip_vqa_v6/`

---

## Outputs

Typical outputs:
- **Seq2Seq / Curriculum:** `*.pth` checkpoints in repo root
- **BLIP:** `data/outputs_blip_vqa_v6/best_model.pth` + `final_results.json` + `training_history.json`

---

## Reproducibility

- Default seed: **42**
- Deterministic split & reproducible runs (within GPU nondeterminism limits)
- Keep the same train/test JSON files for fair comparisons across models

---

## Project Structure

```text
.
├── main_baseline.py                      # Baseline (Model 1)
├── main_advanced_1.py                    # Advanced Seq2Seq (Model 2)
├── main_advanced_2.py
├── main_advanced_3.py
├── run_strategy.py
├── evaluate_real.py
├── blip_vqa_train_v6.py                 # BLIP-VQA fine-tuning (Model 3)
├── data/
│   ├── trainset.json
│   ├── testset.json
│   ├── VQA_RAD Dataset Public.json
│   ├── VQA_RAD Image Folder/
│   └── glove.840B.300d.txt
└── src/
    ├── config.py
    ├── dataset.py
    ├── dataset_advanced.py
    ├── model_baseline.py
    ├── model_advanced.py
    ├── glove_utils.py
    ├── train_baseline.py
    ├── train_advanced_1.py
    ├── train_advanced_2.py
    ├── train_advanced_3.py
    └── train_advanced_4.py
```

---

## Suggested .gitignore

```text
__pycache__/
*.pyc
.DS_Store
.ipynb_checkpoints/

# checkpoints
*.pth

# dataset images (do not upload)
data/VQA_RAD Image Folder/
data/images/
```

---

## Troubleshooting

- **CUDA Out of Memory**
  - Reduce `BATCH_SIZE` (Seq2Seq) or reduce effective batch (BLIP).
- **Dataset Not Found**
  - Check `data/` layout and printed “DATA PATH VALIDATION”.
- **TensorFlow / cuDNN registration warnings in Colab**
  - Common environment warnings; training is usually unaffected.

---

## Notes & Limitations

- **Research/Education use only. Not for clinical deployment.**
- VQA-RAD is small; results vary with augmentation and split strategy.
- Open-ended evaluation is strict (exact match may underestimate correctness).
- Horizontal flip augmentation may not be ideal if laterality matters in radiology.




## References

- VQA-RAD dataset (Lau et al.)
- BLIP: Bootstrapping Language-Image Pre-training (Salesforce)

---
## Acknowledgments

* Dataset: **VQA-RAD** (download via OSF)
* Libraries: PyTorch, Torchvision, HuggingFace Transformers, Sentence-Transformers
* Thanks to the research community for MedVQA baselines and reproducible tooling.
---
