下面就是 **完整可用、格式正确、可以直接覆盖粘贴进 GitHub 的 `README.md` 全文**。
你只要：在 GitHub 的 `README.md` 编辑页面里 **全选 → 删除 → 粘贴下面全部内容 → Commit changes** 就行。

> ✅ 从第一行 `# 🏥 MedVQA-Curriculum` 开始复制，到最后一行结束（不用额外加任何东西）。

---

# 🏥 MedVQA-Curriculum

**Curriculum Learning for Medical Visual Question Answering (VQA-RAD)**
**Devil-to-Rehab Strategy · Anti-Leakage Split · Semantic Evaluation (SBERT Optional)**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

> This repository contains an end-to-end MedVQA research implementation (course/research project).
> It focuses on improving **open-ended** medical VQA performance by using a strict **anti-leakage split** and a two-phase curriculum training strategy: **Devil → Rehab**.

---

## Table of Contents

* [Overview](#overview)
* [Quickstart](#quickstart)
* [Key Contributions](#key-contributions)
* [Method Summary](#method-summary)
* [Results](#results)
* [Dataset Preparation](#dataset-preparation)
* [Installation](#installation)
* [How to Run](#how-to-run)
* [Outputs](#outputs)
* [Reproducibility](#reproducibility)
* [Project Structure](#project-structure)
* [Suggested .gitignore](#suggested-gitignore)
* [Notes & Limitations](#notes--limitations)
* [Acknowledgments](#acknowledgments)

---

## Overview

Medical Visual Question Answering (MedVQA) requires a model to answer natural-language clinical questions based on radiology images.

A common failure mode is **Yes/No bias**: models overfit to frequent short answers (“yes”, “no”) and under-learn image-grounded reasoning for open-ended questions.

This project proposes a curriculum learning approach that intentionally **removes easy Yes/No supervision** first, forcing the model to learn stronger visual reasoning, and then reintroduces Yes/No questions to recover overall balance.

---

## Quickstart

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Download VQA-RAD and place it under `data/` (see [Dataset Preparation](#dataset-preparation))

3. Run the Devil → Rehab curriculum pipeline

```bash
python run_strategy.py
```

---

## Key Contributions

### 1) Strict Anti-Leakage Train/Test Split

Instead of relying on potentially leaky splits, we enforce a **deterministic split** with a fixed seed (Seed = **42**) and ensure the **test subset remains unseen** throughout training.

This is implemented in `run_strategy.py` using a fixed RNG seed for reproducibility.

### 2) Devil-to-Rehab Curriculum Learning (Two-Phase Training)

* **Phase A — Devil Training (Open-only):**
  Train *only on open-ended questions* by filtering out samples whose answers are `yes/no`. This encourages image-grounded reasoning.
* **Phase B — Rehab Training (Mixed):**
  Reintroduce all question types with a very low learning rate to restore Yes/No performance while keeping improved open-ended ability.

### 3) Semantic Evaluation for Open-ended Answers (Optional SBERT)

Open-ended answers can be semantically correct even with different wording.
We provide an `EvalHelper` that supports:

* normalization + substring checks
* optional **Sentence-BERT** similarity matching (threshold = **0.85**) if `sentence-transformers` is installed.

---

## Method Summary

### Architecture (High-level)

* **Vision Encoder:** ImageNet-pretrained **ResNet** feature extractor
* **Question Encoder:** Tokenization via **BERT tokenizer (`bert-base-uncased`)**, then Embedding + LSTM
* **Answer Decoder:** Attention-based decoder with greedy generation (Seq2Seq)

> Note: The code uses the BERT tokenizer for robust tokenization, but the question encoder is an Embedding+LSTM (not a full BERT encoder fine-tuning). This keeps training manageable on limited compute.

---

## Results

Below are example results from our reported best run (Seed = 42, strict split).
(Exact values may vary slightly depending on hardware and environment.)

| Metric          | Baseline | Curriculum (Devil→Rehab) |       Gain |
| --------------- | -------: | -----------------------: | ---------: |
| Open Accuracy   |   31.18% |               **34.71%** | **+3.53%** |
| Closed Accuracy |   68.42% |               **71.58%** | **+3.16%** |
| Total Accuracy  |   50.83% |               **54.17%** | **+3.34%** |

---

## Dataset Preparation

This project uses the **VQA-RAD** dataset.
Due to copyright/licensing, **raw images are not included** in this repository.

### Official Download Links

* **OSF (official source):** [https://osf.io/89k6j/](https://osf.io/89k6j/)
* **Backup (HuggingFace):** [https://huggingface.co/datasets/flaviagiammarino/vqa-rad](https://huggingface.co/datasets/flaviagiammarino/vqa-rad)

### Expected Data Layout (Matches `src/config.py`)

By default, `src/config.py` expects the following names/paths:

```text
data/
├── VQA_RAD Dataset Public.json
└── VQA_RAD Image Folder/
    ├── (all image files...)
```

### Steps

1. Download from OSF and extract.
2. Put the JSON file into `data/`:

   * `VQA_RAD Dataset Public.json`
3. Put the image folder into `data/`:

   * `VQA_RAD Image Folder/`

> If you rename files/folders (e.g., `data/trainset.json` + `data/images/`), make sure you update `DATA_JSON_PATH` and `IMG_DIR_PATH` in `src/config.py`.

---

## Installation

### 1) Create environment (recommended)

```bash
python -m venv .venv
# Windows:
# .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### Recommended `requirements.txt`

(Keep this as a separate file in the repo root.)

```text
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
sentence-transformers>=2.2.2
pillow
numpy
tqdm
```

---

## How to Run

### A) Run the Curriculum Strategy (Main Entry)

This executes the full **Devil → Rehab** pipeline:

```bash
python run_strategy.py
```

What it does:

1. Loads VQA-RAD data and applies a strict **80/20 split** with Seed **42**.
2. Builds the **Devil subset** by filtering out `yes/no` answers from the training split.
3. **Phase A (Devil):** trains on open-only subset and saves `medvqa_specialist.pth` (best open performance during Phase A).
4. **Phase B (Rehab):** trains on the full training set with very low LR and saves `medvqa_ultimate_final.pth`.

### B) (Optional) Train Baseline

```bash
python main_baseline.py
```

This trains a simpler baseline model for comparison.

### C) (Optional) Other Experiment Entries

You may also find earlier experiment entrypoints:

* `main_advanced_1.py`
* `main_advanced_2.py`
* `main_advanced_3.py`

They represent intermediate development versions.

---

## Outputs

Training will typically generate model checkpoints (`.pth`) in the repo root, such as:

* `medvqa_specialist.pth` (Phase A best)
* `medvqa_ultimate_final.pth` (Final model after Phase B)

> **Recommendation:** Do not commit `.pth` to GitHub unless required. Add `*.pth` into `.gitignore`.

---

## Reproducibility

* Split seed is fixed (Seed = **42**) in the curriculum pipeline.
* For stable reproduction, run in a consistent Python/PyTorch environment.
* Semantic evaluation (SBERT) requires `sentence-transformers`. If not installed, evaluation falls back to strict/heuristic matching.

---

## Project Structure

Current typical structure:

```text
.
├── run_strategy.py
├── main_baseline.py
├── main_advanced_1.py
├── main_advanced_2.py
├── main_advanced_3.py
├── evaluate_real.py
├── data/
│   ├── VQA_RAD Dataset Public.json
│   └── VQA_RAD Image Folder/
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

To keep your repo clean and avoid uploading large/private files, create a `.gitignore` in the repo root:

```text
__pycache__/
*.pyc
.DS_Store
.ipynb_checkpoints/

# model checkpoints
*.pth

# dataset images (do not upload)
data/VQA_RAD Image Folder/
data/images/
```

---

## Notes & Limitations

* **Research/Education use only. Not for clinical deployment.**
* VQA-RAD is small; results may vary based on augmentation, fine-tuning, and evaluation criteria.
* Open-ended evaluation is sensitive to matching rules:

  * strict string match vs semantic similarity can produce different accuracy.

---

## Acknowledgments

* Dataset: **VQA-RAD** (download via OSF)
* Libraries: PyTorch, Torchvision, HuggingFace Transformers, Sentence-Transformers
* Thanks to the research community for MedVQA baselines and reproducible tooling.
