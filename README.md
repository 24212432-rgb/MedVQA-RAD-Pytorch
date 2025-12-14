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

## How to Run

### Option A: Google Colab Workflow (Recommended)

We recommend **Google Colab** for training (GPU).  
To reproduce the full experiment pipeline, run the scripts **in the exact order** below:

1) `main_advanced_1.py`  
2) `main_advanced_2.py`  
3) `main_advanced_3.py`  
4) *(Optional)* `run_strategy.py` (Reinforcement + Rehab Curriculum)

---

#### 0) Colab Setup (GPU + Drive)
In Colab:
- **Runtime → Change runtime type → GPU**

Then run:

```python
from google.colab import drive
drive.mount('/content/drive')

# Go to your project folder in Drive
%cd /content/drive/MyDrive/MedVQA-Curriculum

# Install dependencies
!pip install -r requirements.txt
```

✅ Before training, confirm your dataset is placed correctly (see **Dataset Preparation**).  
By default, the code reads paths from `src/config.py`.

---

### Execution Pipeline (Run in Order)

#### Step 1 — `main_advanced_1.py` (Foundation Training / Stability First)
```bash
!python main_advanced_1.py
```

**What this step does**
- Trains the advanced Seq2Seq VQA model with a **stability-first** strategy.
- Builds a strong initial checkpoint by learning a balanced mapping between images, questions, and answers.

**Why it matters**
- This is the **base stage**. You should not apply aggressive curriculum or boosting before the model learns a stable foundation.

**Expected output**
- A best checkpoint saved during training (typical file name): `medvqa_13best.pth`

✅ **Checkpoint alignment (required for Step 2)**  
Step 2 expects a specific filename. Create it like this:
```bash
!cp medvqa_13best.pth medvqa_advanced_bert_best.pth
```

---

#### Step 2 — `main_advanced_2.py` (Open-ended Boost / Penalize Yes-No)
```bash
!python main_advanced_2.py
```

**What this step does**
- Loads the foundation model (`medvqa_advanced_bert_best.pth`).
- Applies a stronger training pressure toward **open-ended reasoning** by **down-weighting “yes/no” tokens** in the loss.
- This reduces the tendency to answer everything as “yes/no” and encourages richer answers.

**Why it matters**
- Open-ended medical answers are harder and require more **image-grounded reasoning**.
- This stage targets the common **Yes/No bias** problem.

**Expected output**
- A boosted checkpoint (typical file name): `medvqa_advanced_bert_final_boost.pth`

✅ **Checkpoint alignment (required for Step 3)**  
Step 3 expects `medvqa_final_boost.pth`. Create it like this:
```bash
!cp medvqa_advanced_bert_final_boost.pth medvqa_final_boost.pth
```

---

#### Step 3 — `main_advanced_3.py` (Unfreeze Vision / CNN Fine-tuning)
```bash
!python main_advanced_3.py
```

**What this step does**
- Loads the boosted checkpoint (`medvqa_final_boost.pth`).
- **Unfreezes the ResNet visual backbone** and fine-tunes it with an **ultra-low learning rate**.
- This adapts visual features from generic ImageNet patterns to radiology-specific cues.

**Why it matters**
- Fine-tuning the CNN too early can overfit (VQA-RAD is small).
- Doing it after language/decoder stabilization often yields better generalization.

**Expected output**
- A stronger final checkpoint (typical file name): `medvqa_ultimate.pth`

---

#### Step 4 (Optional) — `run_strategy.py` (Devil → Rehab Curriculum Learning) 🏆
```bash
!python run_strategy.py
```

**What this step does**
This script executes the **Devil-to-Rehab curriculum**:

- **Phase A (Devil / Open-only):**  
  Filters out all easy Yes/No samples and trains only on open-ended questions.  
  Goal: maximize open-ended reasoning ability.

- **Phase B (Rehab / Mixed):**  
  Reintroduces the full training set with a **very low LR** to recover closed-ended performance without destroying open-ended gains.

**Why it matters**
- This is your **research contribution**: a curriculum strategy designed to reduce language bias and improve open-ended accuracy.

**Expected outputs**
- `medvqa_specialist.pth` (best open-focused checkpoint from Devil phase)
- `medvqa_ultimate_final.pth` (final model after Rehab phase)

---

### One-Cell Colab Run (Optional Convenience)
If you prefer a single cell to run the whole pipeline:

```bash
!python main_advanced_1.py
!cp medvqa_13best.pth medvqa_advanced_bert_best.pth

!python main_advanced_2.py
!cp medvqa_advanced_bert_final_boost.pth medvqa_final_boost.pth

!python main_advanced_3.py

# Optional curriculum reinforcement
!python run_strategy.py
```


<details>
<summary><b>Troubleshooting (Common Issues)</b></summary>

- **CUDA Out of Memory**
  - Reduce `BATCH_SIZE` in `src/config.py` (e.g., 8 → 4).

- **Dataset Not Found**
  - Ensure `data/VQA_RAD Dataset Public.json` and `data/VQA_RAD Image Folder/` exist.
  - Confirm `src/config.py` points to the correct paths.

- **Checkpoint File Not Found**
  - Use the `cp` commands shown above to align filenames between stages.

</details>


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

### Option A: Google Colab Workflow (Recommended)

We recommend **Google Colab** for training (GPU).  
To reproduce the full experiment pipeline, run the scripts **in the exact order** below:

1) `main_advanced_1.py`  
2) `main_advanced_2.py`  
3) `main_advanced_3.py`  
4) *(Optional)* `run_strategy.py` (Reinforcement + Rehab Curriculum)

---

#### 0) Colab Setup (GPU + Drive)
In Colab:
- **Runtime → Change runtime type → GPU**

Then run:

```python
from google.colab import drive
drive.mount('/content/drive')

# Go to your project folder in Drive
%cd /content/drive/MyDrive/MedVQA-Curriculum

# Install dependencies
!pip install -r requirements.txt
```

✅ Before training, confirm your dataset is placed correctly (see **Dataset Preparation**).  
By default, the code reads paths from `src/config.py`.

---

### Execution Pipeline (Run in Order)

#### Step 1 — `main_advanced_1.py` (Foundation Training / Stability First)
```bash
!python main_advanced_1.py
```

**What this step does**
- Trains the advanced Seq2Seq VQA model with a **stability-first** strategy.
- Builds a strong initial checkpoint by learning a balanced mapping between images, questions, and answers.

**Why it matters**
- This is the **base stage**. You should not apply aggressive curriculum or boosting before the model learns a stable foundation.

**Expected output**
- A best checkpoint saved during training (typical file name): `medvqa_13best.pth`

✅ **Checkpoint alignment (required for Step 2)**  
Step 2 expects a specific filename. Create it like this:
```bash
!cp medvqa_13best.pth medvqa_advanced_bert_best.pth
```

---

#### Step 2 — `main_advanced_2.py` (Open-ended Boost / Penalize Yes-No)
```bash
!python main_advanced_2.py
```

**What this step does**
- Loads the foundation model (`medvqa_advanced_bert_best.pth`).
- Applies a stronger training pressure toward **open-ended reasoning** by **down-weighting “yes/no” tokens** in the loss.
- This reduces the tendency to answer everything as “yes/no” and encourages richer answers.

**Why it matters**
- Open-ended medical answers are harder and require more **image-grounded reasoning**.
- This stage targets the common **Yes/No bias** problem.

**Expected output**
- A boosted checkpoint (typical file name): `medvqa_advanced_bert_final_boost.pth`

✅ **Checkpoint alignment (required for Step 3)**  
Step 3 expects `medvqa_final_boost.pth`. Create it like this:
```bash
!cp medvqa_advanced_bert_final_boost.pth medvqa_final_boost.pth
```

---

#### Step 3 — `main_advanced_3.py` (Unfreeze Vision / CNN Fine-tuning)
```bash
!python main_advanced_3.py
```

**What this step does**
- Loads the boosted checkpoint (`medvqa_final_boost.pth`).
- **Unfreezes the ResNet visual backbone** and fine-tunes it with an **ultra-low learning rate**.
- This adapts visual features from generic ImageNet patterns to radiology-specific cues.

**Why it matters**
- Fine-tuning the CNN too early can overfit (VQA-RAD is small).
- Doing it after language/decoder stabilization often yields better generalization.

**Expected output**
- A stronger final checkpoint (typical file name): `medvqa_ultimate.pth`

---

#### Step 4 (Optional) — `run_strategy.py` (Devil → Rehab Curriculum Learning) 🏆
```bash
!python run_strategy.py
```

**What this step does**
This script executes the **Devil-to-Rehab curriculum**:

- **Phase A (Devil / Open-only):**  
  Filters out all easy Yes/No samples and trains only on open-ended questions.  
  Goal: maximize open-ended reasoning ability.

- **Phase B (Rehab / Mixed):**  
  Reintroduces the full training set with a **very low LR** to recover closed-ended performance without destroying open-ended gains.

**Why it matters**
- This is your **research contribution**: a curriculum strategy designed to reduce language bias and improve open-ended accuracy.

**Expected outputs**
- `medvqa_specialist.pth` (best open-focused checkpoint from Devil phase)
- `medvqa_ultimate_final.pth` (final model after Rehab phase)

---

### One-Cell Colab Run (Optional Convenience)
If you prefer a single cell to run the whole pipeline:

```bash
!python main_advanced_1.py
!cp medvqa_13best.pth medvqa_advanced_bert_best.pth

!python main_advanced_2.py
!cp medvqa_advanced_bert_final_boost.pth medvqa_final_boost.pth

!python main_advanced_3.py

# Optional curriculum reinforcement
!python run_strategy.py
```

---

### Should I explain every step in the README?
**Yes — recommended (especially for a university assignment / interview).**

Simply listing commands can look like a copied workflow.  
Explaining **what each step does and why** shows:
- you understand the experimental design,
- your work is reproducible,
- your contribution (Step 4) is clearly justified.

A good practice is:
- Keep **short explanations** in the README (like above).
- Put deeper details in your final report or GitHub Wiki if needed.

---

<details>
<summary><b>Troubleshooting (Common Issues)</b></summary>

- **CUDA Out of Memory**
  - Reduce `BATCH_SIZE` in `src/config.py` (e.g., 8 → 4).

- **Dataset Not Found**
  - Ensure `data/VQA_RAD Dataset Public.json` and `data/VQA_RAD Image Folder/` exist.
  - Confirm `src/config.py` points to the correct paths.

- **Checkpoint File Not Found**
  - Use the `cp` commands shown above to align filenames between stages.

</details>


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
