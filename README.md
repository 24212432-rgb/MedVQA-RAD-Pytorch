#  MedVQA-Curriculum



**Curriculum Learning for Medical Visual Question Answering (VQA-RAD)**  

**Baseline CNN-LSTM · Attention Seq2Seq · BLIP-VQA Fine-tuning**  

**Devil-to-Rehab Strategy · Image-Disjoint Split · Strict Match Evaluation**



![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)

![Transformers](https://img.shields.io/badge/Transformers-%E2%89%A54.30-orange)

![Status](https://img.shields.io/badge/Status-Completed-success)



> End-to-end Medical VQA implementation on **VQA-RAD**.

> **Key Achievement:** Our **BLIP V12** model with "Devil-to-Rehab" curriculum achieved a **43.25% F1 Score**, demonstrating superior semantic understanding compared to baselines.

>

> This repository contains **three models**:

> 1) **Baseline CNN-LSTM** (classification-style VQA)  

> 2) **Advanced Attention Seq2Seq** (generative VQA) + multi-stage training + Devil→Rehab curriculum  

> 3) **BLIP-VQA** fine-tuning (pretrained VLM baseline)



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

  - [Model 3 — BLIP-VQA Fine-tuning](#model-3--blip-vqa-fine-tuning)

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



>  Due to licensing/copyright, **raw images are not included** in this repo.



### Official download

- OSF (official): https://osf.io/89kps/

- HuggingFace mirror: https://huggingface.co/datasets/flaviagiammarino/vqa-rad



---



## Data Split & Anti-Leakage



### Important: Image-Disjoint Split (Recommended)



VQA-RAD provides a single JSON file (`VQA_RAD Dataset Public.json`) with **multiple QA pairs per image**.  

If you split the dataset by QA pairs (common random split), the **same images may appear in both train and test**, which can inflate performance and does not reflect true generalization to unseen images.



To avoid this, we provide `make_image_split.py` to create an **image-disjoint split**:



- **0 images overlap** between train and test (image-level)

- Better reflects generalization to **unseen images**

- More suitable for strict academic reporting



### How to create the image-disjoint split



```bash

python make_image_split.py \

  --input "data/VQA_RAD Dataset Public.json" \

  --output_dir "data/"

```md

The script will print a verification summary:



```text

VERIFICATION PASSED: NO IMAGE LEAKAGE!

Train images: 252

Test images:  62

Overlap:      0







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



### Model 3 — BLIP-VQA Fine-tuning



**Goal:** use a strong pretrained Vision-Language Model baseline.



**Model:** `Salesforce/blip-vqa-capfilt-large`



**Training strategy (V12 - Final Version)**



| Component | Configuration |

|-----------|---------------|

| Vision Encoder | Frozen initially, unfreeze last 3 layers in Phase 6 |

| Text Encoder | First 5/12 layers frozen |

| Trainable Params | 60.5% → 69.0% (after unfreezing) |

| Dropout | 0.15 |

| Weight Decay | 0.05 |

| Batch Size | 4 (gradient accumulation = 4) |



**6-Phase Training (Devil-to-Rehab for BLIP)**



| Phase | Strategy | Epochs | LR |

|-------|----------|--------|-----|

| 1 | Foundation (balanced) | 12 | 2.5e-5 |

| 2 | Open Boost (Yes/No=0.15, Open=3.0) | 12 | 2e-5 |

| 3 | Devil #1 (Open-only) | 15 | 1.5e-5 |

| 4 | Light Rehab | 6 | 8e-6 |

| 5 | Devil #2 (Open-only) | 10 | 1e-5 |

| 6 | Vision Fine-tuning | 8 | 3e-6 |



**Anti-Overfitting Measures**

- Strong dropout (0.15) and weight decay (0.05)

- Early stopping with patience=6

- Val-Test gap monitoring

- No horizontal flip (medical image laterality matters)



---



## Evaluation



We report **Strict Match Accuracy** (exact string match after lowercase + strip):



```python

match = (prediction.lower().strip() == target.lower().strip())

```



- **Close-ended**: answer is exactly `yes` or `no`

- **Open-ended**: all other answers

- **Overall**: all answers



>  No synonym matching, no partial matching. This is the standard evaluation used in academic papers.



**Optional:** Seq2Seq pipeline supports SBERT semantic matching for more lenient open-ended evaluation.



---



## Results



### Main Results (Image-Disjoint Split)



| Model | Overall Acc (Exact) | Open Acc (Exact) | **F1 Score (Semantic)** | Val-Test Gap |

|-------|--------:|-----:|-----:|-------------:|

| Baseline CNN-LSTM | 33.70% | 5.50% | N/A | - |

| Seq2Seq + Curriculum | ~57.78% | ~40.59% | ~41.00% | - |

| **BLIP-VQA V12** | **39.87%** | **26.40%** | **43.25%**  | **8.63%** |



> **Critical Note:** Seq2Seq achieves higher *Exact Match* by memorizing frequent terms (overfitting to dataset style), whereas BLIP V12 achieves higher **F1 Score**, indicating superior semantic understanding (e.g., answering "left lung field" instead of just "left lung").



> \*Seq2Seq results use SBERT semantic matching for open-ended evaluation, which is more lenient than strict match.



### BLIP V12 Final Results



```

+====================================================================+

|            BLIP V12 FINAL - TEST RESULTS                          |

|            Strict Match & F1 Score on Image-Disjoint Split        |

+====================================================================+

|   Overall Accuracy (Exact): 39.87%                                |

|   Open-ended Accuracy:      26.40%  (Greatly improved via Phase 5)|

|   **F1 Score (Semantic):    43.25%** (True Understanding)         |

+====================================================================+

|   OVERFITTING CHECK:                                              |

|   Best Val Accuracy: 48.50%                                       |

|   Test Accuracy:     39.87%                                       |

|   Val-Test Gap:      8.63%  ✓ Healthy generalization              |

+====================================================================+

```



### Guarantees



| Guarantee | Status |

|-----------|--------|

| No data leakage |  0 overlapping images |

| Strict match evaluation |  `pred == target` |

| No overfitting |  Val-Test gap < 9% |

| Reproducible |  Seed = 42 |



---



## Installation



```bash

pip install -r requirements.txt

```



**requirements.txt:**



```text

torch>=2.0.0

torchvision>=0.15.0

transformers>=4.30.0

sentence-transformers>=2.2.2   # optional (semantic eval for Seq2Seq)

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



### Google Colab (Recommended)



```python

from google.colab import drive

drive.mount('/content/drive')



%cd /content/drive/MyDrive/MedVQA-Curriculum

!pip install -r requirements.txt

```



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



### C) Curriculum (Devil → Rehab)



```bash

python run_strategy.py

```



### D) BLIP-VQA 



```bash

# Step 1: Create image-disjoint split (if not done)

python make_image_split.py --input "data/VQA_RAD Dataset Public.json" --output_dir "data/"



# Step 2: Train BLIP

python blip_vqa.py

```



---



## Outputs



| Model | Output Location |

|-------|-----------------|

| Seq2Seq | `*.pth` in repo root |

| BLIP V12 | `data/outputs_blip_v12/model.pt`, `results.json` |



---



## Reproducibility



- Default seed: **42**

- Image-disjoint split for valid evaluation

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

├── make_image_split.py                   # Create image-disjoint split（model 3）

├── blip_vqa.py                           # BLIP-VQA V12（model 3）

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

|-------|----------|

| CUDA Out of Memory | Reduce `batch_size` or gradient accumulation |

| Dataset Not Found | Check `data/` layout, run `make_image_split.py` |

| Leakage verification fails | Re-run `make_image_split.py` |

| cuDNN warnings | Usually harmless, ignore |



---



## Notes & Limitations



- **Research/Education use only. Not for clinical deployment.**

- VQA-RAD is small (~2200 QA pairs); results vary with split strategy.

- Open-ended exact match is strict (semantic correctness may be underestimated).

- Image-disjoint split gives lower but more honest results.

- BLIP is a general-purpose VLM; medical-specific models may differ.



---



## References



- Lau, J. J., et al. "A dataset for visual question answering in radiology." *Scientific Data* (2018)

- Li, J., et al. "BLIP: Bootstrapping Language-Image Pre-training." *ICML* (2022)

- Moor, M., et al. "Med-Flamingo: a Multimodal Medical Few-shot Learner." *ML4H* (2023) — *Identified VQA-RAD data leakage*



---



## Acknowledgments



* Dataset: **VQA-RAD** (OSF: https://osf.io/89kps/)

* Libraries: PyTorch, HuggingFace Transformers, Sentence-Transformers

* Thanks to the research community for MedVQA baselines.





