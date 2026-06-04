# DR-MOEAD-Fusion

# Tri-Scale Retinal Feature Fusion with Multi-Objective Selection for Efficient Diabetic Retinopathy Grading

## Overview

This repository contains the implementation of the proposed diabetic retinopathy (DR) grading framework based on:

* Custom CNN feature extraction
* DenseNet121 feature extraction
* Vision Transformer (ViT) feature extraction
* Deep feature fusion
* MOEA/D-based multi-objective feature selection
* LightGBM classification
* Explainability analysis using Grad-CAM, Grad-CAM++, Integrated Gradients, and Insertion–Deletion evaluation

The framework was evaluated on five publicly available diabetic retinopathy datasets:

1. APTOS 2019
2. DDR
3. Messidor-2
4. Diabetic Balanced Dataset
5. DR Dataset

---

## Proposed Pipeline

Fundus Image
→ CLAHE Preprocessing
→ Custom CNN Feature Extraction
→ DenseNet121 Feature Extraction
→ ViT Feature Extraction
→ Feature Fusion
→ SMOTE (Training Set Only)
→ MOEA/D Feature Selection
→ LightGBM Classification
→ Explainability Analysis



## Experimental Protocol

To prevent data leakage, the following protocol was strictly followed:

1. Stratified train-validation-test split.
2. CLAHE preprocessing applied independently.
3. Custom CNN trained only on the training set.
4. DenseNet121 and ViT used as pretrained feature extractors.
5. Feature extraction performed separately for train, validation, and test sets.
6. Feature fusion applied after extraction.
7. SMOTE applied only on training features.
8. MOEA/D feature selection performed using training and validation data.
9. Final LightGBM classifier trained using selected features.
10. Evaluation performed once on the untouched test set.

---

## Ablation Study

The following configurations were evaluated:

* CNN only
* DenseNet121 only
* ViT only
* CNN + DenseNet121
* CNN + ViT
* DenseNet121 + ViT
* Full Fusion (CNN + DenseNet121 + ViT)
* Full Fusion + PCA
* Full Fusion + LASSO
* Full Fusion + NSGA-II
* Proposed Full Fusion + MOEA/D

---

## Cross-Dataset Validation

Cross-dataset experiments include:

* APTOS → DDR
* APTOS → Messidor-2
* APTOS → DR Dataset
* DDR → APTOS
* DDR → Messidor-2
* DDR → DR Dataset
* Messidor-2 → APTOS
* Messidor-2 → DDR
* Messidor-2 → DR Dataset
* DR Dataset → APTOS
* DR Dataset → DDR
* DR Dataset → Messidor-2

Additionally, combined-dataset training with held-out dataset evaluation was performed.

---

## Explainability Analysis

The framework includes:

* Grad-CAM
* Grad-CAM++
* Integrated Gradients
* Insertion–Deletion Analysis
* Reliability Diagrams
* Calibration Error (ECE)

Visual explanations were analyzed against clinically meaningful retinal lesions including:

* Microaneurysms
* Hemorrhages
* Hard Exudates
* Cotton-Wool Spots
* Neovascularization

---

## Hardware Configuration

Experiments were conducted using:

* GPU: NVIDIA GeForce GTX 1660Ti with Max-Q Design
* CPU: AMD Ryzen 7 4800H with Radeon Graphics  
* RAM: 16 GB
* Input Resolution: 224 × 224
* Batch Size: 32

---

## Datasets

The datasets used in this study are publicly available:

* APTOS 2019
* DDR
* Messidor-2
* Diabetic Balanced Dataset
* DR Dataset

Please download the datasets from their respective repositories and update the dataset paths before running the code.


