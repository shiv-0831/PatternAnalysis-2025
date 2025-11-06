# ADNI ConvNeXtLite Classifier – Problem 8 (COMP3710)

Author: Shivam Garg
Student Number: 47280647

## Table of Contents
1. [Executive Summary](#1-executive-summary)  
2. [Problem Definition](#2-problem-definition)  
   1. [Problem Statement](#21-problem-statement)  
   2. [Dataset Overview](#22-dataset-overview)  
3. [Methodology Overview](#3-methodology-overview)  
4. [Data Pipeline](#4-data-pipeline)  
   1. [Ingestion & Directory Layout](#41-ingestion--directory-layout)  
   2. [Pre-processing](#42-pre-processing)  
   3. [Augmentation](#43-augmentation)  
5. [Model Architecture](#5-model-architecture)  
   1. [TinyCNN Baseline](#51-tinycnn-baseline)  
   2. [ConvNeXtLite Classifier](#52-convnextlite-classifier)  
6. [Training Configuration & Implementation](#6-training-configuration--implementation)  
7. [Evaluation Protocol](#7-evaluation-protocol)  
8. [Experiments & Results](#8-experiments--results)  
   1. [Training Curves](#81-training-curves)  
   2. [Validation & Test Metrics](#82-validation--test-metrics)  
   3. [Ablations & Comparisons](#83-ablations--comparisons)  
9. [Discussion](#9-discussion)  
10. [Usage Guide](#10-usage-guide)  
    1. [Environment Setup](#101-environment-setup)  
    2. [Training Commands](#102-training-commands)  
    3. [Evaluation Commands](#103-evaluation-commands)  
11. [Reproducibility Checklist](#11-reproducibility-checklist)  
12. [Dependencies](#12-dependencies)  
13. [References](#13-references)  

---

## 1. Executive Summary
This project tackles **binary classification of Alzheimer's Disease (abbreviated as AD) vs Cognitively Normal (abbreviated as CN)** from **ADNI MRI 2D slices** data, targeting ≥ **80%** test accuracy on a strictly **patient-wise held-out** dataset. This implementation follows a leakage-safe pipeline including grayscale conversion, 224x224 resizing, normalisation (x-0.5)/0.25, and ligt MRI-appropriate augmentation, paired with strict **subject-wise** splits to prevent data leakage (patient overlap) across training, validation, and testing set.

Two models are implemented to bracket performance and guide design choices. A compact **TinyCNN** provides a clear, reproducible baseline. A **ConvNeXtLite** classifier then scales representational capacity using modern CNN components (eg., depthwise convolutions, LayerNorm, larger kernels) to better capture subtle brain textures. Training is implemented in PyTorch with **Adam**, checkpointing, seeded runs, and automatic curve exports.

## 2. Problem Definition

### 2.1 Problem Statement
The task is **binary classification** of brain MRI slices into AD (Alzheimer's Disease) and CN (Cognitively Normal). Inputs are 2D axial slices derived from the ADNI scans; the output is a single class label per slice, with patient-lavel reporting obtained by aggregating slide predictions per subject. The primary objective is ≥ 0.80 accuracy on a strict patient held out test set (to prevent data leakage).

### 2.2 Dataset Overview
The data is categorised as follows - 
- **Souces and Classes**: The dataset is a two-class subset of ADNI with labels AD and CN. Each subjec contributes a 3D MRI volume from which 2D axial slices are extracted for training and evaluation.
- **Data units**: Trainint operates at the slide level. Evaluation includes bnoth slice-level and patient-level (aggregated) metrics

## 3. Methodology Overview
The approach is an end-to-end pipeline that turn ADNI MRI 2D slices into patient-levl AD?/CN predictions while preventing data leakage and keeping runs easy to reproduce. It combined a transparent TinyCNN baseline with a stronger ConvNeXtLite classifier to bracket performance.

## 4. Data Pipeline

### 4.1 Ingestion & Directory Layout

### 4.2 Pre-processing

### 4.3 Augmentation

## 5. Model Architecture

### 5.1 TinyCNN Baseline

### 5.2 ConvNeXtLite Classifier

## 6. Training Configuration & Implementation

## 7. Evaluation Protocol

## 8. Experiments & Results

### 8.1 Training Curves

### 8.2 Validation & Test Metrics

### 8.3 Ablations & Comparisons

## 9. Discussion

## 10. Usage Guide

### 10.1 Environment Setup

### 10.2 Training Commands

### 10.3 Evaluation Commands

## 11. Reproducibility Checklist

## 12. Dependencies

## 13. References

## Current Implementation Notes (for reference)
- `modules.py`: TinyCNN (baseline) and ConvNeXtLite model
- `dataset.py`: ADNI loaders, preprocessing, augmentation, subject-wise val split; returns `(x, y, subject_id)`
- `train.py`: training loop, validation, checkpointing (best.pt), plots
- `predict.py`: evaluation on test (slice-level and patient-level)
- `results/`: run outputs (config.json, curves, best.pt)
