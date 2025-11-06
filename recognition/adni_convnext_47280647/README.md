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


## 2. Problem Definition

### 2.1 Problem Statement

### 2.2 Dataset Overview

## 3. Methodology Overview

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
