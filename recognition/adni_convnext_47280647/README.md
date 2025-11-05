# ADNI ConvNeXtLite Classifier (Problem 8)

This subproject trains a compact ConvNeXt-like CNN to classify AD vs NC on ADNI MRI slices. It includes a subject-wise validation split to prevent leakage and uses a held-out test/ folder for final evaluation.

## Project layout
- `modules.py`: TinyCNN (baseline) and ConvNeXtLite model
- `dataset.py`: ADNI loaders, preprocessing, augmentation, subject-wise val split; returns `(x, y, subject_id)`
- `train.py`: training loop, validation, checkpointing (best.pt), plots
- `predict.py`: evaluation on test (slice-level and patient-level)
- `results/`: run outputs (config.json, curves, best.pt)

## Data expectations
Directory with separate train/ and test/:
```
AD_NC/
  train/AD/**, train/NC/**
  test/AD/**,  test/NC/**
```

## Preprocessing and split
- Grayscale → [0,1] → resize 224×224 → standardize `(x - 0.5)/0.25`
- Augmentation (train only): horizontal flip p=0.5
- Validation split: subject-wise from the train pool (prevents leakage)
- Test set: entire `test/` folder, never used in training/validation

## Typical training config
ConvNeXtLite, grayscale (1×224×224):
```
python train.py \
  --dataset adni \
  --data_root /home/groups/comp3710/ADNI/AD_NC \
  --model nextlite_tiny \
  --epochs 25 \
  --batch_size 32 --num_workers 1 \
  --lr 3e-4 \
  --head_dropout 0.2 \
  --drop_path_rate 0.15 \
  --layer_scale_init 1e-6 \
  --augment \
  --save_dir results/nextlite_example
```

## Evaluation on test
```
python predict.py \
  --checkpoint results/nextlite_example/best.pt \
  --dataset adni \
  --data_root /home/groups/comp3710/ADNI/AD_NC \
  --batch_size 64 --num_workers 1 \
  --model nextlite_tiny \
  --head_dropout 0.2 --drop_path_rate 0.15 --layer_scale_init 1e-6
```
Outputs slice-level accuracy and patient-level accuracy (subject aggregation).

## Experimental Results

### Key Finding: Validation Leakage Fix
Initial runs used image-wise validation splitting, causing the same subjects to appear in both train and val, inflating validation accuracy to ~0.98–0.99 while test accuracy remained ~0.65–0.70. After switching to subject-wise splitting, validation accuracy dropped to realistic levels (~0.80) and aligned with test performance.

### Detailed Run Results

#### Run 1: `a100_nextlite_p8_v1` (10 epochs, seed=42, lr=3e-4, head_dropout=0.2, drop_path=0.15)
- **Issue**: Image-wise val split (leakage)
- **Best val accuracy**: 0.861 (epoch 10)
- **Test slice accuracy**: ~0.698
- **Observation**: Large gap (16+ pts) indicates leakage

#### Run 2: `a100_nextlite_p8_v1_seed123` (25 epochs, seed=123, same config)
- **Issue**: Image-wise val split (leakage)
- **Best val accuracy**: 0.989 (epoch 25)
- **Test slice accuracy**: 0.645
- **Observation**: Extreme gap (34+ pts) confirms leakage

#### Run 3: `a100_nextlite_subjectsplit_v1` (25 epochs, seed=123, lr=3e-4, head_dropout=0.2, drop_path=0.15)
- **Fix**: Subject-wise val split implemented
- **Best val accuracy**: 0.797 (epoch 19)
- **Test slice accuracy**: 0.653
- **Test patient accuracy**: 0.667 (450 subjects)
- **Observation**: Gap reduced to ~14 pts; plausible given domain shift

#### Run 4: `a100_nextlite_lr1e-4_hd0.3_dp0.2_s42` (25 epochs, seed=42, lr=1e-4, head_dropout=0.3, drop_path=0.2)
- **Best val accuracy**: 0.806 (epoch 21)
- **Test slice accuracy**: 0.653
- **Test patient accuracy**: 0.689 (450 subjects)
- **Observation**: Stronger regularization (higher dropout/drop_path) improved patient-level accuracy slightly

#### Run 5: `a100_nextlite_fix_impl` (10 epochs, seed=42, lr=3e-3, all reg disabled)
- **Issue**: Initial broken config (no regularization, high LR)
- **Result**: Stuck at ~0.693 loss, ~0.50 accuracy (not learning)
- **Fix**: Standardized inputs and restored head normalization enabled learning

### Summary
- **Test slice accuracy**: ~0.65 consistently across runs (subject-wise split)
- **Test patient accuracy**: ~0.67–0.69 (aggregation helps)
- **Validation accuracy**: ~0.80 (realistic after leakage fix)
- **Gap explanation**: ~10–15 pts between val and test likely due to:
  1. Domain shift between train/ and test/ folders (different sites/scanners)
  2. Per-slice label noise (patient aggregation improves results)
  3. Overfitting to train distribution after first val peak

**Recommendation**: Always report test set accuracy as the primary metric. Validation is for model selection only.

## Knobs to reach ~80% target
- Try sweeps: `lr ∈ {1e-4, 2e-4}`, `head_dropout=0.3`, `drop_path_rate ∈ {0.2, 0.25}`
- Optionally add weight decay (e.g., 1e-4) and mild rotations/brightness jitter
- Train 35–50 epochs; rely on best.pt (early stopping by val)

## Reproducibility
- Seeds set for Python/NumPy/Torch; cuDNN deterministic
- All CLI args saved to `results/<run>/config.json`

