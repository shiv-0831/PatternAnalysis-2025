from __future__ import annotations
import os, glob, random
from dataclasses import dataclass
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# ============== Random dataset (for local smoke tests) ==============
class RandomSliceDataset(Dataset):
    def __init__(self, n: int = 192, image_size: Tuple[int, int, int] = (1, 224, 224), num_classes: int = 2, seed: int = 42):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.x = torch.rand((n, *image_size), generator=g)
        self.y = torch.randint(low=0, high=num_classes, size=(n,), generator=g)

    def __len__(self): return self.x.shape[0]
    def __getitem__(self, i): return self.x[i], int(self.y[i])

def build_loaders_random(batch_size: int = 16, seed: int = 42):
    full = RandomSliceDataset(n=192, seed=seed)
    val_size = 64
    train_size = len(full) - val_size
    train_set, val_set = random_split(full, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, None

# ============== ADNI dataset (Rangpur implementation) ==============
@dataclass
class ADNIArgs:
    data_root: str
    labels_csv: str
    plane: str = "axial"         # axial|sagittal|coronal
    slice_mode: str = "center_k" # center_k|step_s|all
    center_k: int = 32
    step_s: int = 2
    resize_hw: Tuple[int,int] = (224,224)
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    batch_size: int = 16
    num_workers: int = 4
    augment: bool = False

def _axis_for_plane(plane: str) -> int:
    plane = plane.lower()
    return {"axial": 2, "sagittal": 0, "coronal": 1}[plane]

def _find_nifti(subject_dir: str) -> str:
    cands = glob.glob(os.path.join(subject_dir, "**", "*.nii*"), recursive=True)
    if not cands:
        raise FileNotFoundError(f"No NIfTI under {subject_dir}")
    return sorted(cands)[0]

def _zscore(x: np.ndarray) -> np.ndarray:
    mask = x != 0
    if mask.sum() < 10:
        return x.astype(np.float32)
    mu = x[mask].mean()
    sd = x[mask].std() + 1e-6
    return ((x - mu) / sd).astype(np.float32)

class ADNISliceDataset(Dataset):
    """
    Returns (image, label, subject_id). Image is (1,H,W) float32.
    Splits are per-subject to avoid leakage.
    """
    def __init__(self, subjects: List[str], labels: Dict[str,int], args: ADNIArgs, split: str):
        super().__init__()
        self.subjects = subjects
        self.labels   = labels
        self.args     = args
        self.split    = split
        self.axis     = _axis_for_plane(args.plane)
        self.index: List[Tuple[str,int]] = []
        self.cache: Dict[str, np.ndarray] = {}

        for sid in self.subjects:
            vol = self._load_volume(sid)
            D = vol.shape[self.axis]
            if self.args.slice_mode == "center_k":
                k = min(self.args.center_k, D); mid = D // 2
                start = max(0, mid - k//2); sl_idx = list(range(start, start + k))
            elif self.args.slice_mode == "step_s":
                s = max(1, self.args.step_s); sl_idx = list(range(0, D, s))
            elif self.args.slice_mode == "all":
                sl_idx = list(range(0, D))
            else:
                raise ValueError(f"Unknown slice_mode={self.args.slice_mode}")
            for z in sl_idx:
                sl = self._get_slice(vol, z)
                if np.count_nonzero(sl) > 20:
                    self.index.append((sid, z))

    def _load_volume(self, subject_id: str) -> np.ndarray:
        if subject_id in self.cache: return self.cache[subject_id]
        path = _find_nifti(os.path.join(self.args.data_root, subject_id))
        arr = nib.load(path).get_fdata(caching="unchanged")
        if arr.ndim == 4: arr = arr[..., 0]
        arr = _zscore(arr)
        self.cache[subject_id] = arr
        return arr

    def _get_slice(self, vol: np.ndarray, idx: int) -> np.ndarray:
        if self.axis == 0: return vol[idx, :, :]
        if self.axis == 1: return vol[:, idx, :]
        return vol[:, :, idx]

    def _to_tensor(self, sl: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(sl).unsqueeze(0)  # (1,H,W)
        t = F.interpolate(t.unsqueeze(0), size=self.args.resize_hw, mode="bilinear", align_corners=False).squeeze(0)
        if self.args.augment and self.split == "train":
            if random.random() < 0.5:
                t = torch.flip(t, dims=[2])
        return t.float()

    def __len__(self): return len(self.index)

    def __getitem__(self, i):
        sid, z = self.index[i]
        vol = self._load_volume(sid)
        sl  = self._get_slice(vol, z)
        x   = self._to_tensor(sl)
        y   = int(self.labels[sid])
        return x, y, sid

def _split_subjects(all_subjects: List[str], val_ratio: float, test_ratio: float, seed: int):
    rng = random.Random(seed)
    subs = all_subjects[:]; rng.shuffle(subs)
    n = len(subs)
    n_test = int(round(n * test_ratio))
    n_val  = int(round(n * val_ratio))
    test_ids = subs[:n_test]
    val_ids  = subs[n_test:n_test+n_val]
    train_ids = subs[n_test+n_val:]
    return train_ids, val_ids, test_ids

def _read_labels(labels_csv: str) -> Dict[str,int]:
    df = pd.read_csv(labels_csv)
    def map_label(v):
        if isinstance(v, str):
            v = v.strip().upper()
            if v in ("CN", "CONTROL", "NORMAL"): return 0
            if v in ("AD", "ALZHEIMERS", "ALZHEIMER'S"): return 1
        return int(v)
    return {str(r["subject_id"]): map_label(r["label"]) for _, r in df.iterrows()}

def build_loaders_adni(args: ADNIArgs):
    assert os.path.isdir(args.data_root), f"data_root not found: {args.data_root}"
    assert os.path.isfile(args.labels_csv), f"labels_csv not found: {args.labels_csv}"
    labels = _read_labels(args.labels_csv)
    subjects = sorted([d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d)) and d in labels])
    if not subjects:
        raise RuntimeError("No subject folders matched labels_csv; check subject_id values and directory names.")
    train_ids, val_ids, test_ids = _split_subjects(subjects, args.val_ratio, args.test_ratio, args.seed)
    train_set = ADNISliceDataset(train_ids, labels, args, split="train")
    val_set   = ADNISliceDataset(val_ids,   labels, args, split="val")
    test_set  = ADNISliceDataset(test_ids,  labels, args, split="test")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader

# ============== Unified entry point ==============
def build_loaders(dataset: str = "random", **kwargs):
    if dataset == "random":
        return build_loaders_random(batch_size=kwargs.get("batch_size", 16), seed=kwargs.get("seed", 42))
    elif dataset == "adni":
        a = ADNIArgs(**kwargs)
        return build_loaders_adni(a)
    else:
        raise ValueError(f"Unknown dataset={dataset}")
