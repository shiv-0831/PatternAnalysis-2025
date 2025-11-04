from __future__ import annotations
import os, glob, random
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# ============== Random dataset (for local smoke testing) ==============
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

# ============== ADNI dataset (actual dataset for the assignment) ==============
@dataclass
class ADNIArgs:
    data_root: str
    labels_csv: Optional[str] = None   # if None -> folder mode
    plane: str = "axial"               # axial|sagittal|coronal
    slice_mode: str = "center_k"       # center_k|step_s|all
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
        self.subjects = subjects            # list of subject folder paths
        self.labels   = labels              # {basename(subject_dir): 0/1}
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

    def _load_volume(self, subject_dir: str) -> np.ndarray:
        if subject_dir in self.cache: return self.cache[subject_dir]
        path = _find_nifti(subject_dir)
        arr = nib.load(path).get_fdata(caching="unchanged")
        if arr.ndim == 4: arr = arr[..., 0]
        arr = _zscore(arr)
        self.cache[subject_dir] = arr
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
        base = os.path.basename(sid)
        y   = int(self.labels[base])
        return x, y, base

# ---------- Subject/label discovery ----------

def _read_labels_csv(labels_csv: str) -> Dict[str,int]:
    df = pd.read_csv(labels_csv)
    def map_label(v):
        if isinstance(v, str):
            v = v.strip().upper()
            if v in ("CN", "NC", "CONTROL", "NORMAL"): return 0
            if v in ("AD", "ALZHEIMERS", "ALZHEIMER'S"): return 1
        return int(v)
    return {str(r["subject_id"]): map_label(r["label"]) for _, r in df.iterrows()}

def _scan_class_folders(root: str, class_map: Dict[str,int] = None):
    """
    Expects root to contain AD/ and/or NC/ with subject folders inside each.
    Returns (list_of_subject_paths, labels_by_subject_basename)
    """
    if class_map is None:
        class_map = {"AD": 1, "NC": 0}

    subjects: List[str] = []
    labels: Dict[str, int] = {}
    for cls_name, cls_label in class_map.items():
        cls_dir = os.path.join(root, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for entry in sorted(os.listdir(cls_dir)):
            subj_dir = os.path.join(cls_dir, entry)
            if os.path.isdir(subj_dir):
                subjects.append(subj_dir)
                labels[entry] = cls_label
    if not subjects:
        raise RuntimeError(f"No subjects found under class folders in: {root}")
    return subjects, labels

def _has_subdirs(path: str, names: List[str]) -> bool:
    return all(os.path.isdir(os.path.join(path, n)) for n in names)

# ---------- Build loaders (auto-detect train/test) ----------

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

def build_loaders_adni(args: ADNIArgs):
    assert os.path.isdir(args.data_root), f"data_root not found: {args.data_root}"

    # If CSV provided, we keep the previous CSV mode.
    if args.labels_csv:
        labels = _read_labels_csv(args.labels_csv)
        subjects = []
        for d in sorted(os.listdir(args.data_root)):
            p = os.path.join(args.data_root, d)
            if os.path.isdir(p) and d in labels:
                subjects.append(p)
        if not subjects:
            raise RuntimeError("No subject folders matched labels_csv under data_root.")
        train_ids, val_ids, test_ids = _split_subjects(subjects, args.val_ratio, args.test_ratio, args.seed)
        basename_labels = {k: v for k, v in labels.items()}

    else:
        root = args.data_root.rstrip("/")

        # Case 1: data_root is the parent containing 'train' and 'test'
        if _has_subdirs(root, ["train", "test"]):
            train_root = os.path.join(root, "train")
            test_root  = os.path.join(root, "test")
            train_subjects, train_labels = _scan_class_folders(train_root, {"AD":1, "NC":0})
            # Validation from TRAIN split
            tr_ids, val_ids, _ = _split_subjects(train_subjects, args.val_ratio, 0.0, args.seed)
            # Test from TEST split
            test_subjects, test_labels = _scan_class_folders(test_root, {"AD":1, "NC":0})
            # Ensure consistent label map (both are {basename: label})
            basename_labels = {**train_labels, **test_labels}
            train_ids, test_ids = tr_ids, test_subjects

        else:
            # Case 2: data_root is .../train or .../test, try to find sibling
            base = os.path.basename(root)
            parent = os.path.dirname(root)
            if base in ("train", "test") and _has_subdirs(parent, ["train", "test"]):
                split_root = root
                other_root = os.path.join(parent, "test" if base == "train" else "train")
                # load the chosen split
                split_subjects, split_labels = _scan_class_folders(split_root, {"AD":1, "NC":0})
                # build val from chosen split
                if base == "train":
                    tr_ids, val_ids, _ = _split_subjects(split_subjects, args.val_ratio, 0.0, args.seed)
                    train_ids = tr_ids
                    # test is sibling test/
                    test_subjects, test_labels = _scan_class_folders(other_root, {"AD":1, "NC":0})
                    test_ids = test_subjects
                    basename_labels = {**split_labels, **test_labels}
                else:
                    # base == "test": we will create val from sibling train, and keep test as this root
                    train_subjects, train_labels = _scan_class_folders(other_root, {"AD":1, "NC":0})
                    tr_ids, val_ids, _ = _split_subjects(train_subjects, args.val_ratio, 0.0, args.seed)
                    train_ids = tr_ids
                    test_subjects, test_labels = _scan_class_folders(split_root, {"AD":1, "NC":0})
                    test_ids = test_subjects
                    basename_labels = {**train_labels, **test_labels}

            else:
                # Case 3: data_root itself has AD/ and NC/ (single split only)
                subjects, basename_labels = _scan_class_folders(root, {"AD":1, "NC":0})
                train_ids, val_ids, test_ids = _split_subjects(subjects, args.val_ratio, args.test_ratio, args.seed)

    # Build datasets
    train_set = ADNISliceDataset(train_ids, basename_labels, args, split="train")
    val_set   = ADNISliceDataset(val_ids,   basename_labels, args, split="val")
    test_set  = ADNISliceDataset(test_ids,  basename_labels, args, split="test")

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
