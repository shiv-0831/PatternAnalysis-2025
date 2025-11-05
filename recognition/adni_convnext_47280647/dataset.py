from __future__ import annotations
import os, glob, random
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image  # image loading

# ===== Random dataset (for local smoke tests) =====
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

# ===== ADNI (for Rangpur) =====
@dataclass
class ADNIArgs:
    data_root: str
    labels_csv: Optional[str] = None   # ignored in image-only mode; kept for CLI compatibility
    plane: str = "axial"               # ignored (MRI-only option)
    slice_mode: str = "center_k"       # ignored (MRI-only option)
    center_k: int = 32                 # ignored
    step_s: int = 2                    # ignored
    resize_hw: Tuple[int,int] = (224,224)
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    batch_size: int = 16
    num_workers: int = 4
    augment: bool = False

_IMG_EXTS = (
    "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff",
    "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.TIF", "*.TIFF"
)

def _has_subdirs(path: str, names: List[str]) -> bool:
    return all(os.path.isdir(os.path.join(path, n)) for n in names)

def _list_images_under(root: str, class_map: Dict[str,int]) -> List[Tuple[str,int]]:
    items: List[Tuple[str,int]] = []
    for cls_name, label in class_map.items():
        cls_dir = os.path.join(root, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for pat in _IMG_EXTS:
            for p in glob.glob(os.path.join(cls_dir, "**", pat), recursive=True):
                if os.path.isfile(p):
                    items.append((p, label))
    return items

def _extract_subject_id(path: str) -> str:
    """Heuristic subject ID from path: prefer folder just under AD/ or NC/; fallback to filename stem prefix before '_'"""
    cls_names = {"AD", "NC"}
    parts = os.path.normpath(path).split(os.sep)
    for i, part in enumerate(parts):
        if part in cls_names:
            if i + 1 < len(parts) - 1:
                return parts[i + 1]
            break
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.split("_")[0]

def _split_items_by_subject(items: List[Tuple[str,int]], val_ratio: float, seed: int):
    from collections import defaultdict
    subj_to_items: Dict[str, List[Tuple[str,int]]] = defaultdict(list)
    for p, y in items:
        s = _extract_subject_id(p)
        subj_to_items[s].append((p, y))
    subjects = list(subj_to_items.keys())
    rng = random.Random(seed)
    rng.shuffle(subjects)
    n_val = int(round(len(subjects) * val_ratio))
    val_subjects = set(subjects[:n_val])
    train_subjects = set(subjects[n_val:])
    tr_items: List[Tuple[str,int]] = []
    va_items: List[Tuple[str,int]] = []
    for s in train_subjects:
        tr_items.extend(subj_to_items[s])
    for s in val_subjects:
        va_items.extend(subj_to_items[s])
    return tr_items, va_items

class ADNIImageDataset(Dataset):
    """Each image file is one sample -> returns (x, y)."""
    def __init__(self, items: List[Tuple[str,int]], args: ADNIArgs, split: str):
        self.items = items
        self.args = args
        self.split = split

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        with Image.open(path).convert("L") as img:
            arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W)
        t = torch.from_numpy(arr).unsqueeze(0)               # (1, H, W)
        t = F.interpolate(
            t.unsqueeze(0), size=self.args.resize_hw, mode="bilinear", align_corners=False
        ).squeeze(0)

        mean, std = 0.5, 0.25
        t = (t - mean) / std

        if self.args.augment and self.split == "train":
            if random.random() < 0.5:
                t = torch.flip(t, dims=[2])
        sid = _extract_subject_id(path)
        return t.float(), int(y), sid

def _split_items(items: List, val_ratio: float, test_ratio: float, seed: int):
    rng = random.Random(seed)
    arr = items[:]; rng.shuffle(arr)
    n = len(arr)
    n_test = int(round(n * test_ratio))
    n_val  = int(round(n * val_ratio))
    test_items = arr[:n_test]
    val_items  = arr[n_test:n_test+n_val]
    train_items = arr[n_test+n_val:]
    return train_items, val_items, test_items

def build_loaders_adni(args: ADNIArgs):
    assert os.path.isdir(args.data_root), f"data_root not found: {args.data_root}"
    class_map = {"AD": 1, "NC": 0}
    root = args.data_root.rstrip("/")

    # Case A: parent contains train/ and test/
    if _has_subdirs(root, ["train", "test"]):
        train_root = os.path.join(root, "train")
        test_root  = os.path.join(root, "test")
        tr_items = _list_images_under(train_root, class_map)
        te_items = _list_images_under(test_root,  class_map)
        if not tr_items: raise RuntimeError(f"No images found under: {train_root}")
        if not te_items: raise RuntimeError(f"No images found under: {test_root}")
        tr_split, va_split = _split_items_by_subject(tr_items, args.val_ratio, args.seed)
        print(f"[ADNI IMAGES] train={len(tr_split)} val={len(va_split)} test={len(te_items)}")
        return (
            DataLoader(ADNIImageDataset(tr_split, args, split="train"), batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=True),
            DataLoader(ADNIImageDataset(va_split, args, split="val"),   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True),
            DataLoader(ADNIImageDataset(te_items,  args, split="test"), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True),
        )

    # Case B: root is .../train or .../test (use sibling as the other split)
    base = os.path.basename(root)
    parent = os.path.dirname(root)
    if base in ("train", "test") and _has_subdirs(parent, ["train", "test"]):
        split_root = root
        other_root = os.path.join(parent, "test" if base == "train" else "train")
        sp_items = _list_images_under(split_root, class_map)
        ot_items = _list_images_under(other_root, class_map)
        if not sp_items: raise RuntimeError(f"No images found under: {split_root}")
        if not ot_items: raise RuntimeError(f"No images found under: {other_root}")
        if base == "train":
            tr_split, va_split = _split_items_by_subject(sp_items, args.val_ratio, args.seed)
            print(f"[ADNI IMAGES] train={len(tr_split)} val={len(va_split)} test={len(ot_items)}")
            return (
                DataLoader(ADNIImageDataset(tr_split, args, split="train"), batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=True),
                DataLoader(ADNIImageDataset(va_split, args, split="val"),   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True),
                DataLoader(ADNIImageDataset(ot_items,  args, split="test"), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True),
            )
        else:
            tr_split, va_split = _split_items_by_subject(ot_items, args.val_ratio, args.seed)
            print(f"[ADNI IMAGES] train={len(tr_split)} val={len(va_split)} test={len(sp_items)}")
            return (
                DataLoader(ADNIImageDataset(tr_split, args, split="train"), batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=True),
                DataLoader(ADNIImageDataset(va_split, args, split="val"),   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True),
                DataLoader(ADNIImageDataset(sp_items,  args, split="test"), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True),
            )

    # Case C: single pool (root has AD/ and NC/ only)
    items = _list_images_under(root, class_map)
    if not items:
        raise RuntimeError(f"No AD/NC images found under: {root}")
    tr_items, va_items, te_items = _split_items(items, args.val_ratio, args.test_ratio, args.seed)
    print(f"[ADNI IMAGES] train={len(tr_items)} val={len(va_items)} test={len(te_items)}")
    return (
        DataLoader(ADNIImageDataset(tr_items, args, split="train"), batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=True),
        DataLoader(ADNIImageDataset(va_items, args, split="val"),   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True),
        DataLoader(ADNIImageDataset(te_items, args, split="test"),  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True),
    )

# ===== Unified entry point =====
def build_loaders(dataset: str = "random", **kwargs):
    if dataset == "random":
        return build_loaders_random(batch_size=kwargs.get("batch_size", 16), seed=kwargs.get("seed", 42))
    elif dataset == "adni":
        a = ADNIArgs(**kwargs)
        return build_loaders_adni(a)
    else:
        raise ValueError(f"Unknown dataset={dataset}")
