"""Dataset indexing, patient-level splitting, image caching and PyTorch datasets.

Raw layout (Kaggle "chest-xray-pneumonia"):
    <root>/{train,val,test}/{NORMAL,PNEUMONIA}/*.jpeg

The official 16-image ``val`` folder is too small to be useful, so it is folded
into the training pool and a patient-level validation split is carved out
instead.
"""

from __future__ import annotations

import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from torchvision.transforms import v2

CLASSES = ("NORMAL", "PNEUMONIA")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

_PNEUMONIA_RE = re.compile(r"^person(\d+)_(bacteria|virus)_")
_NORMAL_RE = re.compile(r"^((?:NORMAL2-)?IM-\d+)")


def parse_filename(filename: str) -> tuple[str, str]:
    """Return ``(patient_id, subtype)`` for a dataset filename.

    Pneumonia files look like ``person1_virus_6.jpeg``. The same person number
    can appear under both ``bacteria`` and ``virus``, so the subtype is left out
    of the patient id. Grouping that way may join two different patients, but
    it can never split one patient across train and validation.
    """
    m = _PNEUMONIA_RE.match(filename)
    if m:
        return f"person{m.group(1)}", m.group(2)
    m = _NORMAL_RE.match(filename)
    if m:
        return m.group(1), "normal"
    raise ValueError(f"Unrecognised filename: {filename}")


def build_manifest(raw_root: str | Path) -> pd.DataFrame:
    """Index every image under ``raw_root`` into a DataFrame."""
    raw_root = Path(raw_root)
    rows = []
    for source_split in ("train", "val", "test"):
        for label, cls in enumerate(CLASSES):
            folder = raw_root / source_split / cls
            if not folder.is_dir():
                continue
            for path in sorted(folder.glob("*.jpeg")):
                patient_id, subtype = parse_filename(path.name)
                rows.append(
                    {
                        "path": str(path.relative_to(raw_root)),
                        "source_split": source_split,
                        "split": "test" if source_split == "test" else "train",
                        "label": label,
                        "subtype": subtype,
                        "patient_id": patient_id,
                    }
                )
    if not rows:
        raise FileNotFoundError(f"No images found under {raw_root}/{{train,test}}/{{NORMAL,PNEUMONIA}}")
    return pd.DataFrame(rows)


def assign_val_split(df: pd.DataFrame, val_frac: float = 0.15, seed: int = 42) -> pd.DataFrame:
    """Move a patient-grouped, label-stratified fraction of ``train`` into ``val``."""
    df = df.copy()
    train_idx = df.index[df["split"] == "train"]
    train = df.loc[train_idx]
    n_splits = max(2, round(1 / val_frac))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, val_pos = next(sgkf.split(train, train["label"], groups=train["patient_id"]))
    df.loc[train_idx[val_pos], "split"] = "val"
    return df


def _load_resized(args: tuple[str, int]) -> np.ndarray:
    path, size = args
    with Image.open(path) as im:
        im.draft("L", (size, size))  # fast JPEG downscale on decode
        # Some JPEGs are stored as RGB; force a single grayscale channel for all.
        im = im.convert("L").resize((size, size), Image.BICUBIC)
        return np.asarray(im, dtype=np.uint8)


def cache_images(df: pd.DataFrame, raw_root: str | Path, out_file: str | Path, size: int = 256, workers: int = 8) -> None:
    """Decode, grayscale and resize every image once into a ``(N, size, size)`` uint8 array.

    Resizing to a square (instead of keeping the aspect ratio) deliberately
    removes the image-shape difference between the NORMAL and PNEUMONIA
    sources, which a model could otherwise exploit as a shortcut.
    """
    raw_root = Path(raw_root)
    jobs = [(str(raw_root / p), size) for p in df["path"]]
    with Pool(workers) as pool:
        images = pool.map(_load_resized, jobs, chunksize=32)
    np.save(out_file, np.stack(images))


def build_transforms(image_size: int = 224, augment: bool = True) -> v2.Compose:
    if augment:
        ops = [
            v2.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1), antialias=True),
            v2.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    else:
        # Resize first so the centre crop keeps the same field of view (224/256) at any --image-size;
        # a bare CenterCrop(112) on a 256 cache would only show the middle of the chest.
        ops = [v2.Resize(round(image_size * 256 / 224), antialias=True), v2.CenterCrop(image_size)]
    # No horizontal flips: the heart sits on the left, so mirroring creates anatomically implausible images.
    return v2.Compose(ops + [v2.ToDtype(torch.float32, scale=True)])


class XrayDataset(Dataset):
    """Serves cached grayscale images as ImageNet-normalised 3-channel tensors."""

    def __init__(self, cache_file: str | Path, indices: np.ndarray, labels: np.ndarray, transform: v2.Compose):
        self.cache_file = str(cache_file)
        self.indices = np.asarray(indices)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.transform = transform
        self._images = None  # opened lazily so each DataLoader worker gets its own memmap

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._images is None:
            self._images = np.load(self.cache_file, mmap_mode="r")
        img = torch.from_numpy(np.array(self._images[self.indices[i]])).unsqueeze(0)
        img = self.transform(img).expand(3, -1, -1)
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return img, torch.tensor(self.labels[i])


def load_split(data_dir: str | Path, split: str, augment: bool, image_size: int = 224) -> tuple[XrayDataset, pd.DataFrame]:
    data_dir = Path(data_dir)
    manifest = pd.read_csv(data_dir / "manifest.csv")
    n_cached = np.load(data_dir / "images.npy", mmap_mode="r").shape[0]
    if n_cached != len(manifest):
        raise ValueError(f"images.npy has {n_cached} images but manifest.csv has {len(manifest)} rows; rerun scripts/prepare_data.py")
    rows = manifest[manifest["split"] == split]
    ds = XrayDataset(data_dir / "images.npy", rows.index.to_numpy(), rows["label"].to_numpy(), build_transforms(image_size, augment))
    return ds, rows
