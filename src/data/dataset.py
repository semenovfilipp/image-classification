"""Dataset и DataLoader'ы для подготовленного Oxford-IIIT Pet.

Ожидаемая структура (создаётся scripts/prepare_data.py):
    <root>/images/*.jpg
    <root>/train.csv, val.csv, test.csv   # колонки: filename,label,class_name
    <root>/classes.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from src.utils.seed import seed_worker

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PetsDataset(Dataset):
    def __init__(self, root: str | Path, split: str, transform=None):
        self.root = Path(root)
        df = pd.read_csv(self.root / f"{split}.csv")
        self.files = df["filename"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        with Image.open(self.root / "images" / self.files[idx]) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


def load_classes(root: str | Path) -> list[str]:
    with open(Path(root) / "classes.json", encoding="utf-8") as f:
        return json.load(f)


def build_transforms(data_cfg: dict, train: bool) -> T.Compose:
    size = data_cfg["img_size"]
    normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        cj = data_cfg.get("color_jitter", 0.0)
        steps = [
            T.RandomResizedCrop(size, scale=tuple(data_cfg.get("rrc_scale", (0.5, 1.0)))),
            T.RandomHorizontalFlip(),
        ]
        if cj:
            steps.append(T.ColorJitter(cj, cj, cj))
        return T.Compose([*steps, T.ToTensor(), normalize])
    return T.Compose([
        T.Resize(int(size * 256 / 224)),
        T.CenterCrop(size),
        T.ToTensor(),
        normalize,
    ])


def build_dataloaders(cfg: dict) -> dict[str, DataLoader]:
    data_cfg = cfg["data"]
    root = data_cfg["root"]
    pin = torch.cuda.is_available()
    nw = data_cfg.get("num_workers", 2)
    generator = torch.Generator().manual_seed(cfg["seed"])

    loaders = {}
    for split in ("train", "val", "test"):
        is_train = split == "train"
        ds = PetsDataset(root, split, build_transforms(data_cfg, train=is_train))
        loaders[split] = DataLoader(
            ds,
            batch_size=data_cfg["batch_size"],
            shuffle=is_train,
            drop_last=is_train,
            num_workers=nw,
            pin_memory=pin,
            persistent_workers=nw > 0,
            worker_init_fn=seed_worker,
            generator=generator if is_train else None,
        )
    return loaders
