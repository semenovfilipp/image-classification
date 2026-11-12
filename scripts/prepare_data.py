"""Скачивает Oxford-IIIT Pet, уменьшает картинки и делает разбиение train/val/test.

Уменьшение до 256px по короткой стороне один раз заранее сильно ускоряет
обучение: в Colab всего 2 CPU, и декодирование больших JPEG становится узким местом.

Запуск:
    python scripts/prepare_data.py
"""
import json
from pathlib import Path

import click
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.datasets import OxfordIIITPet
from tqdm import tqdm


def resize_short_side(img, short_side: int):
    w, h = img.size
    scale = short_side / min(w, h)
    if scale >= 1:
        return img
    return img.resize((round(w * scale), round(h * scale)), resample=Image.BICUBIC)


@click.command()
@click.option("--raw-dir", default="data/raw", show_default=True, help="Куда скачивать исходный датасет")
@click.option("--out-dir", default="data/processed/oxford_pets", show_default=True)
@click.option("--short-side", default=256, show_default=True, help="Размер короткой стороны после ресайза")
@click.option("--val-size", default=0.15, show_default=True, help="Доля trainval, уходящая в val")
@click.option("--seed", default=42, show_default=True)
def main(raw_dir, out_dir, short_side, val_size, seed):
    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    frames = {}
    classes = None
    for split in ("trainval", "test"):
        ds = OxfordIIITPet(raw_dir, split=split, target_types="category", download=True)
        classes = ds.classes
        rows = []
        for i in tqdm(range(len(ds)), desc=f"resize {split}"):
            src = Path(ds._images[i])
            label = int(ds._labels[i])
            name = src.stem + ".jpg"
            dst = img_dir / name
            if not dst.exists():
                img, _ = ds[i]
                resize_short_side(img, short_side).save(dst, quality=95)
            rows.append({"filename": name, "label": label, "class_name": classes[label]})
        frames[split] = pd.DataFrame(rows)

    # официальный test оставляем тестом, trainval делим стратифицированно
    train_df, val_df = train_test_split(
        frames["trainval"], test_size=val_size, stratify=frames["trainval"]["label"], random_state=seed
    )
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    frames["test"].to_csv(out_dir / "test.csv", index=False)
    with open(out_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=1)

    print(f"Готово: {out_dir}")
    print(f"  классов: {len(classes)}")
    print(f"  train {len(train_df)} | val {len(val_df)} | test {len(frames['test'])}")


if __name__ == "__main__":
    main()
