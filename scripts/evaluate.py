"""Оценка обученной модели: метрики, отчёт по классам, матрица ошибок, самые частые путаницы.

Пример:
    python scripts/evaluate.py --config configs/experiments/resnet18_finetune.yaml \
        --output-root /content/drive/MyDrive/pets/outputs
"""
import json
import sys
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import build_dataloaders, load_classes
from src.models.factory import build_model
from src.utils.config import load_config


@click.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--checkpoint", default=None, help="По умолчанию <output.root>/<experiment>/best.pt")
@click.option("--output-root", default=None)
@click.option("--data-root", default=None)
@click.option("--split", default="test", type=click.Choice(["train", "val", "test"]), show_default=True)
@click.option("--device", default="auto", show_default=True)
def main(config_path, checkpoint, output_root, data_root, split, device):
    cfg = load_config(config_path)
    if output_root:
        cfg["output"]["root"] = output_root
    if data_root:
        cfg["data"]["root"] = data_root
    run_dir = Path(cfg["output"]["root"]) / cfg["experiment_name"]
    checkpoint = Path(checkpoint) if checkpoint else run_dir / "best.pt"
    device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else
                          ("cpu" if device == "auto" else device))

    classes = load_classes(cfg["data"]["root"])
    loader = build_dataloaders(cfg)[split]

    # веса берём из чекпоинта, поэтому скачивать ImageNet-веса не нужно
    model, _ = build_model(cfg["model"]["name"], cfg["data"]["num_classes"], "scratch")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()

    preds, targets = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc=split):
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                preds.append(model(x.to(device)).argmax(1).cpu())
            targets.append(y)
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    report = classification_report(targets, preds, target_names=classes, digits=3, output_dict=True)
    print(classification_report(targets, preds, target_names=classes, digits=3))
    eval_dir = run_dir / f"eval_{split}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report).T.to_csv(eval_dir / "classification_report.csv")

    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (targets, preds), 1)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm / cm.sum(1, keepdims=True).clip(min=1), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), classes, rotation=90, fontsize=7)
    ax.set_yticks(range(n), classes, fontsize=7)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_title(f"{cfg['experiment_name']} — нормированная матрица ошибок ({split})")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(eval_dir / "confusion_matrix.png", dpi=120)

    off = cm.copy()
    np.fill_diagonal(off, 0)
    top = np.dstack(np.unravel_index(np.argsort(off.ravel())[::-1][:10], off.shape))[0]
    confusions = [{"true": classes[i], "pred": classes[j], "count": int(off[i, j])} for i, j in top if off[i, j]]
    print("Чаще всего путаются:")
    for c in confusions:
        print(f"  {c['true']:>28} -> {c['pred']:<28} {c['count']}")
    with open(eval_dir / "top_confusions.json", "w", encoding="utf-8") as f:
        json.dump(confusions, f, indent=2, ensure_ascii=False)
    print(f"Сохранено в {eval_dir}")


if __name__ == "__main__":
    main()
