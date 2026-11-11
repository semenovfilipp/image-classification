"""Обучение одной модели по конфигу + оценка лучшего чекпоинта на test.

Примеры:
    python scripts/train.py --config configs/experiments/resnet18_finetune.yaml
    python scripts/train.py --config configs/experiments/resnet18_scratch.yaml \
        --output-root /content/drive/MyDrive/pets/outputs -o train.epochs=5

Если запуск оборвался (например, отвалился Colab), повторный запуск той же
команды продолжит обучение с последней эпохи. Если эксперимент уже завершён
(есть test_metrics.json), он пропускается — используйте --force для перезапуска.
"""
import json
import shutil
import sys
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import build_dataloaders  # noqa: E402
from src.models.factory import build_model, count_params, param_groups  # noqa: E402
from src.training.trainer import Trainer, warmup_cosine_schedule  # noqa: E402
from src.utils.config import load_config, save_config  # noqa: E402
from src.utils.loggers import build_loggers  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def update_results_table(results_path: Path, row: dict) -> None:
    """Добавляет/обновляет строку эксперимента в общей таблице results.csv."""
    if results_path.exists():
        df = pd.read_csv(results_path)
        df = df[df["experiment"] != row["experiment"]]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.sort_values(["model", "mode"]).to_csv(results_path, index=False)


@click.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True), help="YAML-конфиг эксперимента")
@click.option("--output-root", default=None, help="Переопределить output.root (например, папка на Google Drive)")
@click.option("--data-root", default=None, help="Переопределить data.root")
@click.option("-o", "--override", "overrides", multiple=True, help="Переопределить параметр: -o train.epochs=5")
@click.option("--device", default="auto", show_default=True)
@click.option("--force", is_flag=True, help="Начать заново, даже если есть чекпоинты/результаты")
@click.option("--limit-batches", type=int, default=None, help="Только N батчей за эпоху (отладка)")
def main(config_path, output_root, data_root, overrides, device, force, limit_batches):
    cfg = load_config(config_path, overrides)
    if output_root:
        cfg["output"]["root"] = output_root
    if data_root:
        cfg["data"]["root"] = data_root

    name = cfg["experiment_name"]
    out_root = Path(cfg["output"]["root"])
    run_dir = out_root / name
    if force and run_dir.exists():
        shutil.rmtree(run_dir)
    if (run_dir / "test_metrics.json").exists():
        print(f"[{name}] уже завершён, пропускаю (используйте --force для перезапуска)")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")

    set_seed(cfg["seed"])
    device = resolve_device(device)
    print(f"[{name}] device={device}" + (f" ({torch.cuda.get_device_name()})" if device.type == "cuda" else ""))
    if device.type != "cuda":
        print("ВНИМАНИЕ: GPU не найден, обучение на CPU будет очень медленным")

    # ---------------- данные
    loaders = build_dataloaders(cfg)
    print(f"train {len(loaders['train'].dataset)} | val {len(loaders['val'].dataset)} | "
          f"test {len(loaders['test'].dataset)}")

    # ---------------- модель и оптимизатор
    m_cfg, t_cfg = cfg["model"], cfg["train"]
    model, head = build_model(m_cfg["name"], cfg["data"]["num_classes"], m_cfg["mode"])
    total, trainable = count_params(model)
    print(f"{m_cfg['name']} / {m_cfg['mode']}: params {total / 1e6:.1f}M, trainable {trainable / 1e6:.2f}M")

    lr_mult = t_cfg["backbone_lr_mult"] if m_cfg["mode"] == "finetune" else 1.0
    optimizer = torch.optim.AdamW(
        param_groups(model, head, t_cfg["lr"], lr_mult), weight_decay=t_cfg["weight_decay"]
    )
    steps_per_epoch = limit_batches or len(loaders["train"])
    scheduler = warmup_cosine_schedule(
        optimizer,
        warmup_steps=t_cfg["warmup_epochs"] * steps_per_epoch,
        total_steps=t_cfg["epochs"] * steps_per_epoch,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=t_cfg.get("label_smoothing", 0.0))

    loggers = build_loggers(cfg, run_dir)
    trainer = Trainer(
        model, head, optimizer, scheduler, criterion, device,
        num_classes=cfg["data"]["num_classes"],
        run_dir=run_dir,
        amp=t_cfg.get("amp", True),
        grad_clip=t_cfg.get("grad_clip"),
        frozen_backbone=m_cfg["mode"] == "feature_extract",
        loggers=loggers,
        limit_batches=limit_batches,
    )
    trainer.try_resume()

    # ---------------- обучение
    trainer.fit(loaders["train"], loaders["val"], t_cfg["epochs"], t_cfg["early_stopping_patience"])

    # ---------------- тест на лучшем чекпоинте
    trainer.load_best()
    test_m, cm = trainer.evaluate(loaders["test"], desc="test")
    print(f"[{name}] TEST acc {test_m['acc']:.4f} | macro-F1 {test_m['f1']:.4f}")

    np.save(run_dir / "test_confusion_matrix.npy", cm.matrix.numpy())
    hist = pd.DataFrame(trainer.history)
    best_row = hist.loc[hist["val_acc"].idxmax()]
    row = {
        "experiment": name,
        "model": m_cfg["name"],
        "mode": m_cfg["mode"],
        "params_M": round(total / 1e6, 2),
        "trainable_M": round(trainable / 1e6, 2),
        "epochs_run": len(hist),
        "best_epoch": int(best_row["epoch"]) + 1,
        "best_val_acc": round(float(best_row["val_acc"]), 4),
        "test_acc": round(test_m["acc"], 4),
        "test_f1": round(test_m["f1"], 4),
        "sec_per_epoch": round(float(hist["epoch_time"].median()), 1),
        "train_time_min": round(float(hist["epoch_time"].sum()) / 60, 1),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
        "finished_at": time.strftime("%Y-%m-%d %H:%M"),
    }
    with open(run_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2, ensure_ascii=False)
    update_results_table(out_root / "results.csv", row)

    # last.pt с оптимизатором больше не нужен, а место на Google Drive ограничено
    (run_dir / "last.pt").unlink(missing_ok=True)
    for logger in loggers:
        logger.close()


if __name__ == "__main__":
    main()
