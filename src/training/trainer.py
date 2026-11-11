"""Цикл обучения: train/val, AMP, early stopping, чекпоинты и продолжение после обрыва."""
from __future__ import annotations

import math
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.utils.metrics import ConfusionMatrix


def warmup_cosine_schedule(optimizer, warmup_steps: int, total_steps: int):
    """Линейный разогрев, затем косинусное затухание LR до 0 (шаг — батч)."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        head: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        criterion: nn.Module,
        device: torch.device,
        num_classes: int,
        run_dir: Path,
        amp: bool = True,
        grad_clip: float | None = None,
        frozen_backbone: bool = False,
        loggers: list | None = None,
        limit_batches: int | None = None,
    ):
        self.model = model.to(device)
        self.head = head
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.num_classes = num_classes
        self.run_dir = Path(run_dir)
        self.amp = amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.grad_clip = grad_clip
        self.frozen_backbone = frozen_backbone
        self.loggers = loggers or []
        self.limit_batches = limit_batches  # для быстрой отладки

        self.start_epoch = 0
        self.best_acc = -1.0
        self.epochs_no_improve = 0
        self.history: list[dict] = []

    # ------------------------------------------------------------------ режимы
    def _set_train_mode(self) -> None:
        if self.frozen_backbone:
            # замороженный backbone держим в eval, чтобы BatchNorm не менял статистики
            self.model.eval()
            self.head.train()
        else:
            self.model.train()

    # ------------------------------------------------------------------ эпохи
    def train_one_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self._set_train_mode()
        cm = ConfusionMatrix(self.num_classes)
        total_loss, n = 0.0, 0
        pbar = tqdm(loader, desc=f"train {epoch}", leave=False)
        for i, (x, y) in enumerate(pbar):
            if self.limit_batches and i >= self.limit_batches:
                break
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                logits = self.model(x)
                loss = self.criterion(logits, y)
            self.scaler.scale(loss).backward()
            if self.grad_clip:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], self.grad_clip
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            cm.update(logits.argmax(1), y)
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        return {"loss": total_loss / max(1, n), **cm.compute()}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, desc: str = "val") -> tuple[dict[str, float], ConfusionMatrix]:
        self.model.eval()
        cm = ConfusionMatrix(self.num_classes)
        total_loss, n = 0.0, 0
        for i, (x, y) in enumerate(tqdm(loader, desc=desc, leave=False)):
            if self.limit_batches and i >= self.limit_batches:
                break
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                logits = self.model(x)
                loss = self.criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            cm.update(logits.argmax(1), y)
        return {"loss": total_loss / max(1, n), **cm.compute()}, cm

    # ------------------------------------------------------------------ fit
    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, patience: int) -> None:
        for epoch in range(self.start_epoch, epochs):
            t0 = time.time()
            train_m = self.train_one_epoch(train_loader, epoch)
            val_m, _ = self.evaluate(val_loader)
            epoch_time = time.time() - t0

            row = {
                "epoch": epoch,
                "lr": self.optimizer.param_groups[0]["lr"],
                "epoch_time": epoch_time,
                **{f"train_{k}": v for k, v in train_m.items()},
                **{f"val_{k}": v for k, v in val_m.items()},
            }
            self.history.append(row)
            for logger in self.loggers:
                logger.log(row, step=epoch)

            improved = val_m["acc"] > self.best_acc
            if improved:
                self.best_acc = val_m["acc"]
                self.epochs_no_improve = 0
                torch.save({"model": self.model.state_dict(), "epoch": epoch, "val": val_m},
                           self.run_dir / "best.pt")
            else:
                self.epochs_no_improve += 1

            print(
                f"[{epoch + 1:>3}/{epochs}] "
                f"train loss {train_m['loss']:.3f} acc {train_m['acc']:.3f} | "
                f"val loss {val_m['loss']:.3f} acc {val_m['acc']:.3f} f1 {val_m['f1']:.3f} | "
                f"{epoch_time:.1f}s{' *' if improved else ''}"
            )
            self.save_last(epoch)
            pd.DataFrame(self.history).to_csv(self.run_dir / "history.csv", index=False)

            if self.epochs_no_improve >= patience:
                print(f"Early stopping: val acc не растёт {patience} эпох")
                break

    # ------------------------------------------------------------------ чекпоинты
    def save_last(self, epoch: int) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "epoch": epoch,
                "best_acc": self.best_acc,
                "epochs_no_improve": self.epochs_no_improve,
                "history": self.history,
            },
            self.run_dir / "last.pt",
        )

    def try_resume(self) -> bool:
        path = self.run_dir / "last.pt"
        if not path.exists():
            return False
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_acc = ckpt["best_acc"]
        self.epochs_no_improve = ckpt["epochs_no_improve"]
        self.history = ckpt["history"]
        print(f"Продолжаю с эпохи {self.start_epoch} (best val acc {self.best_acc:.3f})")
        return True

    def load_best(self) -> None:
        ckpt = torch.load(self.run_dir / "best.pt", map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
