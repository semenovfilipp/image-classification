"""Тонкие обёртки над tensorboard / wandb с общим интерфейсом log(dict, step)."""
from __future__ import annotations

from pathlib import Path


class TensorboardLogger:
    def __init__(self, log_dir: Path):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=str(log_dir))

    def log(self, metrics: dict, step: int) -> None:
        for key, value in metrics.items():
            if key == "epoch":
                continue
            # train_loss -> loss/train, чтобы train и val были на одном графике
            split, _, name = key.partition("_")
            tag = f"{name}/{split}" if split in ("train", "val", "test") else key
            self.writer.add_scalar(tag, value, step)
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()


class WandbLogger:
    def __init__(self, project: str, name: str, config: dict, run_dir: Path):
        import wandb

        self.wandb = wandb
        self.run = wandb.init(project=project, name=name, config=config, dir=str(run_dir),
                              id=name, resume="allow")

    def log(self, metrics: dict, step: int) -> None:
        self.wandb.log(metrics, step=step)

    def close(self) -> None:
        self.run.finish()


def build_loggers(cfg: dict, run_dir: Path) -> list:
    log_cfg = cfg.get("logging", {})
    loggers = []
    if log_cfg.get("tensorboard", True):
        loggers.append(TensorboardLogger(run_dir / "tb"))
    if log_cfg.get("wandb", False):
        loggers.append(WandbLogger(log_cfg.get("wandb_project", "image-classification"),
                                   cfg["experiment_name"], cfg, run_dir))
    return loggers
