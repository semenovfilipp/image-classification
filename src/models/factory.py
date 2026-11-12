"""Создание моделей из torchvision в трёх режимах.

* scratch         — случайная инициализация, учится вся сеть;
* feature_extract — веса ImageNet, backbone заморожен, учится только новая голова;
* finetune        — веса ImageNet, учится вся сеть (backbone с пониженным LR).
"""
from __future__ import annotations

from torch import nn
from torchvision import models as tvm

MODES = ("scratch", "feature_extract", "finetune")

# где у архитектуры лежит последний Linear: (атрибут, индекс в Sequential или None)
HEAD_LOCATIONS: dict[str, tuple[str, int | None]] = {
    "alexnet": ("classifier", 6),
    "vgg11_bn": ("classifier", 6),
    "vgg16": ("classifier", 6),
    "vgg16_bn": ("classifier", 6),
    "resnet18": ("fc", None),
    "resnet34": ("fc", None),
    "resnet50": ("fc", None),
    "efficientnet_b0": ("classifier", 1),
    "efficientnet_b1": ("classifier", 1),
    "efficientnet_b2": ("classifier", 1),
    "mobilenet_v3_large": ("classifier", 3),
    "convnext_tiny": ("classifier", 2),
}


def _replace_head(model: nn.Module, name: str, num_classes: int) -> nn.Linear:
    attr, idx = HEAD_LOCATIONS[name]
    if idx is None:
        old = getattr(model, attr)
        new = nn.Linear(old.in_features, num_classes)
        setattr(model, attr, new)
    else:
        container = getattr(model, attr)
        old = container[idx]
        new = nn.Linear(old.in_features, num_classes)
        container[idx] = new
    return new


def build_model(name: str, num_classes: int, mode: str) -> tuple[nn.Module, nn.Module]:
    """Возвращает (модель, голова). Голова нужна для разных LR и заморозки."""
    if mode not in MODES:
        raise ValueError(f"mode должен быть одним из {MODES}, получено {mode!r}")
    if name not in HEAD_LOCATIONS:
        raise ValueError(f"Неизвестная архитектура {name!r}. Доступны: {sorted(HEAD_LOCATIONS)}")

    weights = None if mode == "scratch" else "DEFAULT"
    model = tvm.get_model(name, weights=weights)
    head = _replace_head(model, name, num_classes)

    if mode == "feature_extract":
        head_ids = {id(p) for p in head.parameters()}
        for p in model.parameters():
            p.requires_grad = id(p) in head_ids
    return model, head


def param_groups(model: nn.Module, head: nn.Module, lr: float, backbone_lr_mult: float) -> list[dict]:
    """Группы параметров оптимизатора: голова с lr, backbone с lr * mult."""
    head_ids = {id(p) for p in head.parameters()}
    head_params = [p for p in head.parameters() if p.requires_grad]
    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [{"params": head_params, "lr": lr, "name": "head"}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr * backbone_lr_mult, "name": "backbone"})
    return groups


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
