"""Загрузка YAML-конфигов с наследованием (`base:`) и переопределениями из CLI."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict[str, Any]:
    """Читает конфиг; если в нём есть `base: <path>`, сначала грузит базовый.

    Путь в `base` считается относительно текущей директории (корня репозитория),
    а если там файла нет — относительно самого конфига.
    `overrides` — строки вида "train.epochs=5" (значение парсится как YAML).
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    base_path = cfg.pop("base", None)
    if base_path is not None:
        base_path = Path(base_path)
        if not base_path.exists():
            base_path = path.parent / base_path
        base_cfg = load_config(base_path)
        base_cfg.pop("experiment_name", None)  # имя берём только у самого конфига
        cfg = _deep_merge(base_cfg, cfg)

    for item in overrides:
        key, _, raw_value = item.partition("=")
        if not _:
            raise ValueError(f"Override должен быть вида key.sub=value, получено: {item!r}")
        node = cfg
        *parents, leaf = key.strip().split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw_value)

    cfg.setdefault("experiment_name", path.stem)
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
