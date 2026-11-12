import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Фиксирует seed во всех генераторах случайных чисел.

    cudnn.benchmark оставляем включённым ради скорости: результаты между
    запусками могут отличаться в последних знаках, но разбиение данных,
    инициализация и порядок батчей воспроизводимы.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """Seed для воркеров DataLoader (аугментации воспроизводимы)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
