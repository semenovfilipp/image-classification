"""Метрики многоклассовой классификации на основе матрицы ошибок."""
import torch


class ConfusionMatrix:
    """Накопительная матрица ошибок: строки — истинный класс, столбцы — предсказанный."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        idx = targets.cpu() * self.num_classes + preds.cpu()
        self.matrix += torch.bincount(idx, minlength=self.num_classes**2).view(
            self.num_classes, self.num_classes
        )

    def accuracy(self) -> float:
        total = self.matrix.sum().item()
        return self.matrix.diag().sum().item() / total if total else 0.0

    def per_class_f1(self) -> torch.Tensor:
        m = self.matrix.double()
        tp = m.diag()
        fp = m.sum(0) - tp
        fn = m.sum(1) - tp
        denom = 2 * tp + fp + fn
        return torch.where(denom > 0, 2 * tp / denom.clamp(min=1), torch.zeros_like(tp))

    def macro_f1(self) -> float:
        # усредняем только по классам, которые встречаются в выборке
        present = self.matrix.sum(1) > 0
        f1 = self.per_class_f1()
        return f1[present].mean().item() if present.any() else 0.0

    def compute(self) -> dict[str, float]:
        return {"acc": self.accuracy(), "f1": self.macro_f1()}
