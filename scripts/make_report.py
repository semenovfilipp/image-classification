"""Собирает результаты всех экспериментов в таблицы и графики для README.

Читает <output-root>/*/test_metrics.json и history.csv, пишет в results/:
    results.csv        — полная таблица
    results.md         — таблицы в markdown (test accuracy: модель x режим, и подробная)
    curves.png         — val accuracy по эпохам для каждой модели и режима
    test_accuracy.png  — сравнение test accuracy

Пример:
    python scripts/make_report.py --output-root /content/drive/MyDrive/pets/outputs
"""
import json
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

MODE_ORDER = ["scratch", "feature_extract", "finetune"]
MODE_COLORS = {"scratch": "#d62728", "feature_extract": "#1f77b4", "finetune": "#2ca02c"}


@click.command()
@click.option("--output-root", default="outputs", show_default=True)
@click.option("--results-dir", default="results", show_default=True)
def main(output_root, results_dir):
    output_root, results_dir = Path(output_root), Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    rows, histories = [], {}
    for metrics_file in sorted(output_root.glob("*/test_metrics.json")):
        with open(metrics_file, encoding="utf-8") as f:
            row = json.load(f)
        rows.append(row)
        hist_file = metrics_file.parent / "history.csv"
        if hist_file.exists():
            histories[row["experiment"]] = pd.read_csv(hist_file)
    if not rows:
        raise click.ClickException(f"В {output_root} нет завершённых экспериментов")

    df = pd.DataFrame(rows)
    df["mode"] = pd.Categorical(df["mode"], MODE_ORDER, ordered=True)
    df = df.sort_values(["model", "mode"])
    df.to_csv(results_dir / "results.csv", index=False)

    pivot = df.pivot_table(index="model", columns="mode", values="test_acc", observed=False)
    pivot = pivot.reindex(columns=[m for m in MODE_ORDER if m in pivot.columns])
    cols = ["model", "mode", "params_M", "trainable_M", "epochs_run", "best_epoch",
            "best_val_acc", "test_acc", "test_f1", "sec_per_epoch", "train_time_min"]
    with open(results_dir / "results.md", "w", encoding="utf-8") as f:
        f.write("### Test accuracy\n\n")
        f.write(pivot.map(lambda v: "—" if pd.isna(v) else f"{v:.3f}").to_markdown())
        f.write("\n\n### Подробно\n\n")
        f.write(df[cols].to_markdown(index=False, floatfmt=".3f"))
        f.write(f"\n\nGPU: {', '.join(sorted(df['gpu'].unique()))}\n")
    print((results_dir / "results.md").read_text(encoding="utf-8"))

    models = sorted(df["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(4.5 * len(models), 3.8), sharey=True, squeeze=False)
    for ax, model in zip(axes[0], models):
        for _, r in df[df["model"] == model].iterrows():
            h = histories.get(r["experiment"])
            if h is None:
                continue
            ax.plot(h["epoch"] + 1, h["val_acc"], label=f"{r['mode']} val", color=MODE_COLORS[r["mode"]])
            ax.plot(h["epoch"] + 1, h["train_acc"], "--", alpha=0.5, color=MODE_COLORS[r["mode"]])
        ax.set_title(model)
        ax.set_xlabel("эпоха")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("accuracy (сплошная — val, пунктир — train)")
    axes[0][-1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(results_dir / "curves.png", dpi=120)

    fig, ax = plt.subplots(figsize=(8, 4))
    pivot.plot.bar(ax=ax, color=[MODE_COLORS[m] for m in pivot.columns], rot=0)
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="режим")
    fig.tight_layout()
    fig.savefig(results_dir / "test_accuracy.png", dpi=120)
    print(f"Графики сохранены в {results_dir}")


if __name__ == "__main__":
    main()
