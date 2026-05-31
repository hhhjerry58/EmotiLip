"""Plot EmotiLip training curves from metrics.jsonl files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def metrics_path(run: Path) -> Path:
    return run / "metrics.jsonl" if run.is_dir() else run


def run_label(run: Path) -> str:
    return run.name if run.is_dir() else run.parent.name


def values(rows: list[dict], section: str, key: str) -> list[float | None]:
    out = []
    for row in rows:
        value = row.get(section, {}).get(key)
        out.append(float(value) if isinstance(value, (int, float)) else None)
    return out


def scalar_values(rows: list[dict], key: str) -> list[float | None]:
    out = []
    for row in rows:
        value = row.get(key)
        out.append(float(value) if isinstance(value, (int, float)) else None)
    return out


def plot_series(
    ax,
    epochs: list[int],
    ys: list[float | None],
    label: str,
    linestyle: str = "-",
    skip_all_zero: bool = False,
) -> None:
    clean_epochs = [epoch for epoch, y in zip(epochs, ys) if y is not None]
    clean_values = [y for y in ys if y is not None]
    if skip_all_zero and clean_values and all(value == 0 for value in clean_values):
        return
    if clean_values:
        ax.plot(clean_epochs, clean_values, label=label, linewidth=1.8, linestyle=linestyle)


def plot_runs(runs: list[Path], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax_total, ax_components, ax_lr, ax_time = axes.ravel()

    for run in runs:
        path = metrics_path(run)
        rows = load_jsonl(path)
        label = run_label(run)
        epochs = [int(row.get("epoch", i)) for i, row in enumerate(rows)]

        plot_series(ax_total, epochs, values(rows, "train", "total_loss"), f"{label} train")
        plot_series(ax_total, epochs, values(rows, "val", "val_loss"), f"{label} val", linestyle="--")

        plot_series(ax_components, epochs, values(rows, "train", "diffusion_loss"), f"{label} diffusion")
        plot_series(ax_components, epochs, values(rows, "train", "prosody_loss"), f"{label} prosody", linestyle="--")
        plot_series(
            ax_components,
            epochs,
            values(rows, "train", "emotion_consistency_loss"),
            f"{label} emo proxy",
            linestyle=":",
            skip_all_zero=True,
        )
        plot_series(
            ax_components,
            epochs,
            values(rows, "train", "emotion_classifier_loss"),
            f"{label} emo cls",
            linestyle="-.",
            skip_all_zero=True,
        )

        plot_series(ax_lr, epochs, scalar_values(rows, "lr_after_step"), f"{label} lr")
        plot_series(ax_time, epochs, scalar_values(rows, "elapsed_sec"), f"{label} seconds")

    ax_total.set_title("Total Loss")
    ax_total.set_xlabel("Epoch")
    ax_total.set_ylabel("Loss")
    ax_total.grid(True, alpha=0.25)

    ax_components.set_title("Training Loss Components")
    ax_components.set_xlabel("Epoch")
    ax_components.set_ylabel("Loss")
    ax_components.grid(True, alpha=0.25)

    ax_lr.set_title("Learning Rate")
    ax_lr.set_xlabel("Epoch")
    ax_lr.set_ylabel("LR")
    ax_lr.set_yscale("log")
    ax_lr.grid(True, alpha=0.25)

    ax_time.set_title("Epoch Time")
    ax_time.set_xlabel("Epoch")
    ax_time.set_ylabel("Seconds")
    ax_time.grid(True, alpha=0.25)

    for ax in axes.ravel():
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot EmotiLip metrics.jsonl training curves")
    parser.add_argument("runs", nargs="+", type=Path, help="Checkpoint dirs or metrics.jsonl files")
    parser.add_argument("--output", type=Path, default=Path("training_curves.png"))
    args = parser.parse_args()

    plot_runs([run.expanduser() for run in args.runs], args.output.expanduser())
    print(f"Wrote plot: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
