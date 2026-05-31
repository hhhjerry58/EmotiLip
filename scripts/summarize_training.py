"""Summarize EmotiLip training logs as a markdown table."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def maybe_float(value) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def fmt_int(value) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    return ""


def summarize_run(path: Path) -> dict:
    metrics_path = path / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing training metrics: {metrics_path}")
    rows = load_jsonl(metrics_path)
    if not rows:
        raise ValueError(f"No rows found in {metrics_path}")

    summary_path = path / "train_summary.json"
    summary = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    last = rows[-1]
    best = min(rows, key=lambda row: row.get("val", {}).get("val_loss", float("inf")))
    return {
        "run": path.name,
        "epochs": len(rows),
        "best_epoch": best.get("epoch"),
        "best_val_loss": maybe_float(best.get("val", {}).get("val_loss")),
        "last_train_loss": maybe_float(last.get("train", {}).get("total_loss")),
        "last_val_loss": maybe_float(last.get("val", {}).get("val_loss")),
        "last_diffusion_loss": maybe_float(last.get("train", {}).get("diffusion_loss")),
        "last_prosody_loss": maybe_float(last.get("train", {}).get("prosody_loss")),
        "last_emotion_consistency_loss": maybe_float(
            last.get("train", {}).get("emotion_consistency_loss")
        ),
        "last_emotion_classifier_loss": maybe_float(
            last.get("train", {}).get("emotion_classifier_loss")
        ),
        "params": summary.get("model_parameters"),
        "train_samples": summary.get("train_samples"),
        "val_samples": summary.get("val_samples"),
    }


def any_metric(summaries: list[dict], key: str) -> bool:
    return any(maybe_float(row.get(key)) not in (None, 0.0) for row in summaries)


def markdown_table(summaries: list[dict]) -> str:
    optional_columns = []
    for key, title in (
        ("last_diffusion_loss", "Last Diff ↓"),
        ("last_prosody_loss", "Last Prosody ↓"),
        ("last_emotion_consistency_loss", "Last Emo Proxy ↓"),
        ("last_emotion_classifier_loss", "Last Emo Cls ↓"),
    ):
        if any_metric(summaries, key):
            optional_columns.append((key, title))

    headers = [
        "Run",
        "Epochs",
        "Best Epoch",
        "Best Val Loss ↓",
        "Last Train Loss ↓",
        "Last Val Loss ↓",
        *[title for _, title in optional_columns],
        "Params",
        "Train Samples",
        "Val Samples",
    ]
    alignments = ["---", "---:", "---:", "---:", "---:", "---:"]
    alignments.extend(["---:" for _ in optional_columns])
    alignments.extend(["---:", "---:", "---:"])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(alignments) + " |",
    ]
    for row in summaries:
        values = [
            row["run"],
            str(row["epochs"]),
            str(row["best_epoch"]),
            fmt(row["best_val_loss"]),
            fmt(row["last_train_loss"]),
            fmt(row["last_val_loss"]),
        ]
        values.extend(fmt(row[key]) for key, _ in optional_columns)
        values.extend([
            fmt_int(row["params"]),
            fmt_int(row["train_samples"]),
            fmt_int(row["val_samples"]),
        ])
        lines.append(
            "| " + " | ".join(values) + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize EmotiLip checkpoint training logs")
    parser.add_argument("runs", nargs="+", type=Path, help="Checkpoint directories with metrics.jsonl")
    parser.add_argument("--output", type=Path, default=None, help="Optional markdown output path")
    args = parser.parse_args()

    summaries = [summarize_run(path.expanduser()) for path in args.runs]
    table = markdown_table(summaries)
    print(table)
    if args.output is not None:
        output = args.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(table)
        print(f"Wrote summary: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
