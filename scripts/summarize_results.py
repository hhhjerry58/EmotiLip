"""Summarize one or more EmotiLip evaluation result directories as markdown tables."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


QUALITY_METRICS = ("mel_mse", "mel_mae", "mel_rmse", "pesq", "estoi", "mcd", "secs", "wer")


def finite_values(rows: list[dict], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(float(value))
    return values


def finite(value) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def metric_from_summary(eval_summary: dict, key: str) -> float | None:
    metrics = eval_summary.get("metrics")
    if not isinstance(metrics, dict):
        return None
    item = metrics.get(key)
    if not isinstance(item, dict) or not item.get("count"):
        return None
    return finite(item.get("mean"))


def prosody_gap(rows: list[dict], key: str) -> float | None:
    by_emotion: dict[str, list[float]] = {}
    for row in rows:
        emotion = str(row.get("emotion", "unknown"))
        prosody = row.get("prosody")
        if not isinstance(prosody, dict):
            continue
        value = prosody.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            by_emotion.setdefault(emotion, []).append(float(value))

    means = [mean(values) for values in by_emotion.values()]
    means = [value for value in means if value is not None]
    if len(means) < 2:
        return None
    return max(means) - min(means)


def load_results(path: Path) -> list[dict]:
    result_path = path / "results.json" if path.is_dir() else path
    with open(result_path) as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list in {result_path}")
    return rows


def load_eval_summary(path: Path) -> dict:
    summary_path = path / "summary.json" if path.is_dir() else path.parent / "summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path) as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def run_name(path: Path) -> str:
    return path.name if path.is_dir() else path.parent.name


def summarize_run(path: Path) -> dict:
    rows = load_results(path)
    eval_summary = load_eval_summary(path)
    ser = eval_summary.get("ser") if isinstance(eval_summary.get("ser"), dict) else {}
    summary = {
        "run": run_name(path),
        "samples": len(rows),
        "f0_gap": prosody_gap(rows, "f0_mean"),
        "energy_gap": prosody_gap(rows, "energy_mean"),
        "ser_status": ser.get("status", "not_available"),
        "emotion_accuracy": finite(ser.get("emotion_accuracy")),
        "emotion_f1": finite(ser.get("emotion_f1")),
        "ser_failed_samples": ser.get("failed_samples", ""),
    }
    for metric in QUALITY_METRICS:
        summary[metric] = metric_from_summary(eval_summary, metric)
        if summary[metric] is None:
            summary[metric] = mean(finite_values(rows, metric))
    return summary


def markdown_table(summaries: list[dict]) -> str:
    lines = [
        "## Quality Metrics",
        "",
        "| Run | Samples | Mel MSE ↓ | Mel MAE ↓ | Mel RMSE ↓ | PESQ ↑ | ESTOI ↑ | MCD ↓ | SECS ↑ | WER ↓ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['run']} | {row['samples']} | {fmt(row['mel_mse'])} | {fmt(row['mel_mae'])} | "
            f"{fmt(row['mel_rmse'])} | {fmt(row['pesq'])} | {fmt(row['estoi'])} | "
            f"{fmt(row['mcd'])} | {fmt(row['secs'])} | {fmt(row['wer'])} |"
        )

    lines.extend([
        "",
        "## Emotion/Prosody Proxies",
        "",
        "| Run | Samples | SER Status | Emotion Acc ↑ | Emotion F1 ↑ | F0 Gap By Emotion ↑ | Energy Gap By Emotion ↑ |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ])
    for row in summaries:
        lines.append(
            f"| {row['run']} | {row['samples']} | {row['ser_status']} | "
            f"{fmt(row['emotion_accuracy'])} | {fmt(row['emotion_f1'])} | "
            f"{fmt(row['f0_gap'])} | {fmt(row['energy_gap'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize EmotiLip eval_output results")
    parser.add_argument("runs", nargs="+", type=Path, help="Eval directories or results.json files")
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
