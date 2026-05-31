"""Create balanced train/val/test manifest subsets from preprocessed data."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
SPLITS = ("train", "val", "test")


def load_manifest(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Manifest root must be a list: {path}")
    return data


def resolve_split_manifest(input_dir: Path, split: str) -> Path:
    split_path = input_dir / split / "manifest.json"
    if split_path.exists():
        return split_path
    raise FileNotFoundError(f"Missing {split} manifest: {split_path}")


def choose_speakers(entries_by_split: dict[str, list[dict]], num_speakers: int | None) -> set[str] | None:
    if num_speakers is None:
        return None
    train_speakers = sorted({str(e["speaker_id"]) for e in entries_by_split.get("train", [])})
    if len(train_speakers) < num_speakers:
        raise ValueError(f"Requested {num_speakers} speakers, but train split only has {len(train_speakers)}")
    return set(train_speakers[:num_speakers])


def filter_entries(
    entries: list[dict],
    speakers: set[str] | None,
    emotions: set[str] | None,
) -> list[dict]:
    selected = []
    for entry in entries:
        if speakers is not None and str(entry.get("speaker_id")) not in speakers:
            continue
        if emotions is not None and str(entry.get("emotion")) not in emotions:
            continue
        selected.append(entry)
    return selected


def cap_per_bucket(
    entries: list[dict],
    max_per_bucket: int | None,
    rng: random.Random,
) -> list[dict]:
    if max_per_bucket is None:
        return list(entries)

    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for entry in entries:
        key = (str(entry.get("speaker_id", "unknown")), str(entry.get("emotion", "unknown")))
        buckets[key].append(entry)

    selected = []
    for key in sorted(buckets):
        bucket = list(buckets[key])
        rng.shuffle(bucket)
        selected.extend(bucket[:max_per_bucket])

    rng.shuffle(selected)
    return selected


def summarize(entries_by_split: dict[str, list[dict]]) -> dict:
    summary = {}
    for split, entries in entries_by_split.items():
        emotions = Counter(str(e.get("emotion", "unknown")) for e in entries)
        speakers = Counter(str(e.get("speaker_id", "unknown")) for e in entries)
        summary[split] = {
            "samples": len(entries),
            "speakers": len(speakers),
            "emotions": dict(sorted(emotions.items())),
        }
    return summary


def write_manifest(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Select a balanced subset from EmotiLip manifests")
    parser.add_argument("--input_dir", type=Path, default=PROJECT_DIR / "data" / "processed")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_DIR / "data" / "subset")
    parser.add_argument("--speakers", nargs="*", default=None, help="Speaker IDs to keep")
    parser.add_argument("--num_speakers", type=int, default=None, help="Keep the first N train speakers")
    parser.add_argument("--emotions", nargs="*", default=None, help="Emotion names to keep")
    parser.add_argument("--max_per_bucket", type=int, default=None, help="Max samples per split/speaker/emotion")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.speakers and args.num_speakers is not None:
        raise ValueError("Use either --speakers or --num_speakers, not both.")
    if args.max_per_bucket is not None and args.max_per_bucket <= 0:
        raise ValueError("--max_per_bucket must be positive when set.")

    entries_by_split = {
        split: load_manifest(resolve_split_manifest(input_dir, split))
        for split in SPLITS
    }
    speakers = set(args.speakers) if args.speakers else choose_speakers(entries_by_split, args.num_speakers)
    emotions = set(args.emotions) if args.emotions else None

    rng = random.Random(args.seed)
    selected_by_split = {}
    for split, entries in entries_by_split.items():
        selected = filter_entries(entries, speakers=speakers, emotions=emotions)
        selected = cap_per_bucket(selected, args.max_per_bucket, rng)
        selected_by_split[split] = selected
        if not selected:
            raise ValueError(
                f"No samples selected for {split}. Check speaker/emotion filters and split coverage."
            )

    all_entries = []
    for split in SPLITS:
        write_manifest(output_dir / split / "manifest.json", selected_by_split[split])
        all_entries.extend(selected_by_split[split])
    write_manifest(output_dir / "manifest.json", all_entries)

    summary = summarize(selected_by_split)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    for split in SPLITS:
        info = summary[split]
        print(
            f"[{split}] samples={info['samples']} speakers={info['speakers']} "
            f"emotions={info['emotions']} -> {output_dir / split / 'manifest.json'}"
        )
    print(f"[all] samples={len(all_entries)} -> {output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
