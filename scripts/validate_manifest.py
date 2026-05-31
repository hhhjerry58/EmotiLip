"""Validate preprocessed EmotiLip manifest files without heavy dependencies."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
REQUIRED_KEYS = (
    "speaker_id",
    "emotion",
    "emotion_label",
    "utterance_id",
    "lip_path",
    "face_path",
    "mel_path",
    "prosody_path",
)
LOGICAL_SAMPLE_KEY_FIELDS = ("speaker_id", "emotion", "intensity", "utterance_id")


def manifest_entry_key(entry: dict) -> str | None:
    """Return a stable logical-sample key for duplicate and split-leakage checks."""
    if not isinstance(entry, dict):
        return None
    values = []
    for key in LOGICAL_SAMPLE_KEY_FIELDS:
        value = entry.get(key)
        if key == "intensity" and value in (None, ""):
            value = "na"
        if value in (None, ""):
            return None
        values.append(str(value))
    return "|".join(values)


def resolve_path(path_value: str | None, manifest_path: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    for base in (manifest_path.parent, manifest_path.parent.parent, PROJECT_DIR, Path.cwd()):
        candidate = base / path
        if candidate.exists():
            return candidate
    return PROJECT_DIR / path


def validate_manifest(path: Path) -> tuple[list[str], dict]:
    with open(path) as f:
        manifest = json.load(f)

    errors = []
    emotions = Counter()
    speakers = Counter()
    logical_keys = Counter()

    if not isinstance(manifest, list):
        return ["manifest root must be a list"], {}

    for i, entry in enumerate(manifest):
        if not isinstance(entry, dict):
            errors.append(f"entry {i}: must be an object")
            continue

        for key in REQUIRED_KEYS:
            if key not in entry or entry[key] in (None, ""):
                errors.append(f"entry {i}: missing {key}")

        for key in ("lip_path", "face_path", "mel_path", "prosody_path"):
            resolved = resolve_path(entry.get(key), path)
            if resolved is None or not resolved.exists():
                errors.append(f"entry {i}: {key} not found: {entry.get(key)!r}")

        emotions[entry.get("emotion", "unknown")] += 1
        speakers[entry.get("speaker_id", "unknown")] += 1
        key = manifest_entry_key(entry)
        if key is not None:
            logical_keys[key] += 1

    duplicates = {key: count for key, count in logical_keys.items() if count > 1}
    if duplicates:
        preview = ", ".join(f"{key} x{count}" for key, count in sorted(duplicates.items())[:5])
        extra = f"; {len(duplicates) - 5} more" if len(duplicates) > 5 else ""
        errors.append(f"duplicate logical sample keys: {preview}{extra}")

    summary = {
        "samples": len(manifest),
        "speakers": len(speakers),
        "emotions": dict(sorted(emotions.items())),
        "logical_keys": len(logical_keys),
        "duplicate_logical_keys": sum(count - 1 for count in logical_keys.values() if count > 1),
    }
    return errors, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate EmotiLip manifest files")
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--max_errors", type=int, default=20)
    args = parser.parse_args()

    any_errors = False
    for manifest_path in args.manifests:
        manifest_path = manifest_path.expanduser().resolve()
        errors, summary = validate_manifest(manifest_path)
        if errors:
            any_errors = True
            print(f"[FAIL] {manifest_path}")
            for error in errors[:args.max_errors]:
                print(f"  - {error}")
            if len(errors) > args.max_errors:
                print(f"  ... {len(errors) - args.max_errors} more errors")
        else:
            print(
                f"[OK]   {manifest_path} "
                f"samples={summary['samples']} speakers={summary['speakers']} "
                f"emotions={summary['emotions']}"
            )

    return 1 if any_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
