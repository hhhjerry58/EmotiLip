"""Generate tiny preprocessed-style data for end-to-end debug training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


EMOTIONS = {
    "angry": 0,
    "happy": 4,
    "neutral": 5,
    "sad": 6,
}
PROJECT_DIR = Path(__file__).resolve().parent.parent


def resolve_output_dir(path: Path) -> Path:
    """Resolve relative debug output paths under the EmotiLip project dir."""
    path = path.expanduser()
    if path.is_absolute():
        return path
    return PROJECT_DIR / path


def make_sample(
    rng: np.random.Generator,
    sample_dir: Path,
    speaker_id: str,
    emotion: str,
    utterance_id: str,
    n_mels: int,
    video_fps: int,
    sample_rate: int,
    hop_length: int,
    lip_size: int,
    face_size: int,
) -> dict:
    sample_dir.mkdir(parents=True, exist_ok=True)

    video_len = int(rng.integers(10, 18))
    mel_len = max(8, int(round(video_len * (sample_rate / hop_length) / video_fps)))

    lip = rng.normal(0.5, 0.18, size=(video_len, lip_size, lip_size)).clip(0, 1).astype(np.float32)
    face = rng.normal(0.5, 0.2, size=(3, face_size, face_size)).clip(0, 1).astype(np.float32)
    mel = rng.normal(0.0, 1.0, size=(n_mels, mel_len)).astype(np.float32)

    t = np.linspace(0, 1, mel_len, dtype=np.float32)
    emotion_bias = {
        "angry": (0.4, 0.5),
        "happy": (0.25, 0.25),
        "neutral": (0.0, 0.0),
        "sad": (-0.25, -0.35),
    }[emotion]
    pitch = (np.sin(2 * np.pi * t) * 0.1 + emotion_bias[0]).astype(np.float32)
    energy = (np.cos(2 * np.pi * t) * 0.1 + emotion_bias[1]).astype(np.float32)

    paths = {
        "lip_path": sample_dir / "lip.npy",
        "face_path": sample_dir / "face.npy",
        "mel_path": sample_dir / "mel.npy",
        "prosody_path": sample_dir / "prosody.npy",
    }
    np.save(paths["lip_path"], lip)
    np.save(paths["face_path"], face)
    np.save(paths["mel_path"], mel)
    np.save(paths["prosody_path"], {"pitch": pitch, "energy": energy})

    return {
        "speaker_id": speaker_id,
        "emotion": emotion,
        "emotion_label": EMOTIONS[emotion],
        "intensity": "debug",
        "utterance_id": utterance_id,
        "video_path": "dummy",
        "audio_path": None,
        **{key: str(path.resolve()) for key, path in paths.items()},
    }


def write_manifest(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(entries, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create tiny dummy EmotiLip data")
    parser.add_argument("--output_dir", type=Path, default=Path("data/debug"))
    parser.add_argument("--speakers", type=int, default=2)
    parser.add_argument("--train_per_pair", type=int, default=2)
    parser.add_argument("--val_per_pair", type=int, default=1)
    parser.add_argument("--test_per_pair", type=int, default=1)
    parser.add_argument("--n_mels", type=int, default=16)
    parser.add_argument("--video_fps", type=int, default=25)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--lip_size", type=int, default=32)
    parser.add_argument("--face_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    output_dir = resolve_output_dir(args.output_dir).resolve()

    split_counts = {
        "train": args.train_per_pair,
        "val": args.val_per_pair,
        "test": args.test_per_pair,
    }
    speakers = [f"S{i:03d}" for i in range(args.speakers)]
    all_entries = []

    for split, per_pair in split_counts.items():
        entries = []
        for speaker_id in speakers:
            for emotion in EMOTIONS:
                for idx in range(per_pair):
                    utterance_id = f"{split}_{speaker_id}_{emotion}_{idx:02d}"
                    sample_dir = output_dir / split / speaker_id / f"{emotion}_{idx:02d}"
                    entry = make_sample(
                        rng,
                        sample_dir,
                        speaker_id,
                        emotion,
                        utterance_id,
                        args.n_mels,
                        args.video_fps,
                        args.sample_rate,
                        args.hop_length,
                        args.lip_size,
                        args.face_size,
                    )
                    entries.append(entry)
                    all_entries.append(entry)
        write_manifest(output_dir / split / "manifest.json", entries)
        print(f"[{split}] {len(entries)} samples -> {output_dir / split / 'manifest.json'}")

    write_manifest(output_dir / "manifest.json", all_entries)
    print(f"[all] {len(all_entries)} samples -> {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
