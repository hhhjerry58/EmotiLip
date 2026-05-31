"""Export generated demo audio samples from checkpoint(s) and a manifest."""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate import resolve_entry_path
from scripts.inference import estimate_mel_length, get_sample_rate, load_model, speaker_tensor_from_entry


PROJECT_DIR = Path(__file__).resolve().parent.parent


def safe_token(value: object) -> str:
    text = str(value) if value not in (None, "") else "na"
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "na"


def load_manifest(path: Path) -> list[dict]:
    with open(path) as f:
        manifest = json.load(f)
    if not isinstance(manifest, list):
        raise ValueError(f"Manifest root must be a list: {path}")
    return manifest


def run_name(checkpoint: Path) -> str:
    if checkpoint.name in {"best.pt", "last.pt"}:
        return safe_token(checkpoint.parent.name)
    return safe_token(checkpoint.stem)


def sample_name(entry: dict, fallback_index: int) -> str:
    parts = [
        entry.get("sample_id") or entry.get("utterance_id") or f"sample_{fallback_index:04d}",
        entry.get("speaker_id"),
        entry.get("emotion"),
        entry.get("intensity"),
    ]
    return safe_token("_".join(str(part) for part in parts if part not in (None, "")))


def filter_manifest(
    manifest: list[dict],
    emotions: set[str] | None,
    speakers: set[str] | None,
) -> list[tuple[int, dict]]:
    selected = []
    for idx, entry in enumerate(manifest):
        if emotions is not None and str(entry.get("emotion")) not in emotions:
            continue
        if speakers is not None and str(entry.get("speaker_id")) not in speakers:
            continue
        selected.append((idx, entry))
    return selected


def choose_entries(
    manifest: list[dict],
    emotions: set[str] | None,
    speakers: set[str] | None,
    max_samples: int,
    seed: int,
    balanced: bool,
) -> list[tuple[int, dict]]:
    candidates = filter_manifest(manifest, emotions, speakers)
    if not candidates:
        raise ValueError("No manifest entries match the requested filters.")

    rng = random.Random(seed)
    if not balanced:
        candidates = list(candidates)
        rng.shuffle(candidates)
        return candidates[:max_samples]

    buckets: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for idx, entry in candidates:
        buckets[str(entry.get("emotion", "unknown"))].append((idx, entry))
    for bucket in buckets.values():
        rng.shuffle(bucket)

    selected = []
    emotion_order = sorted(buckets)
    while len(selected) < max_samples and any(buckets.values()):
        for emotion in emotion_order:
            if buckets[emotion]:
                selected.append(buckets[emotion].pop())
                if len(selected) >= max_samples:
                    break
    return selected


def load_preprocessed_entry(entry: dict, manifest_path: Path, cfg: dict, device: torch.device):
    lip_path = resolve_entry_path(entry.get("lip_path"), manifest_path)
    face_path = resolve_entry_path(entry.get("face_path"), manifest_path)
    if lip_path is None or face_path is None:
        raise ValueError("Entry must contain lip_path and face_path")

    lip = np.load(lip_path).astype(np.float32)
    face = np.load(face_path).astype(np.float32)
    lip_tensor = torch.from_numpy(lip).unsqueeze(0).unsqueeze(0).to(device)
    face_tensor = torch.from_numpy(face).unsqueeze(0).to(device)

    mel_length = estimate_mel_length(cfg, lip_tensor.shape[2])
    mel_path = resolve_entry_path(entry.get("mel_path"), manifest_path)
    if mel_path:
        mel_length = int(np.load(mel_path, mmap_mode="r").shape[-1])
    return lip_tensor, face_tensor, mel_length


def speaker_tensor(model, entry: dict, device: torch.device) -> torch.Tensor | None:
    return speaker_tensor_from_entry(model, entry, device, source="demo manifest")


def maybe_export_reference(
    entry: dict,
    manifest_path: Path,
    output_path: Path,
    sample_rate: int,
    copy_reference: bool,
) -> str | None:
    ref_path = resolve_entry_path(entry.get("audio_path"), manifest_path)
    if not ref_path:
        return None
    ref_path_obj = Path(ref_path)
    if not ref_path_obj.exists():
        return None
    if not copy_reference:
        return str(ref_path_obj)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if ref_path_obj.suffix.lower() == ".wav":
        shutil.copyfile(ref_path_obj, output_path)
    else:
        import librosa
        ref_audio, _ = librosa.load(ref_path_obj, sr=sample_rate, mono=True)
        sf.write(output_path, ref_audio, sample_rate)
    return str(output_path)


def write_index(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Demo Samples",
        "",
        "| Run | Sample | Speaker | Emotion | Generated WAV | Reference |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        generated = row.get("generated_relpath") or row["generated_path"]
        reference = row.get("reference_relpath") or row.get("reference_path") or ""
        lines.append(
            f"| {row['run']} | {row['sample_name']} | {row.get('speaker_id', '')} | "
            f"{row.get('emotion', '')} | {generated} | {reference} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def export_for_checkpoint(
    checkpoint: Path,
    selected_entries: list[tuple[int, dict]],
    manifest_path: Path,
    output_dir: Path,
    device: torch.device,
    guidance_scale: float,
    copy_reference: bool,
    save_mel: bool,
    vocoder_checkpoint: str | None = None,
    vocoder_config: str | None = None,
    mel_scale: str = "db",
) -> list[dict]:
    name = run_name(checkpoint)
    run_dir = output_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[export] Loading {checkpoint}")
    model, cfg = load_model(str(checkpoint), device)
    sample_rate = get_sample_rate(cfg)

    # Override vocoder if provided
    if vocoder_checkpoint:
        from models.vocoder import HiFiGANVocoder
        mel_dim = cfg.get("melgen", {}).get("mel_dim", cfg.get("data", {}).get("n_mels", 80))
        model.vocoder = HiFiGANVocoder(
            mel_dim=mel_dim,
            checkpoint_path=vocoder_checkpoint,
            config_path=vocoder_config,
            mel_scale=mel_scale,
        ).to(device)

    rows = []
    for local_idx, (manifest_idx, entry) in enumerate(selected_entries):
        base_name = f"{local_idx:03d}_{sample_name(entry, manifest_idx)}"
        generated_path = run_dir / f"{base_name}.wav"
        mel_path = run_dir / f"{base_name}_mel.npy"
        reference_path = run_dir / f"{base_name}_ref.wav"

        print(f"  [{local_idx + 1}/{len(selected_entries)}] {base_name}")
        lip_video, face_crop, mel_length = load_preprocessed_entry(entry, manifest_path, cfg, device)
        with torch.no_grad():
            audio, mel = model.generate(
                lip_video=lip_video,
                face_crop=face_crop,
                speaker_id=speaker_tensor(model, entry, device),
                mel_length=mel_length,
                guidance_scale=guidance_scale,
            )

        audio_np = audio.squeeze(0).cpu().numpy()
        sf.write(generated_path, audio_np, sample_rate)
        saved_mel = None
        if save_mel:
            np.save(mel_path, mel.squeeze(0).cpu().numpy().astype(np.float32))
            saved_mel = str(mel_path)

        ref = maybe_export_reference(entry, manifest_path, reference_path, sample_rate, copy_reference)
        generated_relpath = generated_path.relative_to(output_dir).as_posix()
        reference_relpath = None
        if ref is not None:
            try:
                reference_relpath = Path(ref).resolve().relative_to(output_dir).as_posix()
            except ValueError:
                reference_relpath = None
        mel_relpath = mel_path.relative_to(output_dir).as_posix() if saved_mel else None
        rows.append({
            "run": name,
            "checkpoint": str(checkpoint),
            "manifest_index": manifest_idx,
            "sample_name": base_name,
            "speaker_id": entry.get("speaker_id"),
            "emotion": entry.get("emotion"),
            "intensity": entry.get("intensity"),
            "utterance_id": entry.get("utterance_id"),
            "generated_path": str(generated_path),
            "generated_relpath": generated_relpath,
            "reference_path": ref,
            "reference_relpath": reference_relpath,
            "mel_path": saved_mel,
            "mel_relpath": mel_relpath,
            "sample_rate": sample_rate,
            "duration_sec": float(len(audio_np) / sample_rate),
            "mel_length": mel_length,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Export demo WAV samples from EmotiLip checkpoints")
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=PROJECT_DIR / "demo_output" / "samples")
    parser.add_argument("--max_samples", type=int, default=8)
    parser.add_argument("--emotions", nargs="*", default=None)
    parser.add_argument("--speakers", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--no_balance", action="store_true", help="Disable round-robin emotion balancing")
    parser.add_argument("--copy_reference", action="store_true", help="Copy/resample reference audio when available")
    parser.add_argument("--save_mel", action="store_true", help="Also save generated mel arrays")
    parser.add_argument("--vocoder_checkpoint", type=str, default=None, help="Override vocoder checkpoint path.")
    parser.add_argument("--vocoder_config", type=str, default=None, help="Override vocoder config.json path.")
    parser.add_argument("--mel_scale", type=str, default="db", choices=["db", "log"],
                        help="Mel scale fed to vocoder: 'db' for our preprocess_mead output, 'log' for native HiFi-GAN.")
    args = parser.parse_args()

    if args.max_samples <= 0:
        raise ValueError("--max_samples must be positive")

    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_manifest(manifest_path)
    selected_entries = choose_entries(
        manifest,
        emotions=set(args.emotions) if args.emotions else None,
        speakers=set(args.speakers) if args.speakers else None,
        max_samples=args.max_samples,
        seed=args.seed,
        balanced=not args.no_balance,
    )
    print(f"[export] Selected {len(selected_entries)} manifest entries")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for checkpoint in args.checkpoints:
        all_rows.extend(export_for_checkpoint(
            checkpoint.expanduser().resolve(),
            selected_entries,
            manifest_path,
            output_dir,
            device,
            guidance_scale=args.guidance_scale,
            copy_reference=args.copy_reference,
            save_mel=args.save_mel,
            vocoder_checkpoint=args.vocoder_checkpoint,
            vocoder_config=args.vocoder_config,
            mel_scale=args.mel_scale,
        ))

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=False)
    write_index(output_dir / "index.md", all_rows)
    print(f"[export] Wrote metadata: {metadata_path}")
    print(f"[export] Wrote index: {output_dir / 'index.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
