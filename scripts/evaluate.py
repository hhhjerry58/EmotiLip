"""
Evaluation Script.

Runs all metrics on generated samples vs ground truth.

Usage:
    python evaluate.py --checkpoint best.pt --test_manifest data/test/manifest.json
"""

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SER_MODEL_NAME = "iic/emotion2vec_plus_base"
REFERENCE_TEXT_FIELDS = ("reference_text", "transcript", "text", "sentence", "utterance_text")


def resolve_entry_path(path_value: str | None, manifest_path: Path) -> str | None:
    """Resolve absolute or project-relative paths stored in a manifest entry."""
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return str(path)

    for candidate in (PROJECT_DIR / path, manifest_path.parent / path, Path.cwd() / path):
        if candidate.exists():
            return str(candidate)
    return str(PROJECT_DIR / path)


def compute_mel_metrics(reference_mel: np.ndarray, generated_mel: np.ndarray) -> dict[str, float | int]:
    """Compute dependency-free mel reconstruction proxies on overlapping bins/frames."""
    ref = np.asarray(reference_mel, dtype=np.float32)
    gen = np.asarray(generated_mel, dtype=np.float32)
    if ref.ndim != 2 or gen.ndim != 2:
        raise ValueError(f"Expected 2D mel arrays, got ref={ref.shape}, gen={gen.shape}")

    mel_bins = min(ref.shape[0], gen.shape[0])
    frames = min(ref.shape[1], gen.shape[1])
    if mel_bins <= 0 or frames <= 0:
        raise ValueError(f"Empty mel overlap: ref={ref.shape}, gen={gen.shape}")

    diff = gen[:mel_bins, :frames] - ref[:mel_bins, :frames]
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    return {
        "mel_mse": mse,
        "mel_mae": mae,
        "mel_rmse": float(np.sqrt(mse)),
        "mel_bins_compared": int(mel_bins),
        "mel_frames_compared": int(frames),
    }


def finite_number(value) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def reference_text_from_entry(entry: dict) -> str | None:
    """Return the first non-empty transcript-like field from a manifest row."""
    for key in REFERENCE_TEXT_FIELDS:
        value = entry.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def summarize_metrics(rows: list[dict], metric_names: list[str]) -> dict[str, dict[str, float | int]]:
    summary = {}
    for name in metric_names:
        values = [finite_number(row.get(name)) for row in rows]
        values = [value for value in values if value is not None]
        if values:
            summary[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "count": len(values),
            }
        else:
            summary[name] = {"mean": None, "std": None, "count": 0}
    return summary


def summarize_prosody(rows: list[dict]) -> dict[str, dict[str, float | int]]:
    by_emotion: dict[str, list[dict]] = {}
    for row in rows:
        prosody = row.get("prosody")
        if isinstance(prosody, dict):
            by_emotion.setdefault(str(row.get("emotion", "unknown")), []).append(prosody)

    summary = {}
    for emotion, items in sorted(by_emotion.items()):
        emotion_summary = {"count": len(items)}
        for key in ("f0_mean", "f0_std", "f0_range", "energy_mean", "energy_std", "duration"):
            values = [finite_number(item.get(key)) for item in items]
            values = [value for value in values if value is not None]
            if values:
                emotion_summary[f"{key}_mean"] = float(np.mean(values))
                emotion_summary[f"{key}_std"] = float(np.std(values))
        summary[emotion] = emotion_summary
    return summary


def filter_manifest(
    manifest: list[dict],
    emotions: set[str] | None,
    speakers: set[str] | None,
) -> list[tuple[int, dict]]:
    """Filter manifest entries while preserving original manifest indices."""
    selected = []
    for idx, entry in enumerate(manifest):
        if emotions is not None and str(entry.get("emotion")) not in emotions:
            continue
        if speakers is not None and str(entry.get("speaker_id")) not in speakers:
            continue
        selected.append((idx, entry))
    return selected


def choose_eval_entries(
    manifest: list[dict],
    *,
    max_samples: int,
    seed: int,
    emotions: set[str] | None = None,
    speakers: set[str] | None = None,
    balanced: bool = True,
) -> list[tuple[int, dict]]:
    """Choose evaluation entries with deterministic optional emotion balancing."""
    candidates = filter_manifest(manifest, emotions=emotions, speakers=speakers)
    if not candidates:
        raise ValueError("No manifest entries match the requested evaluation filters.")

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate EmotiLip")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_manifest", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="eval_output")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--emotions", nargs="*", default=None, help="Optional emotion labels to include in evaluation.")
    parser.add_argument("--speakers", nargs="*", default=None, help="Optional speaker IDs to include in evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic evaluation sample selection.")
    parser.add_argument("--no_balance", action="store_true", help="Disable round-robin emotion balancing for sample selection.")
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--vocoder_checkpoint", type=str, default=None, help="Override vocoder checkpoint path.")
    parser.add_argument("--vocoder_config", type=str, default=None, help="Override vocoder config.json path.")
    parser.add_argument("--mel_scale", type=str, default="db", choices=["db", "log"],
                        help="Mel scale fed to vocoder: 'db' for our preprocess_mead output, 'log' for native HiFi-GAN.")
    parser.add_argument(
        "--run_ser_eval",
        action="store_true",
        help="Run optional emotion2vec SER evaluation on generated WAVs.",
    )
    parser.add_argument(
        "--ser_model",
        type=str,
        default=DEFAULT_SER_MODEL_NAME,
        help="FunASR emotion2vec model name used when --run_ser_eval is set.",
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        default="base",
        help="Whisper model size/name used for optional WER when manifest rows include reference text.",
    )
    args = parser.parse_args()

    if args.max_samples <= 0:
        raise ValueError("--max_samples must be positive")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    from scripts.inference import estimate_mel_length, get_sample_rate, load_model, speaker_tensor_from_entry
    from evaluation.metrics import MetricsComputer
    from evaluation.emotion_eval import compute_prosody_stats

    print("Loading model...")
    model, cfg = load_model(args.checkpoint, device)

    # Override vocoder if specified
    if args.vocoder_checkpoint:
        from models.vocoder import HiFiGANVocoder
        vocoder_cfg = cfg.get("vocoder", {})
        mel_dim = cfg.get("melgen", {}).get("mel_dim", cfg.get("data", {}).get("n_mels", 80))
        model.vocoder = HiFiGANVocoder(
            mel_dim=mel_dim,
            checkpoint_path=args.vocoder_checkpoint,
            config_path=args.vocoder_config,
            mel_scale=args.mel_scale,
        ).to(device)

    sample_rate = get_sample_rate(cfg)

    test_manifest = Path(args.test_manifest).expanduser().resolve()
    with open(test_manifest) as f:
        manifest = json.load(f)

    selected_entries = choose_eval_entries(
        manifest,
        max_samples=args.max_samples,
        seed=args.seed,
        emotions=set(args.emotions) if args.emotions else None,
        speakers=set(args.speakers) if args.speakers else None,
        balanced=not args.no_balance,
    )
    if not selected_entries:
        raise ValueError(f"No samples found in {test_manifest}")
    selection_counts = Counter(str(entry.get("emotion", "unknown")) for _, entry in selected_entries)
    print(
        f"Selected {len(selected_entries)} / {len(manifest)} samples "
        f"(balanced={not args.no_balance}, emotions={dict(sorted(selection_counts.items()))})"
    )
    metrics_computer = MetricsComputer(whisper_model_size=args.whisper_model)

    all_results = []
    failed_samples = 0

    for eval_index, (manifest_index, entry) in enumerate(selected_entries):
        print(f"[{eval_index+1}/{len(selected_entries)}] {entry.get('sample_id', entry.get('utterance_id', manifest_index))}")

        try:
            # Load preprocessed data
            lip = np.load(resolve_entry_path(entry["lip_path"], test_manifest)).astype(np.float32)
            lip_tensor = torch.from_numpy(lip).unsqueeze(0).unsqueeze(0).to(device)

            face = np.load(resolve_entry_path(entry["face_path"], test_manifest)).astype(np.float32)
            face_tensor = torch.from_numpy(face).unsqueeze(0).to(device)

            mel_length = estimate_mel_length(cfg, lip_tensor.shape[2])
            mel_path = resolve_entry_path(entry.get("mel_path"), test_manifest)
            if mel_path:
                mel_length = int(np.load(mel_path, mmap_mode="r").shape[-1])

            speaker_id = speaker_tensor_from_entry(model, entry, device, source=f"eval row {manifest_index}")

            # Generate
            with torch.no_grad():
                audio, mel = model.generate(
                    lip_video=lip_tensor,
                    face_crop=face_tensor,
                    speaker_id=speaker_id,
                    mel_length=mel_length,
                    guidance_scale=args.guidance_scale,
                )

            gen_audio = audio.squeeze(0).cpu().numpy()
            gen_relpath = f"gen_{eval_index:04d}.wav"
            gen_path = str(output_dir / gen_relpath)
            sf.write(gen_path, gen_audio, sample_rate)

            # Load reference audio for comparison
            result = {
                "index": eval_index,
                "manifest_index": manifest_index,
                "sample_id": entry.get("sample_id", entry.get("utterance_id", str(manifest_index))),
                "speaker_id": entry.get("speaker_id"),
                "utterance_id": entry.get("utterance_id"),
                "emotion": entry.get("emotion", "unknown"),
                "generated_path": gen_path,
                "generated_relpath": gen_relpath,
                "sample_rate": sample_rate,
                "duration_sec": float(len(gen_audio) / sample_rate),
                "mel_length": mel_length,
            }

            if mel_path:
                result["reference_mel_path"] = mel_path
                ref_mel = np.load(mel_path).astype(np.float32)
                gen_mel = mel.squeeze(0).cpu().numpy().astype(np.float32)
                result.update(compute_mel_metrics(ref_mel, gen_mel))

            if entry.get("mel_path"):
                # Try to get reference audio path
                ref_audio_path = resolve_entry_path(entry.get("audio_path"), test_manifest)
                if ref_audio_path:
                    result["reference_audio_path"] = ref_audio_path
                    reference_text = reference_text_from_entry(entry)
                    if reference_text:
                        result["reference_text"] = reference_text
                    import librosa
                    ref_audio, _ = librosa.load(ref_audio_path, sr=sample_rate, mono=True)
                    metrics = metrics_computer.compute_all(
                        ref_audio, gen_audio,
                        reference_text=reference_text,
                        ref_audio_path=ref_audio_path,
                        gen_audio_path=gen_path,
                        sample_rate=sample_rate,
                    )
                    result.update(metrics)

            # Prosody stats
            try:
                prosody = compute_prosody_stats(gen_path)
                result["prosody"] = prosody
            except Exception:
                pass

            all_results.append(result)

        except Exception as e:
            failed_samples += 1
            print(f"  [ERROR] {e}")

    ser_summary = {
        "status": "not_requested",
        "ser_model": args.ser_model,
        "emotion_accuracy": None,
        "emotion_f1": None,
        "failed_samples": None,
    }
    if args.run_ser_eval:
        try:
            from evaluation.emotion_eval import evaluate_emotion_batch

            ser_summary = evaluate_emotion_batch(
                [str(row["generated_path"]) for row in all_results],
                [str(row.get("emotion", "unknown")) for row in all_results],
                model_name=args.ser_model,
            )
            for row, target, prediction, raw_prediction in zip(
                all_results,
                ser_summary.get("targets", []),
                ser_summary.get("predictions", []),
                ser_summary.get("raw_predictions", []),
            ):
                row["ser_target"] = target
                row["ser_prediction"] = prediction
                row["ser_prediction_raw"] = raw_prediction
        except Exception as exc:
            ser_summary = {
                "status": "failed",
                "ser_model": args.ser_model,
                "emotion_accuracy": None,
                "emotion_f1": None,
                "failed_samples": len(all_results),
                "error": str(exc),
            }

    # Aggregate results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    if failed_samples:
        print(f"Failed samples: {failed_samples}/{len(selected_entries)}")

    metric_names = ["mel_mse", "mel_mae", "mel_rmse", "pesq", "estoi", "mcd", "wer", "secs"]
    metric_summary = summarize_metrics(all_results, metric_names)
    for name in metric_names:
        item = metric_summary[name]
        if item["count"]:
            print(f"  {name:>8s}: {item['mean']:.4f} ± {item['std']:.4f}")

    if args.run_ser_eval:
        status = ser_summary.get("status", "unknown")
        accuracy = finite_number(ser_summary.get("emotion_accuracy"))
        f1 = finite_number(ser_summary.get("emotion_f1"))
        if accuracy is not None and f1 is not None:
            print(f"  SER emotion: acc={accuracy:.4f}, f1={f1:.4f}, status={status}")
        else:
            print(f"  SER emotion: unavailable, status={status}")

    # Per-emotion prosody
    emotions = set(r["emotion"] for r in all_results)
    print("\nPer-emotion prosody:")
    prosody_summary = summarize_prosody(all_results)
    for emo in sorted(emotions):
        emo_results = [r for r in all_results if r["emotion"] == emo and "prosody" in r]
        if emo_results:
            f0_means = [r["prosody"]["f0_mean"] for r in emo_results]
            energy_means = [r["prosody"]["energy_mean"] for r in emo_results]
            print(f"  {emo:>12s}: F0={np.mean(f0_means):.1f}±{np.std(f0_means):.1f} "
                  f"Energy={np.mean(energy_means):.1f}±{np.std(energy_means):.1f}")

    # Save detailed results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDetailed results: {results_path}")

    summary = {
        "checkpoint": str(Path(args.checkpoint).expanduser()),
        "test_manifest": str(test_manifest),
        "output_dir": str(output_dir),
        "sample_rate": sample_rate,
        "guidance_scale": args.guidance_scale,
        "whisper_model": args.whisper_model,
        "manifest_samples": len(manifest),
        "samples_requested": len(selected_entries),
        "samples_evaluated": len(all_results),
        "failed_samples": failed_samples,
        "selection": {
            "balanced": not args.no_balance,
            "seed": args.seed,
            "emotions_filter": args.emotions or [],
            "speakers_filter": args.speakers or [],
            "selected_emotion_counts": dict(sorted(selection_counts.items())),
            "selected_manifest_indices": [idx for idx, _ in selected_entries],
        },
        "metrics": metric_summary,
        "per_emotion_prosody": prosody_summary,
        "ser": ser_summary,
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary: {summary_path}")

    if not all_results:
        print("No samples were evaluated successfully.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
