"""Validate EmotiLip YAML configs without importing torch."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_DIR / "configs"


def positive_number(cfg: dict, path: str, errors: list[str]) -> None:
    value = cfg
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            errors.append(f"missing required key: {path}")
            return
        value = value[part]
    if not isinstance(value, (int, float)) or value <= 0:
        errors.append(f"{path} must be positive, got {value!r}")


def nonnegative_int(cfg: dict, path: str, errors: list[str]) -> None:
    value = cfg
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return
        value = value[part]
    if not isinstance(value, int) or value < 0:
        errors.append(f"{path} must be a non-negative integer, got {value!r}")


def positive_int(cfg: dict, path: str, errors: list[str]) -> None:
    value = cfg
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            errors.append(f"missing required key: {path}")
            return
        value = value[part]
    if not isinstance(value, int) or value <= 0:
        errors.append(f"{path} must be a positive integer, got {value!r}")


def optional_positive_number(cfg: dict, path: str, errors: list[str]) -> None:
    value = cfg
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return
        value = value[part]
    if not isinstance(value, (int, float)) or value <= 0:
        errors.append(f"{path} must be positive when set, got {value!r}")


def optional_nonnegative_number(cfg: dict, path: str, errors: list[str]) -> None:
    value = cfg
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return
        value = value[part]
    if not isinstance(value, (int, float)) or value < 0:
        errors.append(f"{path} must be non-negative when set, got {value!r}")


def one_of(cfg: dict, path: str, choices: set[str], errors: list[str]) -> None:
    value = cfg
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            errors.append(f"missing required key: {path}")
            return
        value = value[part]
    if value not in choices:
        errors.append(f"{path} must be one of {sorted(choices)}, got {value!r}")


def validate_config(path: Path) -> list[str]:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    errors = []
    for section in ("data", "visual_encoder", "emotion_encoder", "melgen", "training"):
        if section not in cfg:
            errors.append(f"missing section: {section}")

    if errors:
        return errors

    positive_int(cfg, "data.sample_rate", errors)
    positive_int(cfg, "data.hop_length", errors)
    positive_number(cfg, "data.video_fps", errors)
    positive_int(cfg, "data.n_mels", errors)
    positive_int(cfg, "data.lip_size", errors)
    positive_int(cfg, "data.face_size", errors)
    positive_int(cfg, "training.batch_size", errors)
    positive_int(cfg, "training.epochs", errors)
    positive_number(cfg, "training.lr", errors)
    optional_positive_number(cfg, "training.lr_min", errors)
    optional_nonnegative_number(cfg, "lambda_emotion_consistency", errors)
    nonnegative_int(cfg, "training.num_workers", errors)
    nonnegative_int(cfg, "training.val_num_workers", errors)
    nonnegative_int(cfg, "training.max_train_batches", errors)
    nonnegative_int(cfg, "training.max_val_batches", errors)
    nonnegative_int(cfg, "speaker_dim", errors)
    nonnegative_int(cfg, "num_speakers", errors)
    one_of(cfg, "visual_encoder.backend", {"resnet3d", "avhubert"}, errors)
    positive_int(cfg, "visual_encoder.output_dim", errors)
    if cfg["visual_encoder"].get("backend") == "resnet3d":
        positive_int(cfg, "visual_encoder.base_channels", errors)
    if cfg["visual_encoder"].get("backend") == "avhubert" and not cfg["visual_encoder"].get("checkpoint_path"):
        errors.append("visual_encoder.checkpoint_path is required when visual_encoder.backend='avhubert'")
    one_of(cfg, "emotion_encoder.backend", {"standin", "emonet"}, errors)
    one_of(cfg, "melgen.schedule", {"linear", "cosine"}, errors)
    one_of(cfg, "melgen.fusion_type", {"film", "adain", "cross_attention"}, errors)

    visual_dim = cfg["visual_encoder"].get("output_dim")
    cond_dim = cfg["melgen"].get("cond_dim")
    if visual_dim != cond_dim:
        errors.append(f"visual_encoder.output_dim ({visual_dim}) must match melgen.cond_dim ({cond_dim})")

    emotion_dim = cfg["emotion_encoder"].get("embed_dim")
    melgen_emotion_dim = cfg["melgen"].get("emotion_dim")
    if emotion_dim != melgen_emotion_dim:
        errors.append(
            f"emotion_encoder.embed_dim ({emotion_dim}) must match "
            f"melgen.emotion_dim ({melgen_emotion_dim})"
        )

    if cfg["melgen"].get("mel_dim") != cfg["data"].get("n_mels"):
        errors.append("melgen.mel_dim must match data.n_mels")

    vocoder_mel_dim = cfg.get("vocoder", {}).get("mel_dim")
    if vocoder_mel_dim is not None and vocoder_mel_dim != cfg["melgen"].get("mel_dim"):
        errors.append("vocoder.mel_dim must match melgen.mel_dim when set")

    prosody_cfg = cfg.get("prosody_predictor", {})
    hidden_dim = prosody_cfg.get("hidden_dim")
    num_heads = prosody_cfg.get("num_heads")
    if hidden_dim is not None and (not isinstance(hidden_dim, int) or hidden_dim < 2):
        errors.append(f"prosody_predictor.hidden_dim must be an integer >= 2, got {hidden_dim!r}")
    if num_heads is not None and (not isinstance(num_heads, int) or num_heads <= 0):
        errors.append(f"prosody_predictor.num_heads must be a positive integer, got {num_heads!r}")
    if isinstance(hidden_dim, int) and isinstance(num_heads, int) and num_heads > 0:
        if hidden_dim % num_heads != 0:
            errors.append("prosody_predictor.hidden_dim must be divisible by prosody_predictor.num_heads")

    hop_length = cfg["data"].get("hop_length")
    sample_rate = cfg["data"].get("sample_rate")
    if isinstance(hop_length, int) and isinstance(sample_rate, int) and hop_length >= sample_rate:
        errors.append("data.hop_length should be smaller than data.sample_rate")

    if cfg.get("speaker_dim", 0) > 0 and cfg.get("num_speakers", 0) <= 0:
        errors.append("num_speakers must be positive when speaker_dim > 0")

    if cfg.get("use_prosody", False) and "prosody_predictor" not in cfg:
        errors.append("use_prosody=true requires a prosody_predictor section")

    emotion_consistency_cfg = cfg.get("emotion_consistency", {})
    if emotion_consistency_cfg:
        positive_int(cfg, "emotion_consistency.num_emotion_classes", errors)
        positive_int(cfg, "emotion_consistency.hidden_dim", errors)
        optional_nonnegative_number(cfg, "emotion_consistency.dropout", errors)
    if cfg.get("use_emotion_consistency", False):
        if not cfg.get("use_emotion", False):
            errors.append("use_emotion_consistency=true requires use_emotion=true")
        if cfg.get("lambda_emotion_consistency", 0.0) <= 0:
            errors.append("use_emotion_consistency=true requires lambda_emotion_consistency > 0")
        if not emotion_consistency_cfg:
            errors.append("use_emotion_consistency=true requires an emotion_consistency section")

    for key in ("train_manifest", "val_manifest", "test_manifest"):
        if not cfg["data"].get(key):
            errors.append(f"data.{key} is required")

    if not cfg["training"].get("output_dir"):
        errors.append("training.output_dir is required")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate EmotiLip config files")
    parser.add_argument("configs", nargs="*", type=Path, default=sorted(CONFIG_DIR.glob("*.yaml")))
    args = parser.parse_args()

    any_errors = False
    for path in args.configs:
        errors = validate_config(path)
        if errors:
            any_errors = True
            print(f"[FAIL] {path}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"[OK]   {path}")

    return 1 if any_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
