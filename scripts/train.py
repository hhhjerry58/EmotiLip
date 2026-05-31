"""
EmotiLip Training Script.

Supports staged training:
  Stage 1: Baseline L2S (no emotion conditioning)
  Stage 2: + Emotion conditioning
  Stage 3: + Prosody predictor
  Stage 4: + mel-level proxy emotion consistency loss

Usage:
    python train.py --config configs/emotion_film.yaml
    python train.py --config configs/base.yaml --stage 1
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.emotilip import EmotiLip
from data.dataset import EmotiLipDataset, collate_fn


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    """Seed DataLoader workers from PyTorch's worker seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    import yaml
    config_path = Path(config_path).expanduser().resolve()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg["_config_path"] = str(config_path)
    cfg["_project_dir"] = str(
        config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    )
    return cfg


def resolve_project_path(cfg: dict, path_value: str | None) -> str | None:
    """Resolve config paths relative to the project root, not the caller cwd."""
    if path_value is None:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return str(path)
    return str(Path(cfg.get("_project_dir", ".")).expanduser().resolve() / path)


def apply_stage_overrides(cfg: dict, stage: int | None) -> None:
    """Apply coarse training-stage switches from the project roadmap."""
    if stage is None:
        return
    if stage == 1:
        cfg["use_emotion"] = False
        cfg["use_prosody"] = False
        cfg["use_emotion_consistency"] = False
        cfg["lambda_prosody"] = 0.0
    elif stage == 2:
        cfg["use_emotion"] = True
        cfg["use_prosody"] = False
        cfg["use_emotion_consistency"] = False
        cfg["lambda_prosody"] = 0.0
    elif stage == 3:
        cfg["use_emotion"] = True
        cfg["use_prosody"] = True
        cfg["use_emotion_consistency"] = False
    elif stage == 4:
        cfg["use_emotion"] = True
        cfg["use_prosody"] = True
        cfg["use_emotion_consistency"] = True
        if cfg.get("lambda_emotion_consistency", 0.0) <= 0:
            cfg["lambda_emotion_consistency"] = 0.1
    else:
        raise ValueError("Stage must be 1, 2, 3, or 4.")


def resolve_checkpoint_paths(cfg: dict) -> None:
    """Resolve optional model checkpoint paths stored inside config sections."""
    for section in ("visual_encoder", "emotion_encoder", "vocoder"):
        section_cfg = cfg.get(section, {})
        for key in ("checkpoint_path", "config_path"):
            if section_cfg.get(key):
                section_cfg[key] = resolve_project_path(cfg, section_cfg[key])


def apply_model_asset_overrides(cfg: dict, args: argparse.Namespace) -> None:
    """Apply CLI overrides for external model assets before path resolution."""
    if args.emotion_encoder_backend is not None:
        cfg.setdefault("emotion_encoder", {})["backend"] = args.emotion_encoder_backend
    if args.emonet_checkpoint is not None:
        emotion_cfg = cfg.setdefault("emotion_encoder", {})
        emotion_cfg["backend"] = "emonet"
        emotion_cfg["checkpoint_path"] = args.emonet_checkpoint
    if args.vocoder_checkpoint is not None:
        cfg.setdefault("vocoder", {})["checkpoint_path"] = args.vocoder_checkpoint
    if args.vocoder_config is not None:
        cfg.setdefault("vocoder", {})["config_path"] = args.vocoder_config


def require_nonempty_split(dataset: EmotiLipDataset, split_name: str, manifest_path: str) -> None:
    """Fail early with a useful message when preprocessing produced no usable samples."""
    if len(dataset) == 0:
        raise ValueError(
            f"{split_name} dataset has 0 usable samples after filtering: {manifest_path}. "
            "Check that each manifest entry has lip_path, face_path, mel_path, and prosody_path."
        )


def build_model(cfg: dict) -> EmotiLip:
    """Build EmotiLip model from config."""
    vocoder_cfg = dict(cfg.get("vocoder", {}))
    vocoder_cfg.setdefault(
        "mel_dim",
        cfg.get("melgen", {}).get("mel_dim", cfg.get("data", {}).get("n_mels", 80)),
    )
    return EmotiLip(
        visual_cfg=cfg.get("visual_encoder", {}),
        emotion_cfg=cfg.get("emotion_encoder", {}),
        prosody_cfg=cfg.get("prosody_predictor", {}),
        melgen_cfg=cfg.get("melgen", {}),
        vocoder_cfg=vocoder_cfg,
        speaker_dim=cfg.get("speaker_dim", 256),
        num_speakers=cfg.get("num_speakers", 60),
        lambda_prosody=cfg.get("lambda_prosody", 1.0),
        lambda_emotion_consistency=cfg.get("lambda_emotion_consistency", 0.0),
        emotion_consistency_cfg=cfg.get("emotion_consistency", {}),
        load_vocoder=cfg.get("load_vocoder", False),
        use_emotion=cfg.get("use_emotion", True),
        use_prosody=cfg.get("use_prosody", True),
        use_emotion_consistency=cfg.get("use_emotion_consistency", False),
    )


def load_model_state_or_fail(model: nn.Module, state_dict: dict, checkpoint_path: str | Path) -> None:
    """Load model weights and fail on architecture/checkpoint mismatches."""
    try:
        incompatible = model.load_state_dict(state_dict, strict=False)
    except Exception as exc:
        raise RuntimeError(f"Could not load model state from {checkpoint_path}: {exc}") from exc

    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        details = []
        if missing:
            shown = ", ".join(missing[:5])
            extra = f"; {len(missing) - 5} more" if len(missing) > 5 else ""
            details.append(f"missing keys: {shown}{extra}")
        if unexpected:
            shown = ", ".join(unexpected[:5])
            extra = f"; {len(unexpected) - 5} more" if len(unexpected) > 5 else ""
            details.append(f"unexpected keys: {shown}{extra}")
        raise RuntimeError(
            f"Checkpoint/model architecture mismatch for {checkpoint_path}: "
            + "; ".join(details)
        )


def build_loader(
    dataset: EmotiLipDataset,
    cfg: dict,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    worker_init_fn=None,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """Build a DataLoader with multiprocessing options guarded for num_workers=0."""
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": pin_memory,
    }
    if worker_init_fn is not None and num_workers > 0:
        kwargs["worker_init_fn"] = worker_init_fn
    if generator is not None:
        kwargs["generator"] = generator
    if num_workers > 0:
        kwargs["persistent_workers"] = cfg["training"].get("persistent_workers", False)
        if "prefetch_factor" in cfg["training"]:
            kwargs["prefetch_factor"] = cfg["training"]["prefetch_factor"]
    return DataLoader(dataset, **kwargs)


def build_grad_scaler(enabled: bool) -> GradScaler:
    """Create a GradScaler without deprecated torch.cuda.amp imports."""
    try:
        return GradScaler("cuda", enabled=enabled)
    except TypeError:
        return GradScaler(enabled=enabled)


def append_jsonl(path: Path, record: dict) -> None:
    """Append one JSON record per line for easy plotting/report parsing."""
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    """Load existing JSONL history when resuming a run."""
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_train_summary(path: Path, summary: dict) -> None:
    """Write the latest compact run summary."""
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def train_one_epoch(
    model: EmotiLip,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    use_amp: bool = True,
    log_interval: int = 50,
    max_batches: int | None = None,
) -> dict:
    """Train for one epoch. Returns average losses."""
    model.train()
    total_loss = 0
    total_diff_loss = 0
    total_pros_loss = 0
    total_emo_cons_loss = 0
    total_emo_cls_loss = 0
    n_batches = 0

    for i, batch in enumerate(dataloader):
        lip_video = batch["lip_video"].to(device, non_blocking=True)
        face_crop = batch["face_crop"].to(device, non_blocking=True)
        mel = batch["mel"].to(device, non_blocking=True)
        pitch = batch["pitch"].to(device, non_blocking=True)
        energy = batch["energy"].to(device, non_blocking=True)
        speaker_id = batch["speaker_id"].to(device, non_blocking=True)
        emotion_label = batch["emotion_label"].to(device, non_blocking=True)
        mel_mask = batch.get("mel_mask")
        if mel_mask is not None:
            mel_mask = mel_mask.to(device, non_blocking=True)

        optimizer.zero_grad()

        amp_enabled = use_amp and device.type == "cuda"
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            output = model(
                lip_video=lip_video,
                face_crop=face_crop,
                mel_target=mel,
                speaker_id=speaker_id,
                pitch_target=pitch,
                energy_target=energy,
                mel_mask=mel_mask,
                emotion_label=emotion_label,
            )
            loss = output.total_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_diff_loss += output.diffusion_loss.item()
        if output.prosody_loss is not None:
            total_pros_loss += output.prosody_loss.item()
        if output.emotion_consistency_loss is not None:
            total_emo_cons_loss += output.emotion_consistency_loss.item()
        if output.emotion_classifier_loss is not None:
            total_emo_cls_loss += output.emotion_classifier_loss.item()
        n_batches += 1

        if (i + 1) % log_interval == 0:
            avg = total_loss / n_batches
            print(f"  Epoch {epoch} [{i+1}/{len(dataloader)}] "
                  f"loss={avg:.4f} diff={output.diffusion_loss.item():.4f} "
                  f"pros={output.prosody_loss.item() if output.prosody_loss is not None else 0:.4f} "
                  f"emo={output.emotion_consistency_loss.item() if output.emotion_consistency_loss is not None else 0:.4f}")

        if max_batches is not None and n_batches >= max_batches:
            break

    return {
        "total_loss": total_loss / max(n_batches, 1),
        "diffusion_loss": total_diff_loss / max(n_batches, 1),
        "prosody_loss": total_pros_loss / max(n_batches, 1),
        "emotion_consistency_loss": total_emo_cons_loss / max(n_batches, 1),
        "emotion_classifier_loss": total_emo_cls_loss / max(n_batches, 1),
        "batches": n_batches,
    }


@torch.no_grad()
def validate(
    model: EmotiLip,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict:
    """Validate on held-out set."""
    model.eval()
    total_loss = 0
    total_emo_cons_loss = 0
    total_emo_cls_loss = 0
    n_batches = 0

    for batch in dataloader:
        lip_video = batch["lip_video"].to(device, non_blocking=True)
        face_crop = batch["face_crop"].to(device, non_blocking=True)
        mel = batch["mel"].to(device, non_blocking=True)
        pitch = batch["pitch"].to(device, non_blocking=True)
        energy = batch["energy"].to(device, non_blocking=True)
        speaker_id = batch["speaker_id"].to(device, non_blocking=True)
        emotion_label = batch["emotion_label"].to(device, non_blocking=True)
        mel_mask = batch.get("mel_mask")
        if mel_mask is not None:
            mel_mask = mel_mask.to(device, non_blocking=True)

        output = model(
            lip_video=lip_video,
            face_crop=face_crop,
            mel_target=mel,
            speaker_id=speaker_id,
            pitch_target=pitch,
            energy_target=energy,
            mel_mask=mel_mask,
            emotion_label=emotion_label,
        )
        total_loss += output.total_loss.item()
        if output.emotion_consistency_loss is not None:
            total_emo_cons_loss += output.emotion_consistency_loss.item()
        if output.emotion_classifier_loss is not None:
            total_emo_cls_loss += output.emotion_classifier_loss.item()
        n_batches += 1

        if max_batches is not None and n_batches >= max_batches:
            break

    return {
        "val_loss": total_loss / max(n_batches, 1),
        "emotion_consistency_loss": total_emo_cons_loss / max(n_batches, 1),
        "emotion_classifier_loss": total_emo_cls_loss / max(n_batches, 1),
        "batches": n_batches,
    }


def main():
    parser = argparse.ArgumentParser(description="Train EmotiLip")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs")
    parser.add_argument("--output_dir", type=str, default=None, help="Override training.output_dir")
    parser.add_argument("--train_manifest", type=str, default=None, help="Override data.train_manifest")
    parser.add_argument("--val_manifest", type=str, default=None, help="Override data.val_manifest")
    parser.add_argument("--test_manifest", type=str, default=None, help="Override data.test_manifest")
    parser.add_argument("--num_workers", type=int, default=None, help="Override training.num_workers")
    parser.add_argument("--val_num_workers", type=int, default=None, help="Override training.val_num_workers")
    parser.add_argument("--max_train_batches", type=int, default=None, help="Debug: limit train batches per epoch")
    parser.add_argument("--max_val_batches", type=int, default=None, help="Debug: limit val batches per epoch")
    parser.add_argument(
        "--emotion_encoder_backend",
        choices=["standin", "emonet"],
        default=None,
        help="Override emotion_encoder.backend; use emonet for reportable emotion-conditioned runs.",
    )
    parser.add_argument(
        "--emonet_checkpoint",
        type=str,
        default=None,
        help="Override emotion_encoder.checkpoint_path and set backend=emonet.",
    )
    parser.add_argument(
        "--vocoder_checkpoint",
        type=str,
        default=None,
        help="Override vocoder.checkpoint_path for inference/evaluation/demo generation.",
    )
    parser.add_argument(
        "--vocoder_config",
        type=str,
        default=None,
        help="Override vocoder.config_path for HiFi-GAN-compatible configs.",
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="Override config for staged training: 1=base, 2=emotion, 3=emotion+prosody, 4=+mel emotion proxy",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_stage_overrides(cfg, args.stage)
    apply_model_asset_overrides(cfg, args)
    resolve_checkpoint_paths(cfg)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.output_dir is not None:
        cfg["training"]["output_dir"] = args.output_dir
    if args.train_manifest is not None:
        cfg["data"]["train_manifest"] = args.train_manifest
    if args.val_manifest is not None:
        cfg["data"]["val_manifest"] = args.val_manifest
    if args.test_manifest is not None:
        cfg["data"]["test_manifest"] = args.test_manifest
    if args.num_workers is not None:
        cfg["training"]["num_workers"] = args.num_workers
    if args.val_num_workers is not None:
        cfg["training"]["val_num_workers"] = args.val_num_workers
    seed = cfg["training"].get("seed", 42)
    set_seed(seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    project_dir = Path(cfg["_project_dir"])

    # Data
    train_manifest = resolve_project_path(cfg, cfg["data"]["train_manifest"])
    val_manifest = resolve_project_path(cfg, cfg["data"]["val_manifest"])
    train_dataset = EmotiLipDataset(
        train_manifest,
        max_mel_len=cfg["data"].get("max_mel_len", 400),
        augment=True,
        path_root=project_dir,
    )
    val_dataset = EmotiLipDataset(
        val_manifest,
        max_mel_len=cfg["data"].get("max_mel_len", 400),
        augment=False,
        speaker_to_idx=train_dataset.speaker_to_idx,
        path_root=project_dir,
    )
    require_nonempty_split(train_dataset, "train", train_manifest)
    require_nonempty_split(val_dataset, "val", val_manifest)
    if cfg.get("num_speakers", 0) > 0:
        cfg["num_speakers"] = max(cfg.get("num_speakers", 0), train_dataset.num_mapped_speakers)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    train_num_workers = cfg["training"].get("num_workers", 4)
    val_num_workers = cfg["training"].get("val_num_workers", train_num_workers)

    train_loader = build_loader(
        train_dataset,
        cfg,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=train_num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )
    val_loader = build_loader(
        val_dataset,
        cfg,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=val_num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
    )

    # Model
    model = build_model(cfg).to(device)
    model_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Model parameters: {model_parameters:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"].get("weight_decay", 0.01),
    )

    # LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
        eta_min=cfg["training"].get("lr_min", 1e-6),
    )

    amp_enabled = cfg["training"].get("amp", True) and device.type == "cuda"
    scaler = build_grad_scaler(enabled=amp_enabled)

    # Resume
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        load_model_state_or_fail(model, ckpt["model"], args.resume)
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
            if hasattr(scheduler, "T_max"):
                scheduler.T_max = cfg["training"]["epochs"]
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        print(f"[train] Resumed from epoch {start_epoch}")

    # Output dir
    output_dir = Path(resolve_project_path(cfg, cfg["training"]["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "train_summary.json"
    if start_epoch == 0 and history_path.exists():
        history_path.unlink()
    history = load_jsonl(history_path)
    try:
        import yaml
        with open(output_dir / "run_config.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    except Exception as exc:
        print(f"[train] Warning: failed to write run_config.yaml: {exc}")
    with open(output_dir / "speaker_to_idx.json", "w") as f:
        json.dump(train_dataset.speaker_to_idx, f, indent=2, ensure_ascii=False)

    # Training loop
    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        t0 = time.time()
        lr_before = optimizer.param_groups[0]["lr"]
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch,
            use_amp=amp_enabled,
            log_interval=cfg["training"].get("log_interval", 50),
            max_batches=args.max_train_batches or cfg["training"].get("max_train_batches"),
        )
        val_metrics = validate(
            model,
            val_loader,
            device,
            max_batches=args.max_val_batches or cfg["training"].get("max_val_batches"),
        )
        scheduler.step()
        lr_after = scheduler.get_last_lr()[0]

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch}: train_loss={train_metrics['total_loss']:.4f} "
            f"val_loss={val_metrics['val_loss']:.4f} "
            f"lr={lr_after:.2e} "
            f"time={elapsed:.0f}s"
        )

        is_best = val_metrics["val_loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["val_loss"]

        epoch_record = {
            "epoch": epoch,
            "elapsed_sec": elapsed,
            "lr_before_step": lr_before,
            "lr_after_step": lr_after,
            "is_best": is_best,
            "best_val_loss": best_val_loss,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_record)
        append_jsonl(history_path, epoch_record)

        best_record = min(history, key=lambda row: row.get("val", {}).get("val_loss", float("inf")))
        write_train_summary(
            summary_path,
            {
                "config_path": cfg.get("_config_path"),
                "output_dir": str(output_dir),
                "device": str(device),
                "amp_enabled": amp_enabled,
                "epochs_configured": cfg["training"]["epochs"],
                "epochs_completed": len(history),
                "last_epoch": epoch,
                "best_epoch": best_record.get("epoch"),
                "best_val_loss": best_val_loss,
                "model_parameters": model_parameters,
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "num_speakers": train_dataset.num_mapped_speakers,
                "train_manifest": train_manifest,
                "val_manifest": val_manifest,
                "metrics_path": str(history_path),
            },
        )

        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "best_epoch": best_record.get("epoch"),
            "speaker_to_idx": train_dataset.speaker_to_idx,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "epoch_record": epoch_record,
            "config": cfg,
        }
        torch.save(ckpt, output_dir / "last.pt")

        if is_best:
            torch.save(ckpt, output_dir / "best.pt")
            print(f"  -> New best model (val_loss={best_val_loss:.4f})")

    print(f"[train] Done. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
