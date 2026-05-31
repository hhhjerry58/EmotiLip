"""
EmotiLip Inference Script.

Generate speech from a silent lip video with emotion conditioning.

Usage:
    python inference.py --checkpoint best.pt --video input.mp4 --output output.wav
    python inference.py --checkpoint best.pt --video input.mp4 --emotion angry
"""

import argparse
from pathlib import Path
import tempfile

import numpy as np
import torch
import soundfile as sf

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


EMOTION_VA = {
    "neutral": (0.0, 0.0),
    "happy": (0.8, 0.6),
    "sad": (-0.6, -0.3),
    "angry": (-0.5, 0.8),
    "fear": (-0.6, 0.7),
    "surprised": (0.2, 0.8),
    "disgusted": (-0.7, 0.3),
    "contempt": (-0.3, 0.1),
}


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained EmotiLip model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    from scripts.train import build_model, load_model_state_or_fail
    model = build_model(cfg).to(device)
    load_model_state_or_fail(model, ckpt["model"], checkpoint_path)
    model.speaker_to_idx = ckpt.get("speaker_to_idx", {})
    model.eval()
    return model, cfg


def estimate_mel_length(cfg: dict, num_video_frames: int) -> int:
    """Estimate mel-frame length from video frames using config data rates."""
    data_cfg = cfg.get("data", {})
    sample_rate = data_cfg.get("sample_rate", 16000)
    hop_length = data_cfg.get("hop_length", 256)
    video_fps = data_cfg.get("video_fps", 25)
    mel_fps = sample_rate / hop_length
    return max(1, int(round(num_video_frames * mel_fps / video_fps)))


def get_sample_rate(cfg: dict) -> int:
    """Return the waveform sample rate expected by the checkpoint config."""
    return int(cfg.get("data", {}).get("sample_rate", 16000))


def has_speaker_embedding(model) -> bool:
    return getattr(model, "speaker_embedding", None) is not None


def speaker_index_tensor(model, speaker_index: int, device: torch.device, source: str = "speaker index") -> torch.Tensor:
    """Validate an already-indexed speaker ID before passing it to the model."""
    embedding = getattr(model, "speaker_embedding", None)
    if embedding is None:
        raise ValueError(f"{source} was provided, but this checkpoint has no speaker embedding.")
    if speaker_index < 0 or speaker_index >= embedding.num_embeddings:
        raise ValueError(
            f"{source}={speaker_index} is outside checkpoint speaker embedding range "
            f"0..{embedding.num_embeddings - 1}."
        )
    return torch.tensor([speaker_index], device=device)


def speaker_tensor_from_entry(model, entry: dict, device: torch.device, source: str = "manifest") -> torch.Tensor | None:
    """Map a manifest speaker_id through the checkpoint mapping without silent fallback."""
    speaker_id = entry.get("speaker_id")
    if not has_speaker_embedding(model):
        return None
    if speaker_id in (None, ""):
        raise ValueError(f"{source} entry is missing speaker_id, but the checkpoint uses speaker embeddings.")

    speaker_to_idx = getattr(model, "speaker_to_idx", None)
    if not isinstance(speaker_to_idx, dict) or not speaker_to_idx:
        raise ValueError(
            f"{source} speaker_id={speaker_id!r} cannot be mapped because the checkpoint has no speaker_to_idx metadata."
        )

    candidates = (speaker_id, str(speaker_id))
    for candidate in candidates:
        if candidate in speaker_to_idx:
            return speaker_index_tensor(model, int(speaker_to_idx[candidate]), device, source=f"{source} speaker_id={speaker_id!r}")

    known = sorted(str(key) for key in speaker_to_idx)[:8]
    suffix = "..." if len(speaker_to_idx) > 8 else ""
    raise ValueError(
        f"{source} speaker_id={speaker_id!r} is not present in checkpoint speaker_to_idx. "
        f"Known speakers: {known}{suffix}"
    )


@torch.no_grad()
def build_emotion_override(
    model,
    face_crop: torch.Tensor,
    emotion_name: str | None,
    device: torch.device,
) -> torch.Tensor | None:
    """Create a rough emotion embedding override from a named V-A target."""
    if emotion_name is None:
        return None
    if emotion_name not in EMOTION_VA:
        valid = ", ".join(sorted(EMOTION_VA))
        raise ValueError(f"Unknown emotion '{emotion_name}'. Choose one of: {valid}")
    if not getattr(model, "use_emotion", True):
        raise ValueError("This checkpoint was trained with use_emotion=false; --emotion is unavailable.")

    emotion_out = model.emotion_encoder(face_crop)
    embedding = emotion_out.embedding.clone()
    valence, arousal = EMOTION_VA[emotion_name]

    # This is an interpretable control knob for demos, not a learned V-A decoder.
    if embedding.shape[-1] >= 2:
        embedding[:, 0] = embedding[:, 0] + 2.0 * valence
        embedding[:, 1] = embedding[:, 1] + 2.0 * arousal
    return embedding.to(device)


def preprocess_video(video_path: str, lip_size: int = 96, face_size: int = 256):
    """Extract lip and face crops from input video."""
    from data.preprocess_mead import extract_lip_crop, extract_face_crop

    with tempfile.TemporaryDirectory() as tmpdir:
        lip_path = str(Path(tmpdir) / "lip.npy")
        face_path = str(Path(tmpdir) / "face.npy")
        lip = extract_lip_crop(video_path, lip_path, crop_size=lip_size)
        face = extract_face_crop(video_path, face_path, crop_size=face_size)

    lip_tensor = torch.from_numpy(lip).unsqueeze(0).unsqueeze(0)  # (1, 1, T, H, W)
    face_tensor = torch.from_numpy(face).unsqueeze(0)              # (1, 3, H, W)

    return lip_tensor, face_tensor


def main():
    parser = argparse.ArgumentParser(description="EmotiLip Inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--video", type=str, required=True, help="Input silent video")
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument(
        "--emotion", type=str, default=None,
        choices=sorted(EMOTION_VA),
        help="Optional named emotion override for demos; default uses the face expression.",
    )
    parser.add_argument("--speaker_id", type=int, default=None, help="Optional speaker embedding index for checkpoints trained with speaker embeddings.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {args.checkpoint}...")
    model, cfg = load_model(args.checkpoint, device)

    print(f"Preprocessing video: {args.video}")
    lip_video, face_crop = preprocess_video(args.video)
    lip_video = lip_video.to(device)
    face_crop = face_crop.to(device)
    mel_length = estimate_mel_length(cfg, lip_video.shape[2])
    sample_rate = get_sample_rate(cfg)

    speaker_id = None
    if args.speaker_id is not None:
        speaker_id = speaker_index_tensor(model, args.speaker_id, device, source="--speaker_id")

    print("Generating speech...")
    with torch.no_grad():
        emotion_override = build_emotion_override(model, face_crop, args.emotion, device)
        audio, mel = model.generate(
            lip_video=lip_video,
            face_crop=face_crop,
            speaker_id=speaker_id,
            mel_length=mel_length,
            guidance_scale=args.guidance_scale,
            emotion_override=emotion_override,
        )

    audio_np = audio.squeeze(0).cpu().numpy()
    sf.write(args.output, audio_np, sample_rate)
    print(f"Saved: {args.output} ({len(audio_np) / sample_rate:.2f}s)")


if __name__ == "__main__":
    main()
