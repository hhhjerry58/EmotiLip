"""
Emotion Interpolation Demo.

Generates speech for the same lip video with continuously varying emotion
by interpolating in the Valence-Arousal space.

Usage:
    python demo_interpolation.py --checkpoint best.pt --video input.mp4 \
        --start_emotion neutral --end_emotion angry --steps 5
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Predefined V-A coordinates for basic emotions (approximate)
EMOTION_VA = {
    "neutral":   (0.0, 0.0),
    "happy":     (0.8, 0.6),
    "sad":       (-0.6, -0.3),
    "angry":     (-0.5, 0.8),
    "fear":      (-0.6, 0.7),
    "surprised": (0.2, 0.8),
    "disgusted": (-0.7, 0.3),
    "contempt":  (-0.3, 0.1),
}


def interpolate_emotions(
    start_va: tuple[float, float],
    end_va: tuple[float, float],
    steps: int,
) -> list[tuple[float, float]]:
    """Linear interpolation in V-A space."""
    va_list = []
    for i in range(steps):
        t = i / max(steps - 1, 1)
        v = start_va[0] * (1 - t) + end_va[0] * t
        a = start_va[1] * (1 - t) + end_va[1] * t
        va_list.append((v, a))
    return va_list


def va_to_embedding(
    model,
    face_crop: torch.Tensor,
    target_va: tuple[float, float],
    device: torch.device,
) -> torch.Tensor:
    """
    Create an emotion embedding that corresponds to target V-A values.

    Strategy: extract base embedding from face, then shift it toward
    the target V-A direction. This is a simplified approach —
    a more sophisticated method would train a V-A → embedding decoder.
    """
    emotion_out = model.emotion_encoder(face_crop)
    base_emb = emotion_out.embedding  # (1, D)

    # Create a directional shift based on target V-A
    D = base_emb.shape[-1]
    va_tensor = torch.tensor([[target_va[0], target_va[1]]], device=device)

    # Simple: use the first 2 dims as V-A channels, scale the embedding
    # In production, train a proper V-A → embedding mapping
    shift = torch.zeros_like(base_emb)
    shift[0, 0] = va_tensor[0, 0] * 2.0  # valence
    shift[0, 1] = va_tensor[0, 1] * 2.0  # arousal

    return base_emb + shift


def main():
    parser = argparse.ArgumentParser(description="Emotion Interpolation Demo")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="demo_output")
    parser.add_argument("--start_emotion", type=str, default="neutral")
    parser.add_argument("--end_emotion", type=str, default="angry")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from scripts.inference import estimate_mel_length, get_sample_rate, load_model, preprocess_video

    print(f"Loading model from {args.checkpoint}...")
    model, cfg = load_model(args.checkpoint, device)

    print(f"Preprocessing video: {args.video}")
    lip_video, face_crop = preprocess_video(args.video)
    lip_video = lip_video.to(device)
    face_crop = face_crop.to(device)
    mel_length = estimate_mel_length(cfg, lip_video.shape[2])
    sample_rate = get_sample_rate(cfg)

    # V-A interpolation
    start_va = EMOTION_VA.get(args.start_emotion, (0, 0))
    end_va = EMOTION_VA.get(args.end_emotion, (0, 0))
    va_points = interpolate_emotions(start_va, end_va, args.steps)

    print(f"Interpolating: {args.start_emotion} → {args.end_emotion} ({args.steps} steps)")
    print(f"V-A trajectory: {[(f'{v:.2f}', f'{a:.2f}') for v, a in va_points]}")

    generated_files = []

    for i, (v, a) in enumerate(va_points):
        print(f"  Step {i+1}/{args.steps}: V={v:.2f}, A={a:.2f}")

        emotion_emb = va_to_embedding(model, face_crop, (v, a), device)

        with torch.no_grad():
            audio, mel = model.generate(
                lip_video=lip_video,
                face_crop=face_crop,
                mel_length=mel_length,
                guidance_scale=args.guidance_scale,
                emotion_override=emotion_emb,
            )

        filename = f"step_{i:02d}_v{v:.2f}_a{a:.2f}.wav"
        filepath = output_dir / filename
        audio_np = audio.squeeze(0).cpu().numpy()
        sf.write(str(filepath), audio_np, sample_rate)
        generated_files.append(str(filepath))
        print(f"    Saved: {filepath}")

    # Concatenate all steps into one file
    all_audio = []
    for f in generated_files:
        wav, _ = sf.read(f)
        all_audio.append(wav)
        # Add 0.3s silence between steps
        all_audio.append(np.zeros(int(sample_rate * 0.3)))

    combined = np.concatenate(all_audio)
    combined_path = output_dir / "interpolation_combined.wav"
    sf.write(str(combined_path), combined, sample_rate)
    print(f"\nCombined audio: {combined_path} ({len(combined)/sample_rate:.1f}s)")
    print("Done!")


if __name__ == "__main__":
    main()
