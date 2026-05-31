"""
Objective speech quality and intelligibility metrics.

Wraps standard speech evaluation metrics with a unified interface.
All metrics operate on 16kHz mono audio.
"""

import numpy as np
import torch


def compute_pesq(ref: np.ndarray, gen: np.ndarray, sr: int = 16000) -> float:
    """Perceptual Evaluation of Speech Quality (ITU P.862.2). Higher is better."""
    from pesq import pesq
    return pesq(sr, ref, gen, "wb")


def compute_estoi(ref: np.ndarray, gen: np.ndarray, sr: int = 16000) -> float:
    """Extended Short-Time Objective Intelligibility. Higher is better (0-1)."""
    from pystoi import stoi
    return stoi(ref, gen, sr, extended=True)


def compute_mcd(ref_path: str, gen_path: str) -> float:
    """Mel Cepstral Distortion (DTW-aligned). Lower is better (dB)."""
    from pymcd.mcd import Calculate_MCD
    mcd_calc = Calculate_MCD(MCD_mode="dtw")
    return mcd_calc.calculate_mcd(ref_path, gen_path)


def compute_wer(gen_audio_path: str, reference_text: str, model_size: str = "base") -> float:
    """Word Error Rate via Whisper ASR. Lower is better."""
    import whisper
    import jiwer

    model = whisper.load_model(model_size)
    result = model.transcribe(gen_audio_path)
    hypothesis = result["text"].strip()

    return jiwer.wer(reference_text.strip(), hypothesis)


def compute_speaker_similarity(
    ref_audio: np.ndarray, gen_audio: np.ndarray, sr: int = 16000
) -> float:
    """Speaker Encoder Cosine Similarity (SECS). Higher is better."""
    from resemblyzer import VoiceEncoder, preprocess_wav
    import io
    import soundfile as sf

    encoder = VoiceEncoder()

    ref_emb = encoder.embed_utterance(ref_audio)
    gen_emb = encoder.embed_utterance(gen_audio)

    return float(np.dot(ref_emb, gen_emb) / (
        np.linalg.norm(ref_emb) * np.linalg.norm(gen_emb)
    ))


class MetricsComputer:
    """
    Batch metrics computation.

    Usage:
        mc = MetricsComputer()
        results = mc.compute_all(ref_audio, gen_audio, ref_text, gen_audio_path)
    """

    def __init__(self, whisper_model_size: str = "base"):
        self.whisper_model_size = whisper_model_size

    def compute_all(
        self,
        ref_audio: np.ndarray,
        gen_audio: np.ndarray,
        reference_text: str | None = None,
        gen_audio_path: str | None = None,
        ref_audio_path: str | None = None,
        sample_rate: int = 16000,
    ) -> dict[str, float]:
        """Compute all available metrics. Returns dict of metric_name -> value."""
        results = {}

        # Ensure same length
        min_len = min(len(ref_audio), len(gen_audio))
        ref = ref_audio[:min_len]
        gen = gen_audio[:min_len]

        # PESQ
        try:
            results["pesq"] = compute_pesq(ref, gen, sr=sample_rate)
        except Exception as e:
            results["pesq"] = float("nan")

        # ESTOI
        try:
            results["estoi"] = compute_estoi(ref, gen, sr=sample_rate)
        except Exception as e:
            results["estoi"] = float("nan")

        # MCD (needs file paths)
        if ref_audio_path and gen_audio_path:
            try:
                results["mcd"] = compute_mcd(ref_audio_path, gen_audio_path)
            except Exception:
                results["mcd"] = float("nan")

        # WER (needs generated audio path + reference text)
        if gen_audio_path and reference_text:
            try:
                results["wer"] = compute_wer(
                    gen_audio_path, reference_text, self.whisper_model_size
                )
            except Exception:
                results["wer"] = float("nan")

        # Speaker similarity
        try:
            results["secs"] = compute_speaker_similarity(ref, gen, sr=sample_rate)
        except Exception:
            results["secs"] = float("nan")

        return results
