"""
Emotion-specific evaluation metrics.

Evaluates whether generated speech carries the intended emotion:
  1. Emotion classification accuracy via pretrained SER.
  2. V-A (Valence-Arousal) consistency.
  3. Prosody statistics per emotion category.
"""

from functools import lru_cache
import re

import numpy as np


DEFAULT_SER_MODEL = "iic/emotion2vec_plus_base"


EMOTION_ALIASES = {
    "angry": "angry",
    "anger": "angry",
    "happy": "happy",
    "happiness": "happy",
    "joy": "happy",
    "sad": "sad",
    "sadness": "sad",
    "neutral": "neutral",
    "calm": "neutral",
    "fear": "fear",
    "fearful": "fear",
    "disgust": "disgust",
    "disgusted": "disgust",
    "surprise": "surprise",
    "surprised": "surprise",
    "contempt": "contempt",
    "unknown": "unknown",
}


@lru_cache(maxsize=2)
def _load_emotion2vec(model_name: str):
    """Cache emotion2vec so batch evaluation does not reload it per file."""
    from funasr import AutoModel
    return AutoModel(model=model_name)


def normalize_emotion_label(label: str | None) -> str:
    """Normalize dataset/SER labels before accuracy/F1 calculation."""
    if label is None:
        return "unknown"
    text = str(label).strip().lower()
    if not text:
        return "unknown"

    compact = re.sub(r"[^a-z]+", "", text)
    if compact in EMOTION_ALIASES:
        return EMOTION_ALIASES[compact]

    tokens = [token for token in re.split(r"[^a-z]+", text) if token]
    for token in tokens:
        if token in EMOTION_ALIASES:
            return EMOTION_ALIASES[token]
    return compact or "unknown"


def classify_emotion_emotion2vec(audio_path: str, model_name: str = DEFAULT_SER_MODEL) -> dict:
    """
    Classify emotion using emotion2vec+ model.

    Returns:
        Dict with 'label' (str), 'scores' (list of floats).
    """
    model = _load_emotion2vec(model_name)
    result = model.generate(audio_path, granularity="utterance", extract_embedding=False)
    return {
        "label": result[0]["labels"][0],
        "scores": result[0]["scores"],
    }


def predict_va_from_audio(audio_path: str) -> dict[str, float]:
    """
    Predict continuous valence-arousal-dominance from audio.

    Placeholder intentionally disabled until a calibrated audio V-A model is
    wired in. Returning constants here would produce invalid project metrics.
    """
    raise NotImplementedError(
        "Audio V-A prediction is not wired yet. Use emotion2vec classification "
        "and prosody statistics, or add a calibrated V-A regressor before reporting this metric."
    )


def compute_prosody_stats(audio_path: str) -> dict[str, float]:
    """
    Extract prosody statistics using Parselmouth (Praat).

    Returns dict with f0_mean, f0_std, f0_range, energy_mean, energy_std, duration.
    """
    import parselmouth

    snd = parselmouth.Sound(audio_path)

    # Pitch
    pitch = snd.to_pitch()
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 != 0]

    # Intensity
    intensity = snd.to_intensity()
    energy = intensity.values[0]

    return {
        "f0_mean": float(np.mean(f0)) if len(f0) > 0 else 0.0,
        "f0_std": float(np.std(f0)) if len(f0) > 0 else 0.0,
        "f0_range": float(np.ptp(f0)) if len(f0) > 0 else 0.0,
        "energy_mean": float(np.mean(energy)),
        "energy_std": float(np.std(energy)),
        "duration": float(snd.duration),
    }


def evaluate_emotion_batch(
    generated_audio_paths: list[str],
    target_emotions: list[str],
    model_name: str = DEFAULT_SER_MODEL,
) -> dict:
    """
    Evaluate emotion accuracy across a batch of generated samples.

    Args:
        generated_audio_paths: Paths to generated audio files.
        target_emotions: Ground truth emotion labels.

    Returns:
        Dict with accuracy, per-class accuracy, confusion matrix.
    """
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    raw_predictions = []
    predictions = []
    failures = []
    for path in generated_audio_paths:
        try:
            result = classify_emotion_emotion2vec(path, model_name=model_name)
            raw_label = result["label"]
            raw_predictions.append(raw_label)
            predictions.append(normalize_emotion_label(raw_label))
        except Exception as exc:
            failures.append({"path": path, "error": str(exc)})
            raw_predictions.append("unknown")
            predictions.append("unknown")

    targets = [normalize_emotion_label(label) for label in target_emotions]
    labels = sorted(set(targets + predictions))
    if not generated_audio_paths:
        status = "empty"
        accuracy = None
        f1 = None
        conf_mat = []
    elif len(failures) == len(generated_audio_paths):
        status = "failed"
        accuracy = None
        f1 = None
        conf_mat = confusion_matrix(targets, predictions, labels=labels).tolist()
    else:
        status = "partial" if failures else "ok"
        accuracy = float(accuracy_score(targets, predictions))
        f1 = float(f1_score(targets, predictions, average="weighted", zero_division=0))
        conf_mat = confusion_matrix(targets, predictions, labels=labels).tolist()

    return {
        "status": status,
        "ser_model": model_name,
        "emotion_accuracy": accuracy,
        "emotion_f1": f1,
        "labels": labels,
        "confusion_matrix": conf_mat,
        "predictions": predictions,
        "raw_predictions": raw_predictions,
        "targets": targets,
        "raw_targets": target_emotions,
        "failed_samples": len(failures),
        "failures": failures[:20],
    }
