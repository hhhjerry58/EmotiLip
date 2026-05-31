"""Check whether the local Python environment can run EmotiLip."""

from __future__ import annotations

import importlib.util
import platform
import sys


REQUIRED = {
    "torch": "model training and inference",
    "numpy": "dataset tensors and preprocessing",
    "yaml": "YAML config loading",
    "soundfile": "writing generated WAV files",
}

PREPROCESSING = {
    "cv2": "video frame extraction",
    "librosa": "mel/prosody extraction",
}

RECOMMENDED = {
    "mediapipe": "landmark-based lip/face crops; center crop fallback is used if missing",
    "pyworld": "higher-quality F0 extraction; zeros are used if missing",
}

OPTIONAL = {
    "emonet": "real EmoNet backend for reportable emotion-conditioned runs",
    "pesq": "PESQ evaluation",
    "pystoi": "ESTOI evaluation",
    "pymcd": "MCD evaluation",
    "jiwer": "WER evaluation",
    "whisper": "ASR-based WER evaluation",
    "parselmouth": "Praat prosody statistics",
    "resemblyzer": "speaker similarity evaluation",
    "frechet_audio_distance": "FAD evaluation",
    "funasr": "emotion2vec evaluation",
    "transformers": "wav2vec2 emotion utilities",
}


def module_status(name: str) -> tuple[bool, str]:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return False, "missing"
    try:
        module = __import__(name)
        version = getattr(module, "__version__", "installed")
    except Exception as exc:
        return False, f"import error: {exc}"
    return True, str(version)


def print_group(title: str, modules: dict[str, str]) -> bool:
    print(f"\n{title}")
    print("-" * len(title))
    all_ok = True
    for name, reason in modules.items():
        ok, status = module_status(name)
        marker = "OK" if ok else "MISS"
        print(f"{marker:4s} {name:14s} {status:20s} {reason}")
        all_ok = all_ok and ok
    return all_ok


def main() -> int:
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")

    required_ok = print_group("Required", REQUIRED)
    preprocessing_ok = print_group("Preprocessing", PREPROCESSING)
    print_group("Recommended", RECOMMENDED)
    print_group("Optional Evaluation", OPTIONAL)

    if required_ok and preprocessing_ok:
        print("\nEnvironment is ready for preprocessing, smoke tests, and training.")
        return 0

    print("\nInstall missing required/preprocessing packages before running experiments.")
    print("Typical setup: pip install -r requirements-core.txt")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
