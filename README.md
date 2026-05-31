# EmotiLip

Our Deep Learning course project: an emotion-aware lip-to-speech framework.

## What Is Implemented

- Visual encoder: lightweight 3D ResNet, with an optional AV-HuBERT wrapper
  that requires an explicit checkpoint.
- Emotion encoder: frozen EmoNet-compatible stand-in, with optional real EmoNet loading.
- Diffusion mel generator with FiLM, AdaIN, and cross-attention emotion fusion.
- Prosody predictor for frame-level pitch and energy.
- Optional Stage 4 mel-level proxy emotion consistency ablation.
- Optional emotion2vec SER evaluation for reportable emotion accuracy/F1 when
  evaluation dependencies are installed.
- MEAD preprocessing, train/val/test manifest splitting, training, inference, and evaluation scripts.

## Environment

```bash
cd EmotiLip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-core.txt
```

Install PyTorch with the CUDA build that matches your machine if the generic
requirement does not select the right wheel. Install `requirements-eval.txt`
only when you need optional metrics, SER, ASR, or experiment tracking.

Check the environment before running experiments:

```bash
python scripts/check_env.py
python scripts/validate_configs.py
```

## Preprocess MEAD

```bash
python data/preprocess_mead.py \
  --mead_root /path/to/MEAD \
  --output_dir data/processed \
  --view front \
  --speakers M001 M002 \
  --emotions neutral happy sad angry \
  --max_per_bucket 20 \
  --sample_rate 16000 \
  --n_mels 80 \
  --hop_length 256 \
  --target_fps 25 \
  --val_ratio 0.1 \
  --test_ratio 0.1 \
  --seed 42
```

The MEAD discovery code accepts both combined and nested emotion/intensity
layouts, for example `video/front/angry_level_1/001.mp4` and
`video/front/angry/level_1/001.mp4`, with audio under either
`audio/front/angry_level_1/001.wav` or `audio/angry/level_1/001.m4a`. For a
small pilot, `--max_per_bucket` caps raw samples per speaker/emotion. If
preprocessing is interrupted, rerun with `--skip_existing`.

This writes:

- `data/processed/manifest.json`
- `data/processed/train/manifest.json`
- `data/processed/val/manifest.json`
- `data/processed/test/manifest.json`

The split is stratified by speaker and emotion so validation/test speakers are
less likely to be missing from the training speaker embedding table.

Validate the split manifests before training:

```bash
python scripts/validate_manifest.py \
  data/processed/train/manifest.json \
  data/processed/val/manifest.json \
  data/processed/test/manifest.json
```

For a fast real-data pilot, select a balanced subset from existing processed
manifests without copying tensor files:

```bash
python scripts/select_manifest_subset.py \
  --input_dir data/processed \
  --output_dir data/subset_2spk_4emo \
  --num_speakers 2 \
  --emotions neutral happy sad angry \
  --max_per_bucket 20 \
  --seed 42
```

## Dummy Data Smoke Test

To test the full `manifest -> Dataset -> DataLoader -> train.py` loop without
MEAD, generate tiny dummy preprocessed data:

```bash
python scripts/make_dummy_data.py --output_dir data/debug
python scripts/validate_manifest.py \
  data/debug/train/manifest.json \
  data/debug/val/manifest.json \
  data/debug/test/manifest.json
python scripts/train.py --config configs/debug.yaml \
  --epochs 1 --max_train_batches 1 --max_val_batches 1
```

## Training Stages

See `EXPERIMENTS.md` for the full course-project run plan and report tables.

```bash
# Stage 1: baseline lip-to-speech, no emotion or prosody conditioning
python scripts/train.py --config configs/base.yaml --stage 1 \
  --output_dir checkpoints/base_stage1

# Stage 2: emotion conditioning only
python scripts/train.py --config configs/emotion_film.yaml --stage 2 \
  --output_dir checkpoints/emotion_film_stage2

# Stage 3: emotion + prosody conditioning
python scripts/train.py --config configs/emotion_film.yaml --stage 3 \
  --output_dir checkpoints/emotion_film_stage3

# Stage 4: Stage 3 plus a mel-level proxy emotion consistency loss
python scripts/train.py --config configs/emotion_film.yaml --stage 4 \
  --output_dir checkpoints/emotion_film_stage4_proxy
```

Stage 4 is a lightweight proxy: it trains a small mel emotion classifier on
ground-truth mels and encourages one-step predicted clean mels to match the
manifest emotion label. Use it as an ablation, not as a replacement for final
SER/FER emotion-transfer evaluation.

Before a full run, use a one-batch sanity pass on real preprocessed data:

```bash
python scripts/train.py --config configs/base.yaml --stage 1 \
  --epochs 1 --max_train_batches 1 --max_val_batches 1 \
  --output_dir checkpoints/debug_stage1
```

To train against a subset manifest without editing YAML, override the manifests
from the command line:

```bash
python scripts/train.py --config configs/base.yaml --stage 1 \
  --train_manifest data/subset_2spk_4emo/train/manifest.json \
  --val_manifest data/subset_2spk_4emo/val/manifest.json \
  --test_manifest data/subset_2spk_4emo/test/manifest.json \
  --output_dir checkpoints/base_subset_stage1
```

Real model assets can be injected from the command line instead of editing
YAML. The resolved values are written into each run's `run_config.yaml`:

```bash
python scripts/train.py --config configs/emotion_film.yaml --stage 3 \
  --emonet_checkpoint /path/to/emonet.pt \
  --vocoder_checkpoint /path/to/hifigan/generator.pt \
  --vocoder_config /path/to/hifigan/config.json \
  --output_dir checkpoints/emotion_film_real_assets
```

Passing `--emonet_checkpoint` sets `emotion_encoder.backend` to `emonet`. Use
the default `standin` backend only for development. Missing or unloadable
AV-HuBERT / EmoNet / HiFi-GAN assets fail during model construction.

For ablations, swap the config:

```bash
python scripts/train.py --config configs/emotion_adain.yaml --stage 3 \
  --output_dir checkpoints/emotion_adain_stage3
python scripts/train.py --config configs/emotion_crossattn.yaml --stage 3 \
  --output_dir checkpoints/emotion_crossattn_stage3
```

Each run writes:

- `metrics.jsonl`: one JSON record per epoch for plotting loss curves.
- `train_summary.json`: compact metadata including best epoch, best val loss,
  sample counts, and parameter count.

Summarize training runs for the report:

```bash
python scripts/summarize_training.py \
  checkpoints/base_stage1 \
  checkpoints/emotion_film_stage3 \
  --output checkpoints/training_summary.md
```

Plot training curves from the same logs:

```bash
python scripts/plot_training_curves.py \
  checkpoints/base_stage1 \
  checkpoints/emotion_film_stage3 \
  --output checkpoints/training_curves.png
```

## Inference and Evaluation

```bash
python scripts/inference.py \
  --checkpoint checkpoints/emotion_film/best.pt \
  --video /path/to/input.mp4 \
  --output output.wav

python scripts/inference.py \
  --checkpoint checkpoints/emotion_film/best.pt \
  --video /path/to/input.mp4 \
  --emotion angry \
  --output output_angry.wav

python scripts/evaluate.py \
  --checkpoint checkpoints/emotion_film/best.pt \
  --test_manifest data/processed/test/manifest.json \
  --output_dir eval_output
```

When `requirements-eval.txt` is installed, add SER evaluation for reportable
emotion accuracy/F1:

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/emotion_film/best.pt \
  --test_manifest data/processed/test/manifest.json \
  --output_dir eval_output/emotion_film_stage3 \
  --run_ser_eval
```

Evaluation always reports dependency-free mel reconstruction proxies
(`mel_mse`, `mel_mae`, `mel_rmse`). Optional metrics such as `PESQ`, `ESTOI`,
`MCD`, `SECS`, and `WER` are reported when the evaluation dependencies and
required models are installed. `WER` additionally requires each manifest row
to include one of `reference_text`, `transcript`, `text`, `sentence`, or
`utterance_text`; set `--whisper_model` to choose the Whisper model used for
ASR.

By default, `evaluate.py` selects `--max_samples` with deterministic
round-robin emotion balancing so small pilots are not biased toward the first
manifest entries. Use `--emotions`, `--speakers`, `--seed`, or `--no_balance`
to make the selection policy explicit.

After evaluating multiple runs, generate markdown tables for the report:

```bash
python scripts/summarize_results.py \
  eval_output/base_stage1 \
  eval_output/emotion_film_stage3 \
  --output eval_output/summary.md
```

Export a compact demo audio set from one or more checkpoints and a manifest:

```bash
python scripts/export_demo_samples.py \
  --checkpoints checkpoints/base_stage1/best.pt checkpoints/emotion_film_stage3/best.pt \
  --manifest data/subset_2spk_4emo/test/manifest.json \
  --output_dir demo_output/final_samples \
  --max_samples 8 \
  --emotions neutral happy sad angry \
  --guidance_scale 1.0
```

This writes per-run WAV files plus `metadata.json` and `index.md`.

For continuous V-A emotion interpolation between two anchors:

```bash
python scripts/demo_interpolation.py \
  --checkpoint checkpoints/emotion_film_stage3/best.pt \
  --video /path/to/input.mp4 \
  --output_dir demo_output/interp_neutral_to_angry \
  --emotion_a neutral --emotion_b angry --steps 5
```

## Current Caveats

- Lip and face crops use MediaPipe FaceMesh when available and fall back to
  center crops when landmarks are unavailable.
- The `standin` emotion encoder is for development only. Use the real EmoNet
  checkpoint for project results.
- Stage 4 uses a mel-level proxy emotion consistency loss. It is useful for an
  ablation, but final emotion-transfer claims still require a real SER/FER
  evaluation path.
- The fallback vocoder is untrained. Use a real HiFi-GAN checkpoint for any
  reportable audio.
