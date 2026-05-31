# Experiment Plan

Use the same train/val/test split for all runs. Keep `seed`, selected
speakers, and emotion subsets fixed across ablations.

## Pipeline Sanity Check

Before using MEAD, run the debug pipeline with generated preprocessed tensors:

```bash
python scripts/make_dummy_data.py --output_dir data/debug
python scripts/train.py --config configs/debug.yaml \
  --epochs 1 --max_train_batches 1 --max_val_batches 1
```

This verifies manifest loading, variable-length collation, masked losses,
checkpoint writing, and the staged training loop without requiring dataset
download or expensive model settings.

## Real-Data Pilot Subset

Preprocess only the pilot slice instead of the full dataset:

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
  --target_fps 25 \
  --seed 42
```

Validate manifests before training:

```bash
python scripts/validate_manifest.py \
  data/processed/train/manifest.json \
  data/processed/val/manifest.json \
  data/processed/test/manifest.json
```

For quick sanity runs, create a fixed small subset:

```bash
python scripts/select_manifest_subset.py \
  --input_dir data/processed \
  --output_dir data/subset_2spk_4emo \
  --num_speakers 2 \
  --emotions neutral happy sad angry \
  --max_per_bucket 20 \
  --seed 42
```

Use this subset for one-batch checks and short pilot training before
committing compute to the full processed split.

## Stage Runs

1. Baseline L2S
   ```bash
   python scripts/train.py --config configs/base.yaml --stage 1 \
     --train_manifest data/subset_2spk_4emo/train/manifest.json \
     --val_manifest data/subset_2spk_4emo/val/manifest.json \
     --test_manifest data/subset_2spk_4emo/test/manifest.json \
     --output_dir checkpoints/base_stage1
   ```
   Purpose: verify lip-to-speech reconstruction without emotion/prosody.

2. Emotion Conditioning
   ```bash
   python scripts/train.py --config configs/emotion_film.yaml --stage 2 \
     --train_manifest data/subset_2spk_4emo/train/manifest.json \
     --val_manifest data/subset_2spk_4emo/val/manifest.json \
     --test_manifest data/subset_2spk_4emo/test/manifest.json \
     --output_dir checkpoints/emotion_film_stage2
   ```
   Purpose: isolate whether face emotion embedding improves emotion-related
   audio statistics without the prosody predictor.

3. Emotion + Prosody
   ```bash
   python scripts/train.py --config configs/emotion_film.yaml --stage 3 \
     --train_manifest data/subset_2spk_4emo/train/manifest.json \
     --val_manifest data/subset_2spk_4emo/val/manifest.json \
     --test_manifest data/subset_2spk_4emo/test/manifest.json \
     --output_dir checkpoints/emotion_film_stage3
   ```
   Purpose: measure the full proposed model with pitch/energy conditioning.

4. Emotion-Proxy Consistency Ablation
   ```bash
   python scripts/train.py --config configs/emotion_film.yaml --stage 4 \
     --train_manifest data/subset_2spk_4emo/train/manifest.json \
     --val_manifest data/subset_2spk_4emo/val/manifest.json \
     --test_manifest data/subset_2spk_4emo/test/manifest.json \
     --output_dir checkpoints/emotion_film_stage4_proxy
   ```
   Purpose: test whether a lightweight mel-level emotion proxy improves
   emotion-label consistency. Do not treat this as a substitute for final
   SER/FER evaluation.

5. Fusion Ablations
   ```bash
   python scripts/train.py --config configs/emotion_adain.yaml --stage 3 \
     --output_dir checkpoints/emotion_adain_stage3
   python scripts/train.py --config configs/emotion_crossattn.yaml --stage 3 \
     --output_dir checkpoints/emotion_crossattn_stage3
   ```
   Purpose: compare FiLM, AdaIN, and cross-attention emotion fusion.

Each training run writes `metrics.jsonl` and `train_summary.json` into its
checkpoint directory. Summarize run health and loss values with:

```bash
python scripts/summarize_training.py \
  checkpoints/base_stage1 \
  checkpoints/emotion_film_stage3 \
  checkpoints/emotion_adain_stage3 \
  --output checkpoints/training_summary.md
```

Generate loss-curve figures for the report:

```bash
python scripts/plot_training_curves.py \
  checkpoints/base_stage1 \
  checkpoints/emotion_film_stage3 \
  checkpoints/emotion_adain_stage3 \
  --output checkpoints/training_curves.png
```

## Evaluation

For each `best.pt` checkpoint:

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/<run_name>/best.pt \
  --test_manifest data/processed/test/manifest.json \
  --output_dir eval_output/<run_name> \
  --max_samples 100
```

Evaluation uses deterministic round-robin emotion balancing by default. Add
`--emotions neutral happy sad angry` or `--speakers <ids...>` when the report
should evaluate a named subset.

For optional SER evaluation, install `requirements-eval.txt` and add
`--run_ser_eval`. If you intend to report WER, add reference text to the
manifest rows using one of `reference_text`, `transcript`, `text`, `sentence`,
or `utterance_text`, and set `--whisper_model` if the default Whisper `base`
model is not appropriate.

Each evaluation directory writes `results.json` with per-sample rows and
`summary.json` with aggregate metric coverage, prosody summaries, and SER
status/accuracy/F1 when SER was requested.

Export qualitative demo samples for listening and slides:

```bash
python scripts/export_demo_samples.py \
  --checkpoints checkpoints/base_stage1/best.pt checkpoints/emotion_film_stage3/best.pt \
  --manifest data/subset_2spk_4emo/test/manifest.json \
  --output_dir demo_output/final_samples \
  --max_samples 8 \
  --emotions neutral happy sad angry \
  --guidance_scale 1.0
```

Summarize evaluated runs into markdown report tables:

```bash
python scripts/summarize_results.py \
  eval_output/base_stage1 \
  eval_output/emotion_film_stage3 \
  eval_output/emotion_adain_stage3 \
  --output eval_output/summary.md
```

Report available objective metrics:

- `mel_mse`, `mel_mae`, `mel_rmse`: dependency-free mel reconstruction
  proxies; lower is better.
- `PESQ`, `ESTOI`, `MCD`: reconstruction quality and intelligibility.
- `SECS`: speaker similarity.
- `WER`: ASR word error rate, only when manifest reference text is present.
- Per-emotion F0 and energy statistics: whether generated speech preserves
  expected prosody differences.

Optional, when SER dependencies are installed:

- emotion2vec classification accuracy/F1.
- V-A consistency only after wiring a calibrated audio V-A regressor. Do not
  report placeholder V-A numbers.

## Tables For Report

Quality table:

| Run | Mel MSE ↓ | Mel MAE ↓ | Mel RMSE ↓ | PESQ ↑ | ESTOI ↑ | MCD ↓ | SECS ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | | | | | | | |
| Emotion-FiLM | | | | | | | |
| Emotion+Prosody-FiLM | | | | | | | |
| Emotion+Prosody-AdaIN | | | | | | | |
| Emotion+Prosody-CrossAttn | | | | | | | |

Emotion/prosody table:

| Run | Emotion Acc ↑ | Emotion F1 ↑ | F0 gap by emotion ↑ | Energy gap by emotion ↑ |
| --- | --- | --- | --- | --- |
| Baseline | | | | |
| Emotion-FiLM | | | | |
| Emotion+Prosody-FiLM | | | | |

## Minimum Course-Project Deliverable

- One trained baseline checkpoint.
- One trained full model checkpoint.
- One fusion ablation if compute allows.
- A demo audio set generated with `scripts/inference.py`, including at least
  neutral, happy, sad, and angry examples.
- A short qualitative discussion of failure cases: lip crop quality, speaker
  leakage, emotion exaggeration, and noisy prosody extraction.
