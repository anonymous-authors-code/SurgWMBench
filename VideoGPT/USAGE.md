# Usage

This document covers the SurgWMBench 20-anchor VideoGPT workflow: create the
standard uv environment, train the 20-frame VQ-VAE, then train a 5-frame
conditioned VideoGPT model that predicts anchor frames 6-20 and their future
trajectory points.

## Environment

Use the locked uv environment from the repository root:

```bash
uv sync
```

Verify the synced environment:

```bash
uv run python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
uv run python -m pytest -q tests
```

The default dataset root used by the scripts is:

```text
/mnt/hdd1/neurips2026_dataset_track/SurgWMBench
```

The training path uses official manifests only. Each sample loads the 20
human-anchor frames identified by `sampled_indices` plus the matching sparse
human trajectory coordinates.

## Python Command Usage

All examples below use direct Python entry points. The recommended form is
`uv run python ...` because it uses the locked `.venv` without manually
activating the environment:

```bash
uv run python scripts/train_surgwmbench_videogpt.py -h
```

If you have already activated the uv environment, the equivalent command is:

```bash
source .venv/bin/activate
python scripts/train_surgwmbench_videogpt.py -h
```

The same rule applies to training and evaluation scripts. For example,
`uv run python scripts/eval_surgwmbench_videogpt.py ...` is equivalent to
`python scripts/eval_surgwmbench_videogpt.py ...` inside the activated `.venv`.
Do not run these scripts with system Python outside the uv environment.

## Changing Dataset Path

Set a different SurgWMBench location with `--dataset-root` in every train or
eval command:

```bash
--dataset-root /path/to/SurgWMBench
```

Manifest paths stay relative to that dataset root. For example, if your dataset
is under `/data/SurgWMBench`, use:

```bash
uv run python scripts/train_surgwmbench_vqvae.py \
  --dataset-root /data/SurgWMBench \
  --train-manifest manifests/train.jsonl \
  --val-manifest manifests/val.jsonl
```

Do not edit the official manifest files or create random splits. Keep using
`manifests/train.jsonl`, `manifests/val.jsonl`, and `manifests/test.jsonl`
from the dataset root.

## VQ-VAE Training

Train the SurgWMBench 20-anchor VQ-VAE before training any VideoGPT variant.

Single GPU:

```bash
uv run python scripts/train_surgwmbench_vqvae.py \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --train-manifest manifests/train.jsonl \
  --val-manifest manifests/val.jsonl \
  --sequence_length 20 \
  --resolution 128 \
  --batch_size 4 \
  --num_workers 8 \
  --max_steps 200000 \
  --accelerator gpu \
  --devices 1
```

Multi GPU:

```bash
uv run python scripts/train_surgwmbench_vqvae.py \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --train-manifest manifests/train.jsonl \
  --val-manifest manifests/val.jsonl \
  --sequence_length 20 \
  --resolution 128 \
  --batch_size 4 \
  --num_workers 8 \
  --max_steps 200000 \
  --accelerator gpu \
  --devices 4 \
  --strategy ddp
```

`--batch_size` is per process/GPU under DDP. Adjust it for GPU memory.

## Joint VideoGPT Training

Use this mode for joint future image and future trajectory prediction. It
optimizes image token loss plus trajectory loss and writes checkpoints with a
`trajectory_head`. The trajectory head is conditioned on the first 5 normalized
trajectory points; during training those input trajectory conditions use
Gaussian noise and random point masking for robustness.

Single GPU:

```bash
uv run python scripts/train_surgwmbench_videogpt.py \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --train-manifest manifests/train.jsonl \
  --val-manifest manifests/val.jsonl \
  --vqvae <path-to-vqvae.ckpt> \
  --sequence_length 20 \
  --n_cond_frames 5 \
  --resolution 128 \
  --batch_size 2 \
  --num_workers 8 \
  --max_steps 200000 \
  --trajectory_head \
  --traj_loss_weight 10.0 \
  --trajectory_condition \
  --traj_condition_noise_std 0.01 \
  --traj_condition_mask_prob 0.15 \
  --accelerator gpu \
  --devices 1
```

Multi GPU:

```bash
uv run python scripts/train_surgwmbench_videogpt.py \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --train-manifest manifests/train.jsonl \
  --val-manifest manifests/val.jsonl \
  --vqvae <path-to-vqvae.ckpt> \
  --sequence_length 20 \
  --n_cond_frames 5 \
  --resolution 128 \
  --batch_size 2 \
  --num_workers 8 \
  --max_steps 200000 \
  --trajectory_head \
  --traj_loss_weight 10.0 \
  --trajectory_condition \
  --traj_condition_noise_std 0.01 \
  --traj_condition_mask_prob 0.15 \
  --accelerator gpu \
  --devices 4 \
  --strategy ddp_find_unused_parameters_false
```

`scripts/train_surgwmbench_videogpt.py` enables the trajectory head and
trajectory conditioning by default. Set `--traj_condition_noise_std 0.0` and
`--traj_condition_mask_prob 0.0` to disable robust condition augmentation while
keeping trajectory conditioning. Use `--no_trajectory_condition` to remove input
trajectory conditioning entirely.

### Disable Trajectory Condition Noise

To keep trajectory conditioning and random masking but disable Gaussian noise,
set only the noise standard deviation to zero:

```bash
--trajectory_condition \
--traj_condition_noise_std 0.0 \
--traj_condition_mask_prob 0.15
```

To disable both Gaussian noise and random masking while still using the first 5
trajectory points as clean conditions:

```bash
--trajectory_condition \
--traj_condition_noise_std 0.0 \
--traj_condition_mask_prob 0.0
```

To remove trajectory conditions entirely:

```bash
--no_trajectory_condition
```

## Image-only VideoGPT Training

Use this mode for image prediction ablations. It disables the trajectory head and
optimizes only the VideoGPT image token loss.

Single GPU:

```bash
uv run python scripts/train_surgwmbench_videogpt.py \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --train-manifest manifests/train.jsonl \
  --val-manifest manifests/val.jsonl \
  --vqvae <path-to-vqvae.ckpt> \
  --sequence_length 20 \
  --n_cond_frames 5 \
  --resolution 128 \
  --batch_size 2 \
  --num_workers 8 \
  --max_steps 200000 \
  --no_trajectory_head \
  --accelerator gpu \
  --devices 1
```

Multi GPU:

```bash
uv run python scripts/train_surgwmbench_videogpt.py \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --train-manifest manifests/train.jsonl \
  --val-manifest manifests/val.jsonl \
  --vqvae <path-to-vqvae.ckpt> \
  --sequence_length 20 \
  --n_cond_frames 5 \
  --resolution 128 \
  --batch_size 2 \
  --num_workers 8 \
  --max_steps 200000 \
  --no_trajectory_head \
  --accelerator gpu \
  --devices 4 \
  --strategy ddp_find_unused_parameters_false
```

Image-only checkpoints do not report `traj_loss`, and evaluation skips
trajectory metrics for those checkpoints.

## Evaluation

Evaluate one VideoGPT checkpoint on the test manifest. The script generates the
future anchor sequence and, when the checkpoint has a trajectory head, predicts
future trajectory points. It reports horizons 5, 10, and 15, corresponding to
anchors 6-10, 6-15, and 6-20:

```bash
uv run python scripts/eval_surgwmbench_videogpt.py \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --manifest manifests/test.jsonl \
  --ckpt <path-to-videogpt.ckpt> \
  --horizons 5 10 15 \
  --output-dir outputs/surgwmbench_videogpt_eval
```

Predictions are restored to original frame resolution before PSNR, SSIM, and
LPIPS are computed. Trajectory metrics are reported as `ADE_px` and `FDE_px` in
the original image pixel coordinate system. Per-sample future trajectories are
written to `outputs/surgwmbench_videogpt_eval/predictions.jsonl`.
Evaluation uses the clean first 5 trajectory points as conditions; noise and
masking are training-only augmentations.

Use `--lpips-downsample 64` for faster CPU smoke checks.

## Smoke Commands

CPU smoke training with one clip:

```bash
uv run python scripts/train_surgwmbench_vqvae.py \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --sequence_length 20 \
  --resolution 32 \
  --batch_size 1 \
  --num_workers 0 \
  --max-clips 1 \
  --max_steps 1 \
  --accelerator cpu \
  --devices 1 \
  --limit_train_batches 1 \
  --limit_val_batches 1 \
  --num_sanity_val_steps 0 \
  --embedding_dim 8 \
  --n_codes 32 \
  --n_hiddens 16 \
  --n_res_layers 1 \
  --downsample 4 4 4
```
