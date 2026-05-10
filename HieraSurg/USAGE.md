# HieraSurg SurgWMBench Usage

Run all commands from the repository root. The examples use the canonical local
SurgWMBench root:

```bash
/mnt/hdd1/neurips2026_dataset_track/SurgWMBench
```

To use another copy of the dataset, replace only `--dataset-root`; keep
`--train-manifest`, `--val-manifest`, and `--manifest` relative to that root.

## uv Sync Environment

Create the Python 3.11 environment and install the locked dependencies:

```bash
uv sync
```

Optional labeler dependencies:

```bash
uv sync --extra labeler
```

Verify the synced Python and PyTorch versions:

```bash
uv run python -c "import sys, torch, torchvision; print(sys.version); print(torch.__version__, torchvision.__version__)"
```

Validate the 20-anchor loader before training:

```bash
uv run python src/tools/validate_surgwmbench_anchor_loader.py \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --manifest manifests/train.jsonl \
  --num-samples 8
```

## Joint Image + Trajectory Training

Use this mode when the input contains anchors 1-5 images and their trajectory
points. The image branch predicts anchors 6-20, and the trajectory head predicts
future points 6-20. The saved checkpoint includes both `transformer/` and
`trajectory_head.pt`. By default, robust trajectory conditioning is enabled:
input trajectory points receive normalized Gaussian noise with std `0.01`, and
each context point is randomly masked with probability `0.15`. Set both values
to `0` for clean trajectory inputs.

Single-GPU command:

```bash
uv run accelerate launch --num_processes 1 \
  src/finetune/train_surgwmbench_anchor_i2v.py \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --train-manifest manifests/train.jsonl \
  --val-manifest manifests/val.jsonl \
  --pretrained_model_name_or_path /path/to/cogvideox-or-hierasurg-base \
  --output_dir outputs/surgwmbench_anchor_i2v \
  --height 288 \
  --width 512 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --trajectory_loss_weight 1.0 \
  --trajectory_coord_noise_std 0.01 \
  --trajectory_coord_mask_prob 0.15 \
  --enable_slicing \
  --enable_tiling
```

Multi-GPU command:

```bash
uv run accelerate launch --multi_gpu --num_processes 4 \
  src/finetune/train_surgwmbench_anchor_i2v.py \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --train-manifest manifests/train.jsonl \
  --val-manifest manifests/val.jsonl \
  --pretrained_model_name_or_path /path/to/cogvideox-or-hierasurg-base \
  --output_dir outputs/surgwmbench_anchor_i2v_ddp \
  --height 288 \
  --width 512 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --trajectory_loss_weight 1.0 \
  --trajectory_coord_noise_std 0.01 \
  --trajectory_coord_mask_prob 0.15 \
  --enable_slicing \
  --enable_tiling
```

Resume joint training:

```bash
uv run accelerate launch --multi_gpu --num_processes 4 \
  src/finetune/train_surgwmbench_anchor_i2v.py \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --train-manifest manifests/train.jsonl \
  --val-manifest manifests/val.jsonl \
  --pretrained_model_name_or_path /path/to/cogvideox-or-hierasurg-base \
  --output_dir outputs/surgwmbench_anchor_i2v_ddp \
  --resume_from_checkpoint latest \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --trajectory_loss_weight 1.0 \
  --trajectory_coord_noise_std 0.01 \
  --trajectory_coord_mask_prob 0.15 \
  --enable_slicing \
  --enable_tiling
```

For a quick smoke run, append:

```bash
--train_limit 2 --max_train_steps 1
```

Disable robust trajectory conditioning while keeping joint image + trajectory
training enabled:

```bash
--trajectory_coord_noise_std 0.0 \
--trajectory_coord_mask_prob 0.0
```

This keeps the trajectory head active, but uses the clean context trajectory
points without Gaussian noise or random masking. Do not confuse this with
`--disable_trajectory_head`, which switches to the image-only baseline.

## Image-Only Training

Use this mode for an image-only baseline. It disables trajectory-head
construction, trajectory forward/loss, and `trajectory_head.pt` checkpoint
output. The model still uses anchors 1-5 images as context and predicts anchors
6-20 images.

Single-GPU command:

```bash
uv run accelerate launch --num_processes 1 \
  src/finetune/train_surgwmbench_anchor_i2v.py \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --train-manifest manifests/train.jsonl \
  --val-manifest manifests/val.jsonl \
  --pretrained_model_name_or_path /path/to/cogvideox-or-hierasurg-base \
  --output_dir outputs/surgwmbench_anchor_i2v_image_only \
  --height 288 \
  --width 512 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --disable_trajectory_head \
  --enable_slicing \
  --enable_tiling
```

Multi-GPU command:

```bash
uv run accelerate launch --multi_gpu --num_processes 4 \
  src/finetune/train_surgwmbench_anchor_i2v.py \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --train-manifest manifests/train.jsonl \
  --val-manifest manifests/val.jsonl \
  --pretrained_model_name_or_path /path/to/cogvideox-or-hierasurg-base \
  --output_dir outputs/surgwmbench_anchor_i2v_image_only_ddp \
  --height 288 \
  --width 512 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --disable_trajectory_head \
  --enable_slicing \
  --enable_tiling
```

Resume image-only training:

```bash
uv run accelerate launch --multi_gpu --num_processes 4 \
  src/finetune/train_surgwmbench_anchor_i2v.py \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --train-manifest manifests/train.jsonl \
  --val-manifest manifests/val.jsonl \
  --pretrained_model_name_or_path /path/to/cogvideox-or-hierasurg-base \
  --output_dir outputs/surgwmbench_anchor_i2v_image_only_ddp \
  --resume_from_checkpoint latest \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --disable_trajectory_head \
  --enable_slicing \
  --enable_tiling
```

## Joint Evaluation

Evaluate against original-resolution target frames and original pixel
coordinates. The training resize is an internal model detail; image predictions
are resized back before metric computation. The output includes `metrics.json`
and `predictions.jsonl`; each prediction row contains a complete 20-point
trajectory where points 1-5 are `context_input` and points 6-20 are `predicted`.

```bash
uv run python src/inference/eval_surgwmbench_anchor_i2v.py \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --manifest manifests/val.jsonl \
  --pretrained_model_name_or_path /path/to/cogvideox-or-hierasurg-base \
  --checkpoint outputs/surgwmbench_anchor_i2v/checkpoint-final \
  --output_dir outputs/surgwmbench_anchor_i2v_eval \
  --eval-horizons 5 10 15 \
  --mixed_precision bf16 \
  --save-videos
```

## Image-Only Evaluation

Use this command for checkpoints trained with `--disable_trajectory_head`. It
does not require `trajectory_head.pt`, writes only `metrics.json`, and reports
only image metrics.

```bash
uv run python src/inference/eval_surgwmbench_anchor_i2v.py \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --manifest manifests/val.jsonl \
  --pretrained_model_name_or_path /path/to/cogvideox-or-hierasurg-base \
  --checkpoint outputs/surgwmbench_anchor_i2v_image_only/checkpoint-final \
  --output_dir outputs/surgwmbench_anchor_i2v_image_only_eval \
  --eval-horizons 5 10 15 \
  --mixed_precision bf16 \
  --disable_trajectory_head
```
