# Usage

This guide covers the SurgWMBench 20-anchor workflow for this fork: create the
Python environment, train on one or more GPUs, and evaluate predictions against
the original-resolution frames.

## Environment

Install dependencies with `uv`:

```bash
uv sync
source .venv/bin/activate
```

The environment uses Python 3.11 and CUDA 13.0 PyTorch wheels from
`pyproject.toml`. TensorFlow and legacy dataset-conversion packages are not
installed by default.

Check the installed PyTorch build:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Dataset

Set the dataset root before training or evaluation:

```bash
export SURGWMBENCH_ROOT=/mnt/hdd1/neurips2026_dataset_track/SurgWMBench
```

The SurgWMBench path uses official sparse 20-anchor manifests. Training uses the
first 5 anchors as context and predicts anchors 6-20. Transformer finetuning can
jointly train image tokens and future anchor trajectories with
`--use_trajectory_head`.

## Checkpoint Inputs

The Python commands below expect these pretrained checkpoints to exist:

```text
pretrained_models/amused/vqvae/
pretrained_models/ivideogpt-oxe-256-act-free/transformer/
```

Download them first if they are missing:

```bash
huggingface-cli download amused/amused-256 \
  --include "vqvae/*" \
  --local-dir pretrained_models/amused

huggingface-cli download thuml/ivideogpt-oxe-256-act-free \
  --local-dir pretrained_models/ivideogpt-oxe-256-act-free
```

## Tokenizer Finetuning

Run tokenizer finetuning first. This checkpoint is shared by both image-only
and joint image+trajectory transformer training.

```bash
CUDA_VISIBLE_DEVICES=0 python -m accelerate.commands.launch \
  --num_processes 1 \
  train_tokenizer.py \
  --exp_name surgwmbench_anchor_tokenizer_256 --output_dir log_vqgan \
  --seed 0 --mixed_precision bf16 \
  --model_type ctx_vqgan \
  --learning_rate 5e-4 --discr_learning_rate 5e-4 \
  --train_batch_size 1 --gradient_accumulation_steps 4 --disc_start 250000 \
  --dataset_format surgwmbench_anchor --surgwmbench_root "${SURGWMBENCH_ROOT}" \
  --resolution 256 --dataloader_num_workers 4 \
  --segment_length 20 --context_length 5 \
  --pretrained_model_name_or_path pretrained_models/amused/vqvae \
  --num_train_epochs 1 --validation_steps 500 --checkpointing_steps 1000
```

For tokenizer multi-GPU finetuning, use the same entrypoint with `--multi_gpu`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m accelerate.commands.launch \
  --multi_gpu --num_processes 4 \
  train_tokenizer.py \
  --exp_name surgwmbench_anchor_tokenizer_256 --output_dir log_vqgan \
  --seed 0 --mixed_precision bf16 \
  --model_type ctx_vqgan \
  --learning_rate 5e-4 --discr_learning_rate 5e-4 \
  --train_batch_size 1 --gradient_accumulation_steps 4 --disc_start 250000 \
  --dataset_format surgwmbench_anchor --surgwmbench_root "${SURGWMBENCH_ROOT}" \
  --resolution 256 --dataloader_num_workers 4 \
  --segment_length 20 --context_length 5 \
  --pretrained_model_name_or_path pretrained_models/amused/vqvae \
  --num_train_epochs 1 --validation_steps 500 --checkpointing_steps 1000
```

Then point `TOKENIZER_DIR` to the tokenizer output:

```bash
export TOKENIZER_DIR=log_vqgan/<timestamp>-surgwmbench_anchor_tokenizer_256
```

## Image-Only Transformer Training

Use this mode when the model should only predict future image tokens. Do not
pass `--use_trajectory_head`; the checkpoint will not contain
`trajectory_head.pt`.

Single-GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m accelerate.commands.launch \
  --num_processes 1 \
  train_gpt.py \
  --exp_name surgwmbench_anchor_transformer_image_only_256 --output_dir log_trm \
  --seed 0 --mixed_precision bf16 \
  --vqgan_type ctx_vqgan \
  --pretrained_model_name_or_path "${TOKENIZER_DIR}" \
  --config_name configs/llama/config_surgwm_anchor.json \
  --pretrained_transformer_path pretrained_models/ivideogpt-oxe-256-act-free/transformer \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 --lr_scheduler_type cosine --num_warmup_steps 100 \
  --dataset_format surgwmbench_anchor --surgwmbench_root "${SURGWMBENCH_ROOT}" \
  --resolution 256 --dataloader_num_workers 4 \
  --segment_length 20 --context_length 5 \
  --weight_decay 0.01 --llama_attn_drop 0.1 --embed_no_wd \
  --num_train_epochs 3 --validation_steps 500 --checkpointing_steps 1000 \
  --max_decode_batchsize 1
```

Multi-GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m accelerate.commands.launch \
  --multi_gpu --num_processes 4 \
  train_gpt.py \
  --exp_name surgwmbench_anchor_transformer_image_only_256 --output_dir log_trm \
  --seed 0 --mixed_precision bf16 \
  --vqgan_type ctx_vqgan \
  --pretrained_model_name_or_path "${TOKENIZER_DIR}" \
  --config_name configs/llama/config_surgwm_anchor.json \
  --pretrained_transformer_path pretrained_models/ivideogpt-oxe-256-act-free/transformer \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 --lr_scheduler_type cosine --num_warmup_steps 100 \
  --dataset_format surgwmbench_anchor --surgwmbench_root "${SURGWMBENCH_ROOT}" \
  --resolution 256 --dataloader_num_workers 4 \
  --segment_length 20 --context_length 5 \
  --weight_decay 0.01 --llama_attn_drop 0.1 --embed_no_wd \
  --num_train_epochs 3 --validation_steps 500 --checkpointing_steps 1000 \
  --max_decode_batchsize 1
```

## Joint Image + Trajectory Transformer Training

Use this mode when the model should predict both future image tokens and future
anchor coordinates. Pass `--use_trajectory_head`; the checkpoint will include
`trajectory_head.pt` and `trajectory_head_config.json`.

Single-GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m accelerate.commands.launch \
  --num_processes 1 \
  train_gpt.py \
  --exp_name surgwmbench_anchor_transformer_joint_256 --output_dir log_trm \
  --seed 0 --mixed_precision bf16 \
  --vqgan_type ctx_vqgan \
  --pretrained_model_name_or_path "${TOKENIZER_DIR}" \
  --config_name configs/llama/config_surgwm_anchor.json \
  --pretrained_transformer_path pretrained_models/ivideogpt-oxe-256-act-free/transformer \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 --lr_scheduler_type cosine --num_warmup_steps 100 \
  --dataset_format surgwmbench_anchor --surgwmbench_root "${SURGWMBENCH_ROOT}" \
  --resolution 256 --dataloader_num_workers 4 \
  --segment_length 20 --context_length 5 \
  --use_trajectory_head \
  --weight_decay 0.01 --llama_attn_drop 0.1 --embed_no_wd \
  --num_train_epochs 3 --validation_steps 500 --checkpointing_steps 1000 \
  --max_decode_batchsize 1
```

Multi-GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m accelerate.commands.launch \
  --multi_gpu --num_processes 4 \
  train_gpt.py \
  --exp_name surgwmbench_anchor_transformer_joint_256 --output_dir log_trm \
  --seed 0 --mixed_precision bf16 \
  --vqgan_type ctx_vqgan \
  --pretrained_model_name_or_path "${TOKENIZER_DIR}" \
  --config_name configs/llama/config_surgwm_anchor.json \
  --pretrained_transformer_path pretrained_models/ivideogpt-oxe-256-act-free/transformer \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 --lr_scheduler_type cosine --num_warmup_steps 100 \
  --dataset_format surgwmbench_anchor --surgwmbench_root "${SURGWMBENCH_ROOT}" \
  --resolution 256 --dataloader_num_workers 4 \
  --segment_length 20 --context_length 5 \
  --use_trajectory_head \
  --weight_decay 0.01 --llama_attn_drop 0.1 --embed_no_wd \
  --num_train_epochs 3 --validation_steps 500 --checkpointing_steps 1000 \
  --max_decode_batchsize 1
```

### Robust Trajectory-Condition Training

Robust condition augmentation is only for joint training. It perturbs the first
5 input trajectory points; future trajectory labels stay unchanged. By default
it is disabled with `--trajectory_condition_noise_std 0.0` and
`--trajectory_condition_mask_prob 0.0`.

Add Gaussian noise to normalized context coordinates:

```bash
  --use_trajectory_head \
  --trajectory_condition_noise_std 0.02
```

Randomly mask context trajectory points with a learned mask condition:

```bash
  --use_trajectory_head \
  --trajectory_condition_mask_prob 0.2
```

Use both augmentations together:

```bash
  --use_trajectory_head \
  --trajectory_condition_noise_std 0.02 \
  --trajectory_condition_mask_prob 0.2
```

Do not pass these flags for image-only training. Evaluation normally uses clean
context trajectories and does not need the robust-training flags.

Keep the per-device batch size small for 256x256 training. The examples use
batch size 1 per GPU with gradient accumulation 4. The training code prepares
train and validation dataloaders with Accelerate so each process receives a
different shard.

## Evaluation

Point `TOKENIZER_DIR` and `TRANSFORMER_DIR` to the trained checkpoint
directories. Evaluation predicts anchors 6-10, 6-15, and 6-20 from the first 5
anchors and resizes image predictions back to each frame's original resolution.

### Image-Only Evaluation

Do not pass `--use_trajectory_head` when evaluating an image-only transformer.

```bash
export TOKENIZER_DIR=log_vqgan/<timestamp>-surgwmbench_anchor_tokenizer_256
export TRANSFORMER_DIR=log_trm/<timestamp>-surgwmbench_anchor_transformer_image_only_256
export OUTPUT_DIR=benchmark/outputs/ivideogpt_surgwmbench_anchor_image_only_test

python tools/evaluate_surgwmbench_anchor_prediction.py \
  --surgwmbench_root "${SURGWMBENCH_ROOT}" \
  --manifest manifests/test.jsonl \
  --tokenizer_path "${TOKENIZER_DIR}" \
  --transformer_path "${TRANSFORMER_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --resolution 256 \
  --context_length 5 \
  --segment_length 20 \
  --num_artifacts 2
```

### Joint Image + Trajectory Evaluation

Pass `--use_trajectory_head` when evaluating a joint checkpoint. This reports
image metrics plus trajectory ADE/FDE for anchors 6-10, 6-15, and 6-20.

```bash
export TOKENIZER_DIR=log_vqgan/<timestamp>-surgwmbench_anchor_tokenizer_256
export TRANSFORMER_DIR=log_trm/<timestamp>-surgwmbench_anchor_transformer_joint_256
export OUTPUT_DIR=benchmark/outputs/ivideogpt_surgwmbench_anchor_joint_test

python tools/evaluate_surgwmbench_anchor_prediction.py \
  --surgwmbench_root "${SURGWMBENCH_ROOT}" \
  --manifest manifests/test.jsonl \
  --tokenizer_path "${TOKENIZER_DIR}" \
  --transformer_path "${TRANSFORMER_DIR}" \
  --use_trajectory_head \
  --output_dir "${OUTPUT_DIR}" \
  --resolution 256 \
  --context_length 5 \
  --segment_length 20 \
  --num_artifacts 2
```

The main metrics file is:

```text
${OUTPUT_DIR}/metrics.json
```

Joint trajectory evaluation also writes:

```text
${OUTPUT_DIR}/predictions.jsonl
```

Visual samples, when enabled, are written under:

```text
${OUTPUT_DIR}/artifacts/
```

Limit evaluation for a quick smoke run:

```bash
python tools/evaluate_surgwmbench_anchor_prediction.py \
  --surgwmbench_root "${SURGWMBENCH_ROOT}" \
  --manifest manifests/test.jsonl \
  --tokenizer_path "${TOKENIZER_DIR}" \
  --transformer_path "${TRANSFORMER_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --resolution 256 \
  --context_length 5 \
  --segment_length 20 \
  --max_clips 2 \
  --num_artifacts 1
```

Add `--use_trajectory_head` to the smoke command only when `TRANSFORMER_DIR`
points to a joint checkpoint.

The shell wrappers in `scripts/finetune/` and `scripts/evaluation/` run the same
joint image+trajectory commands by default. For image-only runs, use the Python
commands above or remove `--use_trajectory_head` from the wrapper command.
