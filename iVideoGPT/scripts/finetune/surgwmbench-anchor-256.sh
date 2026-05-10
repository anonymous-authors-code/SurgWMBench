#!/usr/bin/env bash
set -euo pipefail

SURGWMBENCH_ROOT="${SURGWMBENCH_ROOT:-/mnt/hdd1/neurips2026_dataset_track/SurgWMBench}"
AMUSED_VQVAE="${AMUSED_VQVAE:-pretrained_models/amused/vqvae}"
OXE_TRANSFORMER="${OXE_TRANSFORMER:-pretrained_models/ivideogpt-oxe-256-act-free/transformer}"
TOKENIZER_OUTPUT_ROOT="${TOKENIZER_OUTPUT_ROOT:-log_vqgan}"
TRANSFORMER_OUTPUT_ROOT="${TRANSFORMER_OUTPUT_ROOT:-log_trm}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
GPU_IDS="${GPU_IDS:-}"

ACCELERATE_ARGS=(--num_processes "${NUM_PROCESSES}")
if [ "${NUM_PROCESSES}" -gt 1 ]; then
  ACCELERATE_ARGS=(--multi_gpu "${ACCELERATE_ARGS[@]}")
fi
if [ -n "${GPU_IDS}" ]; then
  ACCELERATE_ARGS+=(--gpu_ids "${GPU_IDS}")
fi

if [ ! -f "${AMUSED_VQVAE}/diffusion_pytorch_model.safetensors" ]; then
  mkdir -p pretrained_models/amused
  huggingface-cli download amused/amused-256 \
    --include "vqvae/*" \
    --local-dir pretrained_models/amused
fi

if [ ! -f "${OXE_TRANSFORMER}/model.safetensors" ]; then
  mkdir -p pretrained_models/ivideogpt-oxe-256-act-free
  huggingface-cli download thuml/ivideogpt-oxe-256-act-free \
    --local-dir pretrained_models/ivideogpt-oxe-256-act-free
fi

accelerate launch "${ACCELERATE_ARGS[@]}" train_tokenizer.py \
  --exp_name surgwmbench_anchor_tokenizer_256 --output_dir "${TOKENIZER_OUTPUT_ROOT}" \
  --seed 0 --mixed_precision bf16 \
  --model_type ctx_vqgan \
  --learning_rate 5e-4 --discr_learning_rate 5e-4 \
  --train_batch_size 1 --gradient_accumulation_steps 4 --disc_start 250000 \
  --dataset_format surgwmbench_anchor --surgwmbench_root "${SURGWMBENCH_ROOT}" \
  --resolution 256 --dataloader_num_workers 4 \
  --segment_length 20 --context_length 5 \
  --pretrained_model_name_or_path "${AMUSED_VQVAE}" \
  --num_train_epochs 1 --validation_steps 500 --checkpointing_steps 1000

TOKENIZER_DIR="${TOKENIZER_DIR:-$(ls -td "${TOKENIZER_OUTPUT_ROOT}"/*-surgwmbench_anchor_tokenizer_256 | head -n 1)}"

accelerate launch "${ACCELERATE_ARGS[@]}" train_gpt.py \
  --exp_name surgwmbench_anchor_transformer_256 --output_dir "${TRANSFORMER_OUTPUT_ROOT}" \
  --seed 0 --mixed_precision bf16 \
  --vqgan_type ctx_vqgan \
  --pretrained_model_name_or_path "${TOKENIZER_DIR}" \
  --config_name configs/llama/config_surgwm_anchor.json \
  --pretrained_transformer_path "${OXE_TRANSFORMER}" \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 --lr_scheduler_type cosine --num_warmup_steps 100 \
  --dataset_format surgwmbench_anchor --surgwmbench_root "${SURGWMBENCH_ROOT}" \
  --resolution 256 --dataloader_num_workers 4 \
  --segment_length 20 --context_length 5 \
  --use_trajectory_head \
  --weight_decay 0.01 --llama_attn_drop 0.1 --embed_no_wd \
  --num_train_epochs 3 --validation_steps 500 --checkpointing_steps 1000 \
  --max_decode_batchsize 1
