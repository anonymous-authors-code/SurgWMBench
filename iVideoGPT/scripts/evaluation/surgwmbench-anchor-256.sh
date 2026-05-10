#!/usr/bin/env bash
set -euo pipefail

SURGWMBENCH_ROOT="${SURGWMBENCH_ROOT:-/mnt/hdd1/neurips2026_dataset_track/SurgWMBench}"
TOKENIZER_DIR="${TOKENIZER_DIR:?Set TOKENIZER_DIR to the trained SurgWMBench tokenizer directory}"
TRANSFORMER_DIR="${TRANSFORMER_DIR:?Set TRANSFORMER_DIR to the trained SurgWMBench transformer directory}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmark/outputs/ivideogpt_surgwmbench_anchor_test}"

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
