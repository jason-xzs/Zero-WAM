#!/usr/bin/env bash
# Single GPU, microbatch 1, accumulation 8. Paths belong to the target server.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${MODEL_PATH:?Set MODEL_PATH to the released zero-wam-pretrain model root}"
: "${HUMAN_GEN_ROOT:?Set HUMAN_GEN_ROOT to the migrated HumanGen dataset}"
: "${ZERO_WAM_RGB_FEATURE_ROOT:?Set ZERO_WAM_RGB_FEATURE_ROOT to the prepared RGB cache}"
: "${ZERO_WAM_SAVE_ROOT:?Set ZERO_WAM_SAVE_ROOT to a new RGB experiment directory}"
export MODEL_PATH HUMAN_GEN_ROOT ZERO_WAM_RGB_FEATURE_ROOT ZERO_WAM_SAVE_ROOT
export PYTHON_BIN=${PYTHON_BIN:-python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=1
export ZERO_WAM_CPU_OFFLOAD=0
export ZERO_WAM_ACTIVATION_OFFLOAD=0
unset ZERO_WAM_SYNC_EACH_MICROBATCH ZERO_WAM_GPU_MEMORY_FRACTION
export CONFIG_NAME=robotwin_subset12_rgb_b300
export NGPU=1
export DATASETS=robotwin:1.0
export LOG_RANK=0
export MASTER_PORT=${MASTER_PORT:-29685}

# Keep the CPU preflight independent of torchrun/CUDA. Other CLI overrides,
# including the bounded smoke-test flags, use the same public training entry.
for argument in "$@"; do
    if [[ "$argument" == --check-config ]]; then
        exec "$PYTHON_BIN" -m wan_va.train --config-name "$CONFIG_NAME" \
            --datasets "$DATASETS" "$@"
    fi
done
exec bash script/train.sh "$@"
