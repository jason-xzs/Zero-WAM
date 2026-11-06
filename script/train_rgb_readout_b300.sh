#!/usr/bin/env bash
# Single B300: microbatch 1 x accumulation 8 = global batch 8.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export RGB_READOUT_VARIANT=${RGB_READOUT_VARIANT:-global_local}
case "$RGB_READOUT_VARIANT" in
    static|state|global_local|teacher) ;;
    *) printf 'Invalid RGB_READOUT_VARIANT: %s\n' "$RGB_READOUT_VARIANT" >&2; exit 2 ;;
esac
# Validate before torchrun creates any GPU workers; values are also checked by
# the config factory. Overrides apply to every variant for budget controls.
readout_default_tokens=32
readout_default_global=8
if [[ "$RGB_READOUT_VARIANT" == global_local ]]; then
    readout_default_tokens=16
    readout_default_global=4
fi
export ZERO_WAM_READOUT_TOKENS=${ZERO_WAM_READOUT_TOKENS-$readout_default_tokens}
export ZERO_WAM_READOUT_GLOBAL_TOKENS=${ZERO_WAM_READOUT_GLOBAL_TOKENS-$readout_default_global}
for budget_name in ZERO_WAM_READOUT_TOKENS ZERO_WAM_READOUT_GLOBAL_TOKENS; do
    budget_value=${!budget_name}
    if [[ ! "$budget_value" =~ ^[0-9]+$ ]] || (( 10#$budget_value < 1 )); then
        printf '%s must be a positive integer\n' "$budget_name" >&2
        exit 2
    fi
done
if (( 10#$ZERO_WAM_READOUT_GLOBAL_TOKENS >= 10#$ZERO_WAM_READOUT_TOKENS )); then
    printf 'ZERO_WAM_READOUT_GLOBAL_TOKENS must be smaller than ZERO_WAM_READOUT_TOKENS\n' >&2
    exit 2
fi
: "${MODEL_PATH:?Set MODEL_PATH to the released zero-wam-pretrain model root}"
: "${HUMAN_GEN_ROOT:?Set HUMAN_GEN_ROOT to the migrated HumanGen dataset}"
: "${ZERO_WAM_RGB_FEATURE_ROOT:?Set ZERO_WAM_RGB_FEATURE_ROOT to the prepared RGB cache}"
: "${ZERO_WAM_SAVE_ROOT:?Set ZERO_WAM_SAVE_ROOT to a new experiment directory}"
if [[ "$RGB_READOUT_VARIANT" == teacher ]]; then
    : "${ZERO_WAM_FUTURE_TEACHER:?Set ZERO_WAM_FUTURE_TEACHER to the trained teacher artifact}"
    export ZERO_WAM_FUTURE_TEACHER
fi
export MODEL_PATH HUMAN_GEN_ROOT ZERO_WAM_RGB_FEATURE_ROOT ZERO_WAM_SAVE_ROOT
export PYTHON_BIN=${PYTHON_BIN:-python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=1
export ZERO_WAM_CPU_OFFLOAD=0
export ZERO_WAM_ACTIVATION_OFFLOAD=0
unset ZERO_WAM_SYNC_EACH_MICROBATCH ZERO_WAM_GPU_MEMORY_FRACTION
export CONFIG_NAME="robotwin_subset12_rgb_readout_${RGB_READOUT_VARIANT}_b300"
export NGPU=1
export DATASETS=robotwin:1.0
export LOG_RANK=0
export MASTER_PORT=${MASTER_PORT:-29690}

for argument in "$@"; do
    if [[ "$argument" == --check-config ]]; then
        exec "$PYTHON_BIN" -m wan_va.train --config-name "$CONFIG_NAME" \
            --datasets "$DATASETS" "$@"
    fi
done
exec bash script/train.sh "$@"
