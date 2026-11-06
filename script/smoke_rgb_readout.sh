#!/usr/bin/env bash
# Bounded four-GPU real-model integration test; no 15,000-step run.
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
export PYTHON_BIN=${PYTHON_BIN:-/share/personal/xiangzishang/envs/zero-wam/bin/python}
export MODEL_PATH=${MODEL_PATH:-/workspace/zero-wam/checkpoints/zero-wam-pretrain}
export HUMAN_GEN_ROOT=${HUMAN_GEN_ROOT:-/workspace/zero-wam/datasets/HumanGen}
export ZERO_WAM_RGB_FEATURE_ROOT=${ZERO_WAM_RGB_FEATURE_ROOT:-/workspace/zero-wam/datasets/rgb_transition_dinov2_base}
export ZERO_WAM_SAVE_ROOT=${ZERO_WAM_SAVE_ROOT:-/workspace/zero-wam/runs/rgb-readout-${RGB_READOUT_VARIANT}-smoke}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=1
export ZERO_WAM_CPU_OFFLOAD=0
export ZERO_WAM_ACTIVATION_OFFLOAD=0
unset ZERO_WAM_SYNC_EACH_MICROBATCH ZERO_WAM_GPU_MEMORY_FRACTION
if [[ "$RGB_READOUT_VARIANT" == teacher ]]; then
    : "${ZERO_WAM_FUTURE_TEACHER:?Set ZERO_WAM_FUTURE_TEACHER to the trained teacher artifact}"
    export ZERO_WAM_FUTURE_TEACHER
fi
REPORT=${RGB_SMOKE_REPORT:-$ZERO_WAM_RGB_FEATURE_ROOT/smoke_report.json}
STEPS=${RGB_SMOKE_STEPS:-6}
if [[ ! "$STEPS" =~ ^[0-9]+$ ]] || (( 10#$STEPS < 1 || 10#$STEPS > 20 )); then
    printf 'RGB_SMOKE_STEPS must be an integer from 1 to 20\n' >&2
    exit 2
fi
SAMPLE_IDS=$("$PYTHON_BIN" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["smoke_subset"] and r["pairs"] == 12 and r["tasks"] == 12; print(",".join(r["sample_ids"]))' "$REPORT")
export CONFIG_NAME="robotwin_subset12_rgb_readout_${RGB_READOUT_VARIANT}_train"
export NGPU=4
export DATASETS=robotwin:1.0
export LOG_RANK=0
export MASTER_PORT=${MASTER_PORT:-29691}
exec bash script/train.sh --smoke-test --rgb-smoke-sample-ids "$SAMPLE_IDS" \
    --num-steps "$STEPS" --save-interval "$STEPS" --disable-wandb --load-worker 0 "$@"
