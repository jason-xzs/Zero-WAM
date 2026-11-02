#!/usr/bin/env bash
# A bounded real-model integration run. Never launches the 15,000-step recipe.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHON_BIN=${PYTHON_BIN:-/share/personal/xiangzishang/envs/zero-wam/bin/python}
export MODEL_PATH=${MODEL_PATH:-/workspace/zero-wam/checkpoints/zero-wam-pretrain}
export HUMAN_GEN_ROOT=${HUMAN_GEN_ROOT:-/workspace/zero-wam/datasets/HumanGen}
export ZERO_WAM_RGB_FEATURE_ROOT=${ZERO_WAM_RGB_FEATURE_ROOT:-/workspace/zero-wam/datasets/rgb_transition_dinov2_base}
export ZERO_WAM_SAVE_ROOT=${ZERO_WAM_SAVE_ROOT:-/workspace/zero-wam/runs/rgb-transition-smoke}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=1
export ZERO_WAM_CPU_OFFLOAD=0
export ZERO_WAM_ACTIVATION_OFFLOAD=0
unset ZERO_WAM_SYNC_EACH_MICROBATCH ZERO_WAM_GPU_MEMORY_FRACTION
REPORT=${RGB_SMOKE_REPORT:-$ZERO_WAM_RGB_FEATURE_ROOT/smoke_report.json}
STEPS=${RGB_SMOKE_STEPS:-6}
SAMPLE_IDS=$("$PYTHON_BIN" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["smoke_subset"] and r["pairs"] == 12 and r["tasks"] == 12; print(",".join(r["sample_ids"]))' "$REPORT")
export CONFIG_NAME=robotwin_subset12_rgb_train
export NGPU=4
export MASTER_PORT=${MASTER_PORT:-29683}
exec bash script/train.sh --smoke-test --rgb-smoke-sample-ids "$SAMPLE_IDS" \
    --num-steps "$STEPS" --save-interval "$STEPS" --disable-wandb --load-worker 0 "$@"
