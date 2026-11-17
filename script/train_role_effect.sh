#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHON_BIN=${PYTHON_BIN:-/share/personal/xiangzishang/envs/zero-wam/bin/python}
export MODEL_PATH=${MODEL_PATH:-/workspace/zero-wam/checkpoints/zero-wam-pretrain}
export HUMAN_GEN_ROOT=${HUMAN_GEN_ROOT:-/workspace/zero-wam/datasets/HumanGen}
export ZERO_WAM_RGB_FEATURE_ROOT=${ZERO_WAM_RGB_FEATURE_ROOT:-/workspace/zero-wam/datasets/rgb_transition_dinov2_base}
export ZERO_WAM_SAVE_ROOT=${ZERO_WAM_SAVE_ROOT:-/workspace/zero-wam/runs/role-effect-subset12}
export ZERO_WAM_METRICS_PATH=${ZERO_WAM_METRICS_PATH:-$ZERO_WAM_SAVE_ROOT/metrics.jsonl}
: "${ZERO_WAM_ROLE_MANIFEST:?Set the reviewed role manifest; full training requires coverage of all selected pairs}"
export CONFIG_NAME=robotwin_subset12_role_effect
export NGPU=4
export MASTER_PORT=${MASTER_PORT:-29717}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=1
# Triton needs a host compiler; this machine keeps it in a separate environment.
# Respect a supplied CC or a working system compiler and do not mutate any env.
if [[ -z "${CC:-}" ]] && ! command -v gcc >/dev/null && ! command -v cc >/dev/null; then
    ROLE_TOOLCHAIN=/share/personal/xiangzishang/envs/lingbot-toolchain/bin
    if [[ -x "$ROLE_TOOLCHAIN/x86_64-conda-linux-gnu-gcc" ]]; then
        export CC="$ROLE_TOOLCHAIN/x86_64-conda-linux-gnu-gcc"
        export CXX="$ROLE_TOOLCHAIN/x86_64-conda-linux-gnu-g++"
        export PATH="$ROLE_TOOLCHAIN:$PATH"
    fi
fi
export ZERO_WAM_CPU_OFFLOAD=0
export ZERO_WAM_ACTIVATION_OFFLOAD=0
exec bash script/train.sh "$@"
