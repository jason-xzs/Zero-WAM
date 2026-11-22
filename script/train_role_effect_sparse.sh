#!/usr/bin/env bash
# Explicit sparse-supervision recipe; exact reviewed ICL pairs and cache hashes.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHON_BIN=${PYTHON_BIN:-python}
: "${MODEL_PATH:?Set original Zero-WAM pretrained weights}"
: "${HUMAN_GEN_ROOT:?Set the HumanGen dataset root}"
: "${ZERO_WAM_RGB_FEATURE_ROOT:?Set the exact original frozen RGB cache root}"
: "${ZERO_WAM_ROLE_MANIFEST:?Run configure_sparse_role_annotations.py and set runtime_manifest.json}"
: "${ZERO_WAM_ICL_MANIFEST:?Set the matching filtered icl_manifest.train.json}"
: "${ZERO_WAM_SAVE_ROOT:?Choose a new training output directory}"
export CONFIG_NAME=${CONFIG_NAME:-robotwin_subset12_role_effect_sparse_b300}
case "$CONFIG_NAME" in
  robotwin_subset12_role_effect_sparse_b300) export NGPU=1 ;;
  robotwin_subset12_role_effect_sparse) export NGPU=4 ;;
  *) echo "Choose an explicit sparse role-effect config" >&2; exit 2 ;;
esac
export MASTER_PORT=${MASTER_PORT:-29719}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=1
export ZERO_WAM_METRICS_PATH=${ZERO_WAM_METRICS_PATH:-$ZERO_WAM_SAVE_ROOT/metrics.jsonl}
exec bash script/train.sh "$@"
