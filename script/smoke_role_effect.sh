#!/usr/bin/env bash
# Real backbone training, explicitly partial annotation, bounded to <=20 steps.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHON_BIN=${PYTHON_BIN:-/share/personal/xiangzishang/envs/zero-wam/bin/python}
export ZERO_WAM_RGB_FEATURE_ROOT=${ZERO_WAM_RGB_FEATURE_ROOT:-/workspace/zero-wam/datasets/rgb_transition_dinov2_base}
: "${ZERO_WAM_ROLE_MANIFEST:?Set the reviewed role manifest, including at least one legal donor pair}"
export ZERO_WAM_SAVE_ROOT=${ZERO_WAM_SAVE_ROOT:-/workspace/zero-wam/runs/role-effect-smoke}
export ZERO_WAM_ROLE_ALLOW_PARTIAL=1
REPORT=${RGB_SMOKE_REPORT:-$ZERO_WAM_RGB_FEATURE_ROOT/smoke_report.json}
STEPS=${ROLE_SMOKE_STEPS:-6}
SAMPLE_IDS=$("$PYTHON_BIN" - "$REPORT" "$ZERO_WAM_ROLE_MANIFEST" <<'PY'
import json, sys
baseline = json.load(open(sys.argv[1]))
roles = json.load(open(sys.argv[2]))
assert baseline['smoke_subset'] and baseline['tasks'] == 12
assert any(record.get('donor_pair_key') for record in roles['records'])
keys = baseline['sample_ids'] + [record['pair_key'] for record in roles['records']]
print(','.join(dict.fromkeys(keys)))
PY
)
exec bash script/train_role_effect.sh --smoke-test --rgb-smoke-sample-ids "$SAMPLE_IDS" \
    --num-steps "$STEPS" --save-interval "$STEPS" --disable-wandb --load-worker 0 --drop-icl 0 "$@"
