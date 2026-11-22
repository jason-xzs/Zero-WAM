"""C1 prototype. Fresh Zero-WAM initialization; reviewed labels are mandatory.

All loss coefficients below are initial experimental choices, not tuned claims.
The smoke recipe explicitly permits partial annotation and reports coverage.
"""
from copy import deepcopy
import os
from pathlib import Path

from .va_robotwin_subset12_rgb_readout_cfg import make_rgb_readout_config


def make_role_effect_config(platform="train"):
    cfg = deepcopy(make_rgb_readout_config("state", platform))
    cfg.__name__ = "Config: role/effect subset12 train"
    cfg.role_manifest_path = os.environ.get(
        "ZERO_WAM_ROLE_MANIFEST", "/workspace/zero-wam/datasets/role_effect/manifest.json"
    )
    partial = os.environ.get("ZERO_WAM_ROLE_ALLOW_PARTIAL", "0")
    if partial not in {"0", "1"}:
        raise ValueError("ZERO_WAM_ROLE_ALLOW_PARTIAL must be 0 or 1")
    cfg.role_require_complete = partial == "0"
    cfg.rgb_transition_config.update(
        role_effect_enabled=True,
        role_effect_temporal_encoding="learned",
        role_num_robot_views=3,
        role_human_weight=1.0,
        role_robot_weight=1.0,
        object_future_weight=1.0,
        relation_future_weight=1.0,
        donor_effect_weight=1.0,
    )
    return cfg


va_robotwin_subset12_role_effect_cfg = make_role_effect_config()


def make_sparse_role_effect_config(platform):
    """Reviewed policy pairs; visible-region and effect labels may be sparse.

    This is a new experimental protocol, not a bypass of the legacy complete
    annotation preflight. Missing review records still fail before training.
    """
    cfg = make_role_effect_config(platform)
    cfg.__name__ = f'Config: role/effect reviewed sparse subset12 {platform}'
    cfg.role_require_complete = False
    cfg.role_require_all_records = True
    cfg.icl_manifest_path = os.environ.get(
        'ZERO_WAM_ICL_MANIFEST', str(Path(cfg.role_manifest_path).parent / 'icl_manifest.train.json'))
    cfg.rgb_transition_config['role_future_target_mode'] = 'all_reviewed'
    return cfg


va_robotwin_subset12_role_effect_sparse_cfg = make_sparse_role_effect_config('train')
va_robotwin_subset12_role_effect_sparse_b300_cfg = make_sparse_role_effect_config('b300')
