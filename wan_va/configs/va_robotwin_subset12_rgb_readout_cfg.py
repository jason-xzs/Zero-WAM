"""Controlled demo-readout variants on the existing subset12 RGB recipes.

All variants keep the same frozen feature space, original human Wan context,
and RGB auxiliary objectives. Global/local defaults to 32 -> 16 readout-token
compression; environment overrides support controlled, matched-budget runs.
Teacher training is a separate stage requiring a compatible trained artifact.
"""
from copy import deepcopy
import os

from .va_robotwin_subset12_rgb_b300_cfg import va_robotwin_subset12_rgb_b300_cfg
from .va_robotwin_subset12_rgb_train_cfg import va_robotwin_subset12_rgb_train_cfg


READOUT_VARIANTS = ("static", "state", "global_local", "teacher")


def readout_budget(variant):
    defaults = (16, 4) if variant == "global_local" else (32, 8)
    values = []
    for name, default in zip(("ZERO_WAM_READOUT_TOKENS", "ZERO_WAM_READOUT_GLOBAL_TOKENS"), defaults):
        raw = os.environ.get(name, str(default))
        if not raw.isascii() or not raw.isdecimal() or int(raw) < 1:
            raise ValueError(f"{name} must be a positive integer")
        values.append(int(raw))
    if values[1] >= values[0]:
        raise ValueError("ZERO_WAM_READOUT_GLOBAL_TOKENS must be smaller than ZERO_WAM_READOUT_TOKENS")
    return tuple(values)


def make_rgb_readout_config(variant, platform):
    if variant not in READOUT_VARIANTS:
        raise ValueError(f"Unknown RGB readout variant: {variant!r}")
    if platform not in {"b300", "train"}:
        raise ValueError(f"Unknown RGB readout platform: {platform!r}")
    source = (va_robotwin_subset12_rgb_b300_cfg if platform == "b300"
              else va_robotwin_subset12_rgb_train_cfg)
    cfg = deepcopy(source)
    cfg.__name__ = f"Config: RGB readout {variant} subset12 {platform}"
    cfg.rgb_init_seed = 42
    tokens, global_tokens = readout_budget(variant)
    cfg.rgb_transition_config.update(
        readout_mode="state" if variant == "teacher" else variant,
        readout_num_tokens=tokens,
        readout_global_tokens=global_tokens,
    )
    cfg.future_teacher_checkpoint = (
        os.environ.get("ZERO_WAM_FUTURE_TEACHER") if variant == "teacher" else None
    )
    # Initial experimental coefficient, not a claimed reproduction of WoG's
    # VLM-space alignment coefficient or a tuned success-rate optimum.
    cfg.future_teacher_loss_weight = 0.1 if variant == "teacher" else 0.0
    return cfg


READOUT_CONFIGS = {
    f"robotwin_subset12_rgb_readout_{variant}_{platform}":
        make_rgb_readout_config(variant, platform)
    for variant in READOUT_VARIANTS
    for platform in ("b300", "train")
}
