"""Single-B300 RGB recipe matching the completed Subset12 baseline schedule.

Source: zero-wam-subset12-single-gpu-training-config.md (2026-11-03).
The model, RGB objectives and task split come from the tested four-GPU recipe;
one process accumulates eight microbatches for the same effective batch of 8.
"""
from copy import deepcopy

from .va_robotwin_subset12_rgb_train_cfg import va_robotwin_subset12_rgb_train_cfg


va_robotwin_subset12_rgb_b300_cfg = deepcopy(va_robotwin_subset12_rgb_train_cfg)
va_robotwin_subset12_rgb_b300_cfg.__name__ = 'Config: RGB transition subset12 single B300'
va_robotwin_subset12_rgb_b300_cfg.expected_world_size = 1
va_robotwin_subset12_rgb_b300_cfg.batch_size = 1
va_robotwin_subset12_rgb_b300_cfg.gradient_accumulation_steps = 8
va_robotwin_subset12_rgb_b300_cfg.expected_effective_batch_size = 8
va_robotwin_subset12_rgb_b300_cfg.num_steps = 15000
va_robotwin_subset12_rgb_b300_cfg.load_worker = 4
va_robotwin_subset12_rgb_b300_cfg.rgb_init_seed = 42
va_robotwin_subset12_rgb_b300_cfg.sync_each_microbatch = True
va_robotwin_subset12_rgb_b300_cfg.save_interval = 5000
va_robotwin_subset12_rgb_b300_cfg.enable_wandb = False
