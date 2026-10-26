# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from .va_franka_cfg import va_franka_cfg
from .va_robotwin_cfg import va_robotwin_cfg
from .va_franka_i2va import va_franka_i2va_cfg
from .va_robotwin_i2va import va_robotwin_i2va_cfg
from .va_robotwin_train_cfg import va_robotwin_train_cfg
from .va_robotwin_full_train_cfg import va_robotwin_full_train_cfg
from .va_robotwin_subset12_train_cfg import va_robotwin_subset12_train_cfg
from .va_agibot_train_cfg import va_agibot_train_cfg
from .va_robocoin_train_cfg import va_robocoin_train_cfg
from .va_robomind_train_cfg import va_robomind_train_cfg
from .va_interna1_train_cfg import va_interna1_train_cfg
from .va_oxe_train_cfg import va_oxe_train_cfg
from .va_demo_cfg import va_demo_cfg
from .va_demo_i2va import va_demo_i2va_cfg

VA_CONFIGS = {
    'robotwin': va_robotwin_cfg,
    'franka': va_franka_cfg,
    'robotwin_i2va': va_robotwin_i2va_cfg,
    'franka_i2va': va_franka_i2va_cfg,
    'robotwin_train': va_robotwin_full_train_cfg,
    'robotwin_subset12_train': va_robotwin_subset12_train_cfg,
    # Keep the original generic recipe for the upstream multi-dataset example.
    'robotwin_mixed_train': va_robotwin_train_cfg,
    'agibot_train': va_agibot_train_cfg,
    'robocoin_train': va_robocoin_train_cfg,
    'robomind_train': va_robomind_train_cfg,
    'interna1_train': va_interna1_train_cfg,
    'oxe_train': va_oxe_train_cfg,
    'demo': va_demo_cfg,
    'demo_i2va': va_demo_i2va_cfg,
}

TRAIN_DATASET_CONFIGS = {
    'robotwin': va_robotwin_train_cfg,
    'agibot': va_agibot_train_cfg,
    'robocoin': va_robocoin_train_cfg,
    'robomind': va_robomind_train_cfg,
    'interna1': va_interna1_train_cfg,
    'oxe': va_oxe_train_cfg,
}
