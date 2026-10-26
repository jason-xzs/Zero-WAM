"""Four-GPU, batch-8 RoboTwin recipe without changing other datasets' defaults."""
from copy import deepcopy

from .va_robotwin_train_cfg import va_robotwin_train_cfg

va_robotwin_full_train_cfg = deepcopy(va_robotwin_train_cfg)
va_robotwin_full_train_cfg.__name__ = 'Config: VA robotwin full four-GPU train'
va_robotwin_full_train_cfg.task_selection_dataset = 'robotwin'
va_robotwin_full_train_cfg.included_task_names = None
va_robotwin_full_train_cfg.expected_num_train_tasks = 43
va_robotwin_full_train_cfg.num_steps = 50000
va_robotwin_full_train_cfg.expected_world_size = 4
va_robotwin_full_train_cfg.expected_effective_batch_size = 8
va_robotwin_full_train_cfg.gradient_accumulation_steps = 2
va_robotwin_full_train_cfg.sync_each_microbatch = True
va_robotwin_full_train_cfg.load_worker = 4
va_robotwin_full_train_cfg.save_interval = 5000
va_robotwin_full_train_cfg.enable_wandb = True
va_robotwin_full_train_cfg.wandb_project = 'zero-wam'
