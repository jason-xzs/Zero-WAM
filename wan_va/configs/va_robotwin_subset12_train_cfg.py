"""The shared four-GPU RoboTwin recipe, restricted to 12 training tasks."""
from copy import deepcopy

from .robotwin_tasks import ROBOTWIN_SUBSET12_TASKS
from .va_robotwin_full_train_cfg import va_robotwin_full_train_cfg

va_robotwin_subset12_train_cfg = deepcopy(va_robotwin_full_train_cfg)
va_robotwin_subset12_train_cfg.__name__ = 'Config: VA robotwin subset12 train'
va_robotwin_subset12_train_cfg.included_task_names = list(ROBOTWIN_SUBSET12_TASKS)
va_robotwin_subset12_train_cfg.expected_num_train_tasks = 12
va_robotwin_subset12_train_cfg.num_steps = 15000
