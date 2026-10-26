# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .zerowam_train_config import zerowam_train_cfg
from .va_robotwin_cfg import va_robotwin_cfg
from .robotwin_tasks import ROBOTWIN_EVAL_TASKS
import os
from pathlib import Path

va_robotwin_train_cfg = EasyDict(__name__='Config: VA robotwin train')
va_robotwin_train_cfg.update(va_robotwin_cfg)
va_robotwin_train_cfg.update(zerowam_train_cfg)

_repo_root = Path(__file__).resolve().parents[2]
_human_gen_root = Path(os.environ.get(
    'HUMAN_GEN_ROOT', str(_repo_root / 'data' / 'HumanGen')
))
va_robotwin_train_cfg.model_path = os.environ.get(
    'MODEL_PATH', '/path/to/zero-wam-pretrain'
)
va_robotwin_train_cfg.dataset_path = str(_human_gen_root / 'robotwin_data')
va_robotwin_train_cfg.action_stats_path = str(
    Path(va_robotwin_train_cfg.dataset_path) / 'meta' / 'action_stats.json'
)
va_robotwin_train_cfg.load_robotwin_stats_from_dataset = True
va_robotwin_train_cfg.icl_manifest_path = str(
    _human_gen_root / 'icl_configs' / 'ICL_config_robotwin.json'
)
va_robotwin_train_cfg.human_latent_path = str(
    _human_gen_root / 'human_latents' / 'robotwin'
)
# Keep the seven zero-shot evaluation tasks out of every Robotwin training run.
# Released task directories may append a dataset variant after a hyphen.
va_robotwin_train_cfg.excluded_task_names = list(ROBOTWIN_EVAL_TASKS)
va_robotwin_train_cfg.expected_num_train_tasks = 43
va_robotwin_train_cfg.empty_emb_path = va_robotwin_cfg.empty_text_emb_path
va_robotwin_train_cfg.save_root = os.environ.get(
    'ZERO_WAM_SAVE_ROOT', '/path/to/your/output'
)
va_robotwin_train_cfg.enable_wandb = False
va_robotwin_train_cfg.init_worker = 1
va_robotwin_train_cfg.load_worker = 16
va_robotwin_train_cfg.save_interval = 1000
va_robotwin_train_cfg.gc_interval = 50
va_robotwin_train_cfg.cfg_prob = va_robotwin_train_cfg.droptext_target

# Training parameters
va_robotwin_train_cfg.learning_rate = 1e-4
va_robotwin_train_cfg.beta1 = 0.9
va_robotwin_train_cfg.beta2 = 0.95
va_robotwin_train_cfg.weight_decay = 0.01
va_robotwin_train_cfg.warmup_steps = 200
va_robotwin_train_cfg.max_norm = 1.0
va_robotwin_train_cfg.skip_step_grad_norm_multiplier = 20.0
va_robotwin_train_cfg.batch_size = 1 
va_robotwin_train_cfg.gradient_accumulation_steps = 1
va_robotwin_train_cfg.num_steps = 50000
