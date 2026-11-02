"""RGB-only WALA transition conditioning on the existing 12-task recipe.

The frozen DINOv2 features are prepared offline. This configuration does not
instantiate a depth estimator or change the video/action/MCP objectives.
"""
import os
from copy import deepcopy

from .va_robotwin_subset12_train_cfg import va_robotwin_subset12_train_cfg


va_robotwin_subset12_rgb_train_cfg = deepcopy(va_robotwin_subset12_train_cfg)
va_robotwin_subset12_rgb_train_cfg.__name__ = 'Config: RGB transition subset12'
va_robotwin_subset12_rgb_train_cfg.enable_rgb_transition = True
va_robotwin_subset12_rgb_train_cfg.rgb_feature_root = os.environ.get(
    'ZERO_WAM_RGB_FEATURE_ROOT', '/workspace/zero-wam/datasets/rgb_transition_dinov2_base'
)
va_robotwin_subset12_rgb_train_cfg.rgb_encoder_id = 'facebook/dinov2-base'
va_robotwin_subset12_rgb_train_cfg.rgb_encoder_revision = 'f9e44c814b77203eaa57a6bdbbd535f21ede1415'
va_robotwin_subset12_rgb_train_cfg.rgb_transition_config = {
    'feature_dim': 768,
    'hidden_dim': 256,
    'num_transition_tokens': 4,
    'encoder_layers': 2,
    'decoder_layers': 2,
    'num_heads': 4,
    'max_clips': 16,
    'max_future_steps': 64,
    'max_patches': 256,
    'feature_normalization': 'layer_norm',
    'l1_weight': 1.0,
    'cosine_weight': 0.1,
    'human_self_weight': 1.0,
    'robot_self_weight': 1.0,
    'transfer_weight': 1.0,
}
va_robotwin_subset12_rgb_train_cfg.rgb_loss_weight = 0.1
