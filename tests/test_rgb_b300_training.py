"""Exercise the single-GPU recipe, real accumulation, and launcher routing."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest
import torch

from wan_va.configs import VA_CONFIGS
from wan_va.train import build_training_config
from test_rgb_training import _args, _loss_input, _trainer, config_data_root


RECIPE = 'robotwin_subset12_rgb_b300'
ROOT = Path(__file__).resolve().parents[1]


def test_single_gpu_recipe_matches_baseline_without_mutating_four_gpu_config(config_data_root):
    original = deepcopy(VA_CONFIGS['robotwin_subset12_rgb_train'])
    cfg = build_training_config(_args(
        config_name=RECIPE, dataset_path=config_data_root,
        smoke_test=False, rgb_smoke_sample_ids=None, num_steps=None,
    ))
    assert (cfg.world_size, cfg.batch_size, cfg.gradient_accumulation_steps) == (1, 1, 8)
    assert cfg.expected_effective_batch_size == 8
    assert (cfg.num_steps, cfg.save_interval, cfg.load_worker) == (15000, 5000, 4)
    assert (cfg.learning_rate, cfg.warmup_steps) == (1e-4, 200)
    assert (cfg.beta1, cfg.beta2, cfg.weight_decay) == (0.9, 0.95, 0.01)
    assert (cfg.max_norm, cfg.skip_step_grad_norm_multiplier) == (1.0, 20.0)
    assert cfg.param_dtype == torch.bfloat16
    assert cfg.rgb_init_seed == 42 and cfg.sync_each_microbatch
    assert cfg.enable_wandb is False
    assert cfg.enable_mcp and cfg.num_mcp_modules == 4
    assert cfg.rgb_transition_config == original.rgb_transition_config
    source = cfg.dataset_sources[0]['config']
    assert source.expected_num_train_tasks == 12
    assert source.included_task_names == original.included_task_names
    assert source.excluded_task_names == original.excluded_task_names
    assert source.enable_rgb_transition and source.world_size == 1
    cfg.rgb_transition_config['hidden_dim'] = 8
    assert VA_CONFIGS['robotwin_subset12_rgb_train'] == original
    assert VA_CONFIGS[RECIPE].rgb_transition_config['hidden_dim'] == 256
    assert (original.expected_world_size, original.gradient_accumulation_steps) == (4, 2)
    assert original.enable_wandb is True


def test_single_gpu_recipe_rejects_old_four_rank_launch(config_data_root, monkeypatch):
    monkeypatch.setenv('WORLD_SIZE', '4')
    with pytest.raises(ValueError, match='requires 1 GPUs'):
        build_training_config(_args(config_name=RECIPE, dataset_path=config_data_root))


@pytest.mark.parametrize('accumulation', [1, 2, 4])
def test_single_gpu_recipe_rejects_smaller_global_batch(config_data_root, accumulation):
    with pytest.raises(ValueError, match='requires global batch 8'):
        build_training_config(_args(config_name=RECIPE, dataset_path=config_data_root,
                                    gradient_accumulation_steps=accumulation))


def test_single_gpu_smoke_remains_bounded_and_wandb_is_opt_in(config_data_root):
    cfg = build_training_config(_args(config_name=RECIPE, dataset_path=config_data_root))
    assert cfg.smoke_test and cfg.num_steps == 2
    assert cfg.world_size * cfg.batch_size * cfg.gradient_accumulation_steps == 8
    assert cfg.dataset_sources[0]['config'].rgb_smoke_sample_ids == cfg.rgb_smoke_sample_ids
    assert not cfg.enable_wandb
    cfg = build_training_config(_args(config_name=RECIPE, dataset_path=config_data_root,
                                     enable_wandb=True))
    assert cfg.enable_wandb and cfg.wandb_project == 'zero-wam'
    with pytest.raises(ValueError, match='num-steps between 1 and 20'):
        build_training_config(_args(config_name=RECIPE, dataset_path=config_data_root, num_steps=15000))


def test_eight_microbatches_update_once_with_the_full_batch_gradient():
    trainer = _trainer(sync_each_microbatch=True)
    trainer.gradient_accumulation_steps = 8
    trainer.config.world_size = 1
    results = []
    for i in range(8):
        results.append(trainer._train_step(_loss_input(i + 1), i))
        if i < 7:
            assert not results[-1]['should_log']
            assert trainer.lr_scheduler.last_epoch == 0
            torch.testing.assert_close(trainer.transformer.backbone, torch.ones(1))
    assert results[-1]['should_log'] and not results[-1]['optimizer_step_skipped']
    assert trainer.lr_scheduler.last_epoch == 1
    assert trainer.transformer.sync_calls == [True] * 8
    # mean(1**2, ..., 8**2)=25.5. The two base losses give gradient 4*25.5;
    # RGB gives 0.1*(1+2+3)*2*25.5. One SGD update uses their full-window mean.
    torch.testing.assert_close(trainer.transformer.backbone, torch.tensor([-0.02]))
    torch.testing.assert_close(trainer.transformer.rgb_transition.encoder_weight,
                               torch.tensor([0.694]))
    torch.testing.assert_close(sum(r['rgb_loss'] for r in results), torch.tensor(15.3))


@pytest.fixture
def launcher_env(tmp_path):
    capture = tmp_path / 'capture.json'
    executable = tmp_path / 'python with spaces'
    executable.write_text(
        '#!/usr/bin/env python3\n'
        'import json, os, sys\n'
        'from pathlib import Path\n'
        'keys = ("NGPU", "CONFIG_NAME", "DATASETS", "CUDA_VISIBLE_DEVICES", "LOG_RANK", '
        '"ZERO_WAM_CPU_OFFLOAD", "ZERO_WAM_ACTIVATION_OFFLOAD", '
        '"ZERO_WAM_SYNC_EACH_MICROBATCH", "ZERO_WAM_GPU_MEMORY_FRACTION", '
        '"MODEL_PATH", "HUMAN_GEN_ROOT", "ZERO_WAM_RGB_FEATURE_ROOT", "ZERO_WAM_SAVE_ROOT")\n'
        'Path(os.environ["LAUNCH_CAPTURE"]).write_text(json.dumps('
        '{"args":sys.argv[1:], "env":{k:os.environ.get(k) for k in keys}}))\n'
    )
    executable.chmod(0o755)
    env = {**os.environ, 'PYTHON_BIN': str(executable), 'LAUNCH_CAPTURE': str(capture),
           'MODEL_PATH': '/workspace/model with spaces', 'HUMAN_GEN_ROOT': '/workspace/data',
           'ZERO_WAM_RGB_FEATURE_ROOT': '/workspace/rgb', 'ZERO_WAM_SAVE_ROOT': '/workspace/new-run',
           'CUDA_VISIBLE_DEVICES': '4', 'NGPU': '4', 'CONFIG_NAME': 'robotwin_train',
           'DATASETS': 'agibot:1.0', 'LOG_RANK': '3', 'ZERO_WAM_CPU_OFFLOAD': '1',
           'ZERO_WAM_ACTIVATION_OFFLOAD': '1', 'ZERO_WAM_SYNC_EACH_MICROBATCH': '0',
           'ZERO_WAM_GPU_MEMORY_FRACTION': '0.1'}
    return env, capture


@pytest.mark.parametrize('check_only', [False, True])
def test_launcher_routes_single_process_and_cpu_preflight(launcher_env, check_only, tmp_path):
    env, capture = launcher_env
    extra = ['--save-root', '/workspace/output with spaces', '--num-steps', '2']
    if check_only:
        extra.append('--check-config')
    subprocess.run(['bash', str(ROOT / 'script/train_rgb_b300.sh'), *extra],
                   cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
    actual = json.loads(capture.read_text())
    args, child_env = actual['args'], actual['env']
    if check_only:
        assert args[:2] == ['-m', 'wan_va.train']
        assert 'torch.distributed.run' not in args
    else:
        assert args[:2] == ['-m', 'torch.distributed.run']
        assert '--nproc_per_node=1' in args
        assert args[args.index('--local-ranks-filter=0')].endswith('=0')
    assert args[args.index('--config-name') + 1] == RECIPE
    assert args[args.index('--datasets') + 1] == 'robotwin:1.0'
    assert args[-len(extra):] == extra
    assert child_env['NGPU'] == '1' and child_env['CUDA_VISIBLE_DEVICES'] == '4'
    assert child_env['MODEL_PATH'] == env['MODEL_PATH']
    assert child_env['ZERO_WAM_CPU_OFFLOAD'] == child_env['ZERO_WAM_ACTIVATION_OFFLOAD'] == '0'
    assert child_env['ZERO_WAM_SYNC_EACH_MICROBATCH'] is None
    assert child_env['ZERO_WAM_GPU_MEMORY_FRACTION'] is None


@pytest.mark.parametrize('missing', ['MODEL_PATH', 'HUMAN_GEN_ROOT',
                                    'ZERO_WAM_RGB_FEATURE_ROOT', 'ZERO_WAM_SAVE_ROOT'])
def test_launcher_requires_explicit_target_server_paths(launcher_env, missing):
    env, capture = launcher_env
    env.pop(missing)
    result = subprocess.run(['bash', str(ROOT / 'script/train_rgb_b300.sh'), '--check-config'],
                            env=env, capture_output=True, text=True)
    assert result.returncode != 0 and missing in result.stderr
    assert not capture.exists()
