"""Check task isolation through the real config -> data-source -> loader path."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import shutil

import pytest

import wan_va.train as train
from wan_va.configs import TRAIN_DATASET_CONFIGS, VA_CONFIGS
from wan_va.configs.robotwin_tasks import ROBOTWIN_EVAL_TASKS, ROBOTWIN_SUBSET12_TASKS
from wan_va.dataset.icl_lerobot_latent_dataset import _construct_icl_dataset, _icl_repo_list


def _write_task(root, name):
    meta = root / name / 'meta'
    meta.mkdir(parents=True)
    (meta / 'info.json').write_text('{}')


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    for key in ('WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'ZERO_WAM_SYNC_EACH_MICROBATCH'):
        monkeypatch.delenv(key, raising=False)
    tasks = [*ROBOTWIN_SUBSET12_TASKS, *ROBOTWIN_EVAL_TASKS,
             *(f'other_task_{index}' for index in range(31))]
    for task in tasks:
        _write_task(tmp_path, task + '-demo_clean')
    (tmp_path / 'meta').mkdir()
    shutil.copyfile(Path(__file__).parents[1] / 'wan_va/assets/norm_stats/robotwin_icl.json',
                    tmp_path / 'meta/action_stats.json')
    return tmp_path


def _args(root, name='robotwin_subset12_train', **overrides):
    values = dict(config_name=name, datasets='robotwin:1.0', check_config=True,
                  dataset_path=str(root), enable_wandb=False, disable_wandb=False)
    for field in ('model_path', 'empty_emb_path', 'learning_rate', 'droptext_target',
                  'drop_icl', 'init_worker', 'load_worker', 'batch_size',
                  'gradient_accumulation_steps', 'num_steps', 'save_interval',
                  'save_root', 'icl_manifest_path', 'human_latent_path', 'robot_latent_path'):
        values[field] = None
    values.update(overrides)
    return SimpleNamespace(**values)


def test_full_subset_full_select_actual_sources_without_registry_mutation(data_root):
    original = deepcopy(TRAIN_DATASET_CONFIGS['robotwin'])
    for config_name, count, steps in [('robotwin_train', 43, 50000),
                                      ('robotwin_subset12_train', 12, 15000),
                                      ('robotwin_train', 43, 50000)]:
        config = train.build_training_config(_args(data_root, config_name))
        report = train.training_config_report(config)
        source = report['dataset_sources'][0]
        assert source['selected_task_count'] == count
        assert source['excluded_roots'] == 50 - count
        assert not set(source['training_tasks']) & set(ROBOTWIN_EVAL_TASKS)
        if count == 12:
            assert set(source['training_tasks']) == set(ROBOTWIN_SUBSET12_TASKS)
        assert report['num_steps'] == steps
        assert report['effective_batch_size'] == 8
        assert report['save_interval'] == 5000
    assert TRAIN_DATASET_CONFIGS['robotwin'] == original


def test_common_entry_cpu_check_never_starts_training(data_root, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail('CPU config check initialized training')
    monkeypatch.setattr(train, 'init_distributed', forbidden)
    monkeypatch.setattr(train, 'Trainer', forbidden)
    monkeypatch.setattr(train.wandb, 'init', forbidden)
    train.run(_args(data_root))
    assert json.loads(capsys.readouterr().out)['dataset_sources'][0]['selected_task_count'] == 12


def test_subsets_reject_a_missing_task_even_if_another_has_two_variants(data_root):
    missing = ROBOTWIN_SUBSET12_TASKS[0]
    shutil.rmtree(data_root / (missing + '-demo_clean'))
    _write_task(data_root, ROBOTWIN_SUBSET12_TASKS[1] + '-another_variant')
    config = train.build_training_config(_args(data_root))
    with pytest.raises(ValueError, match='missing=.*duplicate_variants='):
        train.training_config_report(config)


def test_direct_construction_cannot_bypass_subset_selection(data_root):
    config = train.build_training_config(_args(data_root)).dataset_sources[0]['config']
    with pytest.raises(ValueError, match='held-out training dataset'):
        _construct_icl_dataset(data_root / 'other_task_0-demo_clean', config)


def test_subset_rejects_eval_task_in_allowlist(data_root):
    config = deepcopy(VA_CONFIGS['robotwin_subset12_train'])
    config.dataset_path = str(data_root)
    config.included_task_names[0] = ROBOTWIN_EVAL_TASKS[0]
    with pytest.raises(ValueError, match='overlap held-out'):
        _icl_repo_list(config)


def test_subset_can_use_a_directory_containing_only_its_tasks(data_root):
    for child in data_root.iterdir():
        if child.name != 'meta' and child.name.split('-', 1)[0] not in ROBOTWIN_SUBSET12_TASKS:
            shutil.rmtree(child)
    config = train.build_training_config(_args(data_root))
    report = train.training_config_report(config)['dataset_sources'][0]
    assert report['selected_task_count'] == 12
    assert report['excluded_roots'] == 0


def test_four_gpu_recipe_rejects_accidental_global_batch_four(data_root):
    with pytest.raises(ValueError, match='requires global batch 8, got 4'):
        train.build_training_config(_args(data_root, gradient_accumulation_steps=1))


def test_training_requires_four_actual_processes(data_root, monkeypatch):
    monkeypatch.setenv('WORLD_SIZE', '2')
    with pytest.raises(ValueError, match='requires 4 GPUs'):
        train.build_training_config(_args(data_root, check_config=False))


def test_selected_recipe_cannot_silently_use_another_dataset(data_root):
    with pytest.raises(ValueError, match='requires --datasets robotwin:1.0'):
        train.build_training_config(_args(data_root, datasets='agibot:1.0'))


@pytest.mark.parametrize('config_name,dataset', [
    ('agibot_train', 'agibot'), ('robocoin_train', 'robocoin'),
    ('robomind_train', 'robomind'), ('interna1_train', 'interna1'),
    ('oxe_train', 'oxe'), ('robotwin_mixed_train', 'robotwin'),
])
def test_generic_recipes_do_not_inherit_robotwin_recipe_constraints(data_root, monkeypatch, config_name, dataset):
    monkeypatch.setenv('WORLD_SIZE', '8')
    config = train.build_training_config(_args(data_root, config_name, datasets=dataset + ':1.0'))
    assert getattr(config, 'task_selection_dataset', None) is None
    assert getattr(config, 'expected_world_size', None) is None
    assert config.gradient_accumulation_steps == 1
    assert config.enable_wandb is False
