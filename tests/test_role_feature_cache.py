"""Reject stale/illegal supervision rather than silently training on it."""
import json
from pathlib import Path

import pytest
import torch

from wan_va.dataset.role_feature_cache import RoleManifest, file_sha256, usable_future_pairs


def _fixture(tmp_path):
    records = []
    payloads = []
    for i in range(2):
        key = f'{i}:pair'
        hp, rp = tmp_path / f'h{i}.pth', tmp_path / f'r{i}.pth'
        human = {'features': torch.randn(2, 2, 4, 8), 'frame_indices': torch.tensor([[0, 1], [2, 3]]),
                 'metadata': {'schema_version': 1, 'kind': 'human', 'encoder_id': 'dino',
                              'encoder_revision': 'fixed', 'preprocessing': {'resize': 28},
                              'sample_id': str(i), 'source_video_path': f'human_{i}.mp4'}}
        robot = {'features': torch.randn(3, 4, 8), 'frame_indices': torch.arange(3),
                 'metadata': dict(human['metadata'], kind='robot', source_video_path=f'robot_{i}.mp4')}
        torch.save(human, hp)
        torch.save(robot, rp)
        hashes = {'human_cache_sha256': file_sha256(hp), 'robot_cache_sha256': file_sha256(rp)}
        label = {'metadata': {'schema_version': 1, 'pair_key': key, **hashes}}
        for kind, payload in [('human', human), ('robot', robot)]:
            label[f'{kind}_masks'] = torch.full((*payload['features'].shape[:-2], 2, 4), .25)
            label[f'{kind}_valid'] = torch.ones(*payload['features'].shape[:-2], 2, dtype=torch.bool)
            label[f'{kind}_frame_indices'] = payload['frame_indices']
        ap = tmp_path / f'label{i}.pth'
        torch.save(label, ap)
        records.append({'pair_key': key, 'task_name': 'place_mouse_pad', 'split': 'train',
                        'reviewed': True, 'provenance': 'synthetic unit-test labels; not research data',
                        'annotation_file': str(ap), 'annotation_sha256': file_sha256(ap),
                        'human_cache': str(hp), **hashes, 'stable_roles': True, 'single_effect': True,
                        'effect_signature': {'relation': 'on', 'direction': 'establish',
                                             'source_type': 'mouse', 'reference_type': 'mat',
                                             'affordance': 'support'}})
        payloads.append((human, robot, hp, rp))
    records[0].update(donor_pair_key='1:pair', donor_review='Explicit test pair relation match')
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'schema_version': 1, 'split': 'train', 'records': records}))
    return path, records, payloads


def _write(path, records):
    path.write_text(json.dumps({'schema_version': 1, 'split': 'train', 'records': records}))


def test_real_donor_is_distinct_and_missing_labels_are_not_negatives(tmp_path):
    path, _, payloads = _fixture(tmp_path)
    manifest = RoleManifest(path, ['place_mouse_pad'])
    labels, donor, valid = manifest.load('0:pair', *payloads[0])
    assert valid and torch.equal(donor, payloads[1][0]['features'])
    assert labels['human_valid'].all()
    empty, donor, valid = manifest.load('absent', *payloads[0], require_complete=False)
    assert not valid and not empty['human_valid'].any() and not empty['robot_valid'].any()
    with pytest.raises(ValueError, match='Missing reviewed'):
        manifest.load('absent', *payloads[0])


@pytest.mark.parametrize('mutation,match', [
    (lambda r: r[0].update(split='test'), 'held-out'),
    (lambda r: r[0].update(task_name='heldout'), 'held-out'),
    (lambda r: r[0].update(reviewed=False), 'review'),
    (lambda r: r[0].update(donor_pair_key='0:pair'), 'different'),
    (lambda r: r[0].update(donor_review=''), 'matching task'),
    (lambda r: r[1].update(stable_roles=False), 'stable roles'),
    (lambda r: r[1]['effect_signature'].update(direction='remove'), 'signatures'),
    (lambda r: r[1].update(human_cache_sha256=r[0]['human_cache_sha256']), 'different demonstrated'),
])
def test_illegal_donors_are_rejected(tmp_path, mutation, match):
    path, records, _ = _fixture(tmp_path)
    mutation(records)
    _write(path, records)
    with pytest.raises(ValueError, match=match):
        RoleManifest(path, ['place_mouse_pad'])


def test_timeline_and_source_corruption_are_rejected(tmp_path):
    path, records, payloads = _fixture(tmp_path)
    ap = Path(records[0]['annotation_file'])
    label = torch.load(ap, weights_only=True)
    label['robot_frame_indices'] = label['robot_frame_indices'] + 1
    torch.save(label, ap)
    records[0]['annotation_sha256'] = file_sha256(ap)
    _write(path, records)
    with pytest.raises(ValueError, match='timeline'):
        RoleManifest(path, ['place_mouse_pad']).load('0:pair', *payloads[0])
    payloads[0][2].write_bytes(b'corrupt changed source')
    with pytest.raises(ValueError, match='Stale human'):
        RoleManifest(path, ['place_mouse_pad']).load('0:pair', *payloads[0])


def test_empty_valid_region_is_rejected(tmp_path):
    path, records, payloads = _fixture(tmp_path)
    ap = Path(records[0]['annotation_file'])
    label = torch.load(ap, weights_only=True)
    label['human_masks'].zero_()
    torch.save(label, ap)
    records[0]['annotation_sha256'] = file_sha256(ap)
    _write(path, records)
    with pytest.raises(ValueError, match='nonempty'):
        RoleManifest(path, ['place_mouse_pad']).load('0:pair', *payloads[0])


def test_complete_mode_cannot_be_satisfied_with_empty_validity(tmp_path):
    path, records, payloads = _fixture(tmp_path)
    ap = Path(records[0]['annotation_file'])
    label = torch.load(ap, weights_only=True)
    label['human_valid'].zero_()
    label['robot_valid'].zero_()
    torch.save(label, ap)
    records[0]['annotation_sha256'] = file_sha256(ap)
    _write(path, records)
    with pytest.raises(ValueError, match='Insufficient valid'):
        RoleManifest(path, ['place_mouse_pad']).load('0:pair', *payloads[0])


def test_future_coverage_uses_real_causal_chunks_and_entity_identity():
    valid = torch.zeros(8, 2, dtype=torch.bool)
    valid[[0, 7]] = True
    ids = torch.arange(2).expand(8, 2).clone()
    assert usable_future_pairs(valid, ids, 2) == (0, 0)
    assert usable_future_pairs(valid, ids, 8) == (2, 1)
    ids[7, 0] = 2
    assert usable_future_pairs(valid, ids, 8) == (1, 0)
