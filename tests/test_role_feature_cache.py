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


def _update_annotation(path, records, transform):
    annotation_path = Path(records[0]['annotation_file'])
    annotation = torch.load(annotation_path, weights_only=True)
    transform(annotation)
    torch.save(annotation, annotation_path)
    records[0]['annotation_sha256'] = file_sha256(annotation_path)
    _write(path, records)


def test_changing_human_identity_inside_a_clip_is_rejected(tmp_path):
    path, records, payloads = _fixture(tmp_path)
    records[0].pop('donor_pair_key')
    records[0]['stable_roles'] = False
    def change(annotation):
        annotation['human_entity_ids'] = torch.tensor([[[0, 1], [2, 1]], [[2, 1], [2, 1]]])
        annotation['robot_entity_ids'] = torch.tensor([[0, 1], [2, 1], [2, 1]])
    _update_annotation(path, records, change)
    with pytest.raises(ValueError, match='within.*human clip'):
        RoleManifest(path, ['place_mouse_pad']).load('0:pair', *payloads[0])


def test_different_human_clips_can_have_different_stable_entities(tmp_path):
    path, records, payloads = _fixture(tmp_path)
    records[0].pop('donor_pair_key')
    records[0]['stable_roles'] = False
    def change(annotation):
        annotation['human_entity_ids'] = torch.tensor([[[0, 1], [0, 1]], [[2, 0], [2, 0]]])
        annotation['robot_entity_ids'] = torch.tensor([[0, 1], [0, 1], [2, 0]])
    _update_annotation(path, records, change)
    labels, _, donor_valid = RoleManifest(path, ['place_mouse_pad']).load('0:pair', *payloads[0])
    assert not donor_valid
    assert labels['human_entity_ids'][1, 0].tolist() == [2, 0]


@pytest.mark.parametrize('kind', ['human', 'robot'])
def test_donor_stable_role_declaration_must_match_actual_entity_ids(tmp_path, kind):
    path, records, payloads = _fixture(tmp_path)
    def change(annotation):
        shape = annotation[f'{kind}_valid'].shape
        ids = torch.tensor([0, 1]).expand(shape).clone()
        ids[-1, ..., 0] = 2
        annotation[f'{kind}_entity_ids'] = ids
    _update_annotation(path, records, change)
    with pytest.raises(ValueError, match='stable_roles'):
        RoleManifest(path, ['place_mouse_pad']).load('0:pair', *payloads[0])


def test_unknown_entities_are_not_counted_as_usable_future_targets():
    valid = torch.ones(4, 2, dtype=torch.bool)
    ids = torch.full((4, 2), -1, dtype=torch.int64)
    assert usable_future_pairs(valid, ids, 2) == (0, 0)


def test_sparse_far_targets_respect_per_loss_validity_and_identity():
    visible = torch.zeros(12, 2, dtype=torch.bool)
    visible[[0, 10]] = True
    ids = torch.arange(2).expand(12, 2).clone()
    relation = visible.clone()
    relation[10, 1] = False
    assert usable_future_pairs(visible, ids, 2) == (0, 0)
    assert usable_future_pairs(visible, ids, 2, mode='all_reviewed', relation_valid=relation) == (2, 0)
    assert usable_future_pairs(visible, ids, 2, mode='all_reviewed', max_future_steps=5) == (0, 0)
    ids[10, 0] = 2
    assert usable_future_pairs(visible, ids, 2, mode='all_reviewed') == (1, 0)


def _sparse_fixture(tmp_path):
    path, records, payloads = _fixture(tmp_path)
    records[0].pop('donor_pair_key')
    for record in records:
        record.update(cross_effect_approved=True, supervision_status='usable',
                      supervision_review='Synthetic explicit sparse review for unit testing')
        ap = Path(record['annotation_file'])
        a = torch.load(ap, weights_only=True)
        a['metadata']['schema_version'] = 2
        a['robot_effect_valid'] = a['robot_valid'].clone()
        a['robot_relation_valid'] = torch.zeros_like(a['robot_valid'])
        a['cross_effect_valid'] = torch.tensor(True)
        torch.save(a, ap)
        record['annotation_sha256'] = file_sha256(ap)
    path.write_text(json.dumps(dict(schema_version=2, split='train', records=records)))
    return path, records, payloads


def test_sparse_manifest_requires_every_record_and_explicit_scope(tmp_path):
    path, records, payloads = _sparse_fixture(tmp_path)
    m = RoleManifest(path, ['place_mouse_pad'])
    labels, _, _ = m.load('0:pair', *payloads[0], require_complete=False, require_record=True)
    assert labels['robot_effect_valid'].all() and not labels['robot_relation_valid'].any()
    with pytest.raises(ValueError, match='Missing reviewed'):
        m.load('absent', *payloads[0], require_complete=False, require_record=True)
    records[0]['supervision_status'] = 'excluded'
    path.write_text(json.dumps(dict(schema_version=2, split='train', records=records)))
    with pytest.raises(ValueError, match='contradicts'):
        RoleManifest(path, ['place_mouse_pad']).load('0:pair', *payloads[0], require_complete=False)


@pytest.mark.parametrize('field', ['robot_effect_valid', 'robot_relation_valid', 'cross_effect_valid'])
def test_sparse_missing_loss_scope_is_not_implicitly_approved(tmp_path, field):
    path, records, payloads = _sparse_fixture(tmp_path)
    ap = Path(records[0]['annotation_file'])
    a = torch.load(ap, weights_only=True)
    a.pop(field)
    torch.save(a, ap)
    records[0]['annotation_sha256'] = file_sha256(ap)
    path.write_text(json.dumps(dict(schema_version=2, split='train', records=records)))
    with pytest.raises(ValueError, match='valid|boolean'):
        RoleManifest(path, ['place_mouse_pad']).load('0:pair', *payloads[0], require_complete=False)
