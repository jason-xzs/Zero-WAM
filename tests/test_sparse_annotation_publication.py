"""Sparse reviews must not fabricate unseen masks or approve incompatible pairs."""
from copy import deepcopy
import json

import pytest
import torch

from tools.publish_sparse_role_annotations import apply_scope
from tools.configure_sparse_role_annotations import bind_manifest
from wan_va.dataset.role_feature_cache import file_sha256


def fixture(tmp_path):
    evidence = tmp_path / 'viewed.png'
    evidence.write_bytes(b'synthetic test evidence, not a research annotation')
    a = {}
    for kind in ('human', 'robot'):
        a[kind + '_frame_indices'] = torch.tensor([[0, 16, 32]]) if kind == 'human' else torch.tensor([0, 16, 32])
        shape = a[kind + '_frame_indices'].shape
        a[kind + '_valid'] = torch.ones(*shape, 2, dtype=torch.bool)
        a[kind + '_masks'] = torch.full((*shape, 2, 4), .5)
    scope = dict(candidate_annotation_sha256='candidate', reviewer='synthetic reviewer', notes='synthetic review',
        evidence=[dict(path=str(evidence), sha256=file_sha256(evidence))],
        reviewed_frames={'human': [0, 32], 'robot': [0, 32]},
        keep_frames={'human': {'source': [0, 32], 'reference': [0]},
                     'robot': {'source': [0, 32], 'reference': [0, 32]},
                     'effect': {'source': [0, 32], 'reference': []},
                     'relation': {'source': [], 'reference': []}})
    return a, scope


def test_sparse_review_keeps_only_explicit_frames_and_loss_scopes(tmp_path):
    a, scope = fixture(tmp_path)
    apply_scope(a, scope, 'candidate')
    assert not a['robot_valid'][1].any() and not a['robot_masks'][1].any()
    assert a['robot_valid'][2].all()
    assert a['robot_effect_valid'][2].tolist() == [True, False]
    assert not a['robot_relation_valid'].any()


@pytest.mark.parametrize('case,match', [('unviewed', 'unviewed'), ('auto_invalid', 'automatic-invalid'),
                                      ('effect', 'subset'), ('stale', 'candidate')])
def test_sparse_review_cannot_bypass_evidence_or_validity(tmp_path, case, match):
    a, scope = fixture(tmp_path)
    if case == 'unviewed':
        scope['keep_frames']['human']['source'].append(16)
    elif case == 'auto_invalid':
        a['robot_valid'][2, 0] = False
    elif case == 'effect':
        scope['keep_frames']['effect']['source'].append(16)
    else:
        scope['candidate_annotation_sha256'] = 'different'
    with pytest.raises(ValueError, match=match):
        apply_scope(a, scope, 'candidate')


def test_binding_changes_paths_only_and_refuses_different_source_bytes(tmp_path):
    root = tmp_path / 'rgb'
    for kind in ('human', 'robot'):
        (root / kind).mkdir(parents=True)
        (root / kind / 'cache.pth').write_bytes(kind.encode())
    label = tmp_path / 'label.pth'
    label.write_bytes(b'unchanged synthetic label')
    review = tmp_path / 'review.json'
    review.write_text('{}')
    r = dict(policy_pair_approved=True, annotation_file='label.pth', annotation_sha256=file_sha256(label),
             provenance=dict(original_review_file='review.json', original_review_sha256=file_sha256(review),
                             candidate_evidence=[]))
    for kind in ('human', 'robot'):
        r[kind + '_cache_relative'] = kind + '/cache.pth'
        r[kind + '_cache_sha256'] = file_sha256(root / kind / 'cache.pth')
    source = tmp_path / 'manifest.json'
    source.write_text(json.dumps(dict(schema_version=2, records=[r])))
    destination = tmp_path / 'runtime.json'
    bind_manifest(source, root, destination)
    assert file_sha256(label) == r['annotation_sha256']
    assert json.loads(destination.read_text())['records'][0]['human_cache'] == str(root / 'human/cache.pth')
    (root / 'human/cache.pth').write_bytes(b'different cache')
    with pytest.raises(ValueError, match='digest'):
        bind_manifest(source, root, destination)
