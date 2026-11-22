#!/usr/bin/env python3
"""Describe reviewed target changes; this is neither accuracy nor policy quality.

Reports unique forward frame pairs with matching entity identities. Counts do
not reproduce chunk sampling weights. Image-plane mask centroids can change
because of occlusion, so displacement is a diagnostic, not motion ground truth.
"""
import argparse
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wan_va.dataset.role_feature_cache import file_sha256
from wan_va.dataset.rgb_feature_cache import load_feature_cache
from wan_va.modules.role_effect import RoleEffectConditioner


@lru_cache(maxsize=8)
def robot_features(path, digest):
    if file_sha256(path) != digest:
        raise ValueError('RGB cache digest differs')
    f = load_feature_cache(path, 'robot')['features'].float()
    return F.layer_norm(f, (f.shape[-1],), eps=1e-6)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    torch.set_num_threads(2)
    manifest = json.loads(a.manifest.read_text())
    rows = []
    for r in manifest['records']:
        path = a.manifest.parent / r['annotation_file']
        if file_sha256(path) != r['annotation_sha256']:
            raise ValueError('Annotation digest differs')
        target = torch.load(path, weights_only=True, map_location='cpu')
        masks, ids = target['robot_masks'].float(), target['robot_entity_ids']
        features = robot_features(r['robot_cache'], r['robot_cache_sha256'])
        xy = RoleEffectConditioner._square_grid(features.shape[-2], 'cpu', torch.float32)
        pooled, geometry = RoleEffectConditioner._pool(features, masks, xy)
        starts, ends = torch.triu_indices(len(features), len(features), offset=1)
        identity = (ids[starts] == ids[ends]) & (ids[starts] >= 0)
        feature_valid = target['robot_effect_valid']
        object_ok = feature_valid[starts] & feature_valid[ends] & identity & target['cross_effect_valid']
        geo_valid = target['robot_relation_valid']
        geo_ok = geo_valid[starts] & geo_valid[ends] & identity & target['cross_effect_valid']
        relation_ok = geo_ok.all(-1)
        delta = pooled[ends] - pooled[starts]
        no_change_loss = F.smooth_l1_loss(torch.zeros_like(delta), delta, reduction='none').mean(-1)
        # Convert [-1, 1] coordinates to fractions of the image width/height.
        displacement = (geometry[ends, :, :2] - geometry[starts, :, :2]).norm(dim=-1) / 2
        relation = geometry[:, 0, :2] - geometry[:, 1, :2]
        relation_displacement = (relation[ends] - relation[starts]).norm(dim=-1) / 2
        losses = no_change_loss[object_ok]
        movements = displacement[geo_ok]
        rel_movements = relation_displacement[relation_ok]
        rows.append(dict(pair_key=r['pair_key'], task=r['task_name'], object_targets=int(object_ok.sum()),
            relation_targets=int(relation_ok.sum()),
            no_change_object_loss=float(losses.mean()) if losses.numel() else None,
            max_mask_centroid_displacement=float(movements.max()) if movements.numel() else None,
            max_relative_centroid_displacement=float(rel_movements.max()) if rel_movements.numel() else None,
            any_centroid_shift_over_5pct=bool((movements > .05).any()),
            any_relative_shift_over_5pct=bool((rel_movements > .05).any())))
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['task']].append(r)
    summary = {}
    for task, items in sorted(grouped.items()):
        losses = [r['no_change_object_loss'] for r in items if r['no_change_object_loss'] is not None]
        summary[task] = dict(pairs=len(items),
            pairs_with_object_targets=sum(r['object_targets'] > 0 for r in items),
            pairs_with_relation_targets=sum(r['relation_targets'] > 0 for r in items),
            pairs_with_centroid_shift_over_5pct=sum(r['any_centroid_shift_over_5pct'] for r in items),
            pairs_with_relative_shift_over_5pct=sum(r['any_relative_shift_over_5pct'] for r in items),
            median_per_pair_no_change_object_loss=float(torch.tensor(losses).median()) if losses else None)
    report = dict(manifest_sha256=file_sha256(a.manifest), per_task=summary, records=rows,
        definition='Unique forward same-entity teacher frame pairs; 5% image-coordinate L2 diagnostic threshold.',
        limitations=['Centroid changes are not ground-truth object movement or annotation quality.',
                     'Zero-delta loss is a target statistic, not a trained-model evaluation.',
                     'Auxiliary chunk sampling repeats some starts; these counts are unique frame pairs.'])
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
