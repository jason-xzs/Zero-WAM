#!/usr/bin/env python3
"""Bind portable role targets to exact local RGB caches, then run real preflight."""
from __future__ import annotations

import argparse
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@lru_cache(maxsize=4096)
def _digest(path, size, mtime_ns):
    value = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def file_sha256(path):
    # Do not import wan_va here: its package initializer instantiates configs
    # before main() has installed the caller's dataset path overrides.
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return _digest(str(path), stat.st_size, stat.st_mtime_ns)


def checked(path, sha):
    if not sha or file_sha256(path) != sha:
        raise ValueError(f'File missing or digest differs: {path}')


def bind_manifest(source, feature_root, destination):
    source, feature_root = Path(source).resolve(strict=True), Path(feature_root).resolve(strict=True)
    doc = json.loads(source.read_text())
    if doc.get('schema_version') != 2 or not doc.get('records'):
        raise ValueError('Expected a nonempty sparse v2 manifest')
    doc = deepcopy(doc)
    for r in doc['records']:
        if r.get('policy_pair_approved') is not True:
            raise ValueError('Do not bind the all-pair audit archive for policy training')
        label = source.parent / r['annotation_file']
        checked(label, r['annotation_sha256'])
        r['annotation_file'] = str(label.resolve())
        for kind in ('human', 'robot'):
            relative = Path(r[kind + '_cache_relative'])
            if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != kind:
                raise ValueError('Unsafe RGB source-relative path')
            path = feature_root / relative
            checked(path, r[kind + '_cache_sha256'])
            r[kind + '_cache'] = str(path)
        p = r['provenance']
        checked(source.parent / p['original_review_file'], p['original_review_sha256'])
        evidence = list(p['candidate_evidence'])
        if 'sparse_review_file' in p:
            path = source.parent / p['sparse_review_file']
            checked(path, p['sparse_review_sha256'])
            evidence += json.loads(path.read_text())['evidence']
        if 'policy_exclusion' in p:
            raise ValueError('Excluded pair leaked into recommended policy manifest')
        for e in evidence:
            checked(source.parent / e['path'], e['sha256'])
    doc['runtime_source_manifest_sha256'] = file_sha256(source)
    doc['runtime_rgb_feature_root'] = str(feature_root)
    data = json.dumps(doc, ensure_ascii=False, indent=2) + '\n'
    destination = Path(destination)
    if destination.exists():
        if destination.read_text() != data:
            raise ValueError('Runtime manifest already exists with different contents; choose a fresh --runtime-manifest')
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('x') as f:
            f.write(data)
    return destination.resolve()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations-dir', required=True, type=Path)
    p.add_argument('--feature-root', required=True, type=Path)
    p.add_argument('--human-gen-root', required=True, type=Path)
    p.add_argument('--runtime-manifest', type=Path)
    p.add_argument('--report', type=Path)
    p.add_argument('--config-name', choices=['robotwin_subset12_role_effect_sparse', 'robotwin_subset12_role_effect_sparse_b300'],
                   default='robotwin_subset12_role_effect_sparse_b300')
    args = p.parse_args()
    directory = args.annotations_dir.resolve(strict=True)
    manifest = bind_manifest(directory / 'reviewed_manifest.json', args.feature_root,
                             args.runtime_manifest or directory / 'runtime_manifest.json')
    os.environ.update(HUMAN_GEN_ROOT=str(args.human_gen_root.resolve(strict=True)),
                      ZERO_WAM_ROLE_MANIFEST=str(manifest),
                      ZERO_WAM_RGB_FEATURE_ROOT=str(args.feature_root.resolve(strict=True)),
                      ZERO_WAM_ICL_MANIFEST=str(directory / 'icl_manifest.train.json'))
    # Import after setting path overrides: config modules are instantiated once.
    import torch
    torch.set_num_threads(2)
    from wan_va.configs import VA_CONFIGS
    from wan_va.dataset.rgb_feature_cache import validate_rgb_feature_cache
    from wan_va.dataset.role_feature_cache import validate_role_feature_cache
    config = deepcopy(VA_CONFIGS[args.config_name])
    rgb = validate_rgb_feature_cache(config)
    role = validate_role_feature_cache(config)
    report = dict(protocol='reviewed_sparse_visible_regions_v2', config=args.config_name,
                  formal_training_preflight_passed=True, rgb=rgb, role=role,
                  icl_manifest_sha256=file_sha256(config.icl_manifest_path),
                  note='Passes the sparse protocol, not legacy per-pair complete supervision. No task success claim.')
    path = args.report or directory / 'preflight.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(report=str(path), manifest=str(manifest), pairs=role['pairs'],
                          tasks=rgb['tasks'], formal_training_preflight_passed=True)), flush=True)


if __name__ == '__main__':
    main()
