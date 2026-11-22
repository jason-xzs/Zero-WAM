#!/usr/bin/env python3
"""Publish explicitly reviewed sparse SAM targets, preserving original decisions.

This is a different protocol from manage_role_annotations.py finalize. It never
turns a rejected full-clip candidate into a blanket acceptance. New approvals
must enumerate exact raw frames, roles and loss scopes in a visual review file.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.manage_role_annotations import apply_exclusions, value_hash
from wan_va.dataset.rgb_feature_cache import rgb_pair_key, load_feature_cache
from wan_va.dataset.role_feature_cache import RoleManifest, file_sha256, usable_future_pairs

KINDS = ('human', 'robot')
ROLES = ('source', 'reference')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def checked_file(path, digest):
    if not digest or file_sha256(path) != digest:
        raise ValueError(f'Changed or missing evidence/source: {path}')
    return Path(path)


def checked_candidate(job):
    complete = read(job / 'completed.json')
    directory = job / complete['candidate_dir']
    mp = checked_file(directory / 'candidate_manifest.json', complete['candidate_manifest_sha256'])
    candidate, = read(mp)['records']
    for name in ('records', 'seeds'):
        checked_file(directory.parent / (name + '.json'), complete['inputs'][name])
    record, = read(directory.parent / 'records.json')['records']
    if record['pair_key'] != candidate['pair_key'] or record['task'] != candidate['task_name']:
        raise ValueError('Candidate/record pair or task mismatch')
    path = checked_file(directory / Path(candidate['annotation_file']).name, candidate['annotation_sha256'])
    for item in complete['visual_evidence']:
        checked_file(job / item['path'], item['sha256'])
    return complete, directory, candidate, record, torch.load(path, weights_only=True, map_location='cpu')


def validate_original_review(review, complete, candidate, record):
    if (review.get('candidate_annotation_sha256') != candidate['annotation_sha256']
            or review.get('input_identity_sha256') != value_hash(complete['inputs'])
            or review.get('pair_key') != record['pair_key']
            or review.get('decision') not in ('accept', 'reject')
            or not review.get('reviewer') or not review.get('notes')):
        raise ValueError('Stale or incomplete original visual review')
    for kind in KINDS:
        ids = torch.tensor(record[kind + '_frame_indices']).flatten().tolist()
        actual = review.get('reviewed_frames', {}).get(kind, [])
        if len(actual) != len(set(actual)) or set(actual) != set(ids):
            raise ValueError('Original review does not cover its declared sampled frames')
    # Historical absolute paths may come from another host; content hashes and
    # the pair-specific completed record identify the verified local evidence.
    signature = lambda items: {(x['kind'], Path(x['path']).name, x['sha256']) for x in items}
    if signature(review['evidence']) != signature(complete['visual_evidence']):
        raise ValueError('Original review evidence differs from active candidate')


def apply_scope(annotation, scope, candidate_sha):
    if (scope.get('candidate_annotation_sha256') != candidate_sha
            or not scope.get('reviewer') or not scope.get('notes') or not scope.get('evidence')):
        raise ValueError('Explicit sparse review must identify candidate, reviewer, notes and PNG evidence')
    for item in scope['evidence']:
        checked_file(item['path'], item['sha256'])
    keep = scope['keep_frames']
    if set(keep) != {'human', 'robot', 'effect', 'relation'}:
        raise ValueError('All four per-loss scopes must be explicit')
    for kind in KINDS:
        ids = annotation[kind + '_frame_indices']
        reviewed = scope.get('reviewed_frames', {}).get(kind, [])
        if (any(type(x) is not int for x in reviewed) or len(set(reviewed)) != len(reviewed)
                or not set(reviewed) <= set(ids.flatten().tolist())):
            raise ValueError('Reviewed raw frames must belong to this cached timeline')
        old_valid = annotation[kind + '_valid'].clone()
        for role_index, role in enumerate(ROLES):
            frames = keep[kind][role]
            if (any(type(x) is not int for x in frames) or not set(frames) <= set(reviewed)):
                raise ValueError('Cannot approve unviewed/out-of-timeline role frames')
            selected = torch.isin(ids, torch.tensor(frames, dtype=ids.dtype))
            if (selected & ~old_valid[..., role_index]).any():
                raise ValueError('Cannot restore an automatic-invalid candidate; regenerate and review it')
            annotation[kind + '_valid'][..., role_index] = selected
        annotation[kind + '_masks'][~annotation[kind + '_valid']] = 0
    for scope_kind, field in [('effect', 'robot_effect_valid'), ('relation', 'robot_relation_valid')]:
        ids = annotation['robot_frame_indices']
        valid = torch.zeros_like(annotation['robot_valid'])
        for i, role in enumerate(ROLES):
            frames = keep[scope_kind][role]
            if any(type(x) is not int for x in frames) or not set(frames) <= set(keep['robot'][role]):
                raise ValueError('Effect/relation approval must be a subset of approved visible masks')
            valid[:, i] = torch.isin(ids, torch.tensor(frames, dtype=ids.dtype))
        annotation[field] = valid


def copy_evidence(items, output, prefix):
    result = []
    for i, item in enumerate(items):
        source = checked_file(item['path'], item['sha256'])
        dest = output / prefix / f'{i:02d}_{source.name}'
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        result.append({**item, 'path': str(dest.relative_to(output))})
    return result


def cache_relative(value, kind):
    parts = Path(value).parts
    starts = [i for i, p in enumerate(parts) if p == kind]
    if not starts or '..' in parts:
        raise ValueError('Unsafe/unrecognized RGB cache path')
    return Path(*parts[starts[-1]:])


def publish(args):
    review = read(args.review)
    if review.get('schema_version') != 2 or review.get('protocol') != 'reviewed_sparse_visible_regions_v2':
        raise ValueError('Expected explicit reviewed sparse v2 decisions')
    scopes, exclusions = review['scopes'], review['policy_exclusions']
    index = read(args.bundle / 'annotations/work/index.json')
    keys = {x['pair_key'] for x in index['jobs']}
    if len(keys) != index['pairs'] or set(scopes) - keys or set(exclusions) - keys:
        raise ValueError('Review keys do not match the original dataset')
    master = read(args.icl_manifest)
    samples = {rgb_pair_key(s): s for s in master['samples']}
    if not keys <= samples.keys():
        raise ValueError('Original ICL manifest does not contain all reviewed pair identities')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'input_review.json', review)
    write(output / 'original_index.json', index)
    records, counts = [], Counter()
    for entry in index['jobs']:
        key, label = entry['pair_key'], entry['label']
        job = args.bundle / 'annotations/work/jobs' / label
        complete, directory, candidate, record, annotation = checked_candidate(job)
        original_review_path = job / 'reviews' / (candidate['annotation_sha256'] + '.json')
        original_review = read(original_review_path)
        validate_original_review(original_review, complete, candidate, record)
        history = output / 'reviews' / label
        history.mkdir(parents=True)
        # Preserve the exact reviewed bytes: reformatting JSON changes its hash.
        shutil.copyfile(original_review_path, history / 'original_decision.json')
        write(history / 'original_completed.json', complete)
        original_hash = candidate['annotation_sha256']
        scope = deepcopy(scopes.get(key))
        if scope and scope.get('replacement_job'):
            job = Path(scope['replacement_job'])
            complete, directory, candidate, record, annotation = checked_candidate(job)
            if candidate['pair_key'] != key:
                raise ValueError('Replacement is a different pair')
            shutil.copyfile(directory.parent / 'seeds.json', history / 'replacement_seeds.json')
            write(history / 'replacement_completed.json', complete)
            counts['regenerated_candidates'] += 1
        provenance = {
            'kind': 'reviewed_SAM2_sparse_weak_targets', 'not_ground_truth': True,
            'original_candidate_sha256': original_hash,
            'used_candidate_sha256': candidate['annotation_sha256'],
            'original_decision': original_review['decision'],
            'original_review_file': str((history / 'original_decision.json').relative_to(output)),
            'original_review_sha256': file_sha256(original_review_path),
            'source_hashes': complete['inputs']['sources'], 'teacher': complete['teacher'],
        }
        evidence = [{**e, 'path': str(job / e['path'])} for e in complete['visual_evidence']]
        provenance['candidate_evidence'] = copy_evidence(evidence, output, Path('reviews') / label / 'candidate')
        if scope:
            apply_scope(annotation, scope, candidate['annotation_sha256'])
            scope['evidence'] = copy_evidence(scope['evidence'], output, Path('reviews') / label / 'scope')
            write(history / 'sparse_decision.json', scope)
            provenance['sparse_review_file'] = str((history / 'sparse_decision.json').relative_to(output))
            provenance['sparse_review_sha256'] = file_sha256(history / 'sparse_decision.json')
            supervision_review = scope['notes']
            counts['new_sparse_reviews'] += 1
        else:
            if original_review['decision'] != 'accept':
                raise ValueError(f'Rejected pair lacks a new explicit sparse decision: {key}')
            apply_exclusions(annotation, original_review, record)
            annotation['robot_effect_valid'] = annotation['robot_valid'].clone()
            annotation['robot_relation_valid'] = annotation['robot_valid'].clone()
            supervision_review = 'Reused original reviewed scopes with original exclusions. ' + original_review['notes']
            counts['reused_accepted_reviews'] += 1
        approved = key not in exclusions
        annotation['cross_effect_valid'] = torch.tensor(approved)
        if not approved:
            ex = deepcopy(exclusions[key])
            for kind in KINDS:
                if ex['source_video_sha256'][kind] != complete['inputs']['sources'][kind + '_video']:
                    raise ValueError('Semantic review refers to a different source video')
            ex['evidence'] = copy_evidence([ex['evidence']], output, Path('reviews') / label / 'semantic')[0]
            write(history / 'policy_exclusion.json', ex)
            provenance['policy_exclusion'] = ex
        teacher = complete['teacher']
        teacher_name = 'facebook/sam2.1-hiera-large' if teacher['sam2_config'].endswith('_l.yaml') else 'facebook/sam2.1-hiera-tiny'
        if annotation['metadata'].get('teacher_model') not in (teacher_name, 'sam2 (see teacher config and checkpoint digest)'):
            counts['corrected_teacher_names'] += 1
        annotation['metadata'].update(schema_version=2, teacher_model=teacher_name,
                                      teacher_revision=teacher, provenance=provenance)
        lp = output / 'labels' / (label + '.pth')
        lp.parent.mkdir(exist_ok=True)
        torch.save(annotation, lp)
        out = {**candidate, 'annotation_file': str(lp.relative_to(output)), 'annotation_sha256': file_sha256(lp),
               'reviewed': True, 'provenance': provenance, 'cross_effect_approved': approved,
               'policy_pair_approved': approved, 'supervision_review': supervision_review,
               'supervision_status': 'usable' if any(annotation[k + '_valid'].any() for k in KINDS) else 'excluded'}
        for kind in KINDS:
            relative = cache_relative(candidate[kind + '_cache'], kind)
            source = args.bundle / 'rgb-cache' / relative
            checked_file(source, candidate[kind + '_cache_sha256'])
            out[kind + '_cache_relative'] = str(relative)
            out[kind + '_cache'] = str(source.resolve())
        for field in ('stable_roles', 'single_effect', 'effect_signature', 'donor_pair_key', 'donor_review'):
            out[field] = original_review.get(field)
        if scope:
            # Reusing original role IDs is not a new donor-equivalence review.
            out.update(single_effect=False, donor_pair_key=None, donor_review=None)
            if type(out.get('stable_roles')) is not bool:
                out['stable_roles'] = False
        records.append(out)
        if len(records) % 50 == 0:
            print(f'published {len(records)}/{len(keys)} review records', flush=True)
    by_key = {r['pair_key']: r for r in records}
    for r in records:
        donor = r.get('donor_pair_key')
        if donor and (not r['policy_pair_approved'] or not by_key[donor]['policy_pair_approved']):
            r['donor_pair_key'] = r['donor_review'] = None
            counts['removed_incompatible_donor_links'] += 1
    write(output / 'audit_manifest.all600.json', dict(schema_version=2, split='train', records=records))
    retained = [r for r in records if r['policy_pair_approved']]
    manifest_path = output / 'reviewed_manifest.pending.json'
    write(manifest_path, dict(schema_version=2, split='train', protocol=review['protocol'], records=retained))
    selected = [samples[r['pair_key']] for r in retained]
    write(output / 'icl_manifest.train.json', dict(dataset=master.get('dataset'), samples=selected,
        selection_rule='Exact original subset pairs minus explicit semantic incompatibilities; no replacement demos.',
        audit=dict(original_manifest_sha256=file_sha256(args.icl_manifest), original_pairs=len(records),
                   retained_pairs=len(retained), excluded_pair_keys=sorted(exclusions))))
    tasks = sorted({r['task_name'] for r in records})
    manifest = RoleManifest(manifest_path, tasks)
    coverage = defaultdict(Counter)
    for r in retained:
        hp, rp = Path(r['human_cache']), Path(r['robot_cache'])
        targets, _, donor = manifest.load(r['pair_key'], load_feature_cache(hp, 'human'), load_feature_cache(rp, 'robot'),
                                         hp, rp, require_complete=False, require_record=True)
        c = coverage[r['task_name']]
        c['pairs'] += 1
        c['donor_pairs'] += int(donor)
        for kind in KINDS:
            c[kind + '_valid_roles'] += int(targets[kind + '_valid'].sum())
            c[kind + '_total_roles'] += targets[kind + '_valid'].numel()
        for name in ('effect', 'relation'):
            c['robot_' + name + '_valid_roles'] += int(targets['robot_' + name + '_valid'].sum())
        for size in range(1, 5):
            obj, rel = usable_future_pairs(targets['robot_effect_valid'], targets['robot_entity_ids'], size,
                mode='all_reviewed', relation_valid=targets['robot_relation_valid'])
            c[f'chunk{size}_object_targets'] += obj
            c[f'chunk{size}_relation_targets'] += rel
            c[f'chunk{size}_pairs_with_object_targets'] += obj > 0
            c[f'chunk{size}_pairs_with_relation_targets'] += rel > 0
    report = dict(protocol=review['protocol'], original_review_records=len(records), policy_pairs=len(retained),
                  policy_exclusions=len(exclusions), counts=dict(counts), per_task=dict(coverage),
                  all_labels_and_sources_validated=True, formal_training_preflight_passed=False,
                  limitations=review['limitations'])
    write(output / 'coverage.json', report)
    manifest_path.rename(output / 'reviewed_manifest.json')
    print(json.dumps({k: v for k, v in report.items() if k not in ('per_task', 'limitations')}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--icl-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    publish(args)


if __name__ == '__main__':
    main()
