"""Explicit, reviewed role labels; no task-name-based donor equivalence.

Teacher masks never replace the model's predicted masks. Missing annotations
are represented by false validity, not by background labels. Full training
requires complete coverage unless a partial-supervision experiment is explicit.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import torch

from .rgb_feature_cache import load_feature_cache


@lru_cache(maxsize=4096)
def _digest(path, size, mtime_ns):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def file_sha256(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return _digest(str(path), stat.st_size, stat.st_mtime_ns)


def empty_role_targets(human, robot):
    result = {}
    for kind, payload in (("human", human), ("robot", robot)):
        shape = payload["features"].shape
        result[f"{kind}_masks"] = torch.zeros(*shape[:-2], 2, shape[-2])
        result[f"{kind}_valid"] = torch.zeros(*shape[:-2], 2, dtype=torch.bool)
        result[f"{kind}_entity_ids"] = torch.full((*shape[:-2], 2), -1, dtype=torch.int64)
    result['robot_effect_valid'] = result['robot_valid'].clone()
    result['robot_relation_valid'] = result['robot_valid'].clone()
    result['cross_effect_valid'] = torch.tensor(True)
    return result


def _changing_entities(ids, dim):
    """Unknown/occluded intervals do not create or erase a known identity."""
    known = ids >= 0
    first = ids.masked_fill(~known, torch.iinfo(torch.int64).max).amin(dim)
    last = ids.amax(dim)
    return known.any(dim) & (first != last)


def validate_role_identities(annotation, kind, valid, stable_roles):
    ids = annotation.get(f'{kind}_entity_ids')
    if ids is None and stable_roles:
        ids = torch.arange(2, dtype=torch.int64).expand(valid.shape).clone()
    if (not isinstance(ids, torch.Tensor) or ids.dtype != torch.int64
            or tuple(ids.shape) != tuple(valid.shape) or torch.any(ids < -1)):
        raise ValueError(f"Changing/unknown role identity requires explicit {kind}_entity_ids")
    if torch.any(valid & (ids < 0)):
        raise ValueError("Valid role labels need persistent object identity")
    if torch.any((ids[..., 0] >= 0) & (ids[..., 0] == ids[..., 1])):
        raise ValueError("Source and reference must identify distinct entities")
    if stable_roles and _changing_entities(ids.reshape(-1, 2), dim=0).any():
        raise ValueError(f"stable_roles=True contradicts actual {kind}_entity_ids")
    if kind == 'human' and _changing_entities(ids, dim=1).any():
        raise ValueError(
            "Role identity changes within a human clip: object differences would mix distinct entities. "
            "Use clips/annotations with consistent objects within each clip; the current model does "
            "not support an intra-clip role switch."
        )
    return ids


class RoleManifest:
    """Validate identity, source digests, timelines and explicit donor review."""

    def __init__(self, path, allowed_tasks):
        self.path = Path(path).resolve(strict=True)
        payload = json.loads(self.path.read_text())
        self.schema_version = payload.get("schema_version")
        if self.schema_version not in (1, 2) or payload.get("split") != "train":
            raise ValueError("Role manifest must use schema 1/2 and split=train")
        self.records = {}
        allowed = set(allowed_tasks)
        for record in payload.get("records", []):
            key = record.get("pair_key")
            if not isinstance(key, str) or not key or key in self.records:
                raise ValueError("Role manifest pair_key is missing or duplicated")
            if record.get("task_name") not in allowed or record.get("split") != "train":
                raise ValueError("Role annotation includes a held-out/non-training task")
            if record.get("reviewed") is not True or not record.get("provenance"):
                raise ValueError("Role annotations require an explicit review and provenance")
            if self.schema_version == 2 and type(record.get('cross_effect_approved')) is not bool:
                raise ValueError('Sparse annotations require an explicit cross-effect approval')
            self.records[key] = record
        if not self.records:
            raise ValueError("Role manifest contains no reviewed annotations")
        for key, record in self.records.items():
            donor_key = record.get("donor_pair_key")
            if donor_key is None:
                continue
            donor = self.records.get(donor_key)
            if donor is None or donor_key == key:
                raise ValueError("Donor must be a different, reviewed training pair")
            if not record.get("donor_review"):
                raise ValueError("A matching task name is insufficient: explicit donor review is required")
            for item in (record, donor):
                if self.schema_version == 2 and not item['cross_effect_approved']:
                    raise ValueError('Donor transfer requires compatible human/robot effects at both ends')
                if item.get("stable_roles") is not True or item.get("single_effect") is not True:
                    raise ValueError("First-version donor transfer requires reviewed single effects and stable roles")
            signature = record.get("effect_signature")
            if not isinstance(signature, dict) or not all(
                signature.get(k) for k in ("relation", "direction", "source_type", "reference_type", "affordance")
            ) or signature != donor.get("effect_signature"):
                raise ValueError("Donor effect signatures differ or are incomplete")
            if record.get("human_cache_sha256") == donor.get("human_cache_sha256"):
                raise ValueError("Donor must use a different demonstrated human video")

    def _path(self, value):
        path = Path(value)
        return path if path.is_absolute() else self.path.parent / path

    def load(self, pair_key, human, robot, human_path, robot_path, *, require_complete=True, require_record=False):
        record = self.records.get(pair_key)
        if record is None:
            if require_complete or require_record:
                raise ValueError(f"Missing reviewed role annotation for {pair_key}")
            return empty_role_targets(human, robot), human["features"], torch.tensor(False)
        for kind, path in (("human", human_path), ("robot", robot_path)):
            if file_sha256(path) != record.get(f"{kind}_cache_sha256"):
                raise ValueError(f"Stale {kind} RGB source in role annotation: {pair_key}")
        annotation_path = self._path(record["annotation_file"])
        if file_sha256(annotation_path) != record.get("annotation_sha256"):
            raise ValueError(f"Role annotation digest mismatch: {pair_key}")
        annotation = torch.load(annotation_path, map_location="cpu", weights_only=True)
        metadata = annotation.get("metadata", {})
        if metadata.get("schema_version") != self.schema_version or metadata.get("pair_key") != pair_key:
            raise ValueError("Role annotation schema/pair identity mismatch")
        if metadata.get("role_order", ["source", "reference"]) != ["source", "reference"]:
            raise ValueError("Role annotation order must be [source, reference]")
        result = {}
        for kind, payload in (("human", human), ("robot", robot)):
            if metadata.get(f"{kind}_cache_sha256") != record[f"{kind}_cache_sha256"]:
                raise ValueError("Role annotation and manifest source hashes differ")
            ids = annotation.get(f"{kind}_frame_indices")
            if not isinstance(ids, torch.Tensor) or not torch.equal(ids, payload["frame_indices"]):
                raise ValueError(f"Role {kind} timeline does not match frozen RGB timeline")
            shape = payload["features"].shape
            masks, valid = annotation.get(f"{kind}_masks"), annotation.get(f"{kind}_valid")
            expected = (*shape[:-2], 2, shape[-2])
            if not isinstance(masks, torch.Tensor) or tuple(masks.shape) != expected:
                raise ValueError(f"Invalid {kind} role mask shape")
            if not masks.is_floating_point() or not torch.isfinite(masks).all() or torch.any((masks < 0) | (masks > 1)):
                raise ValueError("Role mask coverage must be finite floats in [0,1]")
            if not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool or tuple(valid.shape) != expected[:-1]:
                raise ValueError(f"Invalid {kind} role validity")
            if torch.any(valid & (masks.sum(-1) <= 0)):
                raise ValueError("Visible valid roles must have nonempty masks; mark occlusions invalid")
            result[f"{kind}_masks"], result[f"{kind}_valid"] = masks.float(), valid
            result[f"{kind}_entity_ids"] = validate_role_identities(
                annotation, kind, valid, stable_roles=record.get('stable_roles') is True)
        effect_valid = annotation.get('robot_effect_valid')
        cross_valid = annotation.get('cross_effect_valid')
        if self.schema_version == 1:
            effect_valid = result['robot_valid'] if effect_valid is None else effect_valid
            cross_valid = torch.tensor(True) if cross_valid is None else cross_valid
        if (not isinstance(effect_valid, torch.Tensor) or effect_valid.dtype != torch.bool
                or effect_valid.shape != result['robot_valid'].shape
                or torch.any(effect_valid & ~result['robot_valid'])):
            raise ValueError('robot_effect_valid must be a boolean subset of visible mask validity')
        if not isinstance(cross_valid, torch.Tensor) or cross_valid.dtype != torch.bool or cross_valid.ndim != 0:
            raise ValueError('cross_effect_valid must be an explicit scalar boolean')
        result['robot_effect_valid'], result['cross_effect_valid'] = effect_valid, cross_valid
        relation_valid = annotation.get('robot_relation_valid')
        if relation_valid is None and self.schema_version == 1:
            relation_valid = effect_valid
        if (not isinstance(relation_valid, torch.Tensor) or relation_valid.dtype != torch.bool
                or relation_valid.shape != result['robot_valid'].shape
                or torch.any(relation_valid & ~result['robot_valid'])):
            raise ValueError('robot_relation_valid must be a boolean subset of visible mask validity')
        result['robot_relation_valid'] = relation_valid
        if self.schema_version == 2:
            if bool(cross_valid) != record['cross_effect_approved']:
                raise ValueError('Cross-effect validity contradicts the reviewed manifest decision')
            status = record.get('supervision_status')
            if status not in ('usable', 'excluded') or not record.get('supervision_review'):
                raise ValueError('Sparse annotations need an explicit supervision status and review')
            nonempty = bool(result['human_valid'].any() or result['robot_valid'].any())
            if (status == 'usable') != nonempty:
                raise ValueError('Supervision status contradicts the actual valid masks')
        if require_complete:
            human_counts = result['human_valid'].reshape(-1, 2).sum(0)
            robot_counts = result['robot_valid'].sum(0)
            joint_robot = result['robot_valid'].all(-1).sum()
            if (human_counts < 1).any() or (robot_counts < 2).any() or joint_robot < 2:
                raise ValueError(f"Insufficient valid role/future supervision for complete training: {pair_key}")
        donor_key = record.get("donor_pair_key")
        if donor_key is None:
            return result, human["features"], torch.tensor(False)
        if not bool(cross_valid):
            raise ValueError('A rejected human/robot effect correspondence cannot receive a donor')
        donor_record = self.records[donor_key]
        donor_path = self._path(donor_record["human_cache"])
        if file_sha256(donor_path) != donor_record["human_cache_sha256"]:
            raise ValueError("Stale donor human RGB features")
        donor = load_feature_cache(donor_path, "human", expected_encoder_id=human["metadata"]["encoder_id"],
                                   expected_revision=human["metadata"]["encoder_revision"])
        if donor["metadata"]["preprocessing"] != human["metadata"]["preprocessing"] or donor["features"].shape != human["features"].shape:
            raise ValueError("Donor preprocessing/shape differs from recipient")
        if donor["metadata"]["source_video_path"] == human["metadata"]["source_video_path"]:
            raise ValueError("Donor source video must differ from recipient")
        return result, donor["features"], torch.tensor(True)


def usable_future_pairs(valid, identities, chunk_size, *, mode='chunk', max_future_steps=64, relation_valid=None):
    """Count only teacher pairs the real causal chunk reader can use."""
    objects = relations = 0
    relation_valid = valid if relation_valid is None else relation_valid
    for start in range(0, valid.shape[0], chunk_size):
        observed = max(start - 1, 0)
        if mode not in ('chunk', 'all_reviewed'):
            raise ValueError('Unknown role future target mode')
        end = min(start + chunk_size, valid.shape[0]) if mode == 'chunk' else min(
            valid.shape[0], observed + max_future_steps + 1)
        first = max(start, observed + 1) if mode == 'chunk' else observed + 1
        for future in range(first, end):
            usable = (valid[observed] & valid[future] & (identities[observed] >= 0)
                      & (identities[observed] == identities[future]))
            objects += int(usable.sum())
            relational = (relation_valid[observed] & relation_valid[future] & (identities[observed] >= 0)
                          & (identities[observed] == identities[future]))
            relations += int(relational.all())
    return objects, relations


def validate_role_feature_cache(config):
    from copy import deepcopy
    from .rgb_feature_cache import expected_rgb_records, rgb_pair_key
    manifest = RoleManifest(config.role_manifest_path, config.included_task_names)
    full_config = deepcopy(config)
    full_config.rgb_smoke_sample_ids = None
    permitted = {rgb_pair_key(record['sample']): record for record in expected_rgb_records(full_config)}
    selected_records = list(expected_rgb_records(config))
    selected_keys = {rgb_pair_key(record['sample']) for record in selected_records}
    missing = selected_keys - manifest.records.keys()
    if (config.role_require_complete or getattr(config, 'role_require_all_records', False)) and missing:
        raise ValueError(
            f"Missing reviewed role annotations: {len(selected_keys) - len(missing)}/{len(selected_keys)} "
            f"selected pairs annotated; {len(missing)} missing. Prepare complete labels before full training. "
            f"First missing pair: {sorted(missing)[0]}"
        )
    if getattr(config, 'role_require_all_records', False) and manifest.schema_version != 2:
        raise ValueError('Reviewed sparse protocol requires schema 2 with explicit per-loss validity')
    if getattr(config, 'role_require_all_records', False) and any(
            r.get('policy_pair_approved') is not True for r in manifest.records.values()):
        raise ValueError('Sparse policy training requires explicitly approved ICL pairs; use the filtered policy manifest, not the all-pair audit archive')
    for key, annotation in manifest.records.items():
        record = permitted.get(key)
        if record is None or record['task_name'].split('-', 1)[0] != annotation['task_name']:
            raise ValueError(f"Role/donor pair does not belong to the actual training manifest: {key}")
        for kind in ('human', 'robot'):
            if file_sha256(record[f'{kind}_cache_path']) != annotation[f'{kind}_cache_sha256']:
                raise ValueError(f"Role/donor source does not match training pair {key}")
    total = annotated = donors = 0
    donor_usable = 0
    human_valid_roles = robot_valid_roles = human_total_roles = robot_total_roles = 0
    robot_effect_valid_roles = robot_relation_valid_roles = incompatible_effect_pairs = excluded_pairs = 0
    object_future_pairs = relation_future_pairs = 0
    fixed_chunk = int(getattr(config, 'frame_chunk_size', 0))
    chunk_sizes = [fixed_chunk] if fixed_chunk > 0 else list(range(1, int(config.max_frame_chunk_size) + 1))
    usable_counts = {size: {'object': 0, 'relation': 0} for size in chunk_sizes}
    known_keys = set()
    # Donors outside a smoke selection still need valid sidecars and consistent
    # identities. A manually entered stable_roles flag alone is insufficient.
    extra_donors = {manifest.records[key].get('donor_pair_key') for key in selected_keys
                    if key in manifest.records} - {None} - selected_keys
    for key in sorted(extra_donors):
        record = permitted[key]
        hp, rp = record['human_cache_path'], record['robot_cache_path']
        manifest.load(key, load_feature_cache(hp, 'human'), load_feature_cache(rp, 'robot'), hp, rp,
                      require_complete=config.role_require_complete)
    for record in selected_records:
        # Same record enumeration and exact pair keys as the RGB preflight.
        key = rgb_pair_key(record["sample"])
        known_keys.add(key)
        human_path, robot_path = record["human_cache_path"], record["robot_cache_path"]
        human, robot = load_feature_cache(human_path, "human"), load_feature_cache(robot_path, "robot")
        roles, _, donor_valid = manifest.load(key, human, robot, human_path, robot_path,
                                              require_complete=config.role_require_complete)
        total += 1
        annotated += int(key in manifest.records)
        donors += int(donor_valid)
        human_valid_roles += int(roles['human_valid'].sum())
        robot_valid_roles += int(roles['robot_valid'].sum())
        human_total_roles += roles['human_valid'].numel()
        robot_total_roles += roles['robot_valid'].numel()
        robot_effect_valid_roles += int(roles['robot_effect_valid'].sum())
        robot_relation_valid_roles += int(roles['robot_relation_valid'].sum())
        incompatible_effect_pairs += int(not roles['cross_effect_valid'])
        excluded_pairs += int(not (roles['human_valid'].any() or roles['robot_valid'].any()))
        valid = roles['robot_effect_valid'] & roles['cross_effect_valid']
        relation_valid = roles['robot_relation_valid'] & roles['cross_effect_valid']
        identities = roles['robot_entity_ids']
        same = identities[:, None] == identities[None, :]
        possible = same & valid[:, None] & valid[None, :]
        later = torch.triu(torch.ones(valid.shape[0], valid.shape[0], dtype=torch.bool), diagonal=1)
        object_future_pairs += int((possible & later[..., None]).sum())
        relation_possible = same & relation_valid[:, None] & relation_valid[None, :]
        relation_future_pairs += int((relation_possible.all(-1) & later).sum())
        pair_counts = [usable_future_pairs(valid, identities, size,
                          mode=config.rgb_transition_config.get('role_future_target_mode', 'chunk'),
                          max_future_steps=config.rgb_transition_config.get('max_future_steps', 64),
                          relation_valid=relation_valid)
                       for size in chunk_sizes]
        for size, (objects, relations) in zip(chunk_sizes, pair_counts):
            usable_counts[size]['object'] += objects
            usable_counts[size]['relation'] += relations
        has_usable_future = any(objects > 0 and relations > 0 for objects, relations in pair_counts)
        if config.role_require_complete and not has_usable_future:
            raise ValueError(f"No future-role targets reachable by configured causal chunks: {key}")
        donor_usable += int(bool(donor_valid) and any(o > 0 or r > 0 for o, r in pair_counts))
    if config.rgb_transition_config.get('donor_effect_weight', 1) > 0 and donors == 0:
        raise ValueError("Donor loss is enabled but no selected training pair has a reviewed donor")
    if not human_valid_roles or not robot_valid_roles or not object_future_pairs or not relation_future_pairs:
        raise ValueError("Role/effect experiment has no usable role or future-relation supervision")
    if not any(value['relation'] > 0 for value in usable_counts.values()):
        raise ValueError("All relation targets lie outside the configured causal chunks")
    if config.rgb_transition_config.get('donor_effect_weight', 1) > 0 and donor_usable == 0:
        raise ValueError("Donor annotations exist but have no usable future supervision")
    return {"pairs": total, "annotated_pairs": annotated, "donor_pairs": donors,
            "require_complete": bool(config.role_require_complete),
            "human_valid_roles": human_valid_roles, "human_total_roles": human_total_roles,
            "robot_valid_roles": robot_valid_roles, "robot_total_roles": robot_total_roles,
            "robot_effect_valid_roles": robot_effect_valid_roles,
            "robot_relation_valid_roles": robot_relation_valid_roles,
            "incompatible_effect_pairs": incompatible_effect_pairs,
            "excluded_role_pairs": excluded_pairs,
            "require_all_review_records": bool(getattr(config, 'role_require_all_records', False)),
            "future_target_mode": config.rgb_transition_config.get('role_future_target_mode', 'chunk'),
            "candidate_object_future_pairs": object_future_pairs,
            "candidate_relation_future_pairs": relation_future_pairs,
            "usable_future_pairs_by_chunk_size": {str(size): counts for size, counts in usable_counts.items()},
            "donor_pairs_with_usable_targets": donor_usable,
            "manifest": str(manifest.path), "manifest_sha256": file_sha256(manifest.path)}
