#!/usr/bin/env python3
"""Prepare *unreviewed* visible-role mask candidates from explicit human seeds.

Run in the existing SAM2 environment; this tool installs nothing. It consumes
real videos and original RGB-cache frame indices. SAM2 predictions are weak
labels, not simulator ground truth. Inspect every sampled-frame overlay and
mark occlusions/tracking failures invalid before approving a candidate manifest.
No task-name rule assigns object boxes, donor validity, or review approval.

Per-video legacy seeds remain valid::
    {"seed_frame": 0, "boxes_xyxy": {"source": [x1,y1,x2,y2],
                                    "reference": [x1,y1,x2,y2]}}

For multiple persistent entities and explicit changes of task roles, use::
    {"seed_frame": 0,
     "objects": {"0": {"box_xyxy": [x1,y1,x2,y2]},
                 "1": {"box_xyxy": [x1,y1,x2,y2]},
                 "2": {"box_xyxy": [x1,y1,x2,y2]}},
     "role_events": [{"start_frame": 0, "end_frame": 40,
                      "source": "0", "reference": "1"},
                     {"start_frame": 50, "end_frame": 100,
                      "source": "2", "reference": "1"}]}

Event bounds are inclusive original-video frame indices. All explicitly
seeded objects are propagated; events select the two roles at each cache frame.
Gaps remain unknown (valid=False, entity_id=-1); roles are never inferred from
task names or neighboring events. Overlapping/out-of-range events, unknown or
identical role IDs, and malformed boxes are rejected. Numeric object IDs remain
unchanged; named IDs receive deterministic integers recorded in role_seed_plan.
Explicit objects may also supply points_xy and point_labels (1=foreground,
0=background), in original seed-frame pixels, alongside their box. These are
observed prompts, never reviewed masks or a substitute for candidate review.
All entity boxes refer to seed_frame (default 0). A later visible seed is
allowed: earlier frames remain unknown, with zero supervision and no backward
propagation. Optional invalid_intervals retain
their legacy role-scoped meaning. Multi-stage support does not imply that any
real multi-stage data have been annotated or reviewed.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import time

import av
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F


ROLES = ("source", "reference")
COLORS = ((255, 70, 60), (0, 205, 235))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def video_source_identity(path, kind):
    """Identify a recognized source while ignoring only its storage root.

    Human identity includes the complete run/sample/file suffix. Robot identity
    includes task variant, videos directory, chunk, camera, and episode file.
    Unknown layouts return None; basename equality never permits relocation.
    """
    if not isinstance(path, (str, Path)) or not str(path):
        raise ValueError("Video source path must be a nonempty path")
    parts = PurePosixPath(str(path).replace("\\", "/")).parts
    if ".." in parts:
        raise ValueError("Video source paths cannot contain parent traversal")
    if kind == "human":
        if (len(parts) >= 4 and parts[-4].startswith("run_")
                and parts[-3] == "samples" and parts[-2] and parts[-1]):
            return tuple(parts[-4:])
        return None
    if kind != "robot":
        raise ValueError(f"Unknown video kind: {kind}")
    if len(parts) < 5:
        return None
    task, directory, chunk, camera, filename = parts[-5:]
    if (re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", task) is None
            or directory != "videos"
            or re.fullmatch(r"chunk-[0-9]{3,}", chunk) is None
            or re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", camera) is None
            or re.fullmatch(r"episode_[0-9]{6,}\.[A-Za-z0-9]+", filename) is None):
        return None
    return task, directory, chunk, camera, filename


def resolve_source_video(record, kind, metadata):
    """Use the current record path, validating it against cache provenance."""
    original = metadata["source_video_path"]
    current = record.get(f"{kind}_video_path", original)
    original_identity = video_source_identity(original, kind)
    current_identity = video_source_identity(current, kind)
    original_path = PurePosixPath(str(original).replace("\\", "/"))
    current_path = PurePosixPath(str(current).replace("\\", "/"))
    if original_path != current_path and (
            original_identity is None or current_identity is None
            or original_identity != current_identity):
        raise ValueError(
            f"{kind} video source identity differs from RGB cache provenance: "
            f"record={current!s}, cache={original!s}"
        )
    return Path(str(current_path))


def _object_id(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("Object IDs must be nonempty strings or nonnegative integers")
    value = str(value)
    if not value or value != value.strip():
        raise ValueError("Object IDs must be nonempty and have no surrounding whitespace")
    if value.lstrip("-").isdigit() and (int(value) < 0 or str(int(value)) != value):
        raise ValueError("Numeric object IDs must be canonical nonnegative integers")
    if value.isdigit() and int(value) >= 2**63-1:
        raise ValueError("Object IDs must fit int64 with room for the SAM ID offset")
    return value


def _validate_points(value, width, height):
    points, labels = value.get('points_xy'), value.get('point_labels')
    if (not isinstance(points, list) or not points or not isinstance(labels, list)
            or len(labels) != len(points) or any(type(x) is not int or x not in (0, 1) for x in labels)):
        raise ValueError('Correction requires matching points_xy and binary point_labels')
    for point in points:
        if (not isinstance(point, (list, tuple)) or len(point) != 2
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x) for x in point)
                or not (0 <= point[0] < width and 0 <= point[1] < height)):
            raise ValueError('Correction points must be finite original-frame pixels')
    return {'points_xy': [list(point) for point in points], 'point_labels': list(labels)}


def normalize_role_seeds(seeds, frame_count, width, height):
    """Validate explicit role events and create persistent entity IDs on CPU."""
    if not isinstance(frame_count, int) or frame_count < 1:
        raise ValueError("Video frame count must be positive")
    seed_frame = seeds.get("seed_frame", 0)
    if (isinstance(seed_frame, bool) or not isinstance(seed_frame, int)
            or not 0 <= seed_frame < frame_count):
        raise ValueError("seed_frame must be an original-video frame inside the video")
    legacy = "objects" not in seeds
    if legacy:
        boxes = seeds.get("boxes_xyxy", {})
        if set(boxes) != set(ROLES) or "role_events" in seeds:
            raise ValueError("Legacy seeds require source/reference boxes and no role_events")
        objects = {str(i): {"box_xyxy": boxes[role]} for i, role in enumerate(ROLES)}
        events = [{"start_frame": seed_frame, "end_frame": frame_count-1,
                   "source": "0", "reference": "1"}]
    else:
        if "boxes_xyxy" in seeds:
            raise ValueError("Use either legacy boxes_xyxy or explicit objects, not both")
        objects = seeds["objects"]
        events = seeds.get("role_events")
        if not isinstance(objects, dict) or len(objects) < 2 or not isinstance(events, list):
            raise ValueError("Multi-entity seeds require >=2 objects and explicit role_events list")
    normalized = {}
    for key, value in objects.items():
        key = _object_id(key)
        if key in normalized or not isinstance(value, dict):
            raise ValueError("Duplicate object ID or malformed object seed")
        box = value.get("box_xyxy")
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x) for x in box)
                or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)):
            raise ValueError(f"Invalid original-pixel box for object {key}: {box}")
        normalized[key] = {"external_id": key, "box_xyxy": list(box)}
        corrections = value.get('corrections', [])
        if not isinstance(corrections, list):
            raise ValueError('corrections must be a list of reviewed frame prompts')
        frames = set()
        checked = []
        for correction in corrections:
            if not isinstance(correction, dict):
                raise ValueError('Malformed correction prompt')
            frame = correction.get('frame')
            if (type(frame) is not int or not seed_frame < frame < frame_count or frame in frames
                    or not isinstance(correction.get('review'), str) or not correction['review'].strip()):
                raise ValueError('Correction needs a unique later frame and explicit visual review')
            checked.append(dict(frame=frame, review=correction['review'],
                                **_validate_points(correction, width, height)))
            frames.add(frame)
        if checked:
            normalized[key]['corrections'] = sorted(checked, key=lambda x: x['frame'])
        if "points_xy" in value or "point_labels" in value:
            points, labels = value.get("points_xy"), value.get("point_labels")
            if (not isinstance(points, list) or not points
                    or not isinstance(labels, list) or len(labels) != len(points)
                    or any(type(label) is not int or label not in (0, 1) for label in labels)):
                raise ValueError(f"Invalid point prompts for object {key}")
            for point in points:
                if (not isinstance(point, (list, tuple)) or len(point) != 2
                        or any(isinstance(x, bool) or not isinstance(x, (int, float))
                               or not np.isfinite(x) for x in point)
                        or not (0 <= point[0] < width and 0 <= point[1] < height)):
                    raise ValueError(f"Invalid original-pixel point for object {key}: {point}")
            normalized[key].update(points_xy=[list(point) for point in points],
                                   point_labels=list(labels))
    used_ids = {int(key) for key in normalized if key.isdigit()}
    entity_ids = {key: int(key) for key in normalized if key.isdigit()}
    for key in sorted(set(normalized)-set(entity_ids)):
        candidate = 0
        while candidate in used_ids:
            candidate += 1
        entity_ids[key] = candidate
        used_ids.add(candidate)
    object_list = []
    for key, value in normalized.items():
        object_list.append({**value, "entity_id": entity_ids[key], "sam_id": entity_ids[key]+1})
    normalized_events = []
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Each role event must be an object")
        start, end = event.get("start_frame"), event.get("end_frame")
        if (isinstance(start, bool) or isinstance(end, bool)
                or not isinstance(start, int) or not isinstance(end, int)
                or not seed_frame <= start <= end < frame_count):
            raise ValueError(f"Role event has invalid/out-of-video bounds: {event}")
        selected = {role: _object_id(event.get(role)) for role in ROLES}
        if any(key not in normalized for key in selected.values()):
            raise ValueError(f"Role event refers to an unknown object: {event}")
        if selected["source"] == selected["reference"]:
            raise ValueError("Source and reference must be different persistent entities")
        normalized_events.append({"start_frame": start, "end_frame": end, **selected})
    normalized_events.sort(key=lambda item: item["start_frame"])
    for previous, current in zip(normalized_events, normalized_events[1:]):
        if current["start_frame"] <= previous["end_frame"]:
            raise ValueError("Inclusive role events overlap")
    intervals = seeds.get("invalid_intervals", {})
    for role in ROLES:
        for interval in intervals.get(role, []):
            if (not isinstance(interval, list) or len(interval) != 2
                    or any(isinstance(x, bool) or not isinstance(x, int) for x in interval)
                    or not 0 <= interval[0] <= interval[1] < frame_count):
                raise ValueError(f"Invalid role-specific exclusion interval: {role}/{interval}")
    return {"legacy_two_role_schema": legacy, "seed_frame": seed_frame, "objects": object_list,
            "role_events": normalized_events, "invalid_intervals": intervals,
            "event_bounds": "inclusive original-video frame indices"}


def select_roles_at_frame(frame, object_masks, object_valid, plan):
    """Select two roles without changing persistent identity or filling gaps.

    Candidate masks remain available for overlays even when visibility is
    invalid. Callers zero invalid coverage before saving training supervision.
    """
    masks = torch.as_tensor(object_masks, dtype=torch.bool)
    valid = torch.as_tensor(object_valid, dtype=torch.bool)
    objects = plan["objects"]
    if masks.ndim != 3 or masks.shape[0] != len(objects) or valid.shape != (len(objects),):
        raise ValueError("Object masks/validity do not match the explicit entity list")
    selected_masks = torch.zeros((2, *masks.shape[-2:]), dtype=torch.bool)
    selected_valid = torch.zeros(2, dtype=torch.bool)
    identities = torch.full((2,), -1, dtype=torch.int64)
    indices = [-1, -1]
    event = next((event for event in plan["role_events"]
                  if event["start_frame"] <= frame <= event["end_frame"]), None)
    if event is None:
        return selected_masks, selected_valid, identities, indices
    by_id = {obj["external_id"]: index for index, obj in enumerate(objects)}
    for role_index, role in enumerate(ROLES):
        index = by_id[event[role]]
        indices[role_index] = index
        selected_masks[role_index] = masks[index]
        identities[role_index] = objects[index]["entity_id"]
        manually_invalid = any(start <= frame <= end for start, end in
                               plan["invalid_intervals"].get(role, []))
        selected_valid[role_index] = valid[index] and not manually_invalid
    return selected_masks, selected_valid, identities, indices


def patch_coverage(masks):
    """Same 224x224 full-frame geometry as DINO; area, not hard patch labels.

    Area interpolation is intentional for masks: bicubic RGB interpolation can
    overshoot [0,1]. Each 14x14 region corresponds to one DINOv2 patch.
    """
    masks = torch.as_tensor(masks, dtype=torch.float32)
    if masks.ndim != 3 or masks.shape[0] != 2:
        raise ValueError("Expected two original-resolution binary masks")
    resized = F.interpolate(masks[:, None], size=(224, 224), mode="area")
    return F.avg_pool2d(resized, kernel_size=14).reshape(2, 256).clamp(0, 1)


def extract_video(video, destination, sampled_ids):
    """Decode every original frame for official SAM2 JPEG-folder input."""
    destination.mkdir(parents=True, exist_ok=True)
    sampled, count, size = {}, 0, None
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        for index, frame in enumerate(container.decode(video=0)):
            image = frame.to_image().convert("RGB")
            if size is None:
                size = image.size
            if image.size != size:
                raise ValueError("Variable-resolution videos are unsupported")
            image.save(destination / f"{index:06d}.jpg", quality=98, subsampling=0)
            if index in sampled_ids:
                sampled[index] = image
            count += 1
    if set(sampled) != set(sampled_ids):
        raise ValueError(f"Video lacks cache frames: {sorted(set(sampled_ids)-set(sampled))}")
    return sampled, {"frames": count, "width": size[0], "height": size[1], "fps": fps}


def object_score(state, frame, obj_id):
    # Official propagation yields masks only; record the stored score rather
    # than substituting mask confidence for the object's presence score.
    index = state["obj_id_to_idx"][obj_id]
    outputs = state["output_dict_per_obj"][index]
    for name in ("cond_frame_outputs", "non_cond_frame_outputs"):
        if frame in outputs[name]:
            value = outputs[name][frame].get("object_score_logits")
            if value is not None:
                return float(value.detach().float().reshape(-1)[0].cpu())
    return None


def validity_reasons(frame, role, area, seed_area, score, overlap_fraction, seeds, args):
    reasons = []
    if score is None or not np.isfinite(score):
        reasons.append("missing_object_score")
    elif float(torch.sigmoid(torch.tensor(score))) < args.min_object_probability:
        reasons.append("low_object_presence_score")
    if area < args.min_mask_pixels:
        reasons.append("empty_or_too_small_visible_mask")
    ratio = area / max(seed_area, 1)
    if not args.min_area_ratio <= ratio <= args.max_area_ratio:
        reasons.append("visible_area_changed_beyond_gate")
    if overlap_fraction > args.max_overlap_fraction:
        reasons.append("ambiguous_role_overlap")
    for interval in seeds.get("invalid_intervals", {}).get(role, []):
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError("invalid_intervals must contain inclusive [start,end] pairs")
        if int(interval[0]) <= frame <= int(interval[1]):
            reasons.append("manual_occlusion_or_tracking_failure")
    return reasons


def save_overlay(image, masks, frame, stats, path):
    pixels = np.asarray(image).astype(np.float32)
    for role_index, color in enumerate(COLORS):
        mask = masks[role_index]
        # Rejected candidates remain visible for review but are never labels.
        tint = np.array(color if stats[role_index]["valid"] else (245, 180, 30))
        pixels[mask] = .58 * pixels[mask] + .42 * tint
    # Keep the entire decoded frame visible, including objects near its top edge.
    tinted = Image.fromarray(np.uint8(np.clip(pixels, 0, 255)))
    overlay = Image.new("RGB", (tinted.width, tinted.height + 52), (15, 15, 15))
    overlay.paste(tinted, (0, 52))
    draw = ImageDraw.Draw(overlay)
    draw.text((7, 5), f"frame={frame} | UNREVIEWED SAM2 weak label | red=source cyan=reference amber=invalid", fill="white")
    for i, item in enumerate(stats):
        text = f"{ROLES[i]} id={item.get('entity_id', i)} valid={item['valid']} p={item['object_probability']} area={item['visible_pixels']}"
        draw.text((7, 21 + 14 * i), text, fill=COLORS[i])
    overlay.save(path)


def save_contact_sheets(images, title, output):
    paths = []
    for start in range(0, len(images), 16):
        group = images[start:start + 16]
        rows = (len(group) + 3) // 4
        sheet = Image.new("RGB", (4 * 384, 42 + rows * 312), (240, 240, 240))
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 9), title + f" | sampled frames {start + 1}-{start + len(group)} | UNREVIEWED", fill="black")
        for j, (frame, path) in enumerate(group):
            with Image.open(path) as source:
                tile = source.copy()
            tile.thumbnail((384, 288), Image.Resampling.LANCZOS)
            x, y = j % 4 * 384, 42 + j // 4 * 312
            sheet.paste(tile, (x + (384-tile.width)//2, y+(288-tile.height)//2))
            draw.text((x + 7, y + 291), f"raw frame {frame}", fill="black")
        path = output / f"overlays_{start // 16 + 1:02d}.png"
        sheet.save(path)
        paths.append(str(path))
    return paths


def process_video(predictor, record, kind, seeds, output, args):
    label = f"{record['label']}_{kind}"
    output = output / label
    output.mkdir(parents=True, exist_ok=True)
    payload = torch.load(record[f"{kind}_cache_path"], map_location="cpu", weights_only=True)
    ids = payload["frame_indices"]
    if payload["features"].shape[-2:] != (256, 768):
        raise ValueError("This mask recipe requires the existing 16x16 DINOv2-base cache")
    preprocessing = payload["metadata"]["preprocessing"]["config"]
    if preprocessing.get("do_center_crop") or preprocessing.get("size") != {"height": 224, "width": 224}:
        raise ValueError("Cache preprocessing is not 224x224 full-frame resize")
    selected = ids.flatten().tolist()
    video = resolve_source_video(record, kind, payload["metadata"])
    sampled, info = extract_video(video, output / "sam2_input_frames", set(selected))
    plan = normalize_role_seeds(seeds, info["frames"], info["width"], info["height"])
    state = predictor.init_state(str(output / "sam2_input_frames"), offload_video_to_cpu=True,
                                 offload_state_to_cpu=True)
    for obj in plan["objects"]:
        point_prompt = {}
        if "points_xy" in obj:
            point_prompt = {"points": np.array(obj["points_xy"], dtype=np.float32),
                            "labels": np.array(obj["point_labels"], dtype=np.int32)}
        predictor.add_new_points_or_box(state, frame_idx=plan["seed_frame"], obj_id=obj["sam_id"],
                                        box=np.array(obj["box_xyxy"], dtype=np.float32),
                                        **point_prompt)
    # SAM2's supported offline interactive refinement. Future prompts are
    # annotation evidence, never inputs to the policy. Re-review ALL resulting
    # sampled masks because conditioning on a correction may change others.
    for obj in plan['objects']:
        for correction in obj.get('corrections', []):
            predictor.add_new_points_or_box(state, frame_idx=correction['frame'], obj_id=obj['sam_id'],
                points=np.array(correction['points_xy'], dtype=np.float32),
                labels=np.array(correction['point_labels'], dtype=np.int32))
    masks_by_frame, valid_by_frame, stats_by_frame = {}, {}, {}
    entity_ids_by_frame, object_stats_by_frame = {}, {}
    overlays, seed_areas = [], None
    # A later observed prompt is not evidence about earlier occluded frames.
    # Retain every cache timestamp and its PNG, but supervise none before it.
    for frame in range(plan["seed_frame"]):
        frame_stats = [{"role": role, "entity_id": -1, "external_object_id": None,
                        "object_score_logit": None, "object_probability": None,
                        "visible_pixels": 0, "raw_mask_pixels": 0,
                        "fraction_of_image": 0., "ratio_to_seed_area": 0.,
                        "overlap_fraction_removed": 0., "valid": False,
                        "invalid_reasons": ["before_first_visible_seed"]} for role in ROLES]
        stats_by_frame[str(frame)] = frame_stats
        object_stats_by_frame[str(frame)] = []
        if frame in sampled:
            masks_by_frame[frame] = torch.zeros((2, 256))
            valid_by_frame[frame] = torch.zeros(2, dtype=torch.bool)
            entity_ids_by_frame[frame] = torch.full((2,), -1, dtype=torch.int64)
            path = output / f"overlay_{frame:06d}.png"
            save_overlay(sampled[frame], np.zeros((2, info["height"], info["width"]), dtype=bool),
                         frame, frame_stats, path)
            overlays.append((frame, path))
    for frame, obj_ids, logits in predictor.propagate_in_video(state, start_frame_idx=plan["seed_frame"]):
        order = [obj_ids.index(obj["sam_id"]) for obj in plan["objects"]]
        raw = (logits[order, 0].float().cpu().numpy() > 0)
        areas_before = raw.sum(axis=(1, 2))
        overlap = raw.sum(axis=0) > 1
        # Ambiguous overlapping pixels supervise no entity. Never fill holes
        # or extrapolate a complete object behind an occluder.
        masks = raw & ~overlap[None]
        areas = masks.sum(axis=(1, 2))
        if seed_areas is None:
            seed_areas = areas.copy()
        entity_stats, accepted = [], []
        for index, obj in enumerate(plan["objects"]):
            score = object_score(state, frame, obj["sam_id"])
            overlap_fraction = float((raw[index] & overlap).sum() / max(int(areas_before[index]), 1))
            # Visibility is evaluated per persistent entity. Role-specific
            # exclusions apply only after the explicit event mapping below.
            reasons = validity_reasons(frame, obj["external_id"], int(areas[index]), int(seed_areas[index]),
                                       score, overlap_fraction, {}, args)
            valid = not reasons
            accepted.append(valid)
            entity_stats.append({"entity_id": obj["entity_id"], "external_object_id": obj["external_id"],
                "object_score_logit": score,
                "object_probability": None if score is None else round(float(torch.sigmoid(torch.tensor(score))), 6),
                "visible_pixels": int(areas[index]), "raw_mask_pixels": int(areas_before[index]),
                "fraction_of_image": float(areas[index] / masks[index].size),
                "ratio_to_seed_area": float(areas[index] / max(seed_areas[index], 1)),
                "overlap_fraction_removed": overlap_fraction, "valid": valid,
                "invalid_reasons": reasons})
        role_masks, role_valid, identities, indices = select_roles_at_frame(frame, masks, accepted, plan)
        frame_stats = []
        for role_index, role in enumerate(ROLES):
            index = indices[role_index]
            if index < 0:
                item = {"role": role, "entity_id": -1, "external_object_id": None,
                    "object_score_logit": None, "object_probability": None,
                    "visible_pixels": 0, "raw_mask_pixels": 0, "fraction_of_image": 0.,
                    "ratio_to_seed_area": 0., "overlap_fraction_removed": 0., "valid": False,
                    "invalid_reasons": ["uncovered_role_event"]}
            else:
                item = {**entity_stats[index], "role": role, "valid": bool(role_valid[role_index]),
                        "invalid_reasons": list(entity_stats[index]["invalid_reasons"])}
                if any(start <= frame <= end for start, end in plan["invalid_intervals"].get(role, [])):
                    item["invalid_reasons"].append("manual_occlusion_or_tracking_failure")
            frame_stats.append(item)
        stats_by_frame[str(frame)] = frame_stats
        object_stats_by_frame[str(frame)] = entity_stats
        if frame in sampled:
            coverage = patch_coverage(role_masks)
            coverage[~role_valid] = 0
            masks_by_frame[frame], valid_by_frame[frame] = coverage, role_valid
            entity_ids_by_frame[frame] = identities
            path = output / f"overlay_{frame:06d}.png"
            save_overlay(sampled[frame], role_masks.numpy(), frame, frame_stats, path)
            overlays.append((frame, path))
    if set(masks_by_frame) != set(selected):
        raise RuntimeError("Propagation did not cover all RGB cache timestamps")
    coverage = torch.stack([masks_by_frame[i] for i in selected]).reshape(*ids.shape, 2, 256)
    valid = torch.stack([valid_by_frame[i] for i in selected]).reshape(*ids.shape, 2)
    entity_ids = torch.stack([entity_ids_by_frame[i] for i in selected]).reshape(*ids.shape, 2)
    sheets = save_contact_sheets(sorted(overlays), label, output)
    report = {"label": label, "video": str(video),
        "cache_source_video_path": payload["metadata"]["source_video_path"],
        "video_source_identity": video_source_identity(video, kind),
        "video_info": info, "seed": seeds, "role_seed_plan": plan,
        "cache_frame_indices": ids.tolist(), "sampled_valid_counts": valid.reshape(-1, 2).sum(0).tolist(),
        "sampled_entity_ids": entity_ids.tolist(),
        "sampled_role_counts": valid.reshape(-1, 2).shape[0], "overlay_contact_sheets": sheets,
        "all_frame_stats": stats_by_frame, "all_object_frame_stats": object_stats_by_frame,
        "warning": "SAM2 presence/area gates do not prove visible-mask accuracy. Review overlays; add explicit invalid_intervals for occlusion or tracking failures. No amodal completion, mask hole filling, or reviewer approval was applied."}
    (output / "mask_report.json").write_text(json.dumps(report, indent=2))
    predictor.reset_state(state)
    del state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(json.dumps({"video_done": label, "sampled_valid": report["sampled_valid_counts"],
                      "sampled_frames": report["sampled_role_counts"], "overlays": sheets}), flush=True)
    return coverage, valid, ids.clone(), entity_ids, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-json", type=Path, required=True)
    parser.add_argument("--seed-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("/workspace/models/sam2.1-hiera-tiny/sam2.1_hiera_tiny.pt"))
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-object-probability", type=float, default=.8)
    parser.add_argument("--min-mask-pixels", type=int, default=32)
    parser.add_argument("--min-area-ratio", type=float, default=.25)
    parser.add_argument("--max-area-ratio", type=float, default=2.0)
    parser.add_argument("--max-overlap-fraction", type=float, default=.2)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Choose a fresh output directory; reviewed artifacts are never overwritten")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = json.loads(args.seed_json.read_text())
    records = json.loads(args.records_json.read_text())["records"]
    if seeds.get("schema_version") != 1:
        raise ValueError("Expected explicit seed schema_version=1")
    import sam2
    from sam2.build_sam import build_sam2_video_predictor
    source_root = Path(sam2.__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
    teacher = {"source_commit": revision, "checkpoint_sha256": file_sha256(args.checkpoint),
               "config": args.sam2_config, "torch_version": str(torch.__version__)}
    predictor = build_sam2_video_predictor(args.sam2_config, str(args.checkpoint),
        device=args.device, mode="eval", apply_postprocessing=False)
    manifest_records, reports = [], []
    began = time.monotonic()
    device_type = torch.device(args.device).type
    autocast = torch.autocast(device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
    with torch.inference_mode(), autocast:
        for record in records:
            annotation = {}
            video_reports = {}
            for kind in ("human", "robot"):
                key = f"{record['label']}_{kind}"
                if key not in seeds["videos"]:
                    raise ValueError(f"No explicit seed annotation for {key}")
                masks, valid, indices, entity_ids, report = process_video(predictor, record, kind, seeds["videos"][key], args.output_dir, args)
                annotation.update({f"{kind}_masks": masks, f"{kind}_valid": valid,
                                   f"{kind}_frame_indices": indices, f"{kind}_entity_ids": entity_ids})
                video_reports[kind] = report
            hashes = {f"{kind}_cache_sha256": file_sha256(record[f"{kind}_cache_path"]) for kind in ("human", "robot")}
            provenance = {"kind": "manual_seed_boxes_plus_official_SAM2_visible_mask_candidates",
                "not_ground_truth": True, "reviewed": False, "seed_file": str(args.seed_json.resolve()),
                "seed_sha256": file_sha256(args.seed_json), "records_file": str(args.records_json.resolve()),
                "records_sha256": file_sha256(args.records_json), "teacher_frames": "all original video frames decoded to JPEG quality=98 subsampling=0",
                "mask_resize": "full-frame area resize to 224x224; 14x14 mean coverage on 16x16 DINO grid",
                "occlusion_policy": "no completion; low-confidence/area/overlap gates and explicit manual invalid_intervals; final visual review required",
                "gate_settings": {key: getattr(args, key) for key in ("min_object_probability", "min_mask_pixels", "min_area_ratio", "max_area_ratio", "max_overlap_fraction")}}
            annotation["metadata"] = {"schema_version": 1, "pair_key": record["pair_key"], **hashes,
                "role_order": list(ROLES), "teacher_model": "sam2 (see teacher config and checkpoint digest)",
                "teacher_revision": teacher, "provenance": provenance,
                "entity_identity_scope": "persistent within each video; explicit role_events only",
                "role_seed_plans": {kind: report["role_seed_plan"] for kind, report in video_reports.items()}}
            name = record["pair_key"].replace(":", "_") + ".pth"
            path = args.output_dir / name
            torch.save(annotation, path)
            manifest_records.append({"pair_key": record["pair_key"], "task_name": record["task"], "split": "train",
                "annotation_file": name, "annotation_sha256": file_sha256(path),
                "human_cache": record["human_cache_path"], "robot_cache": record["robot_cache_path"], **hashes,
                "reviewed": False, "stable_roles": None, "single_effect": None,
                "effect_signature": None, "donor_pair_key": None, "donor_review": None,
                "provenance": provenance})
            reports.append({"pair_key": record["pair_key"], "annotation": str(path), "videos": video_reports})
    (args.output_dir / "candidate_manifest.json").write_text(json.dumps({"schema_version": 1, "split": "train", "records": manifest_records}, indent=2))
    (args.output_dir / "summary.json").write_text(json.dumps({"reviewed": False, "teacher_revision": teacher,
        "elapsed_seconds": time.monotonic()-began, "pairs": reports}, indent=2))
    print(f"Unreviewed candidate manifest: {args.output_dir / 'candidate_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
