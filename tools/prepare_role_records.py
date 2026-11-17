#!/usr/bin/env python3
"""Export real subset12 pairs and unapproved seed templates for SAM2 labels.

All pairs:
  python tools/prepare_role_records.py --output-json local/roles/records.json \
      --seed-template local/roles/seeds.unapproved.json
Selected pairs (no requirement to select all twelve tasks):
  python tools/prepare_role_records.py --pair-key 4:64476c9e70b954f0 \
      --output-json local/roles/mouse.json --preview-dir local/roles/previews

Fill source/reference identities and original-pixel boxes manually before
running prepare_role_masks_sam2.py. This tool never assigns boxes, approves
identities, or asserts donor equivalence. Existing output files are preserved.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def select_records(records, pair_keys):
    """Filter only after full manifest enumeration, never via smoke filters."""
    from wan_va.dataset.rgb_feature_cache import rgb_pair_key

    by_key = {}
    for record in records:
        key = rgb_pair_key(record["sample"])
        if key in by_key:
            raise ValueError(f"Training manifest contains duplicate pair key: {key}")
        by_key[key] = record
    requested = list(pair_keys or [])
    if len(set(requested)) != len(requested):
        raise ValueError("--pair-key entries must be unique")
    unknown = sorted(set(requested) - by_key.keys())
    if unknown:
        raise ValueError(f"Pair keys absent from the configured training manifest: {unknown}")
    allowed = set(requested)
    selected = [record for key, record in by_key.items() if not requested or key in allowed]
    if not selected:
        raise ValueError("No real training pairs were selected")
    return selected


def pair_label(pair_key):
    if re.fullmatch(r"[0-9]+:[0-9a-f]{16}", pair_key) is None:
        raise ValueError(f"Unexpected training pair-key format: {pair_key}")
    return pair_key.replace(":", "_")


def human_video_path(root, source):
    """Resolve the official manifest path beneath HumanGen/human_data."""
    relative = Path(str(source).replace("\\", "/").removeprefix("icl_raw_data/"))
    if relative.is_absolute() or ".." in relative.parts or relative.parts[:2] != ("human_data", "robotwin"):
        raise ValueError(f"Expected an official human_data/robotwin video path: {source}")
    return (root / relative).resolve(strict=True)


@lru_cache(maxsize=1024)
def cache_summary(path, kind, encoder_id, revision, sample_id):
    """Retain metadata/indices only; do not keep 600 feature tensors in RAM."""
    from wan_va.dataset.rgb_feature_cache import load_feature_cache

    payload = load_feature_cache(path, kind, expected_encoder_id=encoder_id,
                                 expected_revision=revision, expected_sample_id=sample_id)
    if tuple(payload["features"].shape[-2:]) != (256, 768):
        raise ValueError(f"SAM2 mask preparation expects DINOv2-base 16x16 patches: {path}")
    preprocessing = payload["metadata"]["preprocessing"].get("config", {})
    if preprocessing.get("do_center_crop") or preprocessing.get("size") != {"height": 224, "width": 224}:
        raise ValueError(f"SAM2 mask preparation requires full-frame 224x224 resize: {path}")
    return {"shape": list(payload["features"].shape),
            "frame_indices": payload["frame_indices"].tolist(),
            "metadata": deepcopy(payload["metadata"])}


@lru_cache(maxsize=1024)
def video_info(path):
    """Verify a real decodable frame without decoding every full video twice."""
    import av

    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError(f"Video has no video stream: {path}")
        stream = container.streams.video[0]
        frame = next(container.decode(video=0), None)
        if frame is None:
            raise ValueError(f"Video contains no decodable frame: {path}")
        return {"width": frame.width, "height": frame.height,
                "fps": float(stream.average_rate) if stream.average_rate else None,
                "container_frame_count": int(stream.frames) or None,
                "validation": "frame_0_decoded; full sampled-frame coverage checked by SAM2 preparation"}


def save_frame_zero(video, target):
    import av

    with av.open(str(video)) as container:
        frame = next(container.decode(video=0), None)
        if frame is None:
            raise ValueError(f"Video contains no decodable frame: {video}")
        # Preserve original pixel coordinates; never infer or draw role boxes.
        frame.to_image().convert("RGB").save(target)


def export_record(record, config, root, preview_dir=None):
    import torch
    from wan_va.dataset.rgb_feature_cache import (
        human_relative_path, latent_rgb_frame_indices, rgb_pair_key,
        _robot_latent_identity,
    )

    sample = record["sample"]
    key = rgb_pair_key(sample)
    label = pair_label(key)
    result = {"label": label, "pair_key": key, "task": record["task_name"].split("-", 1)[0],
              "sample": deepcopy(sample), "episode": deepcopy(record["meta"]),
              "robot_latent_path": str(record["robot_latent_path"].resolve(strict=True)),
              "human_latent_path": str((Path(config.human_latent_path) /
                                        human_relative_path(sample["human_video_path"])).resolve(strict=True)),
              "reviewed": False, "role_identity_approved": False, "videos": {}}
    summaries = {}
    for kind in ("human", "robot"):
        cache = record[f"{kind}_cache_path"].resolve(strict=True)
        summary = cache_summary(str(cache), kind, config.rgb_encoder_id,
                                config.rgb_encoder_revision, str(sample["sample_id"]))
        summaries[kind] = summary
        metadata = summary["metadata"]
        if kind == "human":
            video = human_video_path(root, sample["human_video_path"])
            if human_relative_path(metadata["source_video_path"]) != human_relative_path(sample["human_video_path"]):
                raise ValueError(f"Human cache is not from the selected training pair: {key}")
        else:
            video = record["robot_video_path"].resolve(strict=True)
            original_video = Path(metadata["source_video_path"])
            # Storage roots may move; task variant, chunk, camera and episode
            # must still identify the exact source video.
            if (".." in original_video.parts or len(original_video.parts) < 5
                    or original_video.parts[-4] != "videos"
                    or original_video.parts[-5:] != video.parts[-5:]):
                raise ValueError(f"Robot cache video differs from the selected training pair: {key}")
            saved_latent = Path(metadata.get("source_latent_path", ""))
            actual_latent = Path(result["robot_latent_path"])
            if saved_latent != actual_latent:
                identity = _robot_latent_identity(saved_latent)
                if identity is None or identity != _robot_latent_identity(actual_latent):
                    raise ValueError(f"Robot cache is not from the selected latent interval: {key}")
            # Only frame IDs and the declared frame count are read: mmap avoids
            # copying the much larger Wan feature tensor for every record.
            latent = torch.load(actual_latent, map_location="cpu", weights_only=True, mmap=True)
            if latent_rgb_frame_indices(latent).tolist() != summary["frame_indices"]:
                raise ValueError(f"Robot cache timestamps differ from actual latent anchors: {key}")
        info = dict(video_info(str(video)))
        flattened_ids = torch.as_tensor(summary["frame_indices"], dtype=torch.long).flatten()
        if info["container_frame_count"] and int(flattened_ids.max()) >= info["container_frame_count"]:
            raise ValueError(f"Cache frame exceeds the video's declared frame count: {key}/{kind}")
        result.update({f"{kind}_cache_path": str(cache), f"{kind}_video_path": str(video),
                       f"{kind}_feature_shape": summary["shape"],
                       f"{kind}_frame_indices": summary["frame_indices"],
                       f"{kind}_cache_metadata": metadata})
        result["videos"][kind] = {"path": str(video), **info}
        if preview_dir is not None:
            preview = preview_dir / f"{label}_{kind}_frame_000000.png"
            if preview.exists():
                raise FileExistsError(f"Refusing to overwrite preview: {preview}")
            save_frame_zero(video, preview)
            result["videos"][kind]["frame_0_preview"] = str(preview)
    if summaries["human"]["metadata"]["preprocessing"] != summaries["robot"]["metadata"]["preprocessing"]:
        raise ValueError(f"Human and robot cache preprocessing differs: {key}")
    return result


def seed_template(records):
    return {"schema_version": 1, "reviewed": False, "role_identity_approved": False,
            "coordinate_convention": "Original decoded frame pixels: [x_min,y_min,x_max,y_max]; frame 0 only.",
            "instructions": "Fill each role identity and box manually. Null boxes are intentional and cannot be used for SAM2. Review propagated masks separately; no donor is approved here.",
            "videos": {f"{record['label']}_{kind}": {
                "pair_key": record["pair_key"], "video_path": record[f"{kind}_video_path"],
                "seed_frame": 0, "boxes_xyxy": {"source": None, "reference": None},
                "role_identity": {"source": None, "reference": None},
                "role_identity_approved": False, "reviewed": False,
                "invalid_intervals": {"source": [], "reference": []},
            } for record in records for kind in ("human", "robot")}}


def write_new_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", choices=["robotwin_subset12_rgb_train"], default="robotwin_subset12_rgb_train")
    parser.add_argument("--human-gen-root", type=Path, default=Path(os.environ.get("HUMAN_GEN_ROOT", "/workspace/zero-wam/datasets/HumanGen")))
    parser.add_argument("--feature-root", type=Path, default=Path(os.environ.get("ZERO_WAM_RGB_FEATURE_ROOT", "/workspace/zero-wam/datasets/rgb_transition_dinov2_base")))
    parser.add_argument("--pair-key", action="append", default=[], help="Exact training pair key; repeat for a subset, omit for all pairs")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--seed-template", type=Path, help="Optional new JSON with null boxes and unapproved role identities")
    parser.add_argument("--preview-dir", type=Path, help="Optional empty directory for original-resolution frame-0 PNGs")
    args = parser.parse_args(argv)
    outputs = [path.resolve() for path in (args.output_json, args.seed_template) if path is not None]
    if len(set(outputs)) != len(outputs):
        parser.error("--output-json and --seed-template must be different files")
    if any(path.exists() for path in outputs):
        parser.error("Choose new output files; existing annotations/templates are never overwritten")
    if args.preview_dir is not None:
        args.preview_dir = args.preview_dir.resolve()
        if args.preview_dir.exists() and (not args.preview_dir.is_dir() or any(args.preview_dir.iterdir())):
            parser.error("--preview-dir must be a new or empty directory")
        if args.preview_dir in outputs:
            parser.error("--preview-dir must differ from JSON output paths")

    import torch
    from wan_va.configs import VA_CONFIGS
    from wan_va.dataset.rgb_feature_cache import expected_rgb_records

    torch.set_num_threads(min(torch.get_num_threads(), 4))
    config = deepcopy(VA_CONFIGS[args.config])
    root = args.human_gen_root.resolve(strict=True)
    config.dataset_path = str(root / "robotwin_data")
    config.icl_manifest_path = str(root / "icl_configs/ICL_config_robotwin.json")
    config.human_latent_path = str(root / "human_latents/robotwin")
    config.rgb_feature_root = str(args.feature_root.resolve(strict=True))
    config.rgb_smoke_sample_ids = None
    candidates = list(expected_rgb_records(config))
    selected = select_records(candidates, args.pair_key)
    if args.preview_dir is not None:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"training_manifest_pairs": len(candidates), "selected_pairs": len(selected),
                      "selected_tasks": len({item["task_name"] for item in selected})}), flush=True)
    records = []
    for index, record in enumerate(selected, 1):
        records.append(export_record(record, config, root, args.preview_dir))
        if index % 50 == 0:
            print(json.dumps({"validated_pairs": index, "selected_pairs": len(selected)}), flush=True)
    payload = {"schema_version": 1, "split": "train", "config": args.config,
               "training_manifest_pairs": len(candidates), "selected_pairs": len(records),
               "reviewed": False, "role_identity_approved": False,
               "selection": "Real training pairs; no role boxes or donor semantic approval assigned.",
               "coordinate_convention": "Original decoded frame pixels; previews preserve full resolution.",
               "records": records}
    write_new_json(args.output_json, payload)
    if args.seed_template is not None:
        write_new_json(args.seed_template, seed_template(records))
    print(json.dumps({"records_json": str(args.output_json.resolve()), "pairs": len(records),
                      "seed_template": str(args.seed_template.resolve()) if args.seed_template else None,
                      "reviewed": False}), flush=True)


if __name__ == "__main__":
    main()
