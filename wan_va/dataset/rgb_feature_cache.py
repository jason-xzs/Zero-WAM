"""Validated, frozen RGB patch features shared by training and inference.

Human files follow the original ``run_...`` path. Robot files correspond to a
specific latent interval, and their time axis is exactly the Wan latent axis.
No RGB features are inferred from VAE latents or from future robot inputs.
"""
from pathlib import Path
import hashlib
import posixpath
import re

import torch


SCHEMA_VERSION = 1
DEFAULT_ENCODER_ID = "facebook/dinov2-base"


def rgb_pair_key(sample):
    # A generated human video is deliberately paired with multiple robot
    # episodes; neither numeric sample_id nor run+sample_id identifies a pair.
    robot = (sample.get("status_info") or {}).get("video_rel_path") or sample.get("robot_video_path", "")
    value = posixpath.normpath(str(robot).replace("\\", "/")) + "\n" + posixpath.normpath(str(sample.get("human_video_path", "")).replace("\\", "/"))
    return f"{sample['sample_id']}:{hashlib.sha256(value.encode()).hexdigest()[:16]}"


def human_relative_path(human_path):
    parts = Path(str(human_path).replace("\\", "/")).parts
    start = next((i for i, part in enumerate(parts) if part.startswith("run_")), None)
    if start is None or ".." in parts:
        raise ValueError(f"Human path must contain a safe run_ component: {human_path}")
    return Path(*parts[start:]).with_suffix(".pth")


def human_feature_path(root, human_path):
    return Path(root) / "human" / human_relative_path(human_path)


def robot_feature_path(root, task_name, camera_key, meta):
    return (Path(root) / "robot" / task_name / camera_key /
            f"episode_{int(meta['episode_index']):06d}_{int(meta['start_frame'])}_{int(meta['end_frame'])}.pth")


def latent_rgb_frame_indices(latent_payload):
    """Each causal Wan latent corresponds to raw frame_ids[0,4,8,...]."""
    ids = torch.as_tensor(latent_payload["frame_ids"], dtype=torch.int64)
    count = int(latent_payload["latent_num_frames"])
    if ids.ndim != 1 or len(ids) != 1 + 4 * (count - 1):
        raise ValueError("Robot frame_ids do not match Wan temporal compression (1+4*(T-1))")
    if len(ids) > 1 and not torch.all(ids[1:] > ids[:-1]):
        raise ValueError("Robot frame_ids must be strictly increasing")
    return ids[::4].clone()


def load_feature_cache(path, kind, *, expected_encoder_id=None,
                       expected_sample_id=None, expected_revision=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen RGB {kind} features: {path}; run tools/prepare_rgb_transition_features.py")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not {"features", "frame_indices", "metadata"} <= payload.keys():
        raise ValueError(f"Invalid RGB feature payload: {path}")
    metadata = payload["metadata"]
    required = {"schema_version", "kind", "encoder_id", "encoder_revision", "preprocessing", "sample_id", "source_video_path"}
    if not isinstance(metadata, dict) or not required <= metadata.keys():
        raise ValueError(f"Incomplete RGB metadata: {path}")
    if metadata["schema_version"] != SCHEMA_VERSION or metadata["kind"] != kind:
        raise ValueError(f"RGB cache schema/kind mismatch: {path}")
    for key, expected in [("encoder_id", expected_encoder_id), ("encoder_revision", expected_revision), ("sample_id", expected_sample_id)]:
        if expected is not None and str(metadata[key]) != str(expected):
            raise ValueError(f"RGB cache {key} mismatch in {path}: {metadata[key]!r} != {expected!r}")
    if not metadata["encoder_revision"] or not isinstance(metadata["preprocessing"], dict):
        raise ValueError(f"Unpinned encoder or invalid preprocessing: {path}")
    features, ids = payload["features"], payload["frame_indices"]
    ndim = {"human": 4, "robot": 3}.get(kind)
    if not isinstance(features, torch.Tensor) or features.ndim != ndim or not features.is_floating_point():
        raise ValueError(f"Unexpected {kind} features shape in {path}")
    if not isinstance(ids, torch.Tensor) or ids.dtype != torch.int64 or tuple(ids.shape) != tuple(features.shape[:-2]):
        raise ValueError(f"RGB frame indices/feature shape mismatch: {path}")
    if min(features.shape) <= 0 or torch.any(ids < 0) or not torch.isfinite(features).all():
        raise ValueError(f"Empty, negative-index or nonfinite RGB features: {path}")
    if ids.numel() > 1 and torch.any(ids.flatten()[1:] < ids.flatten()[:-1]):
        raise ValueError(f"Nonmonotonic RGB frame indices: {path}")
    return payload


def check_robot_alignment(payload, latent_payload, *, expected_sample_id=None):
    expected = latent_rgb_frame_indices(latent_payload)
    if not torch.equal(payload["frame_indices"], expected):
        raise ValueError("RGB robot frame indices differ from the sampled robot latent frame_ids[::4]")
    if expected_sample_id is not None and str(payload["metadata"]["sample_id"]) != str(expected_sample_id):
        raise ValueError("RGB robot sample_id mismatch")


def check_feature_limits(human, robot, limits):
    """Check positional-table and feature sizes before loading the large model."""
    human_shape, robot_shape = human["features"].shape, robot["features"].shape
    expected_dim = limits.get("feature_dim", 768)
    max_patches = limits.get("max_patches", 1024)
    max_frames = limits.get("max_future_steps", 64) + 1
    if human_shape[-2:] != robot_shape[-2:] or human_shape[-1] != expected_dim:
        raise ValueError(f"RGB feature width/patch count does not match model config: human={tuple(human_shape)}, robot={tuple(robot_shape)}")
    if human_shape[-2] > max_patches:
        raise ValueError(f"RGB patch count {human_shape[-2]} exceeds max_patches={max_patches}")
    if human_shape[0] > limits.get("max_clips", 32):
        raise ValueError(f"RGB human clip count {human_shape[0]} exceeds max_clips")
    if not 2 <= human_shape[1] <= max_frames or not 2 <= robot_shape[0] <= max_frames:
        raise ValueError(f"RGB timeline exceeds max_future_steps={max_frames - 1}: human={human_shape[1]}, robot={robot_shape[0]}")
    if "frames_per_clip" in limits and human_shape[1] != limits["frames_per_clip"]:
        raise ValueError("RGB human frames_per_clip does not match model config")


def _robot_latent_identity(path):
    """Ignore only the storage root of a recognized RoboTwin latent layout.

    The full task directory (including its variant), chunk, camera and episode
    interval remain part of the identity. Unknown layouts are not portable by
    basename: their original path must still match exactly.
    """
    parts = Path(path).parts
    if len(parts) < 5 or ".." in parts:
        return None
    task, directory, chunk, camera, filename = parts[-5:]
    if (
        directory != "latents"
        or re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", task) is None
        or re.fullmatch(r"chunk-[0-9]{3,}", chunk) is None
        or re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", camera) is None
        or re.fullmatch(r"episode_[0-9]{6,}_[0-9]+_[0-9]+\.pth", filename) is None
    ):
        return None
    return task, directory, chunk, camera, filename


def check_feature_sources(human, robot, *, human_video_path, robot_latent_path):
    if human_relative_path(human["metadata"]["source_video_path"]) != human_relative_path(human_video_path):
        raise ValueError("RGB human source video differs from the selected manifest pair")
    original_path = Path(robot["metadata"].get("source_latent_path", ""))
    current_path = Path(robot_latent_path)
    if original_path != current_path:
        original_identity = _robot_latent_identity(original_path)
        if original_identity is None or original_identity != _robot_latent_identity(current_path):
            raise ValueError("RGB robot source latent differs from the selected interval")


def expected_rgb_records(config):
    """Enumerate the same manifest/episode intervals as the Robotwin loader.

    An explicit smoke allowlist filters samples, never missing cache files.
    Every selected task must still have at least one retained sample.
    """
    import json
    from types import SimpleNamespace
    from .icl_lerobot_latent_dataset import ICLLeRobotLatentDataset, _icl_repo_list, _task_latent_root, load_icl_manifest

    exact, episodes = load_icl_manifest(config.icl_manifest_path)
    allowed = getattr(config, "rgb_smoke_sample_ids", None)
    allowed = None if allowed is None else {str(x) for x in allowed}
    found = set()
    for root in _icl_repo_list(config):
        info = json.loads((root / "meta/info.json").read_text())
        chunk_size = int(info.get("chunks_size", 1000))
        stub = ICLLeRobotLatentDataset.__new__(ICLLeRobotLatentDataset)
        stub.root, stub.latent_path = root, _task_latent_root(root, config)
        stub.meta = SimpleNamespace(get_episode_chunk=lambda idx, size=chunk_size: idx // size)
        stub._icl_exact_index, stub._icl_episode_index = exact, episodes
        count = 0
        with (root / "meta/episodes.jsonl").open() as handle:
            for line in handle:
                episode = json.loads(line)
                for interval in episode.get("action_config", []):
                    meta = {"episode_index": episode["episode_index"], **interval}
                    if episode.get("icl"):
                        meta["icl"] = episode["icl"]
                    sample = stub._lookup_icl_sample(meta)
                    if sample is None:
                        continue
                    pair_key = rgb_pair_key(sample)
                    if allowed is not None and pair_key not in allowed:
                        continue
                    human_latent = Path(config.human_latent_path) / human_relative_path(sample["human_video_path"])
                    if not human_latent.is_file():
                        raise FileNotFoundError(f"Missing original paired human latent: {human_latent}")
                    camera = config.obs_cam_keys[0]
                    latent_path = stub._latent_file(meta, camera)
                    if not latent_path.is_file():
                        raise FileNotFoundError(f"Missing original robot latent: {latent_path}")
                    video_relative = info["video_path"].format(episode_chunk=episode["episode_index"] // chunk_size, episode_index=episode["episode_index"], video_key=camera)
                    count += 1
                    found.add(pair_key)
                    yield {"sample": sample, "meta": meta, "task_name": root.name,
                           "camera_key": camera, "robot_latent_path": latent_path,
                           "robot_video_path": root / video_relative,
                           "human_cache_path": human_feature_path(config.rgb_feature_root, sample["human_video_path"]),
                           "robot_cache_path": robot_feature_path(config.rgb_feature_root, root.name, camera, meta)}
        if count == 0:
            raise ValueError(f"No explicitly selected RGB samples for training task {root.name}")
    if allowed is not None and allowed != found:
        raise ValueError(f"RGB smoke allowlist IDs absent from selected training tasks: {sorted(allowed - found)}")


def validate_rgb_feature_cache(config):
    count = 0
    tasks = set()
    encoders, revisions = set(), set()
    feature_spec = None
    human_shapes, robot_lengths = set(), set()
    limits = getattr(config, "rgb_transition_config", {}) or {}
    for record in expected_rgb_records(config):
        sid = str(record["sample"]["sample_id"])
        kwargs = dict(expected_encoder_id=getattr(config, "rgb_encoder_id", DEFAULT_ENCODER_ID),
                      expected_sample_id=sid, expected_revision=getattr(config, "rgb_encoder_revision", None))
        human = load_feature_cache(record["human_cache_path"], "human", **kwargs)
        robot = load_feature_cache(record["robot_cache_path"], "robot", **kwargs)
        if human["features"].shape[-2:] != robot["features"].shape[-2:]:
            raise ValueError("Human/robot RGB feature dimensions differ")
        for key in ("encoder_id", "encoder_revision", "preprocessing"):
            if human["metadata"][key] != robot["metadata"][key]:
                raise ValueError(f"Human/robot RGB metadata differ: {key}")
        this_spec = {"rgb_encoder_id": human["metadata"]["encoder_id"],
                     "rgb_encoder_revision": human["metadata"]["encoder_revision"],
                     "rgb_preprocessing": human["metadata"]["preprocessing"]}
        if feature_spec is not None and this_spec != feature_spec:
            raise ValueError("RGB encoder revision/preprocessing differs across selected training pairs")
        feature_spec = this_spec
        expected_preprocessing = getattr(config, "rgb_preprocessing", None)
        if expected_preprocessing is not None and expected_preprocessing != this_spec["rgb_preprocessing"]:
            raise ValueError("RGB preprocessing differs from the saved model config")
        check_feature_limits(human, robot, limits)
        check_feature_sources(human, robot, human_video_path=record["sample"]["human_video_path"],
                              robot_latent_path=record["robot_latent_path"])
        latent = torch.load(record["robot_latent_path"], map_location="cpu", weights_only=False)
        check_robot_alignment(robot, latent, expected_sample_id=sid)
        count += 1
        tasks.add(record["task_name"])
        encoders.add(human["metadata"]["encoder_id"])
        revisions.add(human["metadata"]["encoder_revision"])
        human_shapes.add(tuple(human["features"].shape))
        robot_lengths.add(int(robot["features"].shape[0]))
    return {"pairs": count, "tasks": len(tasks), "smoke_subset": getattr(config, "rgb_smoke_sample_ids", None) is not None,
            "rgb_feature_root": str(config.rgb_feature_root), "schema_version": SCHEMA_VERSION,
            "encoder_ids": sorted(encoders), "encoder_revisions": sorted(revisions), "feature_spec": feature_spec,
            "human_shapes": [list(shape) for shape in sorted(human_shapes)], "robot_frame_counts": sorted(robot_lengths)}
