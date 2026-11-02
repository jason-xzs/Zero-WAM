import pytest
import torch

from wan_va.dataset.rgb_feature_cache import (
    check_robot_alignment, human_feature_path, latent_rgb_frame_indices,
    load_feature_cache, robot_feature_path, rgb_pair_key, check_feature_limits,
    check_feature_sources,
)


def _write_cache(path, kind="robot", frame_indices=None, **metadata):
    indices = torch.tensor([100, 116, 132] if frame_indices is None else frame_indices, dtype=torch.int64)
    features = torch.ones((*indices.shape, 2, 4), dtype=torch.float16)
    payload = {
        "features": features, "frame_indices": indices,
        "metadata": {"schema_version": 1, "kind": kind, "encoder_id": "test-rgb",
                     "encoder_revision": "immutable-revision", "preprocessing": {"do_center_crop": False},
                     "sample_id": "12", "source_video_path": "/workspace/video.mp4", **metadata},
    }
    torch.save(payload, path)
    return payload


def test_wan_alignment_retains_nonzero_source_frame_offset():
    latent = {"frame_ids": list(range(100, 133, 4)), "latent_num_frames": 3}
    assert latent_rgb_frame_indices(latent).tolist() == [100, 116, 132]
    payload = {"frame_indices": torch.tensor([100, 116, 132]), "metadata": {"sample_id": "12"}}
    check_robot_alignment(payload, latent, expected_sample_id=12)
    payload["frame_indices"] = torch.tensor([0, 16, 32])
    with pytest.raises(ValueError, match="sampled robot latent"):
        check_robot_alignment(payload, latent)


def test_wan_timeline_cannot_be_truncated_or_invented():
    with pytest.raises(ValueError, match="temporal compression"):
        latent_rgb_frame_indices({"frame_ids": [100, 104, 108], "latent_num_frames": 3})


def test_human_video_and_latent_resolve_to_same_cache(tmp_path):
    a = human_feature_path(tmp_path, "/workspace/human_data/robotwin/run_1/samples/2/generated.mp4")
    b = human_feature_path(tmp_path, "/workspace/human_latents/robotwin/run_1/samples/2/generated.pth")
    assert a == b == tmp_path / "human/run_1/samples/2/generated.pth"
    with pytest.raises(ValueError, match="safe run_"):
        human_feature_path(tmp_path, "run_1/../../other.mp4")


def test_missing_cache_fails_instead_of_synthesizing_features(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing frozen RGB"):
        load_feature_cache(tmp_path / "absent.pth", "human")


@pytest.mark.parametrize("kwargs", [
    {"expected_sample_id": "another-pair"},
    {"expected_encoder_id": "another-encoder"},
    {"expected_revision": "different-weights"},
])
def test_wrong_sample_or_encoder_cannot_be_reused(tmp_path, kwargs):
    path = tmp_path / "cache.pth"
    _write_cache(path)
    with pytest.raises(ValueError, match="mismatch"):
        load_feature_cache(path, "robot", **kwargs)


def test_human_segment_axis_and_frame_indices_are_validated(tmp_path):
    path = tmp_path / "cache.pth"
    _write_cache(path, "human", frame_indices=[[0, 2], [4, 6]])
    assert load_feature_cache(path, "human")["features"].shape == (2, 2, 2, 4)
    payload = torch.load(path, weights_only=True)
    payload["frame_indices"] = payload["frame_indices"].flatten()
    torch.save(payload, path)
    with pytest.raises(ValueError, match="indices/feature shape"):
        load_feature_cache(path, "human")


def test_interval_and_camera_are_part_of_robot_cache_identity(tmp_path):
    first = {"episode_index": 3, "start_frame": 0, "end_frame": 100}
    later = {"episode_index": 3, "start_frame": 50, "end_frame": 150}
    assert robot_feature_path(tmp_path, "task", "cam", first) != robot_feature_path(tmp_path, "task", "cam", later)
    assert robot_feature_path(tmp_path, "task", "left", first) != robot_feature_path(tmp_path, "task", "right", first)


def test_reused_human_demo_pairs_have_distinct_smoke_keys():
    human = "human_data/robotwin/run_1/samples/12/generated.mp4"
    first = {"sample_id": 12, "human_video_path": human, "robot_video_path": "task/episode_001.mp4"}
    second = {**first, "robot_video_path": "task/episode_002.mp4"}
    assert rgb_pair_key(first) != rgb_pair_key(second)
    assert rgb_pair_key(first) == rgb_pair_key({**first, "robot_video_path": "./task/episode_001.mp4"})


@pytest.mark.parametrize("limits,error", [
    ({"feature_dim": 3}, "width/patch"),
    ({"feature_dim": 4, "max_patches": 1}, "patch count"),
    ({"feature_dim": 4, "max_future_steps": 1}, "timeline"),
    ({"feature_dim": 4, "max_clips": 1}, "clip count"),
])
def test_preflight_rejects_model_capacity_mismatch(limits, error):
    human = {"features": torch.zeros(2, 4, 2, 4)}
    robot = {"features": torch.zeros(3, 2, 4)}
    with pytest.raises(ValueError, match=error):
        check_feature_limits(human, robot, limits)


def test_source_identity_rejects_another_pair_with_same_numeric_id():
    human = {"metadata": {"source_video_path": "/workspace/human/run_1/a.mp4"}}
    robot = {"metadata": {"source_latent_path": "/workspace/task/episode_1.pth"}}
    check_feature_sources(human, robot, human_video_path="human_data/robotwin/run_1/a.mp4",
                          robot_latent_path="/workspace/task/episode_1.pth")
    with pytest.raises(ValueError, match="source latent"):
        check_feature_sources(human, robot, human_video_path="human_data/robotwin/run_1/a.mp4",
                              robot_latent_path="/workspace/task/episode_2.pth")


_ROBOT_LATENT_SUFFIX = (
    "place_bread_skillet-demo_clean_collect_200-1000/latents/chunk-000/"
    "observation.images.cam_high/episode_000123_0_128.pth"
)


def test_robot_cache_remains_valid_after_relocating_dataset_root():
    human = {"metadata": {"source_video_path": "/workspace/human/run_1/a.mp4"}}
    robot = {"metadata": {"source_latent_path": "/workspace/HumanGen/robotwin_data/" + _ROBOT_LATENT_SUFFIX}}
    check_feature_sources(
        human, robot, human_video_path="/cluster/data/human/run_1/a.mp4",
        robot_latent_path="/cluster/new-host/robot_latents/" + _ROBOT_LATENT_SUFFIX,
    )


@pytest.mark.parametrize("old,new", [
    ("place_bread_skillet", "place_can_basket"),
    ("demo_clean_collect_200-1000", "demo_randomized_collect_200-1000"),
    ("chunk-000", "chunk-001"),
    ("cam_high", "cam_left_wrist"),
    ("episode_000123", "episode_000124"),
    ("_0_128.pth", "_16_128.pth"),
    ("_0_128.pth", "_0_144.pth"),
])
def test_relocated_robot_cache_keeps_complete_interval_identity(old, new):
    human = {"metadata": {"source_video_path": "/workspace/human/run_1/a.mp4"}}
    robot = {"metadata": {"source_latent_path": "/workspace/data/" + _ROBOT_LATENT_SUFFIX}}
    with pytest.raises(ValueError, match="source latent"):
        check_feature_sources(
            human, robot, human_video_path="/cluster/human/run_1/a.mp4",
            robot_latent_path="/cluster/data/" + _ROBOT_LATENT_SUFFIX.replace(old, new),
        )


@pytest.mark.parametrize("suffix", [
    "task/episode_000123_0_128.pth",
    "task/latents/unknown-layout/cam/episode_000123_0_128.pth",
    "task/latents/chunk-000/cam/episode_000123.pth",
    "task/../task/latents/chunk-000/cam/episode_000123_0_128.pth",
])
def test_unknown_robot_layout_still_requires_exact_source_path(suffix):
    human = {"metadata": {"source_video_path": "/workspace/human/run_1/a.mp4"}}
    robot = {"metadata": {"source_latent_path": "/workspace/data/" + suffix}}
    check_feature_sources(human, robot, human_video_path="run_1/a.mp4",
                          robot_latent_path="/workspace/data/" + suffix)
    with pytest.raises(ValueError, match="source latent"):
        check_feature_sources(human, robot, human_video_path="run_1/a.mp4",
                              robot_latent_path="/cluster/data/" + suffix)


def test_inference_inputs_use_relocated_videos_not_provenance_paths(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from PIL import Image
    from wan_va.configs import VA_CONFIGS
    import wan_va.dataset.rgb_feature_cache as cache_module
    import tools.prepare_rgb_transition_features as preparation
    from tools.smoke_rgb_inference import select_inputs

    root = tmp_path / "new_server/HumanGen"
    records = []
    for sid in (1, 2):
        sample = {"sample_id": sid, "human_video_path": f"human_data/robotwin/run_1/{sid}/video.mp4",
                  "robot_video_path": f"task/videos/chunk-000/observation.images.cam_high/episode_{sid:06d}.mp4",
                  "robot_task_name": "Move the object"}
        human_path, robot_path = tmp_path / f"human_{sid}.pth", tmp_path / f"robot_{sid}.pth"
        _write_cache(human_path, "human", frame_indices=[[0, 1], [2, 3]], sample_id=sid)
        _write_cache(robot_path, sample_id=sid, source_video_path=f"/old_server/{sample['robot_video_path']}",
                     source_latent_frame_ids=[0, 4, 8, 12, 16])
        latent = root / f"human_latents/robotwin/run_1/{sid}/video.pth"
        latent.parent.mkdir(parents=True)
        latent.touch()
        current_video = root / "robotwin_data" / sample["robot_video_path"]
        for camera in VA_CONFIGS["robotwin_subset12_rgb_train"].obs_cam_keys:
            path = current_video.parent.parent / camera / current_video.name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        records.append({"sample": sample, "human_cache_path": human_path, "robot_cache_path": robot_path,
                        "robot_video_path": current_video, "task_name": "task"})
    monkeypatch.setattr(cache_module, "expected_rgb_records", lambda cfg: records)
    decoded_paths = []

    def decode(path, ids):
        decoded_paths.append(path)
        assert path.is_relative_to(root)
        assert ids == [0, 4, 8, 12, 16]
        return [Image.new("RGB", (8, 8)) for _ in ids]

    monkeypatch.setattr(preparation, "decode_rgb_frames", decode)
    args = SimpleNamespace(data_config="robotwin_subset12_rgb_train", human_gen_root=str(root),
                           feature_root=str(tmp_path / "features"), sample_id=None, second_sample_id=None)
    first, _, _, observations, evidence = select_inputs(args)
    assert len(decoded_paths) == 3 and len(observations) == 5
    assert first["robot_metadata"]["source_video_path"].startswith("/old_server/")
    assert all(str(root) in path for path in evidence["camera_videos"].values())


def test_full_rgb_resize_preserves_edges():
    from PIL import Image
    # The real processor config; no model or GPU is needed for this check.
    from transformers import BitImageProcessor
    processor = BitImageProcessor(size={"height": 224, "width": 224}, do_center_crop=False,
                                  do_rescale=False, do_normalize=False)
    image = Image.new("RGB", (480, 320), (0, 0, 0))
    for x in range(20):
        for y in range(320):
            image.putpixel((x, y), (255, 0, 0))
            image.putpixel((479 - x, y), (0, 255, 0))
    pixels = processor(images=image, return_tensors="pt")["pixel_values"]
    assert pixels.shape == (1, 3, 224, 224)
    assert pixels[0, 0, :, 0].min() == 255
    assert pixels[0, 1, :, -1].min() == 255


def test_evaluation_human_only_selection_never_requires_robot_future(tmp_path):
    import json
    from types import SimpleNamespace
    from tools.prepare_rgb_transition_features import human_only_records
    samples = [{"sample_id": 1, "human_video_path": "icl_raw_data/human_data/robotwin/run_1/sample/video.mp4",
                "robot_video_path": "open_microwave-demo_clean_collect_200-1000/videos/missing-robot-video.mp4"}]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"samples": samples}))
    config = SimpleNamespace(icl_manifest_path=manifest, rgb_feature_root=tmp_path / "features", rgb_smoke_sample_ids=None)
    records = human_only_records(config, "open_microwave")
    assert len(records) == 1
    assert "robot_latent_path" not in records[0]
    assert records[0]["human_cache_path"] == tmp_path / "features/human/run_1/sample/video.pth"
    with pytest.raises(ValueError, match="No human-only manifest"):
        human_only_records(config, "absent_task")


def test_eval_mapping_selects_exact_demo_instead_of_first_manifest_sample(tmp_path):
    import json
    from types import SimpleNamespace
    from tools.prepare_rgb_transition_features import human_only_records
    samples = [{"sample_id": i, "human_video_path": f"icl_raw_data/human_data/robotwin/run_1/{i}/video.mp4",
                "robot_video_path": f"open_microwave-demo/videos/missing-{i}.mp4"} for i in (1, 2)]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"samples": samples}))
    mapping = tmp_path / "eval_map.py"
    mapping.write_text("ROBOTWIN_ICL_HUMAN_VIDEOS = {'open_microwave': ['/not/downloaded/run_1/2/video.mp4']}")
    config = SimpleNamespace(icl_manifest_path=manifest, rgb_feature_root=tmp_path / "features", rgb_smoke_sample_ids=None)
    records = human_only_records(config, "", mapping)
    assert len(records) == 1 and records[0]["sample"]["sample_id"] == 2
    assert "robot_cache_path" not in records[0]


def test_archive_http_resume_does_not_repeat_compressed_bytes():
    from types import SimpleNamespace
    import requests
    from tools.prepare_rgb_transition_features import resumable_http_blocks

    class Response:
        status_code = 206
        def __init__(self, retry):
            self.retry = retry
            self.headers = {"Content-Length": "6", "Content-Range": "bytes 3-5/6"}
        def iter_content(self, size):
            yield b"def" if self.retry else b"abc"
            if not self.retry:
                raise requests.ConnectionError("interrupted archive")
        def close(self):
            pass
        def raise_for_status(self):
            pass

    def resume(url, **kwargs):
        assert kwargs["headers"] == {"Range": "bytes=3-"}
        return Response(True)
    assert b"".join(resumable_http_blocks(SimpleNamespace(get=resume), "official-url", Response(False))) == b"abcdef"
