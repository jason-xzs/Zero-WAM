"""CPU-only tests of explicit role annotation semantics, without SAM2 weights."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from tools import prepare_role_masks_sam2 as preparation


HUMAN_SUFFIX = (
    "run_robotwin_20260728_013720_robotwin_n1000/samples/"
    "004_robotwin_Grab_the_mouse_and_place_it_on_a_colored_mat/"
    "generated_video_kling-v3.mp4"
)
ROBOT_SUFFIX = (
    "place_mouse_pad-demo_clean_collect_200-1000/videos/chunk-000/"
    "observation.images.cam_high/episode_000059.mp4"
)


def _seeds():
    return {
        "objects": {
            "0": {"box_xyxy": [1, 1, 5, 5]},
            "1": {"box_xyxy": [6, 1, 10, 5]},
            "2": {"box_xyxy": [11, 1, 15, 5]},
        },
        # Deliberately not chronological: annotation order has no semantics.
        "role_events": [
            {"start_frame": 3, "end_frame": 5, "source": 2, "reference": 1},
            {"start_frame": 0, "end_frame": 1, "source": 0, "reference": 1},
        ],
        "invalid_intervals": {"source": [[5, 5]]},
    }


class FakePredictor:
    """Expose the official predictor interface with deterministic visible masks."""

    def __init__(self):
        self.masks = {}
        self.reset = False

    def init_state(self, path, **kwargs):
        return {"obj_id_to_idx": {}, "output_dict_per_obj": {}}

    def add_new_points_or_box(self, state, frame_idx, obj_id, box):
        assert frame_idx == 0
        x1, y1, x2, y2 = map(int, box)
        mask = torch.zeros(16, 16, dtype=torch.bool)
        mask[y1:y2, x1:x2] = True
        self.masks[obj_id] = mask
        index = len(state["obj_id_to_idx"])
        state["obj_id_to_idx"][obj_id] = index
        state["output_dict_per_obj"][index] = {
            "cond_frame_outputs": {}, "non_cond_frame_outputs": {},
        }

    def propagate_in_video(self, state):
        for frame in range(6):
            for obj_id, index in state["obj_id_to_idx"].items():
                # The second moving object becomes unobservable at frame 4.
                score = -5.0 if frame == 4 and obj_id == 3 else 5.0
                state["output_dict_per_obj"][index]["non_cond_frame_outputs"][frame] = {
                    "object_score_logits": torch.tensor(score),
                }
            # SAM output position must never replace persistent object identity.
            order = list(reversed(self.masks))
            logits = torch.stack([
                torch.where(self.masks[obj_id], 5.0, -5.0) for obj_id in order
            ])[:, None]
            yield frame, order, logits

    def reset_state(self, state):
        self.reset = True


@pytest.fixture(autouse=True)
def _no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def forbidden(*args, **kwargs):
        raise AssertionError("These mask preparation tests must not access CUDA")

    monkeypatch.setattr(torch.cuda, "empty_cache", forbidden)


@pytest.mark.parametrize("kind", ["human", "robot"])
def test_multistage_preparation_preserves_cache_timeline_and_visibility(tmp_path, monkeypatch, kind):
    # Human segments can repeat frames; robot samples have a single time axis.
    indices = (torch.tensor([[0, 1, 2, 3], [2, 3, 4, 5]]) if kind == "human"
               else torch.arange(6))
    suffix = HUMAN_SUFFIX if kind == "human" else ROBOT_SUFFIX
    original, relocated = "/old/data/" + suffix, "/new/data/" + suffix
    cache = tmp_path / "rgb.pth"
    torch.save({
        "features": torch.zeros(*indices.shape, 256, 768, dtype=torch.float16),
        "frame_indices": indices,
        "metadata": {"source_video_path": original, "preprocessing": {"config": {
            "size": {"height": 224, "width": 224}, "do_center_crop": False,
        }}},
    }, cache)

    def fake_extract(video, destination, sampled_ids):
        assert video == Path(relocated)
        assert sampled_ids == set(range(6))
        return ({frame: Image.new("RGB", (16, 16), "white") for frame in sampled_ids},
                {"frames": 6, "width": 16, "height": 16, "fps": 24})

    monkeypatch.setattr(preparation, "extract_video", fake_extract)
    predictor = FakePredictor()
    coverage, valid, returned_indices, identities, report = preparation.process_video(
        predictor,
        {"label": "synthetic", f"{kind}_cache_path": str(cache),
         f"{kind}_video_path": relocated},
        kind, _seeds(), tmp_path / "output",
        SimpleNamespace(min_object_probability=.8, min_mask_pixels=1,
                        min_area_ratio=.25, max_area_ratio=2., max_overlap_fraction=.2),
    )

    expected_ids = torch.tensor([[0, 1], [0, 1], [-1, -1], [2, 1], [2, 1], [2, 1]])
    expected_valid = torch.tensor([
        [True, True], [True, True], [False, False],
        [True, True], [False, True], [False, True],
    ])
    assert torch.equal(returned_indices, indices)
    assert torch.equal(identities, expected_ids[indices])
    assert torch.equal(valid, expected_valid[indices])
    assert coverage.shape == (*indices.shape, 2, 256)
    assert not coverage[~valid].any()
    assert set(predictor.masks) == {1, 2, 3} and predictor.reset

    # A 16x16 input pixel maps exactly to a single DINO patch after resizing.
    expected_coverage = torch.zeros(6, 2, 16, 16)
    expected_coverage[0:2, 0, 1:5, 1:5] = 1
    expected_coverage[3, 0, 1:5, 11:15] = 1
    expected_coverage[[0, 1, 3, 4, 5], 1, 1:5, 6:10] = 1
    assert torch.equal(coverage, expected_coverage.flatten(-2)[indices])

    stats = report["all_frame_stats"]
    assert stats["2"][0]["invalid_reasons"] == ["uncovered_role_event"]
    assert "low_object_presence_score" in stats["4"][0]["invalid_reasons"]
    assert "manual_occlusion_or_tracking_failure" in stats["5"][0]["invalid_reasons"]
    # Invalid supervision still leaves the candidate visible for later review.
    assert stats["4"][0]["visible_pixels"] == stats["5"][0]["visible_pixels"] == 16
    assert all(len(objects) == 3 for objects in report["all_object_frame_stats"].values())
    assert report["cache_source_video_path"] == original
    assert report["video"] == relocated
    assert all(Path(path).is_file() for path in report["overlay_contact_sheets"])


def test_legacy_two_role_seeds_keep_persistent_ids_at_both_video_ends():
    seeds = {"boxes_xyxy": {"source": [1, 1, 5, 5], "reference": [6, 1, 10, 5]}}
    plan = preparation.normalize_role_seeds(seeds, 6, 16, 16)
    masks = torch.zeros(2, 16, 16, dtype=torch.bool)
    masks[0, 1:5, 1:5] = True
    masks[1, 1:5, 6:10] = True
    for frame in (0, 5):
        selected, valid, identities, _ = preparation.select_roles_at_frame(
            frame, masks, [True, True], plan)
        assert torch.equal(selected, masks) and valid.all()
        assert identities.tolist() == [0, 1]


def test_uncovered_video_does_not_infer_roles_from_object_order():
    seeds = _seeds()
    seeds["role_events"] = []
    plan = preparation.normalize_role_seeds(seeds, 6, 16, 16)
    masks, valid, identities, _ = preparation.select_roles_at_frame(
        3, torch.ones(3, 16, 16), [True, True, True], plan)
    assert not masks.any() and not valid.any()
    assert identities.tolist() == [-1, -1]


def test_named_entities_are_stable_when_seed_dictionary_order_changes():
    seeds = _seeds()
    seeds["objects"] = {
        "bowl_c": seeds["objects"]["2"],
        "0": seeds["objects"]["0"],
        "bowl_a": seeds["objects"]["1"],
    }
    seeds["role_events"] = [
        {"start_frame": 0, "end_frame": 5, "source": "bowl_c", "reference": "bowl_a"},
    ]
    mappings = []
    for objects in (seeds["objects"], dict(reversed(list(seeds["objects"].items())))):
        plan = preparation.normalize_role_seeds(dict(seeds, objects=objects), 6, 16, 16)
        mappings.append({obj["external_id"]: obj["entity_id"] for obj in plan["objects"]})
    assert mappings == [{"0": 0, "bowl_a": 1, "bowl_c": 2}] * 2


@pytest.mark.parametrize("mutation", [
    pytest.param(lambda s: s["role_events"][0].update(start_frame=1), id="inclusive-overlap"),
    pytest.param(lambda s: s["role_events"][0].update(source=9), id="unknown-entity"),
    pytest.param(lambda s: s["role_events"][0].update(source=1), id="same-entity-in-both-roles"),
    pytest.param(lambda s: s["role_events"][0].update(end_frame=6), id="past-video-end"),
    pytest.param(lambda s: s["role_events"][1].update(start_frame=-1), id="before-video-start"),
    pytest.param(lambda s: s["role_events"][0].update(source=-1), id="reserved-unknown-entity"),
    pytest.param(lambda s: s["objects"]["0"].update(box_xyxy=[0, 0, 17, 5]), id="box-outside-frame"),
])
def test_ambiguous_or_out_of_video_annotations_are_rejected(mutation):
    seeds = deepcopy(_seeds())
    mutation(seeds)
    with pytest.raises(ValueError):
        preparation.normalize_role_seeds(seeds, 6, 16, 16)


@pytest.mark.parametrize("kind,suffix", [("human", HUMAN_SUFFIX), ("robot", ROBOT_SUFFIX)])
def test_source_root_relocation_preserves_complete_video_identity(kind, suffix):
    metadata = {"source_video_path": "/original/" + suffix}
    assert preparation.resolve_source_video({}, kind, metadata) == Path(metadata["source_video_path"])
    assert preparation.resolve_source_video(
        {f"{kind}_video_path": "/relocated/" + suffix}, kind, metadata
    ) == Path("/relocated/" + suffix)


@pytest.mark.parametrize("kind,original,changed", [
    ("human", HUMAN_SUFFIX, HUMAN_SUFFIX.replace("run_robotwin_", "run_other_", 1)),
    ("human", HUMAN_SUFFIX, HUMAN_SUFFIX.replace("004_robotwin_", "005_robotwin_", 1)),
    ("human", HUMAN_SUFFIX, HUMAN_SUFFIX.replace("kling-v3.mp4", "another.mp4")),
    ("robot", ROBOT_SUFFIX, ROBOT_SUFFIX.replace("place_mouse_pad-", "place_object_scale-", 1)),
    ("robot", ROBOT_SUFFIX, ROBOT_SUFFIX.replace("chunk-000", "chunk-001")),
    ("robot", ROBOT_SUFFIX, ROBOT_SUFFIX.replace("cam_high", "cam_left")),
    ("robot", ROBOT_SUFFIX, ROBOT_SUFFIX.replace("000059.mp4", "000060.mp4")),
    ("human", HUMAN_SUFFIX, "generated_video_kling-v3.mp4"),
    ("robot", ROBOT_SUFFIX, "episode_000059.mp4"),
    ("robot", "unknown.mp4", "unknown.mp4"),
    ("robot", ROBOT_SUFFIX, "../" + ROBOT_SUFFIX),
])
def test_source_identity_cannot_be_replaced_by_matching_basename(kind, original, changed):
    with pytest.raises(ValueError):
        preparation.resolve_source_video(
            {f"{kind}_video_path": "/new/" + changed}, kind,
            {"source_video_path": "/old/" + original},
        )
