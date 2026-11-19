"""Exercise annotation handoff contracts using explicitly synthetic labels."""
from copy import deepcopy
from pathlib import Path
import subprocess
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from tools import manage_role_annotations as workflow


def _workspace(tmp_path, count=2):
    records = []
    for i in range(count):
        record = {"pair_key": f"{i}:{i:016x}", "label": f"old_label_{i}",
                  "task": "place_mouse_pad", "videos": {}}
        for kind in workflow.KINDS:
            video, cache = tmp_path / f"{kind}{i}.mp4", tmp_path / f"{kind}{i}.pth"
            video.write_bytes(f"synthetic source identity {kind}{i}".encode())
            ids = torch.tensor([[0, 1], [2, 3]]) if kind == "human" else torch.arange(4)
            payload = {"features": torch.ones(*ids.shape, 4, 8), "frame_indices": ids,
                       "metadata": {"schema_version": 1, "kind": kind, "encoder_id": "test-encoder",
                                    "encoder_revision": "fixed", "source_video_path": str(video), "sample_id": str(i),
                                    "preprocessing": {"resize": 28}}}
            torch.save(payload, cache)
            record.update({f"{kind}_cache_path": str(cache), f"{kind}_video_path": str(video),
                           f"{kind}_frame_indices": ids.tolist()})
            record["videos"][kind] = {"width": 16, "height": 16, "container_frame_count": 8}
        records.append(record)
    source = tmp_path / "source.json"
    workflow.write_json(source, {"schema_version": 1, "split": "train", "records": records})
    args = SimpleNamespace(records_json=source, work_dir=tmp_path / "work", pair_key=None,
                           task=None, num_shards=1, shard_index=0, limit=10, device="cuda:0",
                           sam_python=Path(__file__), checkpoint=Path(__file__), sam2_config="test",
                           output_dir=tmp_path / "final")
    workflow.init_work(args)
    return args


def _fill_seeds(args):
    for job, record in workflow.jobs(args):
        seeds = workflow.read_json(job / "seeds.json")
        for seed in seeds["videos"].values():
            seed.update(boxes_xyxy={"source": [0, 0, 4, 4], "reference": [5, 0, 9, 4]},
                        role_identity={"source": "synthetic mouse", "reference": "synthetic mat"},
                        role_identity_approved=True, identity_review="unit-test fixture, not a visual review")
        workflow.write_json(job / "seeds.json", seeds, replace=True)


def _mock_sam(command, **kwargs):
    def flag(name):
        return Path(command[command.index(name) + 1])

    record, = workflow.read_json(flag("--records-json"))["records"]
    output = flag("--output-dir")
    output.mkdir()
    hashes = {f"{kind}_cache_sha256": workflow.file_hash(record[f"{kind}_cache_path"]) for kind in workflow.KINDS}
    annotation = {"metadata": {"schema_version": 1, "pair_key": record["pair_key"], **hashes,
                               "seed_sha256": workflow.file_hash(flag("--seed-json"))}}
    for kind in workflow.KINDS:
        ids = torch.tensor(record[f"{kind}_frame_indices"])
        annotation.update({f"{kind}_frame_indices": ids,
                           f"{kind}_masks": torch.full((*ids.shape, 2, 4), .25),
                           f"{kind}_valid": torch.ones(*ids.shape, 2, dtype=torch.bool),
                           f"{kind}_entity_ids": torch.arange(2).expand(*ids.shape, 2).clone()})
        video_dir = output / f"{record['label']}_{kind}"
        video_dir.mkdir()
        sheet = video_dir / "overlays_01.png"
        Image.new("RGB", (16, 16)).save(sheet)
        workflow.write_json(video_dir / "mask_report.json", {
            "cache_frame_indices": ids.tolist(), "overlay_contact_sheets": [str(sheet)]})
    label = output / (record["label"] + ".pth")
    torch.save(annotation, label)
    candidate = {"pair_key": record["pair_key"], "task_name": record["task"], "split": "train",
                 "annotation_file": label.name, "annotation_sha256": workflow.file_hash(label),
                 "reviewed": False, "human_cache": record["human_cache_path"],
                 "robot_cache": record["robot_cache_path"], **hashes,
                 "provenance": {"kind": "synthetic unit-test fixture, not real labels"}}
    workflow.write_json(output / "candidate_manifest.json", {"schema_version": 1, "split": "train", "records": [candidate]})


def _mock_runner(monkeypatch):
    monkeypatch.setattr(workflow, "teacher_identity", lambda args: {"version": "synthetic"})
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return _mock_sam(command, **kwargs)

    monkeypatch.setattr(workflow.subprocess, "run", run)
    return calls


def _review_all(args, donors=False):
    selected = workflow.jobs(args)
    for i, (job, record) in enumerate(selected):
        _, _, candidate = workflow.completed_candidate(job, record)
        path = workflow.review_file(job, candidate)
        review = workflow.read_json(path)
        review.update(decision="accept", reviewer="synthetic unit test",
                      notes="No real visual review: tests on generated constant tensors only.",
                      stable_roles=True, single_effect=True,
                      reviewed_frames=deepcopy(review["expected_sampled_frames"]))
        if donors:
            review.update(donor_pair_key=selected[1-i][1]["pair_key"],
                          donor_review={"approved": True, "reviewer": "unit test", "reason": "Synthetic equivalence fixture"},
                          effect_signature={"relation": "on", "direction": "establish", "source_type": "mouse",
                                            "reference_type": "mat", "affordance": "support"})
        workflow.write_json(path, review, replace=True)


def test_init_refuses_overwrite_and_keeps_all_seeds_unapproved(tmp_path):
    args = _workspace(tmp_path)
    for job, record in workflow.jobs(args):
        assert record["label"] == record["pair_key"].replace(":", "_")
        with pytest.raises(ValueError, match="Inspect"):
            workflow.validate_seeds(job, record)
    with pytest.raises(FileExistsError):
        workflow.init_work(args)


def test_seed_check_rejects_intra_clip_identity_switch_before_sam(tmp_path):
    args = _workspace(tmp_path)
    _fill_seeds(args)
    job, record = workflow.jobs(args)[0]
    seeds = workflow.read_json(job / "seeds.json")
    seed = seeds["videos"][f"{record['label']}_human"]
    seed.pop("boxes_xyxy")
    seed["objects"] = {str(i): {"box_xyxy": [i*5, 0, i*5+4, 4], "description": f"object {i}"} for i in range(3)}
    seed["role_events"] = [{"start_frame": 0, "end_frame": 0, "source": "0", "reference": "1"},
                           {"start_frame": 1, "end_frame": 3, "source": "2", "reference": "1"}]
    with pytest.raises(ValueError, match="within a human clip"):
        workflow.validate_seed(record, "human", seed)
    seed["role_events"][0]["end_frame"] = 1
    seed["role_events"][1]["start_frame"] = 2
    workflow.validate_seed(record, "human", seed)


def test_human_seed_sharing_requires_exact_source_and_preserves_robot(tmp_path):
    args = _workspace(tmp_path, count=3)
    selected = workflow.jobs(args)
    source_job, source_record = selected[0]
    target_job, target_record = selected[1]
    for key in ("human_video_path", "human_cache_path", "human_frame_indices"):
        target_record[key] = deepcopy(source_record[key])
    workflow.write_json(target_job / "records.json", {"records": [target_record]}, replace=True)
    args.pair_key = [source_record["pair_key"]]
    _fill_seeds(args)
    args.pair_key = None
    args.from_pair = source_record["pair_key"]
    workflow.share_human(args)
    copied = workflow.read_json(target_job / "seeds.json")["videos"]
    assert copied[f"{target_record['label']}_human"]["role_identity_approved"] is True
    assert copied[f"{target_record['label']}_robot"]["role_identity_approved"] is False
    unrelated_job, unrelated_record = selected[2]
    assert not workflow.read_json(unrelated_job / "seeds.json")["videos"][f"{unrelated_record['label']}_human"]["role_identity_approved"]
    with pytest.raises(ValueError, match="already edited"):
        workflow.share_human(args)


def test_resume_skips_only_matching_complete_candidates(tmp_path, monkeypatch):
    args = _workspace(tmp_path)
    _fill_seeds(args)
    calls = _mock_runner(monkeypatch)
    workflow.run_jobs(args)
    assert len(calls) == 2
    workflow.run_jobs(args)
    assert len(calls) == 2
    job, record = workflow.jobs(args)[0]
    seeds = workflow.read_json(job / "seeds.json")
    seeds["videos"][f"{record['label']}_human"]["identity_review"] += "; revised"
    workflow.write_json(job / "seeds.json", seeds, replace=True)
    with pytest.raises(ValueError, match="inputs changed"):
        workflow.completed_candidate(job, record)
    workflow.run_jobs(args)
    assert len(calls) == 3
    assert len(list((job / "attempts").iterdir())) == 2


def test_sam_venv_python_symlink_is_not_dereferenced(tmp_path, monkeypatch):
    args = _workspace(tmp_path, count=1)
    _fill_seeds(args)
    calls = _mock_runner(monkeypatch)
    interpreter = tmp_path / "sam-venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(Path(__file__).resolve())
    args.sam_python = interpreter
    workflow.run_jobs(args)
    assert calls[0][0] == str(interpreter.absolute())
    assert calls[0][0] != str(interpreter.resolve())


def test_batch_failure_keeps_other_completed_pairs_and_can_retry(tmp_path, monkeypatch):
    args = _workspace(tmp_path)
    _fill_seeds(args)
    _mock_runner(monkeypatch)
    failed = False

    def run(command, **kwargs):
        nonlocal failed
        if not failed:
            failed = True
            raise subprocess.CalledProcessError(1, command)
        return _mock_sam(command, **kwargs)

    monkeypatch.setattr(workflow.subprocess, "run", run)
    with pytest.raises(SystemExit):
        workflow.run_jobs(args)
    selected = workflow.jobs(args)
    assert not (selected[0][0] / "completed.json").exists()
    assert (selected[1][0] / "completed.json").exists()
    workflow.run_jobs(args)
    assert all((job / "completed.json").exists() for job, _ in selected)


def test_unreviewed_candidate_cannot_be_finalized(tmp_path, monkeypatch):
    args = _workspace(tmp_path)
    _fill_seeds(args)
    _mock_runner(monkeypatch)
    workflow.run_jobs(args)
    workflow.make_reviews(args)
    with pytest.raises(ValueError, match="visually accepted"):
        workflow.finalize(args)
    assert not args.output_dir.exists()


@pytest.mark.parametrize("defect,match", [
    ("partial_frames", "every sampled"), ("stale_candidate", "stale"),
    ("wrong_evidence", "does not match"), ("missing_donor_review", "Donor requires"),
])
def test_review_must_match_candidate_and_full_visual_scope(tmp_path, monkeypatch, defect, match):
    args = _workspace(tmp_path)
    _fill_seeds(args)
    _mock_runner(monkeypatch)
    workflow.run_jobs(args)
    workflow.make_reviews(args)
    _review_all(args)
    job, record = workflow.jobs(args)[0]
    complete, _, candidate = workflow.completed_candidate(job, record)
    review = workflow.read_json(workflow.review_file(job, candidate))
    if defect == "partial_frames":
        review["reviewed_frames"]["human"].pop()
    elif defect == "stale_candidate":
        review["candidate_annotation_sha256"] = "wrong"
    elif defect == "wrong_evidence":
        review["evidence"][0]["path"] = str(tmp_path / "different.png")
    else:
        review["donor_pair_key"] = "1:0000000000000001"
    with pytest.raises(ValueError, match=match):
        workflow.validate_review(review, complete, candidate, record, job)


def test_finalization_preserves_donor_contract_and_zeroes_excluded_masks(tmp_path, monkeypatch):
    args = _workspace(tmp_path)
    _fill_seeds(args)
    _mock_runner(monkeypatch)
    workflow.run_jobs(args)
    workflow.make_reviews(args)
    _review_all(args, donors=True)
    job, record = workflow.jobs(args)[0]
    _, _, candidate = workflow.completed_candidate(job, record)
    path = workflow.review_file(job, candidate)
    review = workflow.read_json(path)
    review["invalidate_intervals"]["human"]["source"] = [[1, 1]]
    workflow.write_json(path, review, replace=True)
    workflow.finalize(args)
    manifest = workflow.read_json(args.output_dir / "reviewed_manifest.json")
    saved = torch.load(args.output_dir / manifest["records"][0]["annotation_file"], weights_only=True)
    assert not saved["human_valid"][0, 1, 0]
    assert saved["human_masks"][0, 1, 0].sum() == 0
    assert saved["human_entity_ids"][0, 1, 0] == 0
    report = workflow.read_json(args.output_dir / "selection_validation.json")
    assert report["pairs"] == report["donor_pairs"] == 2
    assert not report["formal_training_preflight_passed"]
    assert not (args.output_dir / "INCOMPLETE.json").exists()


def test_bad_final_labels_never_publish_a_reviewed_manifest(tmp_path, monkeypatch):
    args = _workspace(tmp_path)
    _fill_seeds(args)
    _mock_runner(monkeypatch)
    workflow.run_jobs(args)
    workflow.make_reviews(args)
    _review_all(args)
    job, record = workflow.jobs(args)[0]
    _, _, candidate = workflow.completed_candidate(job, record)
    path = workflow.review_file(job, candidate)
    review = workflow.read_json(path)
    review["invalidate_intervals"]["robot"]["source"] = [[0, 3]]
    workflow.write_json(path, review, replace=True)
    with pytest.raises(ValueError, match="Insufficient"):
        workflow.finalize(args)
    assert not (args.output_dir / "reviewed_manifest.json").exists()
    assert (args.output_dir / "INCOMPLETE.json").exists()


def test_validation_report_write_failure_cannot_publish_manifest(tmp_path, monkeypatch):
    args = _workspace(tmp_path, count=1)
    _fill_seeds(args)
    _mock_runner(monkeypatch)
    workflow.run_jobs(args)
    workflow.make_reviews(args)
    _review_all(args)
    original_write = workflow.write_json

    def write(path, value, **kwargs):
        if Path(path).name == "selection_validation.json":
            raise OSError("simulated report write failure")
        original_write(path, value, **kwargs)

    monkeypatch.setattr(workflow, "write_json", write)
    with pytest.raises(OSError, match="report write failure"):
        workflow.finalize(args)
    assert not (args.output_dir / "reviewed_manifest.json").exists()
    assert (args.output_dir / "INCOMPLETE.json").exists()


def test_corrupt_label_and_changed_source_invalidate_completion(tmp_path, monkeypatch):
    args = _workspace(tmp_path)
    _fill_seeds(args)
    _mock_runner(monkeypatch)
    workflow.run_jobs(args)
    job, record = workflow.jobs(args)[0]
    _, directory, candidate = workflow.completed_candidate(job, record)
    (directory / candidate["annotation_file"]).write_bytes(b"partial file")
    with pytest.raises(ValueError, match="digest mismatch"):
        workflow.completed_candidate(job, record)
    job, record = workflow.jobs(args)[1]
    Path(record["human_video_path"]).write_bytes(b"different video")
    with pytest.raises(ValueError, match="inputs changed"):
        workflow.completed_candidate(job, record)


def test_shards_are_disjoint_and_cover_all_pairs(tmp_path):
    args = _workspace(tmp_path, count=3)
    args.num_shards = 2
    first = {record["pair_key"] for _, record in workflow.jobs(args)}
    args.shard_index = 1
    second = {record["pair_key"] for _, record in workflow.jobs(args)}
    assert not first & second
    assert len(first | second) == 3
