#!/usr/bin/env python3
"""Prepare, resume and review role annotations without starting policy training.

See README.role-annotations.md. Seeds and SAM2 candidates are never approved
automatically. Only explicit, frame-complete visual review can produce a
reviewed manifest; `check` runs the actual full-training preflight on CPU.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
KINDS = ("human", "robot")
ROLES = ("source", "reference")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if replace:
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        temporary.write_text(data, encoding="utf-8")
        os.replace(temporary, path)
    else:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(data)


@lru_cache(maxsize=4096)
def _file_hash(path, size, mtime):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def file_hash(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return _file_hash(str(path), stat.st_size, stat.st_mtime_ns)


def value_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def flattened(value):
    if isinstance(value, list):
        return [item for part in value for item in flattened(part)]
    return [value]


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def human_group(record):
    # The exact same source, cache and timeline, never just a task name.
    return value_hash({key: record[key] for key in
                       ("human_video_path", "human_cache_path", "human_frame_indices")})


def init_work(args):
    from tools.prepare_role_records import pair_label, seed_template

    document = read_json(args.records_json)
    records = document["records"]
    if document.get("schema_version") != 1 or document.get("split") != "train" or not records:
        raise ValueError("Expected nonempty schema-1 training records from prepare_role_records.py")
    keys = [record["pair_key"] for record in records]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate pair keys")
    records = deepcopy(records)
    for record in records:
        record["label"] = pair_label(record["pair_key"])
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    groups = defaultdict(list)
    for record in records:
        job = work / "jobs" / record["label"]
        write_json(job / "records.json", {"schema_version": 1, "split": "train", "records": [record]})
        seeds = seed_template([record])
        for kind in KINDS:
            seeds["videos"][f"{record['label']}_{kind}"]["identity_review"] = None
        write_json(job / "seeds.json", seeds)
        groups[human_group(record)].append(record["pair_key"])
    index = {"schema_version": 1, "split": "train", "source_records": str(args.records_json.resolve()),
             "source_records_sha256": file_hash(args.records_json), "pairs": len(records),
             "tasks": dict(Counter(record["task"] for record in records)),
             "jobs": [{key: record[key] for key in ("pair_key", "label", "task")} for record in records],
             "shared_human_groups": list(groups.values())}
    write_json(work / "index.json", index)
    print(json.dumps({"work_dir": str(work), "pairs": len(records), "tasks": index["tasks"],
                      "unique_human_sources": len(groups), "approved_seeds": 0}))


def jobs(args):
    work = args.work_dir.resolve(strict=True)
    index = read_json(work / "index.json")
    requested = set(getattr(args, "pair_key", None) or [])
    known = {entry["pair_key"] for entry in index["jobs"]}
    if requested - known:
        raise ValueError(f"Pair keys absent from this work directory: {sorted(requested-known)}")
    shards, shard = getattr(args, "num_shards", 1), getattr(args, "shard_index", 0)
    if not 0 <= shard < shards:
        raise ValueError("Require 0 <= shard-index < num-shards")
    selected = []
    for i, entry in enumerate(index["jobs"]):
        if (requested and entry["pair_key"] not in requested) or i % shards != shard:
            continue
        if getattr(args, "task", None) and entry["task"] != args.task:
            continue
        job = work / "jobs" / entry["label"]
        record, = read_json(job / "records.json")["records"]
        if any(record[key] != entry[key] for key in ("pair_key", "label", "task")):
            raise ValueError(f"Job identity differs from index: {job}")
        selected.append((job, record))
    if not selected:
        raise ValueError("No jobs matched the selection")
    return selected


def validate_seed(record, kind, seed):
    from tools.prepare_role_masks_sam2 import normalize_role_seeds

    if seed.get("pair_key") != record["pair_key"] or seed.get("video_path") != record[f"{kind}_video_path"]:
        raise ValueError("Seed belongs to a different pair/video")
    if seed.get("role_identity_approved") is not True or not nonempty(seed.get("identity_review")):
        raise ValueError("Inspect video/PNG, fill role identities and identity_review before approving seeds")
    if "objects" in seed:
        if not all(nonempty(obj.get("description")) for obj in seed["objects"].values()):
            raise ValueError("Each persistent object requires a visual description")
    elif not all(nonempty(seed.get("role_identity", {}).get(role)) for role in ROLES):
        raise ValueError("Both roles require explicit visual identities")
    info = record["videos"][kind]
    frame_count = info.get("container_frame_count")
    if not frame_count:
        import av
        with av.open(record[f"{kind}_video_path"]) as container:
            frame_count = sum(1 for _ in container.decode(video=0))
    plan = normalize_role_seeds(seed, int(frame_count), info["width"], info["height"])
    if kind == "human":
        for clip in record["human_frame_indices"]:
            for role in ROLES:
                identities = {event[role] for frame in clip for event in plan["role_events"]
                              if event["start_frame"] <= frame <= event["end_frame"]}
                if len(identities) > 1:
                    raise ValueError("Role identity changes within a human clip; use one entity per role per clip")
    return plan


def validate_seeds(job, record):
    seeds = read_json(job / "seeds.json")
    if seeds.get("schema_version") != 1:
        raise ValueError("Expected schema_version=1 seeds")
    for kind in KINDS:
        validate_seed(record, kind, seeds["videos"][f"{record['label']}_{kind}"])
    return seeds


def preview(args):
    import av
    from tools.prepare_role_masks_sam2 import save_contact_sheets

    for job, record in jobs(args):
        for kind in KINDS:
            destination = job / "previews" / kind
            destination.mkdir(parents=True, exist_ok=True)
            ids = set(flattened(record[f"{kind}_frame_indices"])) | {0}
            frames = []
            with av.open(record[f"{kind}_video_path"]) as container:
                for i, frame in enumerate(container.decode(video=0)):
                    if i in ids:
                        path = destination / f"frame_{i:06d}.png"
                        frame.to_image().convert("RGB").save(path)
                        frames.append((i, path))
                    if i >= max(ids):
                        break
            if {i for i, _ in frames} != ids:
                raise ValueError(f"Missing sampled frames: {record['pair_key']}/{kind}")
            sheets = save_contact_sheets(frames, f"{record['label']}_{kind}", destination)
            print(json.dumps({"pair_key": record["pair_key"], "kind": kind,
                              "frame_0": str(destination / "frame_000000.png"), "sheets": sheets}), flush=True)


def share_human(args):
    selected = jobs(args)
    source = next(((j, r) for j, r in selected if r["pair_key"] == args.from_pair), None)
    if source is None:
        raise ValueError("--from-pair must belong to this work directory")
    source_job, source_record = source
    source_seed = read_json(source_job / "seeds.json")["videos"][f"{source_record['label']}_human"]
    validate_seed(source_record, "human", source_seed)
    updates = []
    for job, record in selected:
        if record["pair_key"] == args.from_pair or human_group(record) != human_group(source_record):
            continue
        document = read_json(job / "seeds.json")
        key = f"{record['label']}_human"
        target = document["videos"][key]
        # No overwrite of someone else's annotation, even if it is unapproved.
        if (target.get("objects") or any(target.get("boxes_xyxy", {}).values())
                or any(target.get("role_identity", {}).values())
                or target.get("role_identity_approved") or target.get("identity_review")):
            raise ValueError(f"Human seed already edited; refusing to overwrite {job / 'seeds.json'}")
        copied = deepcopy(source_seed)
        copied["pair_key"] = record["pair_key"]
        copied["copied_from_pair"] = args.from_pair
        copied["copied_seed_sha256"] = file_hash(source_job / "seeds.json")
        document["videos"][key] = copied
        updates.append((job / "seeds.json", document))
    for path, document in updates:
        write_json(path, document, replace=True)
    print(json.dumps({"shared_human_seeds": len(updates), "from_pair": args.from_pair,
                      "robot_seeds_changed": 0}))


def input_identity(job, record):
    return {"records": file_hash(job / "records.json"), "seeds": file_hash(job / "seeds.json"),
            "sources": {f"{kind}_{source}": file_hash(record[f"{kind}_{source}_path"])
                        for kind in KINDS for source in ("video", "cache")}}


def teacher_identity(args):
    # Probe the actual SAM environment once per batch, without printing secrets
    # or importing it into the policy environment.
    code = ("import json,subprocess,sam2,torch; from pathlib import Path; "
            "root=Path(sam2.__file__).resolve().parents[1]; "
            "print(json.dumps({'source_commit':subprocess.check_output("
            "['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),"
            "'torch_version':str(torch.__version__)}))")
    result = subprocess.run([str(args.sam_python), "-c", code], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"SAM2 environment probe failed for {args.sam_python}:\n{result.stderr}")
    return {**json.loads(result.stdout.strip().splitlines()[-1]),
            "checkpoint_sha256": file_hash(args.checkpoint), "sam2_config": args.sam2_config,
            "sam_python": str(args.sam_python.absolute()),
            "preparation_script_sha256": file_hash(ROOT / "tools/prepare_role_masks_sam2.py"),
            "device": args.device}


def completed_candidate(job, record, *, check_inputs=True):
    complete = read_json(job / "completed.json")
    if check_inputs and complete["inputs"] != input_identity(job, record):
        raise ValueError("Candidate inputs changed; rerun SAM2 and review the new result")
    directory = job / complete["candidate_dir"]
    manifest_path = directory / "candidate_manifest.json"
    if file_hash(manifest_path) != complete["candidate_manifest_sha256"]:
        raise ValueError("Candidate manifest changed after generation")
    candidate, = read_json(manifest_path)["records"]
    if candidate["pair_key"] != record["pair_key"] or candidate["task_name"] != record["task"]:
        raise ValueError("Candidate pair/task identity mismatch")
    if file_hash(directory / candidate["annotation_file"]) != candidate["annotation_sha256"]:
        raise ValueError("Candidate annotation digest mismatch")
    for kind in KINDS:
        if candidate[f"{kind}_cache_sha256"] != complete["inputs"]["sources"][f"{kind}_cache"]:
            raise ValueError("Candidate RGB cache does not match the job")
    for item in complete["visual_evidence"]:
        if file_hash(job / item["path"]) != item["sha256"]:
            raise ValueError("Candidate visual evidence changed after generation")
    return complete, directory, candidate


def visual_evidence(job, record, directory):
    evidence = []
    for kind in KINDS:
        video_dir = directory / f"{record['label']}_{kind}"
        report = read_json(video_dir / "mask_report.json")
        if report["cache_frame_indices"] != record[f"{kind}_frame_indices"]:
            raise ValueError("Overlay report and RGB timeline differ")
        for sheet in report["overlay_contact_sheets"]:
            path = video_dir / Path(sheet).name
            evidence.append({"kind": kind, "path": str(path.relative_to(job)), "sha256": file_hash(path)})
    if {item["kind"] for item in evidence} != set(KINDS):
        raise ValueError("Missing visual evidence for a video")
    return evidence


def run_jobs(args):
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    # A venv's python may symlink to a different environment's binary. Following
    # that link changes sys.prefix and can silently select the wrong packages.
    args.sam_python = args.sam_python.absolute()
    if not args.sam_python.is_file():
        raise FileNotFoundError(args.sam_python)
    args.checkpoint = args.checkpoint.resolve(strict=True)
    teacher = teacher_identity(args)
    counts = Counter()
    for job, record in jobs(args):
        with (job / ".run.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                counts["locked"] += 1
                continue
            try:
                validate_seeds(job, record)
            except (ValueError, KeyError, TypeError) as exc:
                counts["seeds_not_ready"] += 1
                print(json.dumps({"pair_key": record["pair_key"], "seeds_not_ready": str(exc)}), flush=True)
                continue
            inputs = input_identity(job, record)
            if (job / "completed.json").exists():
                try:
                    complete, _, _ = completed_candidate(job, record)
                    if complete["teacher"] == teacher:
                        counts["unchanged_completed"] += 1
                        continue
                except (ValueError, OSError, KeyError):
                    pass  # Keep the old attempt; generate a new one, never patch it.
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
            attempt = job / "attempts" / stamp
            attempt.mkdir(parents=True)
            for name in ("records.json", "seeds.json"):
                shutil.copyfile(job / name, attempt / name)
            candidate_dir = attempt / "candidate"
            command = [str(args.sam_python), str(ROOT / "tools/prepare_role_masks_sam2.py"),
                       "--records-json", str(attempt / "records.json"), "--seed-json", str(attempt / "seeds.json"),
                       "--output-dir", str(candidate_dir), "--checkpoint", str(args.checkpoint),
                       "--sam2-config", args.sam2_config, "--device", args.device]
            print(json.dumps({"starting": record["pair_key"], "log": str(attempt / "sam2.log")}), flush=True)
            counts["attempted"] += 1
            try:
                with (attempt / "sam2.log").open("w") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
                candidate, = read_json(candidate_dir / "candidate_manifest.json")["records"]
                if candidate["pair_key"] != record["pair_key"]:
                    raise ValueError("SAM2 returned a different pair")
                if file_hash(candidate_dir / candidate["annotation_file"]) != candidate["annotation_sha256"]:
                    raise ValueError("Incomplete SAM2 output")
                if inputs != input_identity(job, record):
                    raise ValueError("Job inputs were edited while SAM2 was running; rerun with fixed inputs")
                complete = {"inputs": inputs, "teacher": teacher,
                            "candidate_dir": str(candidate_dir.relative_to(job)),
                            "candidate_manifest_sha256": file_hash(candidate_dir / "candidate_manifest.json"),
                            "visual_evidence": visual_evidence(job, record, candidate_dir)}
                write_json(job / "completed.json", complete, replace=True)
                counts["completed"] += 1
            except (subprocess.CalledProcessError, ValueError, OSError, KeyError) as exc:
                write_json(attempt / "failure.json", {"error": str(exc)})
                counts["failed"] += 1
                print(json.dumps({"failed": record["pair_key"], "error": str(exc)}), flush=True)
            if counts["attempted"] >= args.limit:
                break
    print(json.dumps(dict(counts)), flush=True)
    if counts["failed"]:
        raise SystemExit(1)


def review_file(job, candidate):
    return job / "reviews" / (candidate["annotation_sha256"] + ".json")


def make_reviews(args):
    for job, record in jobs(args):
        if not (job / "completed.json").exists():
            continue
        complete, directory, candidate = completed_candidate(job, record)
        path = review_file(job, candidate)
        if path.exists():
            continue
        evidence = [{**item, "path": str(job / item["path"])} for item in complete["visual_evidence"]]
        review = {"schema_version": 1, "pair_key": record["pair_key"], "decision": None,
                  "candidate_annotation_sha256": candidate["annotation_sha256"],
                  "input_identity_sha256": value_hash(complete["inputs"]), "reviewer": None, "notes": None,
                  "expected_sampled_frames": {kind: sorted(set(flattened(record[f"{kind}_frame_indices"]))) for kind in KINDS},
                  "reviewed_frames": {kind: [] for kind in KINDS}, "evidence": evidence,
                  "invalidate_intervals": {kind: {role: [] for role in ROLES} for kind in KINDS},
                  "stable_roles": None, "single_effect": None, "effect_signature": None,
                  "donor_pair_key": None, "donor_review": None,
                  "instructions": "Inspect every sampled-frame overlay. Evidence is prefilled for navigation, not approval. Set accept only after actual visual review; list reviewed raw frame IDs, reviewer and notes. Exclusions only remove supervision; fixing masks needs new seeds/SAM2. Donors require separate explicit semantic review."}
        write_json(path, review)
        print(str(path), flush=True)


def validate_review(review, complete, candidate, record, job):
    if review.get("schema_version") != 1 or review.get("pair_key") != record["pair_key"]:
        raise ValueError("Review schema/pair mismatch")
    if review.get("decision") != "accept":
        raise ValueError(f"Pair has not been visually accepted: {record['pair_key']}")
    if (review.get("candidate_annotation_sha256") != candidate["annotation_sha256"]
            or review.get("input_identity_sha256") != value_hash(complete["inputs"])):
        raise ValueError("Review is stale relative to the candidate/inputs")
    if not nonempty(review.get("reviewer")) or not nonempty(review.get("notes")):
        raise ValueError("Visual review requires an identified reviewer and concrete notes")
    for flag in ("stable_roles", "single_effect"):
        if type(review.get(flag)) is not bool:
            raise ValueError(f"Review must explicitly decide {flag}")
    for kind in KINDS:
        actual = review.get("reviewed_frames", {}).get(kind, [])
        expected = set(flattened(record[f"{kind}_frame_indices"]))
        if any(type(i) is not int for i in actual) or set(actual) != expected or len(actual) != len(expected):
            raise ValueError(f"Review does not cover every sampled {kind} frame")
    evidence = review.get("evidence", [])
    if not evidence or {item.get("kind") for item in evidence} != set(KINDS):
        raise ValueError("Review must retain PNG evidence for both videos")
    expected_evidence = {(item["kind"], str(job / item["path"]), item["sha256"])
                         for item in complete["visual_evidence"]}
    if {(item["kind"], item["path"], item["sha256"]) for item in evidence} != expected_evidence:
        raise ValueError("Review evidence does not match this candidate's complete contact sheets")
    for item in evidence:
        if Path(item["path"]).suffix.lower() != ".png" or file_hash(item["path"]) != item["sha256"]:
            raise ValueError("Review PNG evidence changed or is missing")
    donor = review.get("donor_pair_key")
    if donor is not None:
        decision = review.get("donor_review")
        if (not isinstance(decision, dict) or decision.get("approved") is not True
                or not nonempty(decision.get("reviewer")) or not nonempty(decision.get("reason"))):
            raise ValueError("Donor requires explicit approval, reviewer and semantic equivalence reason")


def apply_exclusions(annotation, review, record):
    """Conservative edits only; retain IDs so invalidity cannot hide an ID swap."""
    exclusions = review.get("invalidate_intervals", {})
    if set(exclusions) - set(KINDS):
        raise ValueError("Unknown exclusion video kind")
    for kind, roles in exclusions.items():
        if set(roles) - set(ROLES):
            raise ValueError("Unknown exclusion role")
        ids = annotation[f"{kind}_frame_indices"]
        frame_count = record["videos"][kind].get("container_frame_count")
        for role, intervals in roles.items():
            role_index = ROLES.index(role)
            for interval in intervals:
                if (not isinstance(interval, list) or len(interval) != 2
                        or any(type(i) is not int for i in interval)
                        or not 0 <= interval[0] <= interval[1]
                        or (frame_count and interval[1] >= frame_count)):
                    raise ValueError("Invalid exclusion interval; use inclusive raw frame indices")
                selected = (ids >= interval[0]) & (ids <= interval[1])
                if not bool(selected.any()):
                    raise ValueError("Exclusion interval affects no cached frame; check raw frame numbering")
                annotation[f"{kind}_valid"][..., role_index][selected] = False
                annotation[f"{kind}_masks"][..., role_index, :][selected] = 0


def validate_selection(manifest_path, records):
    import torch
    from wan_va.configs import VA_CONFIGS
    from wan_va.dataset.rgb_feature_cache import load_feature_cache
    from wan_va.dataset.role_feature_cache import RoleManifest, usable_future_pairs

    torch.set_num_threads(min(torch.get_num_threads(), 4))
    config = VA_CONFIGS["robotwin_subset12_role_effect"]
    manifest = RoleManifest(manifest_path, config.included_task_names)
    counts = Counter()
    per_task = defaultdict(Counter)
    chunk_sizes = ([config.frame_chunk_size] if config.frame_chunk_size > 0 else
                   list(range(1, config.max_frame_chunk_size + 1)))
    for record in records:
        hp, rp = record["human_cache_path"], record["robot_cache_path"]
        targets, _, donor = manifest.load(record["pair_key"], load_feature_cache(hp, "human"),
                                          load_feature_cache(rp, "robot"), hp, rp, require_complete=True)
        if not any(objects > 0 and relations > 0 for objects, relations in
                   (usable_future_pairs(targets["robot_valid"], targets["robot_entity_ids"], size)
                    for size in chunk_sizes)):
            raise ValueError(f"No causal future targets for {record['pair_key']}")
        counts["pairs"] += 1
        counts["donor_pairs"] += int(donor)
        task = per_task[record["task"]]
        task["pairs"] += 1
        task["donor_pairs"] += int(donor)
        for kind in KINDS:
            task[f"{kind}_valid_roles"] += int(targets[f"{kind}_valid"].sum())
            task[f"{kind}_total_roles"] += targets[f"{kind}_valid"].numel()
    return {**dict(counts), "per_task": dict(per_task),
            "complete_selected_labels": True, "formal_training_preflight_passed": False,
            "next": "Run check against the actual complete training manifest; selected labels are not full-training approval."}


def finalize(args):
    import torch

    selected = jobs(args)
    prepared = []
    # Validate all decisions before creating an output directory.
    for job, record in selected:
        complete, directory, candidate = completed_candidate(job, record)
        path = review_file(job, candidate)
        review = read_json(path)
        validate_review(review, complete, candidate, record, job)
        prepared.append((record, directory, candidate, path, review))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest_records = []
    try:
        for record, directory, candidate, review_path, review in prepared:
            review_dir = output / "reviews" / record["label"]
            review_dir.mkdir(parents=True)
            shutil.copyfile(review_path, review_dir / "decision.json")
            evidence = []
            for i, item in enumerate(review["evidence"]):
                target = review_dir / f"{i:02d}_{item['kind']}_{Path(item['path']).name}"
                shutil.copyfile(item["path"], target)
                evidence.append({**item, "path": str(target.relative_to(output))})
            annotation = torch.load(directory / candidate["annotation_file"], map_location="cpu", weights_only=True)
            apply_exclusions(annotation, review, record)
            provenance = {**candidate["provenance"], "reviewed": True, "not_ground_truth": True,
                          "reviewer": review["reviewer"], "review_notes": review["notes"],
                          "candidate_annotation_sha256": candidate["annotation_sha256"],
                          "review_file": str((review_dir / "decision.json").relative_to(output)),
                          "review_sha256": file_hash(review_path), "evidence": evidence,
                          "invalidate_intervals": review.get("invalidate_intervals", {})}
            annotation["metadata"]["provenance"] = provenance
            label_path = output / "labels" / (record["label"] + ".pth")
            label_path.parent.mkdir(exist_ok=True)
            torch.save(annotation, label_path)
            result = {**candidate, "annotation_file": str(label_path.relative_to(output)),
                      "annotation_sha256": file_hash(label_path), "reviewed": True, "provenance": provenance}
            for key in ("stable_roles", "single_effect", "effect_signature", "donor_pair_key", "donor_review"):
                result[key] = review.get(key)
            manifest_records.append(result)
        pending = output / "manifest.pending.json"
        write_json(pending, {"schema_version": 1, "split": "train", "records": manifest_records})
        report = validate_selection(pending, [record for _, record in selected])
        write_json(output / "selection_validation.json", report)
        os.replace(pending, output / "reviewed_manifest.json")
    except Exception as exc:
        write_json(output / "INCOMPLETE.json", {"error": str(exc), "training_ready": False})
        raise
    print(json.dumps({"manifest": str(output / "reviewed_manifest.json"), **report}), flush=True)


def full_check(args):
    import torch
    from wan_va.configs import VA_CONFIGS
    from wan_va.dataset.rgb_feature_cache import validate_rgb_feature_cache
    from wan_va.dataset.role_feature_cache import validate_role_feature_cache

    torch.set_num_threads(min(torch.get_num_threads(), 4))
    config = deepcopy(VA_CONFIGS["robotwin_subset12_role_effect"])
    root = args.human_gen_root.resolve(strict=True)
    config.dataset_path = str(root / "robotwin_data")
    config.icl_manifest_path = str(root / "icl_configs/ICL_config_robotwin.json")
    config.human_latent_path = str(root / "human_latents/robotwin")
    config.rgb_feature_root = str(args.feature_root.resolve(strict=True))
    config.role_manifest_path = str(args.manifest.resolve(strict=True))
    config.rgb_smoke_sample_ids = None
    config.role_require_complete = True
    # Missing role coverage should fail before scanning all RGB tensors.
    role = validate_role_feature_cache(config)
    rgb = validate_rgb_feature_cache(config)
    if rgb["pairs"] != 600 or rgb["tasks"] != 12:
        raise ValueError(f"Expected the official 600-pair/12-task selection, got {rgb['pairs']}/{rgb['tasks']}")
    report = {"formal_training_preflight_passed": True, "rgb": rgb, "role": role,
              "meaning": "Data contracts passed; not a guarantee of label semantics or policy performance."}
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def status(args):
    counts = Counter()
    for job, record in jobs(args):
        counts["pairs"] += 1
        try:
            validate_seeds(job, record)
            counts["seeds_ready"] += 1
        except (ValueError, KeyError, TypeError):
            pass
        if (job / "completed.json").exists():
            counts["completion_records"] += 1
            complete = read_json(job / "completed.json")
            directory = job / complete["candidate_dir"]
            candidate, = read_json(directory / "candidate_manifest.json")["records"]
            path = review_file(job, candidate)
            if path.exists() and read_json(path).get("decision") == "accept":
                counts["accept_decisions"] += 1
    print(json.dumps({**dict(counts), "note": "Status counts only; finalize/check verify hashes, reviews and training coverage."}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("init", help="Create per-pair jobs with empty seeds")
    p.add_argument("--records-json", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.set_defaults(function=init_work)
    for name, function in (("preview", preview), ("share-human", share_human), ("run", run_jobs),
                           ("review-template", make_reviews), ("finalize", finalize), ("status", status)):
        p = commands.add_parser(name)
        p.add_argument("--work-dir", type=Path, required=True)
        if name not in ("share-human", "finalize"):
            p.add_argument("--pair-key", action="append")
            p.add_argument("--task")
        p.set_defaults(function=function)
        if name == "share-human":
            p.add_argument("--from-pair", required=True)
        if name == "run":
            p.add_argument("--sam-python", type=Path, required=True)
            p.add_argument("--checkpoint", type=Path, required=True)
            p.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
            p.add_argument("--device", default="cuda:0")
            p.add_argument("--limit", type=int, default=10, help="Maximum newly attempted pairs this invocation")
            p.add_argument("--num-shards", type=int, default=1)
            p.add_argument("--shard-index", type=int, default=0)
        if name == "finalize":
            p.add_argument("--output-dir", type=Path, required=True)
    p = commands.add_parser("check", help="Run full 600-pair training data preflight on CPU")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--human-gen-root", type=Path, default=Path(os.environ.get("HUMAN_GEN_ROOT", "/workspace/zero-wam/datasets/HumanGen")))
    p.add_argument("--feature-root", type=Path, default=Path(os.environ.get("ZERO_WAM_RGB_FEATURE_ROOT", "/workspace/zero-wam/datasets/rgb_transition_dinov2_base")))
    p.add_argument("--report", type=Path, required=True)
    p.set_defaults(function=full_check)
    args = parser.parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
