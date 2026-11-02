#!/usr/bin/env python3
"""Prepare frozen RGB features for the selected RoboTwin training pairs.

Example (all 12 training tasks):
  python tools/prepare_rgb_transition_features.py --download-human --download-encoder
Smoke only, one real pair per task:
  python tools/prepare_rgb_transition_features.py --download-human --smoke-one-per-task
The printed sample IDs must be explicitly supplied to the training smoke run.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import shutil
import subprocess
import tarfile
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from wan_va.dataset.rgb_feature_cache import (
    DEFAULT_ENCODER_ID, SCHEMA_VERSION, expected_rgb_records,
    latent_rgb_frame_indices, load_feature_cache, validate_rgb_feature_cache, rgb_pair_key,
)

HUMANGEN_REPO = "Robbyant-Research/HumanGen"
HUMANGEN_REVISION = "971d12e942bf101c0d47716f928b1c044d02ac71"
ENCODER_REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"


def resumable_http_blocks(session, url, response, retries=3):
    """Resume the compressed byte stream without buffering a large archive."""
    import requests
    consumed = 0
    total = int(response.headers.get("Content-Length", 0))
    current = response
    try:
        for attempt in range(retries + 1):
            try:
                for block in current.iter_content(1024 * 1024):
                    if block:
                        yield block
                        consumed += len(block)
                if total and consumed != total:
                    raise requests.ConnectionError(f"Truncated archive HTTP body: {consumed}/{total}")
                return
            except requests.RequestException:
                current.close()
                if attempt == retries:
                    raise
                print(f"resuming official archive HTTP stream at compressed byte {consumed}", flush=True)
                current = session.get(url, headers={"Range": f"bytes={consumed}-"}, stream=True, timeout=(30, 120))
                current.raise_for_status()
                if current.status_code != 206 or not current.headers.get("Content-Range", "").startswith(f"bytes {consumed}-"):
                    raise ValueError("Archive server did not honor the exact Range resume offset")
    finally:
        current.close()


def download_encoder(model_path):
    """Download the three official pinned encoder files into /workspace."""
    import requests
    root = Path(model_path)
    if not root.resolve().is_relative_to(Path("/workspace")):
        raise ValueError("Downloaded encoder weights must live under /workspace")
    root.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    response = session.get(f"https://huggingface.co/api/models/{DEFAULT_ENCODER_ID}/tree/{ENCODER_REVISION}", timeout=30)
    response.raise_for_status()
    entries = {entry["path"]: entry for entry in response.json()}
    for name in ("config.json", "preprocessor_config.json", "model.safetensors"):
        entry = entries[name]
        path = root / name
        size = int(entry["size"])
        sha = (entry.get("lfs") or {}).get("oid")
        if path.is_file() and path.stat().st_size == size:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                    digest.update(block)
            if sha is None or digest.hexdigest() == sha:
                continue
        response = session.get(f"https://huggingface.co/{DEFAULT_ENCODER_ID}/resolve/{ENCODER_REVISION}/{name}",
                               stream=True, timeout=(30, 120))
        response.raise_for_status()
        digest = hashlib.sha256()
        try:
            with path.open("wb") as handle:
                for block in response.iter_content(4 * 1024 * 1024):
                    handle.write(block)
                    digest.update(block)
            if path.stat().st_size != size or (sha is not None and digest.hexdigest() != sha):
                raise ValueError(f"Encoder download size/hash mismatch: {path}")
        except Exception:
            path.unlink(missing_ok=True)
            raise
        print(f"downloaded pinned encoder file {name} bytes={size}", flush=True)
    (root / "encoder_provenance.json").write_text(json.dumps(
        {"encoder_id": DEFAULT_ENCODER_ID, "encoder_revision": ENCODER_REVISION}, indent=2))


def extract_human_archives(root, manifest_paths):
    """Stream official zstd shards, writing only explicitly selected videos.

    HumanGen's pinned release contains archives, not individual downloadable
    MP4s. RoboTwin is at the end of the lexically packed shards. We inspect
    shards in reverse order and stop as soon as every selected file is found.
    """
    import requests
    root = Path(root)
    if not root.resolve().is_relative_to(Path("/workspace")):
        raise ValueError("Downloaded RGB videos must live under /workspace")
    wanted = {str(x).removeprefix("icl_raw_data/") for x in manifest_paths}
    if any(not x.startswith("human_data/robotwin/") or ".." in Path(x).parts for x in wanted):
        raise ValueError("Only explicitly selected RoboTwin human archive members are permitted")
    wanted = {x for x in wanted if not (root / x).is_file()}
    if not wanted:
        return
    zstd = shutil.which("zstd")
    if not zstd:
        raise RuntimeError("Streaming HumanGen archives requires the zstd executable")
    session = requests.Session()
    session.trust_env = False
    base = f"https://huggingface.co/datasets/{HUMANGEN_REPO}/resolve/{HUMANGEN_REVISION}/"
    response = session.get(base + "human_data/manifest.tsv", timeout=30)
    response.raise_for_status()
    shards = [line.split("\t")[0] for line in response.text.splitlines() if line.strip()]
    for shard in reversed(shards):
        if not wanted:
            break
        print(f"streaming human_data/{shard}; retaining only {len(wanted)} requested videos", flush=True)
        archive_url = base + "human_data/" + shard
        response = session.get(archive_url, stream=True, timeout=(30, 120))
        response.raise_for_status()
        began, last_report = time.monotonic(), time.monotonic()
        progress = {"compressed_bytes": 0}
        total_bytes = int(response.headers.get("Content-Length", 0))
        process = subprocess.Popen([zstd, "-d", "-c", "-q"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        errors = []
        def feed():
            try:
                for block in resumable_http_blocks(session, archive_url, response):
                    process.stdin.write(block)
                    progress["compressed_bytes"] += len(block)
            except BrokenPipeError:
                pass
            except Exception as exc:
                errors.append(exc)
            finally:
                try:
                    process.stdin.close()
                except (BrokenPipeError, ValueError):
                    pass
        thread = threading.Thread(target=feed, daemon=True)
        thread.start()
        first_member = True
        try:
            with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
                for member in archive:
                    name = member.name.removeprefix("./")
                    now = time.monotonic()
                    if first_member or now - last_report >= 30:
                        consumed = progress["compressed_bytes"]
                        rate = consumed / max(now - began, .01)
                        print(json.dumps({"shard": shard, "member": name, "compressed_bytes": consumed,
                                          "MB_per_second": round(rate / 1e6, 2),
                                          "remaining_shard_seconds_upper_bound": round(max(total_bytes - consumed, 0) / rate) if rate > 0 else None}), flush=True)
                        first_member, last_report = False, now
                    if name not in wanted:
                        continue
                    if not member.isfile():
                        raise ValueError(f"Expected a regular video archive member: {name}")
                    target = root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = archive.extractfile(member)
                    try:
                        with source, target.open("wb") as handle:
                            shutil.copyfileobj(source, handle, length=1024 * 1024)
                        if target.stat().st_size != member.size:
                            raise ValueError(f"Truncated selected human archive member: {name}")
                    except Exception:
                        target.unlink(missing_ok=True)
                        raise
                    wanted.remove(name)
                    print(f"extracted {name} bytes={member.size}; remaining={len(wanted)}", flush=True)
                    if not wanted:
                        break
            if errors:
                raise RuntimeError(f"Human archive download failed: {errors[0]}")
        except (tarfile.ReadError, EOFError) as exc:
            if errors:
                raise RuntimeError(f"Human archive download failed: {errors[0]}") from exc
            raise
        finally:
            response.close()
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            thread.join(timeout=1)
    if wanted:
        raise FileNotFoundError(f"Selected human videos absent from official shards: {sorted(wanted)}")


def video_frame_count(path):
    import av
    with av.open(str(path)) as container:
        count = int(container.streams.video[0].frames)
        if count:
            return count
        return sum(1 for _ in container.decode(video=0))


def decode_rgb_frames(path, indices):
    import av
    requested = [int(i) for i in indices]
    wanted, found = set(requested), {}
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                found[index] = frame.to_image().convert("RGB")
            if len(found) == len(wanted):
                break
    missing = sorted(wanted - found.keys())
    if missing:
        raise ValueError(f"Video {path} lacks requested source frame indices: {missing[:8]}")
    return [found[i] for i in requested]


class FrozenRGBEncoder:
    def __init__(self, model_path, device="cuda:0", batch_size=16):
        from transformers import AutoImageProcessor, AutoModel
        model_path = Path(model_path)
        provenance = json.loads((model_path / "encoder_provenance.json").read_text())
        self.encoder_id = provenance["encoder_id"]
        self.revision = provenance["encoder_revision"]
        if self.encoder_id != DEFAULT_ENCODER_ID:
            raise ValueError("This preparation recipe is validated for facebook/dinov2-base")
        self.processor = AutoImageProcessor.from_pretrained(str(model_path), local_files_only=True, use_fast=False)
        # Preserve objects at the edge of widescreen demonstrations. Both
        # embodiments use the identical full-frame RGB resize, without crops.
        self.processor.size = {"height": 224, "width": 224}
        self.processor.do_center_crop = False
        self.model = AutoModel.from_pretrained(str(model_path), local_files_only=True).to(device).eval()
        self.model.requires_grad_(False)
        self.device, self.batch_size = device, batch_size
        self.preprocessing = {"processor_class": type(self.processor).__name__,
                              "config": self.processor.to_dict(),
                              "output": "last_hidden_state_without_cls", "rgb": True}

    @torch.inference_mode()
    def encode(self, frames):
        results = []
        for start in range(0, len(frames), self.batch_size):
            pixels = self.processor(images=frames[start:start + self.batch_size], return_tensors="pt")["pixel_values"].to(self.device)
            outputs = self.model(pixel_values=pixels).last_hidden_state[:, 1:]
            if outputs.shape[1:] != (256, 768):
                raise ValueError(f"Unexpected DINOv2 patch features: {tuple(outputs.shape)}")
            results.append(outputs.to(device="cpu", dtype=torch.float16))
        return torch.cat(results)


def save_features(path, features, indices, *, encoder, kind, sample_id, source_video_path, **extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"schema_version": SCHEMA_VERSION, "kind": kind, "encoder_id": encoder.encoder_id,
                "encoder_revision": encoder.revision, "preprocessing": encoder.preprocessing,
                "sample_id": str(sample_id), "source_video_path": str(source_video_path),
                "source_data_revision": HUMANGEN_REVISION, **extra}
    # Explicit close before verification: /workspace may be object-backed FUSE.
    with path.open("wb") as handle:
        torch.save({"features": features.contiguous(), "frame_indices": indices.to(torch.int64), "metadata": metadata}, handle)
    load_feature_cache(path, kind, expected_encoder_id=encoder.encoder_id, expected_sample_id=sample_id,
                       expected_revision=encoder.revision)


def prepare_human_feature(record, root, encoder, args):
    sample = record["sample"]
    path = record["human_cache_path"]
    human_video = root / str(sample["human_video_path"]).removeprefix("icl_raw_data/")
    if not human_video.is_file():
        raise FileNotFoundError(f"Missing human RGB {human_video}; enable --download-human")
    if args.overwrite or not path.is_file():
        n = video_frame_count(human_video)
        ids = torch.linspace(0, n - 1, args.human_segments * args.human_frames_per_segment).round().long()
        human = encoder.encode(decode_rgb_frames(human_video, ids.tolist()))
        shape = (args.human_segments, args.human_frames_per_segment)
        save_features(path, human.reshape(*shape, *human.shape[1:]), ids.reshape(*shape),
                      encoder=encoder, kind="human", sample_id=sample["sample_id"], source_video_path=human_video,
                      sampling="uniform_full_video_segments", source_video_frames=n)
    payload = load_feature_cache(path, "human", expected_encoder_id=encoder.encoder_id,
                                 expected_revision=encoder.revision, expected_sample_id=sample["sample_id"])
    if payload["metadata"]["preprocessing"] != encoder.preprocessing:
        raise ValueError(f"Existing human cache preprocessing differs; regenerate with --overwrite: {path}")


def human_only_records(config, task_names, eval_human_map=None):
    """Evaluation preparation reads demonstration RGB only, never robot futures."""
    from wan_va.dataset.rgb_feature_cache import human_feature_path, human_relative_path
    tasks = {x for x in task_names.split(",") if x}
    selected_paths = None
    if eval_human_map:
        import runpy
        mapping = runpy.run_path(str(eval_human_map))["ROBOTWIN_ICL_HUMAN_VIDEOS"]
        tasks = tasks or set(mapping)
        if tasks - mapping.keys():
            raise ValueError(f"Tasks absent from evaluation human map: {sorted(tasks - mapping.keys())}")
        selected_paths = {str(human_relative_path(path)): task
                          for task in tasks for path in mapping[task]}
    if not tasks:
        raise ValueError("--human-only requires explicit --task-names or --eval-human-map")
    samples = json.loads(Path(config.icl_manifest_path).read_text())["samples"]
    records, found = [], set()
    allowlist = set(config.rgb_smoke_sample_ids or [])
    for sample in samples:
        robot = (sample.get("status_info") or {}).get("video_rel_path") or sample.get("robot_video_path", "")
        task = robot.split("/")[0]
        if task.split("-", 1)[0] not in tasks or (allowlist and rgb_pair_key(sample) not in allowlist):
            continue
        relative = str(human_relative_path(sample["human_video_path"]))
        if selected_paths is not None and selected_paths.get(relative) != task.split("-", 1)[0]:
            continue
        found.add(task.split("-", 1)[0])
        records.append({"sample": sample, "task_name": task,
                        "human_cache_path": human_feature_path(config.rgb_feature_root, sample["human_video_path"])})
    if found != tasks:
        raise ValueError(f"No human-only manifest samples for requested tasks: {sorted(tasks - found)}")
    if selected_paths is not None:
        unique_records = {str(human_relative_path(x["sample"]["human_video_path"])): x for x in records}
        if selected_paths.keys() != unique_records.keys():
            raise ValueError(f"Evaluation human map videos missing from official manifest: {sorted(selected_paths.keys() - unique_records.keys())}")
        records = list(unique_records.values())
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="robotwin_subset12_rgb_train")
    parser.add_argument("--human-gen-root", default="/workspace/zero-wam/datasets/HumanGen")
    parser.add_argument("--feature-root", default="/workspace/zero-wam/datasets/rgb_transition_dinov2_base")
    parser.add_argument("--model-path", default="/workspace/zero-wam/checkpoints/dinov2-base")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--human-segments", type=int, default=8)
    parser.add_argument("--human-frames-per-segment", type=int, default=4)
    parser.add_argument("--sample-ids", default="")
    parser.add_argument("--smoke-one-per-task", action="store_true")
    parser.add_argument("--download-human", action="store_true")
    parser.add_argument("--download-encoder", action="store_true")
    parser.add_argument("--download-only", action="store_true", help="Download selected raw human RGB, without loading the encoder or using a GPU")
    parser.add_argument("--human-only", action="store_true", help="Prepare human demonstration caches for evaluation without reading robot videos/latents")
    parser.add_argument("--task-names", default="", help="Explicit comma separated tasks; training preparation permits only a subset of the configured training tasks")
    parser.add_argument("--eval-human-map", help="Use the exact ROBOTWIN_ICL_HUMAN_VIDEOS evaluation mapping, with --human-only")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report-path", help="Optional report destination under /workspace; preserves other smoke reports")
    args = parser.parse_args()
    if not Path(args.feature_root).resolve().is_relative_to(Path("/workspace")):
        raise ValueError("Feature data must live under /workspace")
    if args.report_path and not Path(args.report_path).resolve().is_relative_to(Path("/workspace")):
        raise ValueError("Feature preparation reports must live under /workspace")
    from wan_va.configs import VA_CONFIGS
    config = deepcopy(VA_CONFIGS[args.config])
    root = Path(args.human_gen_root)
    config.dataset_path = str(root / "robotwin_data")
    config.icl_manifest_path = str(root / "icl_configs/ICL_config_robotwin.json")
    config.human_latent_path = str(root / "human_latents/robotwin")
    config.rgb_feature_root, config.rgb_encoder_id = args.feature_root, DEFAULT_ENCODER_ID
    config.enable_rgb_transition = True
    config.rgb_smoke_sample_ids = [x for x in args.sample_ids.split(",") if x] or None
    if args.task_names and not args.human_only:
        from wan_va.dataset.icl_lerobot_latent_dataset import _icl_repo_list
        selected_tasks = [x for x in args.task_names.split(",") if x]
        configured_tasks = {path.name.split("-", 1)[0] for path in _icl_repo_list(config)}
        if not selected_tasks or len(set(selected_tasks)) != len(selected_tasks) or set(selected_tasks) - configured_tasks:
            raise ValueError("Offline task subset must be unique and contained in the configured training tasks")
        config.included_task_names = selected_tasks
        config.expected_num_train_tasks = len(selected_tasks)
    if args.eval_human_map and not args.human_only:
        raise ValueError("--eval-human-map requires --human-only")
    records = human_only_records(config, args.task_names, args.eval_human_map) if args.human_only else list(expected_rgb_records(config))
    human_owners = {}
    for record in records:
        path, sid = str(record["human_cache_path"]), str(record["sample"]["sample_id"])
        previous = human_owners.setdefault(path, sid)
        if previous != sid:
            raise ValueError(f"Human cache path reused by distinct sample IDs {previous} and {sid}: {path}")
    if args.smoke_one_per_task:
        selected = {}
        for record in records:
            selected.setdefault(record["task_name"], record)
        records = list(selected.values())
        config.rgb_smoke_sample_ids = [rgb_pair_key(x["sample"]) for x in records]
    selected_ids = [rgb_pair_key(x["sample"]) for x in records]
    selection = {"selected_pairs": len(records), "tasks": len({x["task_name"] for x in records}),
                 "smoke": config.rgb_smoke_sample_ids is not None}
    if len(selected_ids) <= 32:
        selection["sample_ids"] = selected_ids
    print(json.dumps(selection), flush=True)
    if args.validate:
        if args.human_only:
            for record in records:
                load_feature_cache(record["human_cache_path"], "human", expected_encoder_id=DEFAULT_ENCODER_ID,
                                   expected_sample_id=record["sample"]["sample_id"], expected_revision=ENCODER_REVISION)
            print(json.dumps({"human_only": True, "validated_unique_human_videos": len({str(x['human_cache_path']) for x in records})}), flush=True)
            return
        print(json.dumps(validate_rgb_feature_cache(config)), flush=True)
        return
    if args.download_human:
        extract_human_archives(root, [x["sample"]["human_video_path"] for x in records])
    if args.download_encoder:
        download_encoder(args.model_path)
    if args.download_only:
        if not args.download_human:
            raise ValueError("--download-only requires --download-human")
        print(json.dumps({"download_only": True, "selected_pairs": len(records),
                          "unique_human_videos": len({str(x['human_cache_path']) for x in records})}), flush=True)
        return
    encoder = FrozenRGBEncoder(args.model_path, args.device, args.batch_size)
    config.rgb_encoder_revision = encoder.revision
    for index, record in enumerate(records):
        sample = record["sample"]
        sid = str(sample["sample_id"])
        prepare_human_feature(record, root, encoder, args)
        if args.human_only:
            continue
        meta = record["meta"]
        if args.overwrite or not record["robot_cache_path"].is_file():
            latent = torch.load(record["robot_latent_path"], map_location="cpu", weights_only=False)
            ids = latent_rgb_frame_indices(latent)
            robot = encoder.encode(decode_rgb_frames(record["robot_video_path"], ids.tolist()))
            save_features(record["robot_cache_path"], robot, ids, encoder=encoder, kind="robot", sample_id=sid,
                          source_video_path=record["robot_video_path"], source_latent_path=str(record["robot_latent_path"]),
                          source_latent_frame_ids=[int(x) for x in latent["frame_ids"]],
                          latent_num_frames=int(latent["latent_num_frames"]), start_frame=int(meta["start_frame"]),
                          end_frame=int(meta["end_frame"]), camera_key=record["camera_key"],
                          sampling="source_latent_frame_ids_every_four")
        print(f"prepared {index + 1}/{len(records)} sample={sid} task={record['task_name']}", flush=True)
    if args.human_only:
        report = {"human_only": True, "unique_human_videos": len({str(x['human_cache_path']) for x in records})}
    else:
        report = validate_rgb_feature_cache(config)
    report.update(encoder_id=encoder.encoder_id, encoder_revision=encoder.revision, sample_ids=selected_ids)
    Path(args.feature_root).mkdir(parents=True, exist_ok=True)
    report_name = "human_only_report.json" if args.human_only else ("smoke_report.json" if config.rgb_smoke_sample_ids else "preparation_report.json")
    report_path = Path(args.report_path) if args.report_path else Path(args.feature_root) / report_name
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({**report, "sample_ids": selected_ids if len(selected_ids) <= 32 else f"{len(selected_ids)} IDs in {report_path}"}), flush=True)


if __name__ == "__main__":
    main()
