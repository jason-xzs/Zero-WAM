# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""LeRobot latent dataset with one paired human-video ICL condition.

This loader supports one LeRobot sample and one paired human-video condition
per item.
"""

import json
import os
import re
from collections import Counter
from bisect import bisect_right
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import torch
from einops import rearrange
from tqdm import tqdm

from .lerobot_latent_dataset import (
    LatentLeRobotDataset,
    dataset_index_cache_path,
    dataset_index_fingerprint,
    dataset_video_keys,
    recursive_find_file,
)


_EPISODE_FILE_RE = re.compile(r"^(episode_\d{6})(?:_\d+_\d+)?(\.[^.]+)$")


def _normalize_path(value):
    return os.path.normpath(str(value or "")).replace("\\", "/")


def _video_key_without_view(value):
    parts = [part for part in _normalize_path(value).split("/") if part]
    if "videos" in parts:
        video_idx = parts.index("videos")
        if (
            len(parts) > video_idx + 3
            and parts[video_idx + 1].startswith("chunk-")
            and parts[video_idx + 3].startswith("episode_")
        ):
            parts = parts[: video_idx + 2] + parts[video_idx + 3 :]
    return "/".join(parts)


def _episode_key_without_interval(value):
    parts = _video_key_without_view(value).split("/")
    if not parts:
        return ""
    match = _EPISODE_FILE_RE.match(parts[-1])
    if match:
        parts[-1] = f"{match.group(1)}{match.group(2)}"
    return "/".join(parts)


def load_icl_manifest(path):
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    samples = payload.get("samples", []) if isinstance(payload, dict) else []
    exact_index = {}
    episode_index = {}
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        status_info = sample.get("status_info") or {}
        robot_video = status_info.get("video_rel_path") or sample.get(
            "robot_video_path", ""
        )
        exact_key = _video_key_without_view(robot_video)
        episode_key = _episode_key_without_interval(robot_video)
        if exact_key:
            exact_index.setdefault(exact_key, sample)
        if episode_key:
            episode_index.setdefault(episode_key, sample)
    if not episode_index:
        raise ValueError(f"No valid ICL pairs found in {path}")
    return exact_index, episode_index


def _task_latent_root(task_root, config=None):
    collection_root = getattr(config, "robot_latent_path", "") if config else ""
    if collection_root:
        return Path(collection_root) / Path(task_root).name / "latents"
    return Path(task_root) / "latents"


def _construct_icl_dataset(repo_id, config):
    if _is_excluded_task_root(repo_id, config):
        raise ValueError(
            f"Refusing to construct held-out training dataset: {Path(repo_id).name}"
        )
    return ICLLeRobotLatentDataset(
        repo_id=repo_id,
        latent_root=_task_latent_root(repo_id, config),
        config=config,
    )


def _is_excluded_task_root(dataset_root, config):
    root_name = Path(dataset_root).name
    included_tasks = getattr(config, 'included_task_names', None)
    if included_tasks is not None and root_name.split('-', 1)[0] not in included_tasks:
        return True
    return any(
        root_name == task_name or root_name.startswith(f"{task_name}-")
        for task_name in getattr(config, "excluded_task_names", [])
    )


def _partition_icl_repos(config):
    repo_list = sorted(
        Path(path).parents[1]
        for path in recursive_find_file(config.dataset_path, "info.json")
    )
    included = []
    excluded = []
    for dataset_root in repo_list:
        target = excluded if _is_excluded_task_root(dataset_root, config) else included
        target.append(dataset_root)
    return included, excluded


def _validate_training_repo_count(config, included, excluded):
    requested = getattr(config, 'included_task_names', None)
    if requested is not None:
        requested_set = set(requested)
        if not requested or len(requested_set) != len(requested):
            raise ValueError('included_task_names must be nonempty and unique')
        overlap = requested_set.intersection(getattr(config, 'excluded_task_names', []))
        if overlap:
            raise ValueError(f'Training tasks overlap held-out tasks: {sorted(overlap)}')
        counts = Counter(path.name.split('-', 1)[0] for path in included)
        missing = sorted(requested_set - counts.keys())
        duplicates = sorted(name for name, count in counts.items() if count > 1)
        if missing or duplicates:
            raise ValueError(
                f'Expected exactly one dataset per selected task; '
                f'missing={missing}, duplicate_variants={duplicates}'
            )
    expected = getattr(config, "expected_num_train_tasks", None)
    if expected is not None and len(included) != int(expected):
        raise ValueError(
            f"Expected {expected} training task datasets under {config.dataset_path}, "
            f"found {len(included)} after excluding {len(excluded)} held-out datasets"
        )


def _icl_repo_list(config):
    included, excluded = _partition_icl_repos(config)
    _validate_training_repo_count(config, included, excluded)
    return included


def icl_dataset_indexes_ready(config):
    if not getattr(config, "enable_dataset_index_cache", True):
        return False
    if getattr(config, "rebuild_dataset_index_cache", False):
        return False
    for dataset_root in _icl_repo_list(config):
        latent_root = _task_latent_root(dataset_root, config)
        fingerprint = dataset_index_fingerprint(
            dataset_root,
            latent_root,
            dataset_video_keys(dataset_root, config.obs_cam_keys),
        )
        cache_path = dataset_index_cache_path(dataset_root, fingerprint)
        try:
            with cache_path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return False
        cache_files = payload.get("hf_cache_files")
        if payload.get("fingerprint") != fingerprint:
            return False
        if not isinstance(payload.get("valid_metas"), list):
            return False
        if not cache_files or not all(Path(path).is_file() for path in cache_files):
            return False
    return True


class ICLLeRobotLatentDataset(LatentLeRobotDataset):
    """A single LeRobot task with manifest-selected ICL pairs."""

    def __init__(self, repo_id, latent_root, config):
        if _is_excluded_task_root(repo_id, config):
            raise ValueError(
                f"Refusing to load held-out training dataset: {Path(repo_id).name}"
            )
        self._icl_exact_index, self._icl_episode_index = load_icl_manifest(
            config.icl_manifest_path
        )
        self.human_latent_path = Path(config.human_latent_path)
        super().__init__(repo_id=repo_id, config=config, latent_root=latent_root)
        self.new_metas = [
            meta
            for meta in self.new_metas
            if (
                (sample := self._lookup_icl_sample(meta)) is not None
                and self._human_latent_candidate(sample).is_file()
            )
        ]
        if not self.new_metas:
            raise ValueError(f"No manifest-matched ICL samples found in {self.root}")

    def _task_name(self):
        return self.root.name

    def _candidate_keys(self, meta):
        source_video = meta.get("source_video_rel_path")
        if source_video:
            return [source_video], source_video
        episode_index = int(meta["episode_index"])
        chunk = self.meta.get_episode_chunk(episode_index)
        filename = f"episode_{episode_index:06d}.mp4"
        base = f"{self._task_name()}/videos/chunk-{chunk:03d}/{filename}"
        exact = []
        start = meta.get("start_frame")
        end = meta.get("end_frame")
        if start is not None and end is not None:
            exact.append(
                base.replace(
                    filename,
                    f"episode_{episode_index:06d}_{int(start)}_{int(end)}.mp4",
                )
            )
            if int(end) > int(start):
                exact.append(
                    base.replace(
                        filename,
                        f"episode_{episode_index:06d}_{int(start)}_{int(end) - 1}.mp4",
                    )
                )
        return exact, base

    def _lookup_icl_sample(self, meta):
        embedded_sample = meta.get("icl")
        if isinstance(embedded_sample, dict) and embedded_sample.get(
            "human_video_path"
        ):
            return embedded_sample
        exact_keys, episode_key = self._candidate_keys(meta)
        for key in exact_keys:
            sample = self._icl_exact_index.get(_video_key_without_view(key))
            if sample is not None:
                return sample
        return self._icl_episode_index.get(
            _episode_key_without_interval(episode_key)
        )

    def _human_latent_candidate(self, sample):
        human_video = _normalize_path(sample.get("human_video_path", ""))
        parts = [part for part in human_video.split("/") if part]
        run_index = next(
            (index for index, part in enumerate(parts) if part.startswith("run_")),
            None,
        )
        if run_index is None:
            raise ValueError(f"Invalid human_video_path in ICL manifest: {human_video}")
        relative = Path(*parts[run_index:]).with_suffix(".pth")
        return self.human_latent_path / relative

    def _human_latent_file(self, sample):
        latent_file = self._human_latent_candidate(sample)
        if not latent_file.is_file():
            raise FileNotFoundError(f"Missing paired human latent: {latent_file}")
        return latent_file

    @staticmethod
    def _load_human_latent(path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        required = (
            "latent",
            "latent_num_frames",
            "latent_height",
            "latent_width",
            "text_emb",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise KeyError(f"Human latent {path} is missing keys: {missing}")
        latent = payload["latent"]
        frames = int(payload["latent_num_frames"])
        height = int(payload["latent_height"])
        width = int(payload["latent_width"])
        if latent.ndim == 2:
            latent = rearrange(latent, "(f h w) c -> c f h w", f=frames, h=height, w=width)
        elif latent.ndim == 4:
            if latent.shape[0] != 48:
                raise ValueError(f"Unsupported human latent layout in {path}: {latent.shape}")
        else:
            raise ValueError(f"Unsupported human latent rank in {path}: {latent.ndim}")
        if tuple(latent.shape[1:]) != (frames, height, width):
            raise ValueError(
                f"Human latent shape metadata mismatch in {path}: "
                f"tensor={tuple(latent.shape)}, metadata={(frames, height, width)}"
            )
        # Wan pth files already contain model-normalized latents.
        return latent.contiguous(), payload["text_emb"]

    def __getitem__(self, idx):
        idx %= len(self.new_metas)
        meta = self.new_metas[idx]
        sample = self._lookup_icl_sample(meta)
        if sample is None:
            raise RuntimeError("ICL manifest index changed after dataset initialization")
        item = super().__getitem__(idx)
        human_latent_file = self._human_latent_file(sample)
        icl_latents, icl_text_emb = self._load_human_latent(human_latent_file)
        item.update(
            icl_latents=icl_latents,
            icl_text_emb=icl_text_emb,
            icl_sample_id=str(sample.get("sample_id", "")),
            icl_human_latent_path=str(human_latent_file),
        )
        return item


class MultiICLLeRobotLatentDataset(torch.utils.data.Dataset):
    def __init__(self, config, num_init_worker=None):
        repo_list, excluded_repo_list = _partition_icl_repos(config)
        _validate_training_repo_count(config, repo_list, excluded_repo_list)
        self.excluded_dataset_roots = excluded_repo_list
        if not repo_list:
            raise FileNotFoundError(
                f"No LeRobot datasets found under {config.dataset_path}"
            )
        if excluded_repo_list and getattr(config, "rank", 0) == 0:
            print(
                "Excluded held-out task datasets: "
                + ", ".join(path.name for path in excluded_repo_list),
                flush=True,
            )
        if num_init_worker is None:
            num_init_worker = getattr(config, "init_worker", 8)
        num_init_worker = min(max(int(num_init_worker), 1), len(repo_list))
        constructor = partial(_construct_icl_dataset, config=config)
        if num_init_worker == 1:
            iterator = map(constructor, repo_list)
            if getattr(config, "rank", 0) == 0:
                iterator = tqdm(
                    iterator, total=len(repo_list), desc="Loading ICL dataset indexes"
                )
            self._datasets = list(iterator)
        else:
            with Pool(num_init_worker) as pool:
                iterator = pool.imap(constructor, repo_list)
                if getattr(config, "rank", 0) == 0:
                    iterator = tqdm(
                        iterator,
                        total=len(repo_list),
                        desc="Loading ICL dataset indexes",
                    )
                self._datasets = list(iterator)
        self.index_cache_hits = sum(x.index_cache_hit for x in self._datasets)
        self.index_cache_misses = len(self._datasets) - self.index_cache_hits
        self.hf_cache_hits = sum(x.hf_cache_hit for x in self._datasets)
        self._offsets = []
        total = 0
        for dataset in self._datasets:
            self._offsets.append(total)
            total += len(dataset)
        self._length = total

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        if idx < 0:
            idx += self._length
        if not 0 <= idx < self._length:
            raise IndexError(idx)
        for dataset, offset in zip(reversed(self._datasets), reversed(self._offsets)):
            if idx >= offset:
                return dataset[idx - offset]
        raise IndexError(idx)


class MixedICLLeRobotLatentDataset(torch.utils.data.Dataset):
    """Flatten multiple independently configured ICL LeRobot collections."""

    def __init__(self, dataset_sources):
        if not dataset_sources:
            raise ValueError("At least one dataset source is required")

        self.dataset_names = []
        self.dataset_weights = []
        self.dataset_groups = []
        self.dataset_lengths = []
        self.dataset_offsets = []
        self._datasets = []
        self.excluded_dataset_roots = []
        total = 0
        for source in dataset_sources:
            name = source["name"]
            weight = float(source["weight"])
            dataset = MultiICLLeRobotLatentDataset(config=source["config"])
            length = len(dataset)
            if length <= 0:
                raise ValueError(f"Dataset {name!r} contains no valid samples")
            self.dataset_names.append(name)
            self.dataset_weights.append(weight)
            self.dataset_groups.append(dataset)
            self.dataset_lengths.append(length)
            self.dataset_offsets.append(total)
            self._datasets.extend(dataset._datasets)
            self.excluded_dataset_roots.extend(
                getattr(dataset, "excluded_dataset_roots", [])
            )
            total += length

        self._length = total
        self.index_cache_hits = sum(
            dataset.index_cache_hits for dataset in self.dataset_groups
        )
        self.index_cache_misses = sum(
            dataset.index_cache_misses for dataset in self.dataset_groups
        )
        self.hf_cache_hits = sum(
            dataset.hf_cache_hits for dataset in self.dataset_groups
        )

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        if idx < 0:
            idx += self._length
        if not 0 <= idx < self._length:
            raise IndexError(idx)
        dataset_id = bisect_right(self.dataset_offsets, idx) - 1
        return self.dataset_groups[dataset_id][
            idx - self.dataset_offsets[dataset_id]
        ]


__all__ = [
    "ICLLeRobotLatentDataset",
    "MixedICLLeRobotLatentDataset",
    "MultiICLLeRobotLatentDataset",
    "icl_dataset_indexes_ready",
    "load_icl_manifest",
]
