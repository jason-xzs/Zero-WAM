"""Verified sequential checkpoint writes for object-storage FUSE mounts."""
import hashlib
import json
import logging
import time
from pathlib import Path

import torch.distributed as dist
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import save
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

logger = logging.getLogger(__name__)


def write_verified(path, data):
    """Avoid ftruncate/rename, then verify bytes after closing the object."""
    path = Path(path)
    expected = hashlib.sha256(data).hexdigest()
    view = memoryview(data)
    with path.open("wb") as handle:
        for start in range(0, len(view), 8 * 1024 * 1024):
            block = view[start:start + 8 * 1024 * 1024]
            if handle.write(block) != len(block):
                raise OSError(f"Short write: {path}")
    actual = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            actual.update(block)
    if path.stat().st_size != len(data) or actual.hexdigest() != expected:
        raise OSError(f"Checkpoint read-back mismatch: {path}")
    return {"file": path.name, "size": len(data), "sha256": expected}


def save_sharded_safetensors(state_dict, output_dir, max_shard_size="3GB"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split = split_torch_state_dict_into_shards(
        state_dict,
        filename_pattern="diffusion_pytorch_model{suffix}.safetensors",
        max_shard_size=max_shard_size,
    )
    files = []
    for filename, names in split.filename_to_tensors.items():
        shard = {name: state_dict[name] for name in names}
        serialized = save(shard, metadata={"format": "pt"})
        files.append(write_verified(output_dir / filename, serialized))
        del serialized
        logger.info("Saved and SHA256-verified checkpoint shard %s", filename)
    if split.is_sharded:
        index = {"metadata": split.metadata, "weight_map": split.tensor_to_filename}
        files.append(write_verified(
            output_dir / "diffusion_pytorch_model.safetensors.index.json",
            json.dumps(index, indent=2).encode(),
        ))
    return files


def save_checkpoint_verified(trainer):
    """Keep live model state and retry a failed save on all ranks together."""
    # Reuse the trainer's existing BF16/storage-alias handling.
    prepare = trainer.__class__.__module__
    import sys
    prepare = sys.modules[prepare]._prepare_safetensors_state_dict
    state_dict = get_model_state_dict(
        trainer.transformer,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )
    state_dict_bf16 = prepare(state_dict)
    del state_dict
    checkpoint_dir = trainer.save_dir / f"checkpoint_step_{trainer.step}"
    attempt = 0
    while True:
        attempt += 1
        result = [None]
        if trainer.config.rank == 0:
            try:
                transformer_dir = checkpoint_dir / "transformer"
                transformer_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Saving verified checkpoint at step %d (attempt %d)", trainer.step, attempt)
                files = save_sharded_safetensors(state_dict_bf16, transformer_dir)
                config = dict(trainer.transformer.config)
                config.pop("_name_or_path", None)
                files.append(write_verified(
                    transformer_dir / "config.json", json.dumps(config, indent=2).encode(),
                ))
                manifest = {
                    "step": trainer.step,
                    "timestamp": time.time(),
                    "tensor_count": len(state_dict_bf16),
                    "parameter_count": sum(t.numel() for t in state_dict_bf16.values()),
                    "weights_only": True,
                    "files": files,
                }
                write_verified(checkpoint_dir / "COMPLETE.json", json.dumps(manifest, indent=2).encode())
                result[0] = {"ok": True}
                logger.info("Checkpoint saved successfully and verified at step %d", trainer.step)
            except Exception as error:
                result[0] = {"ok": False, "error": str(error)}
                logger.exception("Checkpoint save failed; retaining live state and retrying")
            if trainer.config.enable_wandb:
                try:
                    trainer.wandb.run.summary["checkpoint/save_ok"] = result[0]["ok"]
                    trainer.wandb.run.summary["checkpoint/attempt_step"] = trainer.step
                    if result[0]["ok"]:
                        trainer.wandb.run.summary["checkpoint/last_verified_step"] = trainer.step
                        trainer.wandb.run.summary["checkpoint/path"] = str(checkpoint_dir)
                    else:
                        trainer.wandb.run.summary["checkpoint/error"] = result[0]["error"]
                except Exception:
                    logger.exception("Could not publish checkpoint status to W&B")
        if dist.is_initialized():
            dist.broadcast_object_list(result, src=0)
        if result[0]["ok"]:
            return
        # All ranks wait together; never report training progress past a failed save.
        time.sleep(30)
