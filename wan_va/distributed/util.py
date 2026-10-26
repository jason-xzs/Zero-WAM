# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
import torch
import torch.distributed as dist


def _configure_model(model, shard_fn, param_dtype, device, eval_mode=True):
    """
    TODO
    """
    if eval_mode:
        model.eval().requires_grad_(False)
    if dist.is_initialized():
        dist.barrier()

    if dist.is_initialized():
        model = shard_fn(model)
    else:
        model.to(param_dtype)
        model.to(device)

    return model


def init_distributed(world_size, local_rank, rank):
    torch.cuda.set_device(local_rank)
    memory_fraction = os.environ.get("ZERO_WAM_GPU_MEMORY_FRACTION")
    if memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(float(memory_fraction), local_rank)
    device = torch.device(f"cuda:{local_rank}")
    backend = (
        "cpu:gloo,cuda:nccl"
        if os.environ.get("ZERO_WAM_CPU_OFFLOAD", "0") == "1"
        else "nccl"
    )
    dist.init_process_group(backend=backend,
                            init_method="env://",
                            rank=rank,
                            world_size=world_size,
                            device_id=device)

def dist_mean(local_tensor):
    if dist.is_initialized():
        dist.all_reduce(local_tensor, op=dist.ReduceOp.AVG)
    return local_tensor

def dist_max(local_tensor):
    if dist.is_initialized():
        dist.all_reduce(local_tensor, op=dist.ReduceOp.MAX)
    return local_tensor
