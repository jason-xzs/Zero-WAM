# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
from copy import deepcopy
from contextlib import nullcontext
import os
import time
from pathlib import Path
import wandb

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from .checkpoint_io import (
    save_checkpoint_verified,
    save_sharded_safetensors as _save_sharded_safetensors,
)
import json

from .configs import TRAIN_DATASET_CONFIGS, VA_CONFIGS
from .configs.va_robotwin_cfg import load_robotwin_norm_stat
from .distributed.fsdp import shard_model, apply_ac
from .distributed.util import (
    _configure_model, 
    init_distributed, 
    dist_mean, 
    dist_max
)
from einops import rearrange
from .modules.utils import (
    load_transformer,
    resolve_model_component,
)
from .utils import (
    init_logger, 
    logger, 
    get_mesh_id, 
    sample_timestep_id,
    data_seq_to_patch,
    warmup_constant_lambda,
    FlowMatchScheduler
)

from .dataset import (
    DistributedDatasetMixtureSampler,
    MixedICLLeRobotLatentDataset,
    MultiICLLeRobotLatentDataset,
    icl_dataset_indexes_ready,
    normalized_dataset_weights,
    parse_dataset_mixture,
)
from .mcp import shift_latents_for_mcp, validate_mcp_settings
import gc


def _safe_mean(tensor):
    return tensor.sum() / max(tensor.numel(), 1)


_DATASET_PATH_OVERRIDES = (
    'dataset_path',
    'icl_manifest_path',
    'human_latent_path',
    'robot_latent_path',
)


def _build_dataset_sources(config, args, rank, local_rank, world_size):
    entries = parse_dataset_mixture(
        args.datasets,
        available_names=TRAIN_DATASET_CONFIGS,
    )
    path_overrides = {
        key: getattr(args, key)
        for key in _DATASET_PATH_OVERRIDES
        if getattr(args, key) is not None
    }
    if len(entries) > 1 and path_overrides:
        override_names = ', '.join(
            f"--{key.replace('_', '-')}" for key in path_overrides
        )
        raise ValueError(
            f"Dataset path overrides ({override_names}) are only supported when "
            "DATASETS selects one dataset"
        )

    sources = []
    for name, weight in entries:
        dataset_config = deepcopy(TRAIN_DATASET_CONFIGS[name])
        # The active experiment owns task selection. Do not mutate the global
        # dataset registry or let its full-task defaults overwrite a subset.
        if name == getattr(config, 'task_selection_dataset', None):
            for key in ('included_task_names', 'excluded_task_names', 'expected_num_train_tasks'):
                dataset_config[key] = deepcopy(getattr(config, key))
        if len(entries) == 1:
            dataset_config.update(path_overrides)

        # These settings are shared by every selected data source, while camera,
        # action, latent, and manifest settings remain dataset-specific.
        dataset_config.empty_emb_path = config.empty_emb_path
        dataset_config.init_worker = config.init_worker
        dataset_config.cfg_prob = config.droptext_target
        dataset_config.rank = rank
        dataset_config.local_rank = local_rank
        dataset_config.world_size = world_size
        for key in ('enable_dataset_index_cache', 'rebuild_dataset_index_cache'):
            if hasattr(config, key):
                dataset_config[key] = config[key]

        if getattr(dataset_config, 'load_robotwin_stats_from_dataset', False):
            action_stats_path = (
                Path(dataset_config.dataset_path) / 'meta' / 'action_stats.json'
            )
            dataset_config.action_stats_path = str(action_stats_path)
            dataset_config.norm_stat = load_robotwin_norm_stat(action_stats_path)
        sources.append(
            {'name': name, 'weight': weight, 'config': dataset_config}
        )
    return sources


def _prepare_safetensors_state_dict(state_dict):
    """Cast to BF16 and break only storage aliases rejected by safetensors."""
    output = {}
    seen_storages = set()
    for name, value in state_dict.items():
        tensor = value.to(torch.bfloat16).contiguous()
        storage = tensor.untyped_storage()
        storage_id = (tensor.device.type, storage.data_ptr(), storage.nbytes())
        if storage_id in seen_storages:
            tensor = tensor.clone()
            storage = tensor.untyped_storage()
            storage_id = (tensor.device.type, storage.data_ptr(), storage.nbytes())
        seen_storages.add(storage_id)
        output[name] = tensor
    return output


class Trainer:
    @staticmethod
    def _resolve_transformer_path(path):
        transformer_path = Path(resolve_model_component(path, 'transformer'))
        if not (transformer_path / 'config.json').is_file():
            raise FileNotFoundError(
                f"Expected a Transformer config at {transformer_path}/config.json"
            )
        return str(transformer_path)

    def __init__(self, config):
        if config.enable_wandb and config.rank == 0:
            self.wandb = wandb
            self.wandb.init(
                entity=os.getenv('WANDB_ENTITY') or os.getenv('WANDB_TEAM_NAME'),
                project=os.getenv('WANDB_PROJECT', getattr(config, 'wandb_project', 'va_robotwin')),
                dir=os.getenv("WANDB_DIR"),
                config=config,
                mode=os.getenv("WANDB_MODE", "online"),
                name=os.getenv("WANDB_NAME", "robotwin-posttrain"),
            )
            self.wandb.config.update({
                'effective_batch_size': config.world_size * config.batch_size * config.gradient_accumulation_steps,
                'sync_each_microbatch': getattr(config, 'sync_each_microbatch', False),
            })
            logger.info("WandB logging enabled: %s", self.wandb.run.url)
        self.step = 0
        self.config = config
        self.device = torch.device(f"cuda:{config.local_rank}")
        self.dtype = config.param_dtype
        self.patch_size = config.patch_size
        self.enable_mcp = getattr(config, 'enable_mcp', True)
        if int(config.batch_size) != 1:
            raise ValueError(
                "Zero-WAM ICL training requires batch_size=1 per rank"
            )
        if self.enable_mcp:
            validate_mcp_settings(
                num_mcp_modules=config.num_mcp_modules,
                mcp_blocks_per_group=config.mcp_blocks_per_group,
                mcp_hidden_collect_layers=config.mcp_hidden_collect_layers,
                mcp_loss_weights=config.mcp_loss_weights,
            )

        # Load models
        logger.info("Loading models...")

        # Load and shard transformer with FSDP
        logger.info("Loading transformer...")

        if hasattr(config, 'resume_from') and config.resume_from:
            transformer_path = self._resolve_transformer_path(config.resume_from)
            if config.rank == 0:
                logger.info(f"Resuming from checkpoint: {transformer_path}")
        else:
            transformer_path = self._resolve_transformer_path(config.model_path)

        self.transformer = load_transformer(
            transformer_path,
            torch_dtype=torch.float32,
            torch_device='cpu',
            disable_mcp=not self.enable_mcp,
            icl_model=True,
            enable_mcp=self.enable_mcp,
            num_mcp_modules=config.num_mcp_modules,
            mcp_blocks_per_group=config.mcp_blocks_per_group,
            mcp_hidden_collect_layers=tuple(config.mcp_hidden_collect_layers),
        )
        if self.enable_mcp:
            validate_mcp_settings(
                num_mcp_modules=config.num_mcp_modules,
                mcp_blocks_per_group=config.mcp_blocks_per_group,
                mcp_hidden_collect_layers=config.mcp_hidden_collect_layers,
                mcp_loss_weights=config.mcp_loss_weights,
                num_layers=len(self.transformer.blocks),
            )
            logger.info(
                "MCP architecture: %d modules x %d block, hidden layers=%s",
                config.num_mcp_modules,
                config.mcp_blocks_per_group,
                list(config.mcp_hidden_collect_layers),
            )

        logger.info("Setting up activation checkpointing ...")
        apply_ac(self.transformer)

        logger.info("Setting up FSDP...")
        shard_fn = shard_model
        self.transformer = _configure_model(
            model=self.transformer,
            shard_fn=shard_fn,
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=False,
        )
        self.transformer.train()
        self.transformer.requires_grad_(True)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            [p for p in self.transformer.parameters() if p.requires_grad],
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
            weight_decay=config.weight_decay,
            fused=True,
            foreach=False,
        )

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, 
            lr_lambda=lambda step: warmup_constant_lambda(step, warmup_steps=config.warmup_steps))

        # Setup dataloaders
        logger.info("Setting up datasets...")
        dataset_sources = config.dataset_sources
        dataset_configs = [source['config'] for source in dataset_sources]
        cache_ready = False
        if (
            config.world_size > 1
            and getattr(config, 'enable_dataset_index_cache', True)
        ):
            cache_ready_flag = torch.zeros(1, device=self.device, dtype=torch.int)
            if config.rank == 0:
                cache_ready_flag.fill_(
                    int(
                        all(
                            icl_dataset_indexes_ready(dataset_config)
                            for dataset_config in dataset_configs
                        )
                    )
                )
            dist.broadcast(cache_ready_flag, src=0)
            cache_ready = bool(cache_ready_flag.item())

        use_rank_zero_indexing = (
            config.world_size > 1
            and getattr(config, 'enable_dataset_index_cache', True)
            and not cache_ready
        )
        if config.rank == 0 and cache_ready:
            logger.info(
                "Dataset index and Arrow caches are complete; "
                "loading all ranks concurrently"
            )
        if use_rank_zero_indexing and config.rank != 0:
            dist.barrier()
            # Rank 0 has completed any requested rebuild at this point.
            for dataset_config in dataset_configs:
                dataset_config.rebuild_dataset_index_cache = False

        if len(dataset_sources) == 1:
            train_dataset = MultiICLLeRobotLatentDataset(
                config=dataset_sources[0]['config']
            )
        else:
            train_dataset = MixedICLLeRobotLatentDataset(dataset_sources)

        if use_rank_zero_indexing:
            if config.rank == 0:
                dist.barrier()
            dist.barrier()

        if config.rank == 0:
            mixture = ', '.join(
                f'{name}={weight:.4f}'
                for name, weight in normalized_dataset_weights(
                    [
                        (source['name'], source['weight'])
                        for source in dataset_sources
                    ]
                )
            )
            logger.info(
                "Dataset ready: %d samples from %d datasets "
                "(%d index cache hits, %d direct Arrow loads, %d rebuilt)",
                len(train_dataset),
                len(train_dataset._datasets),
                train_dataset.index_cache_hits,
                train_dataset.hf_cache_hits,
                train_dataset.index_cache_misses,
            )
            logger.info("Dataset sampling probabilities: %s", mixture)
        if len(dataset_sources) > 1:
            train_sampler = DistributedDatasetMixtureSampler(
                dataset_lengths=train_dataset.dataset_lengths,
                dataset_weights=train_dataset.dataset_weights,
                num_replicas=config.world_size,
                rank=config.rank,
                seed=42,
            )
        elif config.world_size > 1:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=config.world_size,
                rank=config.rank,
                shuffle=True,
                seed=42,
            )
        else:
            train_sampler = None
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=(train_sampler is None), 
            num_workers=config.load_worker,
            sampler=train_sampler,
        )

        self.train_scheduler_latent = FlowMatchScheduler(shift=self.config.snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_latent.set_timesteps(1000, training=True)
        self.train_scheduler_action = FlowMatchScheduler(shift=self.config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_action.set_timesteps(1000, training=True)
        self.train_scheduler_mcp = None
        if self.enable_mcp:
            self.train_scheduler_mcp = FlowMatchScheduler(
                shift=self.config.mcp_snr_shift,
                sigma_min=0.0,
                extra_one_step=True,
            )
            self.train_scheduler_mcp.set_timesteps(1000, training=True)

        self.save_dir = Path(config.save_root) / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
        self.train_loader_iter = None
        # if hasattr(config, 'resume_from') and config.resume_from:
        #     self._load_training_state(config.resume_from)
    
    def _get_next_batch(self):
        """Get next batch from iterator, reset if epoch is finished."""
        if self.train_loader_iter is None:
            self.train_loader_iter = iter(self.train_loader)
        
        try:
            batch = next(self.train_loader_iter)
        except StopIteration:
            # Reset sampler and iterator when epoch finishes
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(self.train_loader.sampler.epoch + 1)
            self.train_loader_iter = iter(self.train_loader)
            batch = next(self.train_loader_iter)
        
        return batch

    @torch.no_grad()
    def _add_noise(
        self,
        latent,
        train_scheduler,
        action_mask=None,
        action_mode=False,
        noisy_cond_prob=0.0,
        noisy_cond_min_timestep_bd=0.5,
        noisy_cond_max_timestep_bd=1.0,
        frame_shift=0,
    ):
        B, C, F, H, W = latent.shape

        timestep_ids = sample_timestep_id(batch_size=F, num_train_timesteps=train_scheduler.num_train_timesteps)
        noise = torch.zeros_like(latent).normal_()
        timesteps = train_scheduler.timesteps[timestep_ids].to(device=self.device)
        noisy_latents =train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        targets =train_scheduler.training_target(latent, noise, timesteps)

        patch_f, patch_h, patch_w = self.patch_size
        if action_mode:
            patch_f = patch_h = patch_w = 1
        
        latent_grid_id = get_mesh_id(
            latent.shape[-3] // patch_f,  # F
            latent.shape[-2] // patch_h,  # H
            latent.shape[-1] // patch_w,  # W
            t=1 if action_mode else 0,  # 1 for action mode (0 for latent), not used
            f_w=1,
            f_shift=frame_shift,
            action=False,
        ).to(self.device)  # shape: [4, seq_len]
        latent_grid_id = latent_grid_id[None].repeat(B, 1, 1)

        if torch.rand(1).item() < noisy_cond_prob:
            cond_timestep_ids = sample_timestep_id(
                batch_size=F,
                min_timestep_bd=noisy_cond_min_timestep_bd,
                max_timestep_bd=noisy_cond_max_timestep_bd,
                num_train_timesteps=train_scheduler.num_train_timesteps,
            )
            noise = torch.zeros_like(latent).normal_()
            cond_timesteps = train_scheduler.timesteps[cond_timestep_ids].to(device=self.device)
            latent = train_scheduler.add_noise(latent, noise, cond_timesteps, t_dim=2)
        else:
            cond_timesteps = torch.zeros_like(timesteps)

        if action_mask is not None:
            noisy_latents *= action_mask.float()
            targets *= action_mask.float()
            latent *= action_mask.float()

        return dict(
            timesteps=timesteps[None].repeat(B, 1),
            noisy_latents=noisy_latents,
            targets=targets,
            latent=latent,
            cond_timesteps=cond_timesteps[None].repeat(B, 1),
            grid_id=latent_grid_id,
        )

    @torch.no_grad()
    def _prepare_input_dict(self, batch_dict):
        """Build the single-sample VA training streams and ICL condition."""
        configured_chunk = int(getattr(self.config, 'frame_chunk_size', 0))
        if configured_chunk > 0:
            chunk_size = configured_chunk
        else:
            chunk_size = torch.randint(
                1, int(self.config.max_frame_chunk_size) + 1, (1,)
            ).item()
        latent_dict = self._add_noise(
            latent=batch_dict['latents'], 
            train_scheduler=self.train_scheduler_latent, 
            action_mask=None, 
            action_mode=False,
            noisy_cond_prob=float(self.config.noisy_img_prob),
            noisy_cond_min_timestep_bd=float(
                self.config.noisy_cond_min_timestep_bd),
            noisy_cond_max_timestep_bd=float(
                self.config.noisy_cond_max_timestep_bd),
        )
        
        action_dict = self._add_noise(
            latent=batch_dict['actions'], 
            train_scheduler=self.train_scheduler_action, 
            action_mask=None,
            action_mode=True,
            noisy_cond_prob=0.0)

        action_dict['actions_mask'] = batch_dict['actions_mask']

        use_icl = torch.rand(1).item() >= float(self.config.drop_icl)
        icl_latent_dict = None
        target_text_emb = batch_dict['text_emb']
        target_text_length = target_text_emb.shape[1]
        text_parts = [target_text_emb]
        encoder_seq_parts = [
            torch.zeros(target_text_length, device=self.device, dtype=torch.int)
        ]
        if use_icl:
            icl_latents = batch_dict['icl_latents']
            batch_size, _, frames, height, width = icl_latents.shape
            icl_grid_id = get_mesh_id(
                frames // self.patch_size[0],
                height // self.patch_size[1],
                width // self.patch_size[2],
                t=0,
                h_shift=int(self.config.icl_rope_h),
            ).to(self.device)
            icl_latent_dict = {
                'latent': icl_latents,
                'timesteps': torch.zeros(
                    batch_size, frames, device=self.device, dtype=torch.float32
                ),
                'grid_id': icl_grid_id[None].repeat(batch_size, 1, 1),
            }
            icl_text_emb = batch_dict['icl_text_emb']
            text_parts.append(icl_text_emb)
            encoder_seq_parts.append(
                torch.ones(
                    icl_text_emb.shape[1], device=self.device, dtype=torch.int
                )
            )

        max_window = int(getattr(self.config, 'max_attn_window', 64))
        configured_window = int(getattr(self.config, 'attn_window', 0))
        window_size = (
            configured_window
            if configured_window != 0
            else torch.randint(4, max_window + 1, (1,)).item()
        )
        input_dict = {
            'latent_dict': latent_dict,
            'action_dict': action_dict,
            'icl_latent_dict': icl_latent_dict,
            'text_emb': torch.cat(text_parts, dim=1),
            'encoder_seq_ids': torch.cat(encoder_seq_parts),
            'chunk_size': chunk_size,
            'max_frame_chunk_size': int(self.config.max_frame_chunk_size),
            'window_size': window_size,
        }
        if self.enable_mcp:
            mcp_latent_dicts = []
            for module_index in range(self.config.num_mcp_modules):
                future_chunk_offset = (
                    1 + module_index * int(self.config.future_chunk_stride)
                )
                frame_shift = future_chunk_offset * chunk_size
                shifted_latents, valid_mask = shift_latents_for_mcp(
                    batch_dict['latents'], frame_shift)
                mcp_latent_dict = self._add_noise(
                    latent=shifted_latents,
                    train_scheduler=self.train_scheduler_mcp,
                    action_mask=None,
                    action_mode=False,
                    noisy_cond_prob=float(self.config.noisy_img_prob),
                    noisy_cond_min_timestep_bd=float(
                        self.config.noisy_cond_min_timestep_bd),
                    noisy_cond_max_timestep_bd=float(
                        self.config.noisy_cond_max_timestep_bd),
                    frame_shift=frame_shift,
                )
                mcp_latent_dict['valid_mask'] = valid_mask
                mcp_latent_dicts.append(mcp_latent_dict)
            input_dict['mcp_latent_dicts'] = mcp_latent_dicts
        return input_dict

    def convert_input_format(self, input_dict):
        """Convert input dict to match transformer input format if needed."""
        for key, value in input_dict.items():
            if torch.is_tensor(value):
                input_dict[key] = value.to(self.device)
        return input_dict

    def compute_loss(self,
        input_dict,
        pred
    ):
        if self.enable_mcp:
            latent_pred, action_pred, mcp_pred_list = pred
        else:
            latent_pred, action_pred = pred
            mcp_pred_list = []
        if self.enable_mcp and len(mcp_pred_list) != self.config.num_mcp_modules:
            raise RuntimeError(
                "MCP output count must match num_mcp_modules")
        action_pred = rearrange(action_pred, 'b (f n) c -> b c f n 1', f=input_dict['action_dict']['targets'].shape[-3])
        latent_pred = data_seq_to_patch(
                        self.patch_size, latent_pred,
                        input_dict['latent_dict']['targets'].shape[-3], input_dict['latent_dict']['targets'].shape[-2],
                        input_dict['latent_dict']['targets'].shape[-1], batch_size=latent_pred.shape[0])
        batch_size, num_frames = input_dict['latent_dict']['timesteps'].shape
        latent_loss_weight = self.train_scheduler_latent.training_weight(
            input_dict['latent_dict']['timesteps'].flatten()
        ).reshape(batch_size, num_frames)
        if not bool(self.config.video_loss_reweight):
            latent_loss_weight = torch.ones_like(latent_loss_weight)

        latent_loss = F.mse_loss(
            latent_pred.float(),
            input_dict['latent_dict']['targets'].float().detach(),
            reduction='none',
        )
        latent_loss = _safe_mean(
            latent_loss * latent_loss_weight[:, None, :, None, None]
        )

        action_loss_weight = self.train_scheduler_action.training_weight(
            input_dict['action_dict']['timesteps'].flatten()
        ).reshape(batch_size, num_frames)
        if not bool(self.config.action_loss_reweight):
            action_loss_weight = torch.ones_like(action_loss_weight)
        action_loss = F.mse_loss(
            action_pred.float(),
            input_dict['action_dict']['targets'].float().detach(),
            reduction='none',
        )
        action_loss = _safe_mean(
            action_loss
            * action_loss_weight[:, None, :, None, None]
            * input_dict['action_dict']['actions_mask'].float()
        )

        mcp_losses = []
        for mcp_pred, mcp_latent_dict in zip(
                mcp_pred_list, input_dict.get('mcp_latent_dicts', [])):
            mcp_pred = data_seq_to_patch(
                self.patch_size,
                mcp_pred,
                mcp_latent_dict['targets'].shape[-3],
                mcp_latent_dict['targets'].shape[-2],
                mcp_latent_dict['targets'].shape[-1],
                batch_size=mcp_pred.shape[0],
            )
            mcp_batch_size, mcp_num_frames = mcp_latent_dict[
                'timesteps'].shape
            mcp_loss_weight = self.train_scheduler_mcp.training_weight(
                mcp_latent_dict['timesteps'].flatten()).reshape(
                    mcp_batch_size, mcp_num_frames)
            if not bool(self.config.video_loss_reweight):
                mcp_loss_weight = torch.ones_like(mcp_loss_weight)
            mcp_loss = F.mse_loss(
                mcp_pred.float(),
                mcp_latent_dict['targets'].float().detach(),
                reduction='none',
            )
            mcp_loss = mcp_loss * mcp_loss_weight[:, None, :, None, None]
            valid_mask = mcp_latent_dict['valid_mask'].to(
                device=mcp_loss.device, dtype=mcp_loss.dtype)
            valid_count = valid_mask.expand_as(mcp_loss).sum()
            mcp_loss = (mcp_loss * valid_mask).sum() / valid_count.clamp_min(1.)
            mcp_losses.append(mcp_loss / self.gradient_accumulation_steps)

        return (
            latent_loss / self.gradient_accumulation_steps,
            action_loss / self.gradient_accumulation_steps,
            mcp_losses,
        )

    def _train_step(self, batch, batch_idx):
        """Train a single batch, returns losses for logging."""
        first_batch = self.step == 0 and batch_idx == 0 and self.config.rank == 0
        if first_batch:
            logger.info("First batch shapes: %s", {
                key: tuple(value.shape) for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            })
        batch = self.convert_input_format(batch)
        input_dict = self._prepare_input_dict(batch)
        
        should_update = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        # Syncing each microbatch retains sharded gradients during accumulation.
        # Loss scaling and optimizer/scheduler updates still use the full window.
        sync_each_microbatch = getattr(self.config, 'sync_each_microbatch', False)
        self.transformer.set_requires_gradient_sync(should_update or sync_each_microbatch)

        activation_context = (
            torch.autograd.graph.save_on_cpu(pin_memory=True)
            if os.environ.get("ZERO_WAM_ACTIVATION_OFFLOAD", "0") == "1"
            else nullcontext()
        )
        with activation_context:
            output = self.transformer(input_dict, train_mode=True)
        if first_batch:
            logger.info("First forward completed; computing losses and backward")
        latent_loss, action_loss, mcp_losses = self.compute_loss(
            input_dict, output)
        mcp_loss = sum(
            weight * depth_loss
            for weight, depth_loss in zip(
                self.config.mcp_loss_weights, mcp_losses)
        ) if mcp_losses else latent_loss.new_zeros(())
        loss = latent_loss + action_loss + mcp_loss

        loss.backward()
        if first_batch:
            logger.info("First backward completed; optimizer update due=%s", should_update)

        losses = {
            'latent_loss': latent_loss.detach(),
            'action_loss': action_loss.detach(),
            'mcp_losses': [depth_loss.detach() for depth_loss in mcp_losses],
            'mcp_loss': mcp_loss.detach(),
        }
        
        # Only update weights after accumulating gradients
        if should_update:
            max_norm = float(self.config.max_norm)
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.transformer.parameters(), max_norm)
            skip_threshold = (
                float(self.config.skip_step_grad_norm_multiplier) * max_norm
            )
            skip_optimizer_step = (
                not bool(torch.isfinite(total_norm).item())
                or (
                    float(self.config.skip_step_grad_norm_multiplier) > 0
                    and total_norm.item() > skip_threshold
                )
            )
            if not skip_optimizer_step:
                self.optimizer.step()
            elif self.config.rank == 0:
                logger.warning(
                    "Skipping optimizer step: grad_norm=%s threshold=%s",
                    float(total_norm.item()),
                    skip_threshold,
                )
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
            losses['total_norm'] = total_norm
            losses['optimizer_step_skipped'] = skip_optimizer_step
            losses['should_log'] = True
        else:
            losses['should_log'] = False

        return losses

    def save_checkpoint(self,):
        """Save and verify all shards; retain live state and retry on I/O errors."""
        return save_checkpoint_verified(self)

    def _load_training_state(self, checkpoint_path):
        """Load training state (optimizer + step) after FSDP and optimizer creation."""
        checkpoint_dir = Path(checkpoint_path)
        training_state_path = checkpoint_dir / "training_state.pt"

        if not training_state_path.exists():
            if self.config.rank == 0:
                logger.warning(f"Training state not found: {training_state_path}, starting from step 0")
            return

        if self.config.rank == 0:
            logger.info(f"Loading training state from {training_state_path}")

        # All ranks load the training state directly
        training_state = torch.load(training_state_path, map_location='cpu', weights_only=False)

        # All ranks load optimizer state (required for FSDP)
        set_optimizer_state_dict(
            self.transformer, self.optimizer,
            optim_state_dict=training_state['optimizer_state_dict'],
            options=StateDictOptions(full_state_dict=True, strict=False)
        )
        self.step = training_state.get('step', 0)

        if self.config.rank == 0:
            logger.info(f"Training state loaded, resuming from step {self.step}")

        # Synchronize all ranks
        if dist.is_initialized():
            dist.barrier()

    def train(self):
        """Main training loop - train by steps instead of epochs."""
        logger.info(f"Starting training for {self.config.num_steps} steps...")
        effective_batch = self.config.world_size * self.config.batch_size * self.gradient_accumulation_steps
        logger.info(
            "Training batch: world_size=%d per_rank_batch=%d gradient_accumulation=%d effective_batch=%d sync_each_microbatch=%s",
            self.config.world_size, self.config.batch_size, self.gradient_accumulation_steps,
            effective_batch, int(getattr(self.config, 'sync_each_microbatch', False)),
        )
        metrics_path = os.environ.get("ZERO_WAM_METRICS_PATH")
        if metrics_path and self.config.rank == 0:
            Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        step_started = time.perf_counter()
        self.transformer.train()

        progress_bar = tqdm(
            total=self.config.num_steps,
            desc="Training",
            disable=(self.config.rank != 0),
            leave=True,
            dynamic_ncols=True,
            initial=self.step
        )

        self.optimizer.zero_grad()
        accumulated_latent_losses = []
        accumulated_action_losses = []
        accumulated_mcp_losses = [
            [] for _ in range(self.config.num_mcp_modules)
        ] if self.enable_mcp else []
        accumulated_mcp_total_losses = []
        step_in_accumulation = 0

        while self.step < self.config.num_steps:
            # Get next batch (handles epoch reset automatically)
            batch = self._get_next_batch()
            
            losses = self._train_step(batch, step_in_accumulation)
            
            # Accumulate losses for logging
            accumulated_latent_losses.append(losses['latent_loss'])
            accumulated_action_losses.append(losses['action_loss'])
            for depth, mcp_loss in enumerate(losses['mcp_losses']):
                accumulated_mcp_losses[depth].append(mcp_loss)
            accumulated_mcp_total_losses.append(losses['mcp_loss'])
            step_in_accumulation += 1

            # Log and checkpoint when optimizer steps
            if losses['should_log']:
                lr = self.lr_scheduler.get_last_lr()[0]

                # Average accumulated losses
                latent_loss_show = dist_mean(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                action_loss_show = dist_mean(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()
                max_latent_loss_show = dist_max(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                max_action_loss_show = dist_max(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()
                mcp_loss_shows = [
                    dist_mean(torch.stack(depth_losses).sum()).detach().cpu().item()
                    for depth_losses in accumulated_mcp_losses
                ]
                mcp_total_loss_show = dist_mean(
                    torch.stack(accumulated_mcp_total_losses).sum()
                ).detach().cpu().item()

                # Clear accumulated losses
                accumulated_latent_losses = []
                accumulated_action_losses = []
                accumulated_mcp_losses = [
                    [] for _ in range(self.config.num_mcp_modules)
                ] if self.enable_mcp else []
                accumulated_mcp_total_losses = []
                step_in_accumulation = 0

                torch.cuda.synchronize()
                if self.step % self.config.gc_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                if self.config.rank == 0:
                    total_norm = losses['total_norm']
                    step_seconds = time.perf_counter() - step_started
                    if metrics_path:
                        record = {
                            'timestamp': time.time(), 'step': self.step + 1,
                            'step_seconds': step_seconds, 'effective_batch': effective_batch,
                            'gradient_accumulation_steps': self.gradient_accumulation_steps,
                            'latent_loss': latent_loss_show, 'action_loss': action_loss_show,
                            'mcp_loss': mcp_total_loss_show, 'grad_norm': total_norm.item(),
                            'next_step_lr': lr, 'optimizer_step_skipped': losses['optimizer_step_skipped'],
                            'gpu_peak_allocated_gib_rank0': torch.cuda.max_memory_allocated() / 2**30,
                        }
                        with open(metrics_path, 'a', encoding='utf-8') as metrics_file:
                            metrics_file.write(json.dumps(record) + '\n')
                    progress_bar.n += 1
                    postfix = {
                        'latent_loss': f'{latent_loss_show:.4f}',
                        'action_loss': f'{action_loss_show:.4f}',
                        'step': self.step,
                        'grad_norm': f'{total_norm.item():.2f}',
                        'lr': f'{lr:.2e}',
                        'step_s': f'{step_seconds:.1f}',
                    }
                    if losses['optimizer_step_skipped']:
                        postfix['optimizer_step'] = 'skipped'
                    if self.enable_mcp:
                        postfix['mcp_loss'] = f'{mcp_total_loss_show:.4f}'
                    progress_bar.set_postfix(postfix)
                    if self.config.enable_wandb:
                        log_values = {
                            'performance/step_seconds': step_seconds,
                            'performance/samples_per_second': effective_batch / step_seconds,
                            'train/effective_batch_size': effective_batch,
                            'train/optimizer_step': self.step + 1,
                            'train/optimizer_step_skipped': int(losses['optimizer_step_skipped']),
                            'loss_metrics/global_avg_video_loss': latent_loss_show,
                            'loss_metrics/global_avg_action_loss': action_loss_show,
                            'loss_metrics/global_max_video_loss': max_latent_loss_show,
                            'loss_metrics/global_max_action_loss': max_action_loss_show,
                            'grad_norm': total_norm.item(),
                            'lr': lr,
                        }
                        if self.enable_mcp:
                            log_values['loss_metrics/mcp_weighted_total'] = (
                                mcp_total_loss_show)
                            for depth, mcp_loss_show in enumerate(
                                    mcp_loss_shows):
                                log_values[
                                    f'loss_metrics/mcp_depth_{depth + 1}'] = (
                                        mcp_loss_show)
                        self.wandb.log(log_values, step=self.step + 1)
                
                self.step += 1
                
                if self.step % self.config.save_interval == 0:
                    if self.config.rank == 0:
                        logger.info(f"Starting save model at step {self.step}")
                    self.save_checkpoint()
                step_started = time.perf_counter()

            if dist.is_initialized():
                dist.barrier()

        progress_bar.close()
        logger.info("Training completed!")


def build_training_config(args):
    """Resolve CLI overrides and data sources without initializing CUDA or W&B."""
    config = deepcopy(VA_CONFIGS[args.config_name])

    overrides = {
        'model_path': args.model_path,
        'empty_emb_path': args.empty_emb_path,
        'learning_rate': args.learning_rate,
        'droptext_target': args.droptext_target,
        'drop_icl': args.drop_icl,
        'init_worker': args.init_worker,
        'load_worker': args.load_worker,
        'batch_size': args.batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'num_steps': args.num_steps,
        'save_interval': args.save_interval,
        'save_root': args.save_root,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    config.cfg_prob = config.droptext_target

    if args.enable_wandb:
        config.enable_wandb = True
    if args.disable_wandb:
        config.enable_wandb = False

    if 'ZERO_WAM_SYNC_EACH_MICROBATCH' in os.environ:
        value = os.environ['ZERO_WAM_SYNC_EACH_MICROBATCH']
        if value not in ('0', '1'):
            raise ValueError('ZERO_WAM_SYNC_EACH_MICROBATCH must be 0 or 1')
        config.sync_each_microbatch = value == '1'

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    check_config = getattr(args, 'check_config', False)
    expected_world_size = getattr(config, 'expected_world_size', None)
    default_world_size = (expected_world_size or 1) if check_config else 1
    world_size = int(os.environ.get('WORLD_SIZE', default_world_size))
    if expected_world_size is not None and world_size != expected_world_size:
        raise ValueError(f'{args.config_name} requires {expected_world_size} GPUs, got WORLD_SIZE={world_size}')
    effective_batch = world_size * config.batch_size * config.gradient_accumulation_steps
    expected_batch = getattr(config, 'expected_effective_batch_size', None)
    if expected_batch is not None and effective_batch != expected_batch:
        raise ValueError(f'{args.config_name} requires global batch {expected_batch}, got {effective_batch}')
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    config.datasets = args.datasets
    config.dataset_sources = _build_dataset_sources(
        config,
        args,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )
    if getattr(config, 'task_selection_dataset', None) is not None:
        names = [source['name'] for source in config.dataset_sources]
        if names != [config.task_selection_dataset]:
            raise ValueError(f'{args.config_name} requires --datasets {config.task_selection_dataset}:1.0')
    return config


def training_config_report(config):
    """Inspect the exact dataset configs that the Trainer will construct."""
    from .dataset.icl_lerobot_latent_dataset import _icl_repo_list, _partition_icl_repos

    sources = []
    for source in config.dataset_sources:
        dataset_config = source['config']
        selected = _icl_repo_list(dataset_config)
        _, excluded = _partition_icl_repos(dataset_config)
        sources.append({
            'name': source['name'], 'selected_task_count': len(selected),
            'excluded_roots': len(excluded),
            'training_tasks': [path.name.split('-', 1)[0] for path in selected],
            'selected_roots': [str(path) for path in selected],
            'excluded_task_names': dataset_config.excluded_task_names,
        })
    return {
        'world_size': config.world_size,
        'batch_size': config.batch_size,
        'gradient_accumulation_steps': config.gradient_accumulation_steps,
        'effective_batch_size': config.world_size * config.batch_size * config.gradient_accumulation_steps,
        'num_steps': config.num_steps, 'save_interval': config.save_interval,
        'learning_rate': config.learning_rate, 'warmup_steps': config.warmup_steps,
        'sync_each_microbatch': getattr(config, 'sync_each_microbatch', False),
        'enable_wandb': config.enable_wandb,
        'wandb_project': os.getenv('WANDB_PROJECT', getattr(config, 'wandb_project', 'va_robotwin')),
        'cpu_offload': os.getenv('ZERO_WAM_CPU_OFFLOAD', '0') == '1',
        'activation_offload': os.getenv('ZERO_WAM_ACTIVATION_OFFLOAD', '0') == '1',
        'dataset_sources': sources,
    }


def run(args):
    """Common training entry point for full-task and subset configurations."""
    config = build_training_config(args)
    if getattr(args, 'check_config', False):
        print(json.dumps(training_config_report(config), indent=2))
        return
    rank, local_rank, world_size = config.rank, config.local_rank, config.world_size

    init_distributed(world_size, local_rank, rank)

    if rank == 0:
        logger.info(f"Using config: {args.config_name}")
        logger.info(f"Using DATASETS: {args.datasets}")
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")

    try:
        trainer = Trainer(config)
        trainer.train()
    except BaseException:
        # Report the original error before collective teardown, which can wait
        # for peers that are still in a forward/backward collective.
        logger.exception("Training failed on rank %d", rank)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Train WAN model for robotics")
    parser.add_argument(
        "--config-name",
        type=str,
        default='robotwin_train',
        help="Config name",
    )
    parser.add_argument(
        '--check-config', action='store_true',
        help='Validate the selected task split and recipe without initializing CUDA, model, or W&B',
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="robotwin:1.0",
        help=(
            "Comma-separated dataset sampling weights, for example "
            "agibot:0.5,robotwin:0.1"
        ),
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default=None,
        help="Root directory for saving checkpoints",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Model root containing the transformer directory",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Root directory containing open-format LeRobot datasets",
    )
    parser.add_argument("--icl-manifest-path", type=str, default=None)
    parser.add_argument("--human-latent-path", type=str, default=None)
    parser.add_argument("--robot-latent-path", type=str, default=None)
    parser.add_argument(
        "--empty-emb-path",
        type=str,
        default=None,
        help="Path to the empty-text embedding bundled with Zero-WAM",
    )
    parser.add_argument(
        "--enable-wandb",
        action="store_true",
        help="Enable Weights & Biases logging using WANDB environment settings",
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--droptext-target", type=float, default=None)
    parser.add_argument("--drop-icl", type=float, default=None)
    parser.add_argument("--init-worker", type=int, default=None)
    parser.add_argument("--load-worker", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()
