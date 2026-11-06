#!/usr/bin/env python3
"""Train a small action-shaped RGB future teacher separately from Zero-WAM.

This is a source-grounded research adaptation of WoG's QFormer, not a WoG
reproduction. The default 2,000 steps and learning rate are initial research
settings, not a paper recipe. Production training reads all subset12 samples;
the explicit --smoke-test switch allows only 1--20 optimizer steps and marks
its artifact unusable for formal policy training. No large policy is loaded.
"""
import argparse
from contextlib import nullcontext
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-gen-root", default="/workspace/zero-wam/datasets/HumanGen")
    parser.add_argument("--feature-root", default="/workspace/zero-wam/datasets/rgb_transition_dinov2_base")
    parser.add_argument("--output", type=Path, required=True, help="New artifact directory under /workspace")
    parser.add_argument("--num-steps", type=int, default=2000)
    parser.add_argument("--num-tokens", type=int, default=32,
                        help="Future query count; must match policy readout_num_tokens")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--smoke-report", type=Path, help="Preparation JSON containing explicit sample_ids")
    selection.add_argument("--sample-ids", nargs="+", help="Explicit RGB pair keys; requires --smoke-test")
    args = parser.parse_args(argv)
    if args.num_steps < 1 or args.gradient_accumulation_steps < 1 or args.num_tokens < 1:
        parser.error("Steps, gradient accumulation and query count must be positive")
    if any(not math.isfinite(x) or x <= 0 for x in (args.learning_rate, args.max_grad_norm)):
        parser.error("Learning rate and gradient clipping must be finite and positive")
    if args.smoke_test and args.num_steps > 20:
        parser.error("--smoke-test requires explicit --num-steps between 1 and 20")
    if args.smoke_test != bool(args.smoke_report or args.sample_ids):
        parser.error("Smoke tests require --smoke-report or --sample-ids; formal runs forbid both")
    args.output = args.output.resolve()
    if not args.output.is_relative_to(Path("/workspace")):
        parser.error("Teacher checkpoints must be stored under /workspace")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error(f"Refusing to overwrite existing output: {args.output}")
    metrics_path = args.output.with_name(args.output.name + ".metrics.jsonl")
    if metrics_path.exists():
        parser.error(f"Refusing to overwrite existing training metrics: {metrics_path}")
    return args


def select_training_chunk(batch, *, max_future_steps=4, generator=None, start=None, end=None):
    """Preserve the loader's latent-anchor/action alignment without frame shifts.

    An anchor f has its own 16-action block. The current observation is f-1;
    future observations and target action blocks use precisely [f:end]. The
    initial observation is never trained as a future. Explicit indices support
    deterministic alignment tests.
    """
    import torch

    rgb, actions, mask = (batch[key] for key in ("rgb_robot_features", "actions", "actions_mask"))
    if (rgb.ndim != 4 or rgb.shape[0] != 1 or actions.ndim != 5 or actions.shape[0] != 1
            or actions.shape[-1] != 1 or actions.shape[2] != rgb.shape[1] or mask.shape != actions.shape):
        raise ValueError("Expected aligned batch-size-one RGB [B,F,P,D] and actions [B,A,F,N,1]")
    frames = rgb.shape[1]
    if frames < 2 or max_future_steps < 1:
        raise ValueError("Teacher training needs at least one current and one future observation")
    if start is None:
        valid_anchors = mask[0, :, 1:, :, 0].bool().any(dim=0).any(dim=-1).nonzero().flatten() + 1
        if not valid_anchors.numel():
            raise ValueError("Sample has no valid future action targets")
        choice = int(torch.randint(valid_anchors.numel(), (), generator=generator))
        start = int(valid_anchors[choice])
    if end is None:
        horizon = int(torch.randint(1, min(max_future_steps, frames - start) + 1, (), generator=generator))
        end = start + horizon
    if not 1 <= start < end <= frames or end - start > max_future_steps:
        raise ValueError("Invalid future interval")
    # [B,A,F,N,1] -> [B,F*N,A], preserving within-anchor action order.
    selected_actions = actions[:, :, start:end, :, 0].permute(0, 2, 3, 1).flatten(1, 2)
    selected_mask = mask[:, :, start:end, :, 0].permute(0, 2, 3, 1).flatten(1, 2)
    return (rgb[:, start - 1], rgb[:, start:end], selected_actions, selected_mask), (start, end)


def main(argv=None):
    args = arguments(argv)
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from wan_va.configs import VA_CONFIGS
    from wan_va.configs.va_robotwin_cfg import load_robotwin_norm_stat
    from wan_va.dataset import MultiICLLeRobotLatentDataset
    from wan_va.dataset.rgb_feature_cache import validate_rgb_feature_cache
    from wan_va.modules.future_teacher import ActionFutureTeacher, save_future_teacher, load_future_teacher

    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Teacher training supports cpu and cuda devices")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA teacher training requires BF16 support")
    cfg = deepcopy(VA_CONFIGS["robotwin_subset12_rgb_b300"])
    source = Path(args.human_gen_root)
    cfg.dataset_path = str(source / "robotwin_data")
    cfg.action_stats_path = str(source / "robotwin_data/meta/action_stats.json")
    cfg.norm_stat = load_robotwin_norm_stat(cfg.action_stats_path)
    cfg.icl_manifest_path = str(source / "icl_configs/ICL_config_robotwin.json")
    cfg.human_latent_path = str(source / "human_latents/robotwin")
    cfg.rgb_feature_root = str(Path(args.feature_root))
    cfg.rank, cfg.init_worker, cfg.load_worker = 0, 1, 0
    if args.smoke_test:
        ids = args.sample_ids if args.sample_ids else json.loads(args.smoke_report.read_text())["sample_ids"]
        if not isinstance(ids, list) or not ids or not all(isinstance(x, str) and x for x in ids):
            raise ValueError("Smoke sample_ids must be a nonempty list of explicit RGB pair keys")
        cfg.rgb_smoke_sample_ids = ids
    elif hasattr(cfg, "rgb_smoke_sample_ids"):
        del cfg.rgb_smoke_sample_ids
    print("Validating original paired RGB cache and all 12 training tasks...", flush=True)
    preflight = validate_rgb_feature_cache(cfg)
    if preflight["tasks"] != 12:
        raise RuntimeError("Future teacher training must use exactly the subset12 training tasks")
    cfg.rgb_preprocessing = preflight["feature_spec"]["rgb_preprocessing"]
    dataset = MultiICLLeRobotLatentDataset(cfg, num_init_worker=1)
    generator = torch.Generator().manual_seed(args.seed)
    chunk_generator = torch.Generator().manual_seed(args.seed + 1)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, generator=generator)
    iterator = iter(loader)
    model = ActionFutureTeacher(
        feature_dim=cfg.rgb_transition_config["feature_dim"], action_dim=cfg.action_dim,
        num_tokens=args.num_tokens,
        max_future_steps=4, max_action_steps=4 * cfg.action_per_frame,
        max_patches=cfg.rgb_transition_config["max_patches"],
    ).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(.9, .95), weight_decay=.01)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output.with_name(args.output.name + ".metrics.jsonl")
    training = dict(
        stage="standalone_action_flow_future_teacher", recipe="initial_research_settings_not_paper_recipe",
        num_steps=args.num_steps, num_tokens=args.num_tokens,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        effective_batch_size=args.gradient_accumulation_steps, learning_rate=args.learning_rate,
        optimizer="AdamW", betas=[.9, .95], weight_decay=.01, max_grad_norm=args.max_grad_norm,
        seed=args.seed, batch_size=1, device=str(device), precision="bf16_autocast" if device.type == "cuda" else "float32",
        preflight=preflight, included_tasks=list(cfg.included_task_names),
        action_stats_path=cfg.action_stats_path, max_future_steps=4,
        future_encoder_inputs="future_rgb_only", heldout_tasks_used=False,
        smoke_sample_ids=getattr(cfg, "rgb_smoke_sample_ids", None),
    )
    seen_ids, step_metrics = set(), []
    started = time.monotonic()
    with metrics_path.open("x") as metrics:
        for step in range(1, args.num_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            total_loss, intervals = 0., []
            step_start = time.monotonic()
            for _ in range(args.gradient_accumulation_steps):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                cpu_inputs, interval = select_training_chunk(batch, generator=chunk_generator)
                inputs = tuple(t.to(device=device, non_blocking=False) for t in cpu_inputs)
                seen_ids.update(batch["icl_sample_id"])
                intervals.append(list(interval))
                amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
                with amp:
                    loss = model.action_loss(*inputs)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Nonfinite teacher action loss at optimizer step {step}")
                (loss / args.gradient_accumulation_steps).backward()
                total_loss += float(loss.detach()) / args.gradient_accumulation_steps
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, error_if_nonfinite=True)
            encoder_grad = model.visual_projection.weight.grad
            if encoder_grad is None or not bool(torch.isfinite(encoder_grad).all()) or not bool(encoder_grad.abs().sum() > 0):
                raise RuntimeError("Action loss did not train the future RGB encoder")
            optimizer.step()
            record = dict(step=step, action_flow_loss=total_loss, grad_norm=float(grad_norm),
                          future_encoder_grad_norm=float(encoder_grad.norm()),
                          step_seconds=time.monotonic() - step_start, intervals=intervals)
            metrics.write(json.dumps(record) + "\n")
            metrics.flush()
            step_metrics.append(record)
            print(json.dumps(record), flush=True)
    training.update(elapsed_seconds=time.monotonic() - started, seen_sample_ids=sorted(seen_ids),
                    first_step=step_metrics[0], final_step=step_metrics[-1])
    # Save only after every requested optimizer step succeeded, then exercise
    # the exact checksummed reload API used by policy training.
    metadata = save_future_teacher(model, args.output, feature_spec=preflight["feature_spec"],
                                   successful_steps=args.num_steps, smoke_test=args.smoke_test, training=training)
    restored, _ = load_future_teacher(args.output, device=device, allow_smoke=args.smoke_test,
                                      expected_feature_spec=preflight["feature_spec"])
    with torch.no_grad():
        original_target = model.eval().encode_future(inputs[1])
        restored_target = restored.encode_future(inputs[1])
    torch.testing.assert_close(original_target, restored_target, rtol=0, atol=0)
    report = dict(artifact=str(args.output), metrics=str(metrics_path), successful_steps=args.num_steps,
                  smoke_test=args.smoke_test, reload_exact=True, weights_sha256=metadata["weights_sha256"],
                  seen_sample_ids=sorted(seen_ids), elapsed_seconds=time.monotonic() - started)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    main()
