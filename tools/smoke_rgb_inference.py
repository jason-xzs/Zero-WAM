#!/usr/bin/env python3
"""Exercise the real RGB-conditioned server on recorded RoboTwin observations.

This is an offline API/cache smoke test, not a rollout or a task-success score.
Default sampling is the official 50 video / 50 action steps. Use --steps 2 for
a short pipeline check. A model bundle must contain transformer, vae, tokenizer,
and text_encoder directories; the transformer must be a saved RGB checkpoint.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import socket
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", help="Bundle containing the saved RGB transformer")
    parser.add_argument("--feature-root", default="/workspace/zero-wam/datasets/rgb_transition_dinov2_base")
    parser.add_argument("--human-gen-root", default="/workspace/zero-wam/datasets/HumanGen")
    parser.add_argument("--data-config", default="robotwin_subset12_rgb_train")
    parser.add_argument("--sample-id", help="Explicit pair key from smoke_report.json, or unique sample_id")
    parser.add_argument("--second-sample-id", help="A different cached human demonstration for reset checks")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--video-steps", type=int)
    parser.add_argument("--action-steps", type=int)
    parser.add_argument("--second-chunk", action="store_true")
    parser.add_argument("--icl-guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--inspect-only", action="store_true", help="Validate and decode real inputs on CPU, without constructing the model")
    args = parser.parse_args()
    if not args.inspect_only and not args.model_path:
        parser.error("--model-path is required unless --inspect-only")
    if min(value for value in (args.steps, args.video_steps, args.action_steps) if value is not None) < 1:
        parser.error("Denoising steps must be positive")
    if args.icl_guidance_scale <= 1:
        parser.error("The CFG isolation check requires --icl-guidance-scale > 1")
    return args


def select_inputs(args):
    import numpy as np
    from wan_va.configs import VA_CONFIGS
    from wan_va.dataset.rgb_feature_cache import (
        expected_rgb_records, human_relative_path, load_feature_cache, rgb_pair_key,
    )
    from tools.prepare_rgb_transition_features import decode_rgb_frames

    cfg = deepcopy(VA_CONFIGS[args.data_config])
    source = Path(args.human_gen_root)
    cfg.dataset_path = str(source / "robotwin_data")
    cfg.icl_manifest_path = str(source / "icl_configs/ICL_config_robotwin.json")
    cfg.human_latent_path = str(source / "human_latents/robotwin")
    cfg.rgb_feature_root = args.feature_root
    preparation = Path(args.feature_root) / "smoke_report.json"
    if preparation.is_file():
        cfg.rgb_smoke_sample_ids = json.loads(preparation.read_text())["sample_ids"]
    records = [record for record in expected_rgb_records(cfg)
               if record["human_cache_path"].is_file() and record["robot_cache_path"].is_file()]

    def choose(key, candidates):
        matches = candidates if key is None else [
            item for item in candidates
            if key in {rgb_pair_key(item["sample"]), str(item["sample"]["sample_id"])}
        ]
        if not matches or (key is not None and len(matches) != 1):
            raise ValueError(f"Expected a unique available RGB pair for {key!r}; found {len(matches)}")
        return matches[0]

    first = choose(args.sample_id, records)
    second = choose(args.second_sample_id, [
        item for item in records if item["human_cache_path"] != first["human_cache_path"]
    ])
    evidence = []
    for record in (first, second):
        sample = record["sample"]
        human = load_feature_cache(record["human_cache_path"], "human", expected_sample_id=sample["sample_id"])
        robot = load_feature_cache(record["robot_cache_path"], "robot", expected_sample_id=sample["sample_id"])
        for key in ("encoder_id", "encoder_revision", "preprocessing"):
            if human["metadata"][key] != robot["metadata"][key]:
                raise ValueError(f"Human/robot feature-space mismatch: {key}")
        latent = Path(cfg.human_latent_path) / human_relative_path(sample["human_video_path"])
        if not latent.is_file():
            raise FileNotFoundError(latent)
        record["human_latent_path"] = str(latent)
        record["robot_metadata"] = robot["metadata"]
        evidence.append({
            "pair_key": rgb_pair_key(sample), "task": record["task_name"],
            "human_latent_path": str(latent), "human_feature_path": str(record["human_cache_path"]),
            "human_feature_shape": list(human["features"].shape),
            "robot_feature_path": str(record["robot_cache_path"]),
            "encoder_id": human["metadata"]["encoder_id"],
            "encoder_revision": human["metadata"]["encoder_revision"],
        })
    metadata = first["robot_metadata"]
    frame_ids = [int(value) for value in metadata["source_latent_frame_ids"][:5]]
    if len(frame_ids) != 5:
        raise ValueError("Need five real raw frames: initial observation plus one four-frame cache update")
    # Cached metadata records where features were originally extracted. Resolve
    # runtime input through the current dataset record after moving servers.
    main_video = Path(first["robot_video_path"])
    camera_paths = {camera: main_video.parent.parent / camera / main_video.name for camera in cfg.obs_cam_keys}
    decoded = {}
    for camera, path in camera_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        decoded[camera] = [np.asarray(frame, dtype=np.uint8).copy()
                           for frame in decode_rgb_frames(path, frame_ids)]
    observations = [{camera: decoded[camera][index] for camera in cfg.obs_cam_keys}
                    for index in range(len(frame_ids))]
    prompt = first["sample"].get("robot_task_name") or first["sample"].get("robot_dense_caption")
    if not prompt:
        raise ValueError("The paired manifest has no target robot task description")
    return first, second, prompt, observations, {
        "pairs": evidence, "target_prompt": prompt, "raw_frame_ids": frame_ids,
        "camera_videos": {key: str(value) for key, value in camera_paths.items()},
        "camera_frame_shapes": {key: list(value[0].shape) for key, value in decoded.items()},
        "observation_source": "recorded dataset frames; no actions executed in an environment",
    }


def read_saved_gate(transformer_path):
    """Read the serialized gate itself, before loading or casting the model."""
    from safetensors import safe_open
    import torch

    key = "rgb_transition.gate"
    indexes = list(transformer_path.glob("*.safetensors.index.json"))
    if indexes:
        if len(indexes) != 1:
            raise ValueError(f"Ambiguous safetensors indexes in {transformer_path}")
        weight_map = json.loads(indexes[0].read_text())["weight_map"]
        filename = transformer_path / weight_map[key]
    else:
        files = list(transformer_path.glob("*.safetensors"))
        if len(files) != 1:
            raise ValueError(f"Need a complete safetensors checkpoint in {transformer_path}")
        filename = files[0]
    with safe_open(filename, framework="pt", device="cpu") as saved:
        gate = saved.get_tensor(key)
    if gate.numel() != 1 or not torch.isfinite(gate).all() or gate.item() == 0:
        raise AssertionError("The trained checkpoint must contain a finite, nonzero RGB gate")
    return gate


def ensure_probe_observation(server, observations):
    """Use the real initial observation before a readout-only offline probe.

    Normal deployment installs this in ``_infer_icl``. The smoke test also runs
    probes *before* that call, so it must explicitly supply the same observation;
    an unobserved generated video or cached robot future would be invalid here.
    """
    if not getattr(server.transformer, "uses_rgb_readout", False):
        return None
    if server.readout_observation_latent is None:
        latent = server._encode_initial_obs({"obs": observations[:1]})
        server.readout_observation_latent = latent[:, :, -1:].detach().clone()
    return server.readout_observation_latent


def validate_readout_probe(mode, first, changed, global_tokens):
    """A controlled intervention tests the readout mechanism, not task quality."""
    import torch

    if first.shape != changed.shape or first.ndim != 3:
        raise AssertionError("Readout probe output shapes differ")
    if not torch.isfinite(first).all() or not torch.isfinite(changed).all():
        raise AssertionError("Nonfinite readout probe output")
    delta = first.float() - changed.float()
    result = {"shape": list(first.shape), "max_abs": float(delta.abs().max())}
    if mode == "static":
        torch.testing.assert_close(first, changed, atol=0, rtol=0)
        result["state_invariant"] = True
    elif mode == "global_local":
        torch.testing.assert_close(first[:, :global_tokens], changed[:, :global_tokens], atol=0, rtol=0)
        if torch.equal(first[:, global_tokens:], changed[:, global_tokens:]):
            raise AssertionError("Local readout ignored a state intervention")
        result.update(global_invariant=True, local_state_sensitive=True)
    elif mode == "state":
        if torch.equal(first, changed):
            raise AssertionError("State readout ignored a state intervention")
        result["state_sensitive"] = True
    else:
        raise ValueError(f"Unsupported readout probe mode: {mode}")
    return result


def run_gpu(args, first, second, prompt, observations, report):
    import numpy as np
    import torch
    import torch.distributed as dist
    from wan_va.configs import VA_CONFIGS
    from wan_va.distributed.util import init_distributed
    from wan_va.modules.icl_model import ICL_CACHE_TYPE, OBSERVATION_CACHE_TYPE, PREDICTION_CACHE_TYPE
    from wan_va.utils import init_logger
    from wan_va.wan_va_server import VA_Server

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This script is a single-GPU smoke test")
    if not torch.cuda.is_available():
        raise RuntimeError("A visible CUDA GPU is required")
    model_root = Path(args.model_path)
    for component in ("transformer", "vae", "tokenizer", "text_encoder"):
        if not (model_root / component).is_dir():
            raise FileNotFoundError(model_root / component)
    saved_config = json.loads((model_root / "transformer/config.json").read_text())
    if not saved_config.get("enable_rgb_transition"):
        raise ValueError("The supplied bundle is not a saved RGB-transition checkpoint")
    feature_spec = saved_config.get("rgb_transition_config") or {}
    for key in ("rgb_encoder_id", "rgb_encoder_revision", "rgb_preprocessing"):
        if not feature_spec.get(key):
            raise ValueError(f"RGB checkpoint is missing pinned feature metadata: {key}")
    saved_gate = read_saved_gate(model_root / "transformer")
    report["saved_rgb_gate"] = {"value": float(saved_gate.float().item()),
                                "dtype": str(saved_gate.dtype), "finite": True, "nonzero": True}

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if "MASTER_PORT" not in os.environ:
        with socket.socket() as temporary:
            temporary.bind(("127.0.0.1", 0))
            os.environ["MASTER_PORT"] = str(temporary.getsockname()[1])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    init_logger()
    init_distributed(1, local_rank, 0)
    hooks = []
    try:
        torch.manual_seed(args.seed)
        cfg = deepcopy(VA_CONFIGS["robotwin"])
        cfg.model_path, cfg.rgb_feature_root = str(model_root), args.feature_root
        cfg.rank, cfg.local_rank, cfg.world_size = 0, local_rank, 1
        cfg.enable_offload = False
        cfg.num_inference_steps = args.video_steps or args.steps
        cfg.action_num_inference_steps = args.action_steps or args.steps
        cfg.icl_guidance_scale, cfg.guidance_scale = args.icl_guidance_scale, -1
        cfg.save_root = str(args.output_json.parent if args.output_json else model_root)
        report["sampling"] = {"video_steps": cfg.num_inference_steps,
                              "action_steps": cfg.action_num_inference_steps,
                              "icl_guidance_scale": cfg.icl_guidance_scale, "target_text_cfg": -1}
        started = time.monotonic()
        server = VA_Server(cfg)
        report["model_load_seconds"] = time.monotonic() - started
        loading = server.transformer.rgb_transition_loading_info
        if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError(f"Saved RGB checkpoint failed strict load: {loading}")
        report["checkpoint_loading_info"] = loading
        gate = server.transformer.rgb_transition.gate
        gate = gate.full_tensor() if hasattr(gate, "full_tensor") else gate
        if not torch.isfinite(gate).all() or gate.item() == 0:
            raise AssertionError("Loaded trained RGB fusion gate must be finite and nonzero")
        torch.testing.assert_close(gate.detach().cpu(), saved_gate.to(dtype=gate.dtype), atol=0, rtol=0)
        report["rgb_gate"] = float(gate.detach().float().cpu().item())
        report["checkpoint_transformer_path"] = str((model_root / "transformer").resolve())
        report["checkpoint_feature_spec"] = feature_spec
        calls = []
        readout_calls = []
        captured_readouts = []
        capture_readout = False
        uses_readout = bool(server.transformer.uses_rgb_readout)

        def rgb_call(_module, inputs, kwargs):
            if kwargs.get("compute_aux", False) or kwargs.get("robot_features") is not None:
                raise AssertionError("Inference must not use robot future features or auxiliary targets")
            calls.append({"human_feature_shape": list(inputs[0].shape)})

        hooks.append(server.transformer.rgb_transition.register_forward_pre_hook(rgb_call, with_kwargs=True))
        if uses_readout:
            def readout_call(_module, inputs, output):
                readout_calls.append({"memory_shape": list(inputs[0].shape),
                                      "state_shape": list(inputs[1].shape),
                                      "output_shape": list(output.shape)})
                if capture_readout:
                    captured_readouts.append(output.detach().float().cpu().clone())
            hooks.append(server.transformer.rgb_transition.readout.register_forward_hook(readout_call))
            report["readout"] = {"mode": server.transformer.rgb_transition.readout_mode,
                                 "num_tokens": server.transformer.rgb_transition.readout.num_tokens,
                                 "global_tokens": server.transformer.rgb_transition.readout.global_tokens,
                                 "human_wan_context_retained": True}

        def counts():
            return {str(key): int(value) for key, value in server.transformer.cache_counts().items()}

        def reset(record, use_icl=True):
            server._reset(prompt=prompt, use_icl=use_icl,
                          icl_latent_path=record["human_latent_path"] if use_icl else "",
                          video_guidance_scale=-1, icl_guidance_scale=args.icl_guidance_scale)
            if server.chunk_idx != 0 or server.last_predicted_actions is not None:
                raise AssertionError("Reset retained robot history")
            if uses_readout:
                if server.readout_observation_latent is not None:
                    raise AssertionError("Reset retained readout observation")
                memory = server.transformer.rgb_readout_memory
                if use_icl and server.transformer.enable_rgb_transition:
                    assert memory is not None, "Reset did not install human readout memory"
                else:
                    assert memory is None, "Reset retained human readout memory"
            if use_icl:
                assert server.transformer.cache_counts().get(ICL_CACHE_TYPE, 0) > 0
                assert server.transformer.cache_counts().get(OBSERVATION_CACHE_TYPE, 0) == 0
                assert server.transformer.cache_counts().get(PREDICTION_CACHE_TYPE, 0) == 0
            else:
                assert not server.transformer.cache_counts()

        def check_actions(actions):
            expected = (len(cfg.used_action_channel_ids), cfg.frame_chunk_size, cfg.action_per_frame)
            if tuple(actions.shape) != expected or not np.isfinite(actions).all():
                raise AssertionError(f"Invalid action output: {actions.shape}, expected {expected}")
            return {"shape": list(actions.shape), "finite": True,
                    "min": float(actions.min()), "max": float(actions.max())}

        # Match the production server. FSDP2 unsharding preserves version
        # counters and is incompatible with inference-mode tensor allocation.
        with torch.no_grad():
            reset(first)
            report["first_reset_cache"] = counts()
            assert len(calls) == 1
            # Identical model input/text for the two human contexts and no-ICL
            # control. No probe is written into the persistent KV cache.
            generator = torch.Generator(device=server.device).manual_seed(args.seed + 1000)
            probe = torch.randn(1, 48, cfg.frame_chunk_size, server.latent_height, server.latent_width,
                                generator=generator, device=server.device, dtype=server.dtype)

            def predict_probe(force_cfg=None):
                ensure_probe_observation(server, observations)
                previous = server.use_icl_cfg
                if force_cfg is not None:
                    server.use_icl_cfg = force_cfg
                try:
                    inputs = server._prepare_icl_stream(probe.clone(), 500, "video", PREDICTION_CACHE_TYPE)
                finally:
                    server.use_icl_cfg = previous
                assert "rgb_transition" not in inputs
                result = server.transformer(inputs, update_cache=0, cache_name=server.cache_name,
                                            mode="forward_latent_only").float().cpu()
                if not torch.isfinite(result).all():
                    raise AssertionError("A CFG/no-ICL probe produced nonfinite output")
                return result

            def difference(first, other):
                delta = first - other
                return {"max_abs": float(delta.abs().max()), "mean_abs": float(delta.abs().mean()),
                        "rms": float(delta.square().mean().sqrt()),
                        "reference_rms": float(first.square().mean().sqrt()),
                        "allclose_1e2": torch.allclose(first, other, atol=1e-2, rtol=1e-2)}

            capture_readout = uses_readout
            conditional_a, negative_a = predict_probe().chunk(2, dim=1)
            if uses_readout:
                assert len(captured_readouts) == 1
                original_state = server.readout_observation_latent
                # Keep the main policy query, human memory and KV cache fixed.
                # This synthetic control changes only the new readout input.
                perturbation = torch.linspace(-0.25, 0.25, original_state.numel(),
                                              device=original_state.device).reshape_as(original_state)
                server.readout_observation_latent = original_state + perturbation.to(original_state)
                try:
                    conditional_changed, negative_changed = predict_probe().chunk(2, dim=1)
                finally:
                    server.readout_observation_latent = original_state
                assert len(captured_readouts) == 2
                readout = server.transformer.rgb_transition.readout
                report["readout"]["state_intervention"] = {
                    **validate_readout_probe(readout.mode, *captured_readouts, readout.global_tokens),
                    "input": "synthetic perturbation only to readout state; main query and caches fixed",
                    "conditional_output_change": difference(conditional_a, conditional_changed),
                    "negative_output_change": difference(negative_a, negative_changed),
                }
                torch.testing.assert_close(negative_a, negative_changed, atol=1e-2, rtol=1e-2)
            capture_readout = False
            started = time.monotonic()
            readouts_before_inference = len(readout_calls)
            actions = server._infer_icl({"obs": observations[:1]})
            torch.cuda.synchronize()
            report["first_chunk"] = {**check_actions(actions), "seconds": time.monotonic() - started,
                                      "cache": counts()}
            if uses_readout:
                assert len(readout_calls) > readouts_before_inference, "Policy skipped demo readout"
                report["first_chunk"]["readout_calls"] = len(readout_calls) - readouts_before_inference
                observed_before_update = server.readout_observation_latent.detach().clone()
                memory_before_update = server.transformer.rgb_readout_memory
                readouts_before_update = len(readout_calls)
            server._compute_icl_kv_cache({"obs": observations[1:]})
            observed_counts = server.transformer.cache_counts()
            assert observed_counts.get(ICL_CACHE_TYPE, 0) > 0
            assert observed_counts.get(OBSERVATION_CACHE_TYPE, 0) > 0
            assert observed_counts.get(PREDICTION_CACHE_TYPE, 0) == 0
            assert server.chunk_idx == 1 and len(calls) == 1
            report["real_observation_cache_update"] = {"cache": counts(), "chunk_idx": server.chunk_idx}
            if uses_readout:
                assert len(readout_calls) > readouts_before_update
                assert server.transformer.rgb_readout_memory is memory_before_update
                report["real_observation_cache_update"].update(
                    readout_calls=len(readout_calls) - readouts_before_update,
                    human_memory_preserved=True,
                    observed_state_change=difference(observed_before_update.float().cpu(),
                                                     server.readout_observation_latent.float().cpu()))
            if args.second_chunk:
                started = time.monotonic()
                actions = server._infer_icl({"obs": observations[1:]})
                torch.cuda.synchronize()
                report["second_chunk"] = {**check_actions(actions), "seconds": time.monotonic() - started}
            reset(second)
            assert len(calls) == 2
            report["second_human_reset_cache"] = counts()
            conditional_b, negative_b = predict_probe().chunk(2, dim=1)
            checks = report["cfg_and_reset"] = {
                "rgb_conditioner_calls": calls,
                "negative_human_change": difference(negative_a, negative_b),
                "conditional_human_change": difference(conditional_a, conditional_b),
                "bf16_assert_atol": 1e-2, "bf16_assert_rtol": 1e-2,
            }
            reset(first, use_icl=False)
            no_icl = predict_probe()
            packed_no_icl = predict_probe(force_cfg=True).chunk(2, dim=1)[1]
            checks.update(no_icl_cache=counts(),
                          negative_vs_no_cache_same_packing=difference(negative_a, packed_no_icl),
                          negative_vs_single_stream=difference(negative_a, no_icl))
            assert len(calls) == 2
            # A single-stream query changes matrix/kernel shapes throughout
            # the BF16 backbone. Diagnose that difference using this identical
            # trained checkpoint with RGB fusion disabled, rather than weakening
            # the same-packing isolation invariant or assuming equivalence.
            server.transformer.enable_rgb_transition = False
            try:
                reset(first)
                baseline_conditional, baseline_negative = predict_probe().chunk(2, dim=1)
                reset(first, use_icl=False)
                baseline_single = predict_probe()
                baseline_packed = predict_probe(force_cfg=True).chunk(2, dim=1)[1]
            finally:
                server.transformer.enable_rgb_transition = True
            rgb_delta = negative_a - no_icl
            baseline_delta = baseline_negative - baseline_single
            checks["rgb_disabled_same_checkpoint"] = {
                "negative_vs_rgb_enabled": difference(baseline_negative, negative_a),
                "conditional_vs_rgb_enabled": difference(baseline_conditional, conditional_a),
                "negative_vs_no_cache_same_packing": difference(baseline_negative, baseline_packed),
                "negative_vs_single_stream": difference(baseline_negative, baseline_single),
                "single_stream_delta_vs_rgb_enabled": difference(baseline_delta, rgb_delta),
                "rgb_conditioner_calls_unchanged": len(calls) == 2,
            }
            # Keep all raw controls in the report even if an invariant fails.
            torch.testing.assert_close(negative_a, negative_b, atol=1e-2, rtol=1e-2)
            torch.testing.assert_close(negative_a, packed_no_icl, atol=1e-2, rtol=1e-2)
            torch.testing.assert_close(baseline_negative, baseline_packed, atol=1e-2, rtol=1e-2)
            torch.testing.assert_close(baseline_negative, negative_a, atol=1e-2, rtol=1e-2)
            torch.testing.assert_close(baseline_delta, rgb_delta, atol=1e-2, rtol=1e-2)
            assert len(calls) == 2
            if uses_readout:
                report["readout"]["forward_calls"] = readout_calls
        report["peak_gpu_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
    finally:
        for hook in hooks:
            hook.remove()
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    args = arguments()
    report = {"status": "started", "test": "real_rgb_server_offline_smoke", "model_path": args.model_path,
              "feature_root": args.feature_root, "seed": args.seed, "task_success_evaluated": False}
    started = time.monotonic()
    try:
        first, second, prompt, observations, provenance = select_inputs(args)
        report["inputs"] = provenance
        if args.inspect_only:
            report["status"] = "inputs_validated_cpu_only"
        else:
            run_gpu(args, first, second, prompt, observations, report)
            report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["total_seconds"] = time.monotonic() - started
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
