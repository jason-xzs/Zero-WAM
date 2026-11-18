#!/usr/bin/env python3
"""Controlled target-language sensitivity on a real base Zero-WAM checkpoint.

This is a denoiser intervention, not a rollout or instruction-following score.
Human video/cached caption, observed image, observed-video KV encoding text,
noise, timestep and ICL CFG are held fixed. Every condition clears all caches.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def tensor_digest(value):
    return hashlib.sha256(value.detach().float().cpu().numpy().tobytes()).hexdigest()


def difference(left, right):
    import torch
    delta = left.float() - right.float()
    rms = float(delta.square().mean().sqrt())
    ref = float(left.float().square().mean().sqrt())
    return {'max_abs': float(delta.abs().max()), 'rms': rms,
            'relative_rms': rms / max(ref, 1e-12), 'exact_equal': bool(torch.equal(left, right))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', default='/workspace/zero-wam/checkpoints/zero-wam-pretrain')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {'status': 'started', 'model_path': args.model_path, 'seed': args.seed,
              'task_success_evaluated': False, 'instruction_correctness_evaluated': False,
              'test': 'fixed_input_language_denoiser_sensitivity'}
    started = time.monotonic()
    import numpy as np
    import torch
    import torch.distributed as dist
    from types import SimpleNamespace
    from wan_va.configs import VA_CONFIGS
    from wan_va.distributed.util import init_distributed
    from wan_va.modules.icl_model import OBSERVATION_CACHE_TYPE, PREDICTION_CACHE_TYPE
    from wan_va.utils import init_logger
    from wan_va.wan_va_server import VA_Server
    from tools.smoke_rgb_inference import select_inputs
    source_args = SimpleNamespace(data_config='robotwin_subset12_rgb_train',
        human_gen_root='/workspace/zero-wam/datasets/HumanGen',
        feature_root='/workspace/zero-wam/datasets/rgb_transition_dinov2_base',
        sample_id='322:f3f143357177f925', second_sample_id=None)
    first, _, factual, observations, evidence = select_inputs(source_args)
    report['inputs'] = evidence
    human_payload = torch.load(first['human_latent_path'], map_location='cpu', weights_only=False, mmap=True)
    if not torch.is_tensor(human_payload.get('text_emb')):
        raise ValueError('A fixed cached human caption is required; prompt fallback would confound the probe')
    report['human_caption'] = {'text_field': human_payload.get('text'),
                             'embedding_sha256': tensor_digest(human_payload['text_emb'])}
    prompts = {'factual': factual,
               'paraphrase': 'Use the hammer to strike the block on the table.',
               'conflict': 'Leave the hammer on the table and do not strike the block.'}
    report['prompts'] = prompts
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    if 'MASTER_PORT' not in os.environ:
        with socket.socket() as temporary:
            temporary.bind(('127.0.0.1', 0))
            os.environ['MASTER_PORT'] = str(temporary.getsockname()[1])
    init_logger()
    init_distributed(1, 0, 0)
    try:
        torch.manual_seed(args.seed)
        cfg = deepcopy(VA_CONFIGS['robotwin'])
        cfg.model_path = args.model_path
        cfg.rank, cfg.local_rank, cfg.world_size = 0, 0, 1
        cfg.enable_offload = False
        cfg.save_root = str(args.output_dir)
        cfg.icl_guidance_scale = 5
        server = VA_Server(cfg)
        report['model_load_seconds'] = time.monotonic() - started
        # Empty-text encoding is itself nonzero: compare exact values instead
        # of interpreting a nonzero norm as evidence of caption information.
        if server.empty_text_emb is not None:
            report['human_caption']['comparison_to_empty'] = difference(
                server.empty_text_emb[0].cpu(), human_payload['text_emb'])
        report['rgb_transition_enabled'] = bool(getattr(server.transformer, 'enable_rgb_transition', False))
        rows, outputs, text_inputs = [], {}, {}
        reference_observation = None
        noise_video = noise_action = None
        with torch.no_grad():
            for text_scale in [-1, 1]:
                for name, prompt in prompts.items():
                    torch.manual_seed(args.seed)
                    server._reset(prompt=prompt, use_icl=True,
                        icl_latent_path=first['human_latent_path'],
                        video_guidance_scale=text_scale, icl_guidance_scale=5)
                    assert server.chunk_idx == 0
                    observed = server._encode_initial_obs({'obs': observations[:1]})
                    if reference_observation is None:
                        reference_observation = observed.detach().clone()
                    else:
                        torch.testing.assert_close(observed, reference_observation, atol=0, rtol=0)
                    server.readout_observation_latent = observed[:, :, -1:].detach().clone()
                    # A fixed real observation KV is installed with EMPTY target
                    # text under every condition. No generated future is cached.
                    cache_input = server._prepare_icl_stream(observed, 0, 'video', OBSERVATION_CACHE_TYPE)
                    empty = server.negative_prompt_embeds
                    count = cache_input['text_emb'].shape[1] // empty.shape[1]
                    cache_input['text_emb'] = empty.repeat(1, count, 1).to(server.dtype)
                    cache_digest = tensor_digest(cache_input['text_emb'])
                    server.transformer(cache_input, update_cache=1, cache_name=server.cache_name,
                                       mode='forward_latent_only')
                    if noise_video is None:
                        generator = torch.Generator(device=server.device).manual_seed(args.seed + 1000)
                        noise_video = torch.randn(1, 48, cfg.frame_chunk_size, server.latent_height,
                            server.latent_width, generator=generator, device=server.device, dtype=server.dtype)
                        noise_video[:, :, :1] = reference_observation[:, :, :1]
                        noise_action = torch.randn(1, cfg.action_dim, cfg.frame_chunk_size,
                            cfg.action_per_frame, 1, generator=generator, device=server.device, dtype=server.dtype)
                        noise_action[:, ~server.action_mask] = 0
                    values = {}
                    key = f'{text_scale}:{name}'
                    for mode, noise in [('video', noise_video), ('action', noise_action)]:
                        inputs = server._prepare_icl_stream(noise.clone(), 500, mode, PREDICTION_CACHE_TYPE)
                        if mode == 'video':
                            inputs['latent_res_lst']['timesteps'][::cfg.frame_chunk_size] = 0
                            text_inputs[key] = inputs['text_emb'].detach().float().cpu()
                        value = server.transformer(inputs, update_cache=0, cache_name=server.cache_name,
                            mode='forward_latent_only' if mode == 'video' else 'forward_action_only').detach().float().cpu()
                        if not torch.isfinite(value).all():
                            raise FloatingPointError('Nonfinite denoiser output')
                        values[mode] = value
                    outputs[key] = values
                    rows.append({'target_text_scale': text_scale, 'prompt_kind': name,
                        'target_text_sha256': tensor_digest(server.target_prompt_embeds),
                        'fixed_observation_text_sha256': cache_digest,
                        'cached_human_caption_sha256': report['human_caption']['embedding_sha256'],
                        'cache_counts': {str(k): int(v) for k,v in server.transformer.cache_counts().items()}})
                    print(json.dumps({'finished_condition': key}), flush=True)
        comparisons = []
        for scale in [-1, 1]:
            reference = f'{scale}:factual'
            for name in ['paraphrase', 'conflict']:
                key = f'{scale}:{name}'
                comparisons.append({'target_text_scale': scale, 'comparison': name,
                    'text_input': difference(text_inputs[reference], text_inputs[key]),
                    'video_denoiser': difference(outputs[reference]['video'], outputs[key]['video']),
                    'action_denoiser': difference(outputs[reference]['action'], outputs[key]['action'])})
        for item in comparisons:
            if item['target_text_scale'] == -1:
                assert all(item[field]['exact_equal'] for field in ['text_input','video_denoiser','action_denoiser'])
        report.update(status='passed', conditions=rows, comparisons=comparisons,
            fixed_protocol={'human_video_and_caption': True, 'observed_frame': True,
                'observed_kv_encoding_uses_empty_text': True, 'noise_and_timestep': True,
                'clear_cache_before_each_condition': True, 'timestep': 500, 'icl_cfg': 5,
                'no_generated_future_cached': True},
            limitations=['Sensitivity does not establish instruction correctness.',
                'No environment rollout or task success evaluation.',
                'Base checkpoint; this is not the remotely trained RGB checkpoint.'])
        np.savez_compressed(args.output_dir/'denoiser_outputs.npz', **{
            f'{key}:{mode}': value.numpy() for key, values in outputs.items() for mode, value in values.items()})
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
        for ax, field, title in zip(axes, ['video_denoiser','action_denoiser'], ['Video denoiser','Action denoiser']):
            labels = [f"text {'off' if c['target_text_scale']<0 else 'on'}\n{c['comparison']}" for c in comparisons]
            values = [c[field]['relative_rms'] for c in comparisons]
            ax.bar(labels, values, color=['#a3aab5','#a3aab5','#4675b9','#d78439'])
            ax.set_ylabel('RMS difference / factual RMS')
            ax.set_title(title)
            for i, val in enumerate(values): ax.text(i,val,f'{val:.3g}',ha='center',va='bottom',fontsize=8)
        fig.suptitle('Base Zero-WAM: fixed-input language sensitivity (not task success)')
        fig.savefig(args.output_dir/'language_sensitivity.svg')
        fig.savefig(args.output_dir/'language_sensitivity.png',dpi=160)
        plt.close(fig)
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['total_seconds'] = time.monotonic() - started
        (args.output_dir/'language_probe.json').write_text(json.dumps(report,indent=2)+'\n')
        if dist.is_initialized(): dist.destroy_process_group()


if __name__ == '__main__':
    main()
