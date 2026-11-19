#!/usr/bin/env python3
"""Bounded in-sample learning diagnostic on real, reviewed role pairs.

Fresh role/RGB parameters; frozen released Wan patch projection and cached DINO.
This intentionally overfits the selected pairs to detect disconnected or
unlearnable heads. It is not policy training, held-out evaluation, or evidence
of generalization. No teacher masks enter the deployed memory/readout path.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.nn import functional as F

from tools.analyze_role_checkpoint import robot_patches, subset_weights
from wan_va.configs.va_robotwin_subset12_role_effect_cfg import make_role_effect_config
from wan_va.dataset.rgb_feature_cache import load_feature_cache
from wan_va.dataset.role_feature_cache import RoleManifest, file_sha256
from wan_va.modules.role_effect import ROLE_LOSS_KEYS, RoleEffectConditioner


def zero_change_losses(robot, targets, chunk_size):
    """Independent no-change predictor, with the same causal supervision mask.

    ``robot`` has already received the model's per-patch normalization.
    Compare against this baseline, not just a randomly initialized decoder.
    """
    masks = targets['robot_masks'].float()
    weights = masks / masks.sum(-1, keepdim=True).clamp_min(1e-6)
    pooled = (weights[..., None] * robot[:, :, None].float()).sum(-2)
    side = int(robot.shape[-2] ** 0.5)
    axis = torch.linspace(-1, 1, side, device=robot.device)
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    xy = torch.stack([xx.flatten(), yy.flatten()], -1)
    centers = (weights[..., None] * xy).sum(-2)
    relation = centers[..., 0, :] - centers[..., 1, :]
    valid, ids = targets['robot_valid'], targets['robot_entity_ids']
    values = {'object_future': [], 'relation_future': []}
    for start in range(0, robot.shape[1], chunk_size):
        observed = max(start - 1, 0)
        future = slice(max(start, observed + 1), min(start + chunk_size, robot.shape[1]))
        usable = (valid[:, future] & valid[:, observed:observed + 1]
                  & (ids[:, future] == ids[:, observed:observed + 1]) & (ids[:, future] >= 0))
        delta = pooled[:, future] - pooled[:, observed:observed + 1]
        relative = relation[:, future] - relation[:, observed:observed + 1]
        for key, target, mask in [('object_future', delta, usable),
                                  ('relation_future', relative, usable.all(-1))]:
            error = F.smooth_l1_loss(torch.zeros_like(target), target, reduction='none').mean(-1)
            values[key].append((error * mask).sum() / mask.sum().clamp_min(1))
    return {key: float(torch.stack(items).mean()) for key, items in values.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-transformer', type=Path, required=True)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--chunk-size', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    args = parser.parse_args()
    if not 1 <= args.steps <= 1000 or args.chunk_size < 1:
        parser.error('Use 1..1000 steps and a positive chunk size')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / 'learning.json'
    if report_path.exists():
        raise FileExistsError(f'Preserve previous results: {report_path}')
    torch.manual_seed(42)
    began = time.monotonic()
    base_config = json.loads((args.base_transformer / 'config.json').read_text())
    if base_config.get('patch_size') != [1, 2, 2]:
        raise ValueError('This diagnostic requires the actual Wan [1,2,2] patch layout')
    if base_config.get('enable_rgb_transition', False):
        raise ValueError('Use the original released base weights for this fresh-initialization diagnostic')
    weights = subset_weights(args.base_transformer)
    rgb_config = dict(make_role_effect_config().rgb_transition_config)
    model = RoleEffectConditioner(rgb_config, weights['patch_embedding_mlp.weight'].shape[0]).to(args.device)
    records = json.loads(args.records.read_text())['records']
    manifest = RoleManifest(args.manifest, sorted({r['task'] for r in records}))
    samples = []
    for record in records:
        hp, rp = record['human_cache_path'], record['robot_cache_path']
        human, robot = load_feature_cache(hp, 'human'), load_feature_cache(rp, 'robot')
        targets, donor, donor_valid = manifest.load(record['pair_key'], human, robot, hp, rp)
        state, positions, _ = robot_patches(record, weights)
        samples.append(dict(
            human=human['features'][None].to(args.device).float(),
            robot=robot['features'][None].to(args.device).float(),
            targets={k: v[None].to(args.device) for k, v in targets.items()},
            donor=donor[None].to(args.device).float(), donor_valid=donor_valid[None].to(args.device),
            state=state.detach().to(args.device), positions=positions.to(args.device),
        ))
    if not samples:
        raise ValueError('Select at least one real reviewed pair')

    def sample_loss(sample):
        memory, losses, context = model(
            sample['human'], sample['robot'], True, role_targets=sample['targets'],
            donor_human_features=sample['donor'], donor_valid=sample['donor_valid'], return_aux_context=True)
        chunk_losses = []
        frames = sample['state'].shape[0]
        for start in range(0, frames, args.chunk_size):
            observed = max(start - 1, 0)
            _, values = model.training_read_tokens(
                memory, sample['state'][observed:observed + 1], sample['positions'],
                aux_context=context, observed_index=observed,
                future_indices=list(range(max(start, observed + 1), min(start + args.chunk_size, frames))))
            chunk_losses.append(values)
        for key in chunk_losses[0]:
            value = torch.stack([chunk[key] for chunk in chunk_losses]).mean()
            losses[key] = value
            losses['total'] = losses['total'] + model.role_loss_weights[key] * value
        return losses

    @torch.no_grad()
    def evaluate():
        model.eval()
        totals = {}
        for sample in samples:
            for key, value in sample_loss(sample).items():
                totals[key] = totals.get(key, 0) + float(value) / len(samples)
        # Isolate the new effect slots, so original WALA tokens cannot account
        # for sensitivity to temporal order or a different video.
        human = model._prepare_features(samples[0]['human'])
        effect = model._human_role_memory(human)[1]
        permutation = list(range(human.shape[2]))
        if len(permutation) >= 4:
            permutation[1:-1] = reversed(permutation[1:-1])
        shuffled = model._human_role_memory(human[:, :, permutation])[1]
        totals['effect_order_rms_difference'] = float((effect - shuffled).square().mean().sqrt())
        return totals

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    initial = evaluate()
    baselines = [zero_change_losses(model._prepare_features(s['robot']), s['targets'], args.chunk_size)
                 for s in samples]
    zero_baseline = {key: sum(b[key] for b in baselines) / len(baselines) for key in baselines[0]}
    history = [{'step': 0, **initial}]
    gradient_norms = {}
    required_gradients = (
        'role_patch_projection.weight', 'effect_projection.weight', 'effect_time_embedding',
        'robot_query_projection.weight', 'bound_effect_attention.in_proj_weight',
        'object_decoder.3.weight', 'relation_decoder.3.weight',
    )
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for sample in samples:
            losses = sample_loss(sample)
            if not all(torch.isfinite(value).all() for value in losses.values()):
                raise FloatingPointError(f'Nonfinite loss at diagnostic step {step}')
            (losses['total'] / len(samples)).backward()
        if step == 1:
            for name, parameter in model.named_parameters():
                if name in required_gradients:
                    gradient_norms[name] = float(parameter.grad.norm()) if parameter.grad is not None else 0
            if set(gradient_norms) != set(required_gradients) or not all(v > 0 for v in gradient_norms.values()):
                raise AssertionError(f'Disconnected role/effect path: {gradient_norms}')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % 25 == 0 or step == args.steps:
            row = {'step': step, **evaluate()}
            history.append(row)
            print(json.dumps({k: row[k] for k in ('step', *ROLE_LOSS_KEYS)}), flush=True)
    final = history[-1]
    decreased = {key: final[key] < initial[key] for key in ROLE_LOSS_KEYS}
    beats_zero = {key: final[key] < value for key, value in zero_baseline.items()}
    report = dict(
        status='passed' if all(decreased.values()) and all(beats_zero.values()) else 'failed',
        diagnostic='in-sample optimization only; no policy or generalization evaluation',
        backbone_frozen=True, initialization='fresh role/RGB module; released Wan patch projection',
        precision='FP32', seed=42, steps=args.steps, learning_rate=args.learning_rate,
        warmup_steps=0, chunk_size=args.chunk_size,
        pairs=[r['pair_key'] for r in records], manifest_sha256=file_sha256(args.manifest),
        rgb_config=rgb_config, initial=initial, final=final,
        all_role_losses_decreased=decreased, first_step_gradient_norms=gradient_norms,
        zero_change_baseline=zero_baseline, future_predictions_beat_zero_change=beats_zero,
        history=history, elapsed_seconds=time.monotonic() - began,
    )
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    torch.save({'config': rgb_config, 'state_dict': model.cpu().state_dict(),
                'diagnostic_only': True, 'steps': args.steps}, args.output_dir / 'head_diagnostic.pt')
    if report['status'] != 'passed':
        raise AssertionError(f'An auxiliary head did not improve; inspect {report_path}')
    print(json.dumps({'status': report['status'], 'report': str(report_path)}), flush=True)


if __name__ == '__main__':
    main()
