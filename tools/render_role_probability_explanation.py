#!/usr/bin/env python3
"""Large-label, fixed-scale view of real saved role probabilities (no HTML)."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from tools.analyze_role_checkpoint import subset_weights, robot_patches
from tools.prepare_rgb_transition_features import decode_rgb_frames
from wan_va.dataset.role_feature_cache import RoleManifest
from wan_va.dataset.rgb_feature_cache import load_feature_cache
from wan_va.modules.rgb_transition import build_rgb_transition_conditioner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--records', type=Path, default=Path('local/role_effect_smoke/records.json'))
    parser.add_argument('--manifest', type=Path, default=Path('local/role_effect_smoke/reviewed_v2/reviewed_manifest_v2.json'))
    parser.add_argument('--font', type=Path, default=Path('/tmp/role_svg_fonts/NotoSansCJKsc-Regular.otf'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    font_manager.fontManager.addfont(str(args.font))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=args.font).get_name(),
                         'font.size': 21, 'axes.unicode_minus': False})
    torch.set_num_threads(4)
    config = json.loads((args.checkpoint/'config.json').read_text())
    if config['patch_size'] != [1, 2, 2]:
        raise ValueError('This diagnostic expects patch size [1,2,2]')
    step = json.loads((args.checkpoint.parent/'COMPLETE.json').read_text())['step']
    weights = subset_weights(args.checkpoint)
    model = build_rgb_transition_conditioner(config['rgb_transition_config'],
        int(weights['patch_embedding_mlp.weight'].shape[0])).eval()
    model.load_state_dict({k.removeprefix('rgb_transition.'):v for k,v in weights.items()
                          if k.startswith('rgb_transition.')}, strict=True)
    records = json.loads(args.records.read_text())['records']
    record = records[0]
    human = load_feature_cache(record['human_cache_path'], 'human')
    robot = load_feature_cache(record['robot_cache_path'], 'robot')
    labels = RoleManifest(args.manifest, sorted({r['task'] for r in records}))
    targets, _, _ = labels.load(record['pair_key'], human, robot,
                                record['human_cache_path'], record['robot_cache_path'])
    with torch.no_grad():
        memory, _, logits = model._memory(human['features'][None].float())
        human_probability = logits[0,0,0,0].sigmoid().reshape(16,16)
        state, positions, grid = robot_patches(record, weights)
        _, _, robot_logits = model._bind(memory, state[:1], positions)
        robot_probability = robot_logits[0,0].sigmoid().reshape(grid)
    teacher_robot = F.interpolate(targets['robot_masks'][0,0].reshape(1,1,16,16),
                                   size=grid, mode='area')[0,0]
    fig, axes = plt.subplots(2,3,figsize=(14,9),layout='constrained')
    predictions = []
    for row, (domain, probability, teacher, cache) in enumerate([
        ('人类示范', human_probability, targets['human_masks'][0,0,0].reshape(16,16), human),
        ('机器人画面', robot_probability, teacher_robot, robot)]):
        frame = np.asarray(decode_rgb_frames(cache['metadata']['source_video_path'], [0])[0])
        axes[row,0].imshow(frame)
        axes[row,1].imshow(teacher, vmin=0, vmax=1, cmap='Greys', interpolation='nearest')
        im = axes[row,2].imshow(probability, vmin=0, vmax=1, cmap='Greys', interpolation='nearest')
        axes[row,0].set_ylabel(domain, fontsize=23)
        predictions.append({'domain':domain, 'min':float(probability.min()),
                            'max':float(probability.max()), 'mean':float(probability.mean())})
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    for ax, title in zip(axes[0], ['原始画面', '鼠标区域弱标签', '模型预测的区域概率']):
        ax.set_title(title, fontsize=23, pad=15)
    bar = fig.colorbar(im, ax=axes[:,2], shrink=.75, ticks=[0,.5,1], pad=.03)
    bar.set_label('0：不属于　1：属于', fontsize=21)
    fig.suptitle(f'C1 训练 {step} 步：预测仍接近均匀', fontsize=28)
    fig.savefig(args.output_dir/'7_六步角色预测.png', dpi=140)
    plt.close(fig)
    result = {'checkpoint':str(args.checkpoint),'checkpoint_step':step,'pair_key':record['pair_key'],
              'role':'mouse/source', 'frame':0, 'teacher_used_for_model_input':False,
              'shared_probability_color_limits':[0,1], 'predictions':predictions,
              'interpretation':'In-sample weak-label diagnostic; not convergence or action quality.'}
    (args.output_dir/'7_六步角色预测.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    main()
