#!/usr/bin/env python3
"""Inspect an actual role checkpoint without loading the full Wan transformer.

CPU FP32 computation uses the saved BF16 tensors promoted to FP32. Teacher
masks are used only for scoring/plots, never passed into the role predictions.
The six-step checkpoint is a plumbing test, not a trained semantic segmenter.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from safetensors import safe_open
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from wan_va.modules.rgb_transition import build_rgb_transition_conditioner
from wan_va.dataset.role_feature_cache import RoleManifest
from wan_va.dataset.rgb_feature_cache import load_feature_cache
from tools.prepare_rgb_transition_features import decode_rgb_frames


def subset_weights(root):
    index = json.loads((root/'diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
    groups = defaultdict(list)
    for name, shard in index.items():
        if name.startswith('rgb_transition.') or name in {'patch_embedding_mlp.weight', 'patch_embedding_mlp.bias'}:
            groups[shard].append(name)
    result = {}
    for shard, names in groups.items():
        with safe_open(root/shard, framework='pt', device='cpu') as handle:
            for name in names: result[name] = handle.get_tensor(name).float()
    return result


def score(prediction, masks, valid):
    values = []
    for p, m in zip(prediction[valid], masks[valid]):
        area = float(m.mean())
        mass = float((p*m).sum()/p.sum().clamp_min(1e-8))
        values.append({'mask_area_fraction':area, 'predicted_mass_in_mask':mass,
                       'mass_enrichment_over_uniform':mass/max(area,1e-8),
                       'probability_min':float(p.min()),'probability_max':float(p.max()),
                       'probability_std':float(p.std()),
                       'top1_inside':bool(m[p.argmax()] > .1)})
    return {'valid_region_frames':len(values), 'mean':{k:float(np.mean([r[k] for r in values])) for k in values[0]} if values else {}, 'frames':values}


def robot_patches(record, weights):
    main=Path(record['robot_latent_path'])
    arrays=[]
    for camera in ('observation.images.cam_high','observation.images.cam_left_wrist','observation.images.cam_right_wrist'):
        p=torch.load(main.parent.parent/camera/main.name, map_location='cpu', weights_only=False)
        arrays.append(rearrange(p['latent'], '(f h w) c -> 1 c f h w', f=int(p['latent_num_frames']),
                                h=int(p['latent_height']),w=int(p['latent_width'])))
    latent=torch.cat(arrays,-1).float()
    raw=rearrange(latent,'b c f (h p) (w q) -> (b f) (h w) (c p q)',p=2,q=2)
    state=F.linear(raw,weights['patch_embedding_mlp.weight'],weights['patch_embedding_mlp.bias'])
    h,w=latent.shape[-2]//2,latent.shape[-1]//2
    yy,xx=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
    return state,torch.stack((yy.flatten(),xx.flatten()),-1),(h,w//3)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--records',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--metrics',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    config=json.loads((args.checkpoint/'config.json').read_text())
    if config.get('patch_size') != [1, 2, 2]:
        raise ValueError('This diagnostic expects the current Wan patch size [1, 2, 2]')
    completion_path=args.checkpoint.parent/'COMPLETE.json'
    step=json.loads(completion_path.read_text())['step'] if completion_path.exists() else None
    weights=subset_weights(args.checkpoint)
    model=build_rgb_transition_conditioner(config['rgb_transition_config'],int(weights['patch_embedding_mlp.weight'].shape[0])).eval()
    model.load_state_dict({k.removeprefix('rgb_transition.'):v for k,v in weights.items() if k.startswith('rgb_transition.')},strict=True)
    records=json.loads(args.records.read_text())['records']
    labels=RoleManifest(args.manifest,sorted({r['task'] for r in records}))
    result={'checkpoint':str(args.checkpoint),'checkpoint_steps':step,'computation':'CPU FP32 on saved BF16 weights',
            'teacher_used_as_model_input':False,'role_labels':'SAM2 weak labels with Codex visual review; not human GT',
            'interpretation':'in-sample diagnostic on the two labelled smoke pairs, not held-out accuracy', 'pairs':[]}
    fig,axes=plt.subplots(len(records)*2,5,figsize=(15,3.2*len(records)*2),squeeze=False)
    with torch.no_grad():
        for ri,record in enumerate(records):
            human=load_feature_cache(record['human_cache_path'],'human');robot=load_feature_cache(record['robot_cache_path'],'robot')
            targets,_,_=labels.load(record['pair_key'],human,robot,record['human_cache_path'],record['robot_cache_path'])
            memory,_,logits=model._memory(human['features'][None].float())
            human_prob=logits[0].sigmoid()
            state,positions,grid=robot_patches(record,weights)
            robot_prob=[]
            for frame in state:
                _,_,logit=model._bind(memory,frame[None],positions)
                robot_prob.append(logit[0].sigmoid())
            robot_prob=torch.stack(robot_prob)
            teacher=F.interpolate(targets['robot_masks'].reshape(-1,1,16,16),size=grid,mode='area').reshape(robot_prob.shape)
            pair={'pair_key':record['pair_key'],'human':{},'robot':{}}
            for role,name in enumerate(('source','reference')):
                pair['human'][name]=score(human_prob[...,role,:],targets['human_masks'][...,role,:],targets['human_valid'][...,role])
                pair['robot'][name]=score(robot_prob[:,role],teacher[:,role],targets['robot_valid'][:,role])
            result['pairs'].append(pair)
            for kind,prob,mask,shape,row in [('human',human_prob[0,0],targets['human_masks'][0,0],(16,16),ri*2),
                                           ('robot',robot_prob[0],teacher[0],grid,ri*2+1)]:
                video=(human if kind=='human' else robot)['metadata']['source_video_path']
                axes[row,0].imshow(np.asarray(decode_rgb_frames(video,[0])[0]));axes[row,0].set_title(f'{record["label"]} {kind}: frame 0')
                for role in range(2):
                    axes[row,1+2*role].imshow(mask[role].reshape(shape),vmin=0,vmax=1,cmap='viridis')
                    axes[row,1+2*role].set_title(f'{("source","reference")[role]} weak target')
                    im=axes[row,2+2*role].imshow(prob[role].reshape(shape),vmin=0,vmax=1,cmap='viridis')
                    axes[row,2+2*role].set_title(f'predicted p, range {prob[role].min():.3f}-{prob[role].max():.3f}')
    for ax in axes.flat: ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle(f'C1 checkpoint step {step} | predicted masks use no teacher input | SAME probability scale [0,1]',fontsize=15)
    fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(args.output_dir/'c1_role_probabilities.png',dpi=150);plt.close(fig)
    rows=[json.loads(x) for x in args.metrics.read_text().splitlines()]
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for key in ['rgb_role_human','rgb_role_robot']: axes[0].plot([r['step'] for r in rows],[r[key] for r in rows],marker='o',label=key)
    for key in ['rgb_object_future','rgb_relation_future','rgb_donor_effect']: axes[1].plot([r['step'] for r in rows],[r[key] for r in rows],marker='o',label=key)
    axes[2].plot([r['step'] for r in rows],[r['rgb_gate'] for r in rows],marker='o',label='logged gate')
    for ax in axes: ax.set_xlabel('Optimizer step');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Actual training logs | changing minibatches and partial supervision | NOT convergence evidence\n'
                 'An auxiliary loss of zero can mean no labelled sample in that step, not a perfect prediction')
    fig.tight_layout();fig.savefig(args.output_dir/'c1_six_step_training.png',dpi=150);plt.close(fig)
    result['gate']=float(model.gate);result['training_metrics']=rows
    (args.output_dir/'c1_checkpoint_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'gate':result['gate'],'pairs':[{ 'pair_key':r['pair_key'],**{k:{n:s['mean'] for n,s in r[k].items()} for k in ['human','robot']}} for r in result['pairs']]},indent=2))


if __name__=='__main__': main()
