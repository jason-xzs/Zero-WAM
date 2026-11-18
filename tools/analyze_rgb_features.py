#!/usr/bin/env python3
"""Frozen-DINO diagnostics on real paired videos; no policy-quality claim.

Fits ONE PCA over all selected human/robot frames, compares first-frame feature
change maps, and uses explicitly supervised human regions as correspondence
queries. Every valid robot role/frame contributes to the retrieval report.
SAM2/Codex weak masks are evaluation references, not ground truth. No policy,
robot action, task-language encoder, or transition-adapter checkpoint is used.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import av
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wan_va.dataset.role_feature_cache import file_sha256


def decode(path, frame_ids):
    wanted = set(int(x) for x in frame_ids)
    result = {}
    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in wanted:
                result[i] = np.asarray(frame.to_image().resize((224, 224), Image.Resampling.BICUBIC))
            if len(result) == len(wanted):
                break
    if set(result) != wanted:
        raise ValueError(f'Missing video frames: {path}: {wanted-set(result)}')
    return np.stack([result[int(i)] for i in frame_ids])


def normalized(features):
    # Matches the non-affine patch normalization used by RGBTransitionConditioner.
    x = F.layer_norm(features.float(), (features.shape[-1],), eps=1e-6)
    return x.numpy(), F.normalize(x, dim=-1).numpy()


def pool(features, weights):
    value = (features * weights[:, None]).sum(0) / max(float(weights.sum()), 1e-8)
    return value / max(float(np.linalg.norm(value)), 1e-8)


def soft_auc(score, mask):
    """AUROC with patch coverage as positive mass, 1-coverage as negative mass."""
    positive, negative = mask.astype(np.float64), 1-mask.astype(np.float64)
    comparison = (score[:, None] > score[None]).astype(float)
    comparison += .5 * (score[:, None] == score[None])
    denom = positive.sum()*negative.sum()
    return float((comparison*positive[:, None]*negative[None]).sum()/denom)


def retrieval_metrics(score, mask):
    k = math.ceil(len(score)*.05)
    top = np.argsort(score)[-k:]
    return {'soft_auc': soft_auc(score, mask),
            'top5_mask_mass_fraction': float(mask[top].sum()/mask.sum()),
            'top5_lift_over_random': float(mask[top].sum()/mask.sum()/(k/len(score))),
            'argmax_mask_coverage': float(mask[np.argmax(score)]),
            'pointing_hit_at_coverage_005': float(mask[np.argmax(score)] > .05)}


def aggregate(rows, field='metrics'):
    return {key: float(np.mean([row[field][key] for row in rows]))
            for key in rows[0][field]} if rows else {}


def axes_image(ax, frame, title=''):
    ax.imshow(frame, extent=(0, 16, 16, 0))
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def contour(ax, mask, color):
    if mask.max() > .05:
        ax.contour(np.arange(16)+.5, np.arange(16)+.5, mask.reshape(16,16),
                   levels=[.05], colors=[color], linewidths=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, default=Path('local/role_effect_smoke/records.json'))
    parser.add_argument('--role-manifest', type=Path,
                        default=Path('local/role_effect_smoke/reviewed_v2/reviewed_manifest_v2.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    rng = np.random.default_rng(args.seed)
    records = json.loads(args.records.read_text())['records']
    manifest = json.loads(args.role_manifest.read_text())
    labels = {x['pair_key']: x for x in manifest['records']}
    streams, pairs, provenance = [], {}, []
    for record in records:
        annotation_record = labels[record['pair_key']]
        annotation_path = args.role_manifest.parent/annotation_record['annotation_file']
        assert file_sha256(annotation_path) == annotation_record['annotation_sha256']
        annotation = torch.load(annotation_path, map_location='cpu', weights_only=True)
        pairs[record['label']] = {}
        for domain in ['human', 'robot']:
            cache_path = Path(record[f'{domain}_cache_path'])
            assert file_sha256(cache_path) == annotation_record[f'{domain}_cache_sha256']
            cache = torch.load(cache_path, map_location='cpu', weights_only=True)
            assert torch.equal(cache['frame_indices'], annotation[f'{domain}_frame_indices'])
            frames = cache['frame_indices'].flatten().numpy()
            features, unit = normalized(cache['features'].flatten(0, 1) if domain == 'human' else cache['features'])
            if features.shape[-2:] != (256, 768):
                raise ValueError('This analysis expects the existing 16x16 DINOv2-base grid')
            masks = annotation[f'{domain}_masks'].reshape(-1, 2, 256).float().numpy()
            valid = annotation[f'{domain}_valid'].reshape(-1, 2).numpy()
            video = record.get(f'{domain}_video_path', cache['metadata']['source_video_path'])
            stream = {'name': f"{record['label']} {domain}", 'pair': record['label'], 'domain': domain,
                      'features': features, 'unit': unit, 'masks': masks, 'valid': valid,
                      'frames': frames, 'rgb': decode(video, frames)}
            stream['delta'] = np.maximum(0, 1-(unit*unit[:1]).sum(-1))
            streams.append(stream); pairs[record['label']][domain] = stream
            provenance.append({'pair_key': record['pair_key'], 'domain': domain,
                               'cache': str(cache_path), 'cache_sha256': file_sha256(cache_path),
                               'video': video, 'frame_ids': frames.tolist(),
                               'encoder': cache['metadata']['encoder_id'],
                               'encoder_revision': cache['metadata']['encoder_revision']})

    # A single shared PCA, with a single color range; no per-frame fit/rescaling.
    x = np.concatenate([s['features'].reshape(-1, 768) for s in streams])
    mean = x.mean(0)
    centered = x-mean
    covariance = centered.T@centered/(len(x)-1)
    values, vectors = np.linalg.eigh(covariance)
    projection = vectors[:, -3:][:, ::-1]
    # Fix the arbitrary eigenvector signs for deterministic figure colors.
    for i in range(3):
        if projection[np.argmax(np.abs(projection[:,i])),i] < 0:
            projection[:,i] *= -1
    colors = centered@projection
    low, high = np.quantile(colors, [.02, .98], axis=0)
    for s in streams:
        p = (s['features']-mean)@projection
        s['pca'] = np.clip((p-low)/(high-low), 0, 1).reshape(-1,16,16,3)
    np.savez(args.output_dir/'shared_pca.npz', mean=mean, projection=projection, low=low, high=high)
    fig, axs = plt.subplots(len(streams), 6, figsize=(15, 10), constrained_layout=True)
    for row, s in enumerate(streams):
        for col, index in enumerate([0, len(s['frames'])//2, len(s['frames'])-1]):
            axes_image(axs[row,2*col], s['rgb'][index], f"{s['name']} | frame {s['frames'][index]}")
            axes_image(axs[row,2*col+1], s['pca'][index], 'Shared PCA (16 x 16 patches)')
    fig.suptitle('Frozen DINOv2 features: ONE PCA and ONE color scale across both domains\n'
                 'All original frames resized as the DINO input; colors are not object/role labels', fontsize=13)
    fig.savefig(args.output_dir/'01_shared_pca.png', dpi=150); plt.close(fig)

    changes = []
    for s in streams:
        for t in range(1, len(s['frames'])):
            score=s['delta'][t]
            for role in range(2):
                if not s['valid'][t,role]: continue
                mask=s['masks'][t,role]
                inside=float((score*mask).sum()/mask.sum())
                outside=float((score*(1-mask)).sum()/(1-mask).sum())
                top=np.argsort(score)[-math.ceil(.2*len(score)):]
                changes.append({'stream':s['name'], 'domain':s['domain'], 'role':['mouse','mat'][role],
                                'frame':int(s['frames'][t]), 'inside_delta':inside, 'outside_delta':outside,
                                'enrichment':inside/max(outside,1e-8),
                                'top20_role_mass_fraction':float(mask[top].sum()/mask.sum())})
    vmax=float(np.quantile(np.concatenate([s['delta'][1:].ravel() for s in streams]), .98))
    fig, axs=plt.subplots(len(streams),6,figsize=(15,10),constrained_layout=True)
    heat=None
    for row,s in enumerate(streams):
        for col,index in enumerate([1,len(s['frames'])//2,len(s['frames'])-1]):
            axes_image(axs[row,2*col],s['rgb'][index],f"{s['name']} | frame {s['frames'][index]}")
            ax=axs[row,2*col+1]
            heat=ax.imshow(s['delta'][index].reshape(16,16),vmin=0,vmax=vmax,cmap='magma',extent=(0,16,16,0))
            ax.set_xticks([]);ax.set_yticks([])
            good=[]
            for role,color in enumerate(['lime','cyan']):
                if s['valid'][index,role]: contour(ax,s['masks'][index,role],color);good.append(['mouse','mat'][role])
            ax.set_title('DINO change | valid: '+','.join(good),fontsize=8)
    fig.colorbar(heat,ax=axs.ravel().tolist(),shrink=.55,label='1 - cosine(F_t, F_0), shared scale')
    fig.suptitle('Feature change relative to frame 0: includes robot/hand, shadows and appearance changes\n'
                 'Contours: weak visible-region labels only; this is NOT a learned transition-token visualization',fontsize=12)
    fig.savefig(args.output_dir/'02_feature_change.png',dpi=150);plt.close(fig)

    axis=(np.arange(16)+.5)/16
    yy,xx=np.meshgrid(axis,axis,indexing='ij')
    xy=np.stack([xx.ravel(),yy.ravel()],-1)
    retrieval=[]; visual=[]
    for source_label, source_pair in pairs.items():
        human=source_pair['human']
        for role in range(2):
            assert human['valid'][0,role]
            mask=human['masks'][0,role]
            query=pool(human['features'][0],mask)
            global_query=pool(human['features'][0],np.ones(256))
            center=(xy*mask[:,None]).sum(0)/mask.sum()
            spatial=np.exp(-((xy-center)**2).sum(-1)/(2*.15**2))
            for target_label,target_pair in pairs.items():
                robot=target_pair['robot']
                similarity=robot['unit']@query
                global_similarity=robot['unit']@global_query
                for t in range(len(robot['frames'])):
                    if not robot['valid'][t,role]:continue
                    targetmask=robot['masks'][t,role]
                    randoms=[retrieval_metrics(rng.random(256),targetmask) for _ in range(64)]
                    random_metrics={k:float(np.mean([r[k] for r in randoms])) for k in randoms[0]}
                    random_exact={'soft_auc':.5,'top5_mask_mass_fraction':math.ceil(.05*256)/256,
                                  'top5_lift_over_random':1.,'argmax_mask_coverage':float(targetmask.mean()),
                                  'pointing_hit_at_coverage_005':float((targetmask>.05).mean())}
                    retrieval.append({'human_query':source_label,'robot_target':target_label,
                                      'cross_layout':source_label!=target_label,'role':['mouse','mat'][role],
                                      'robot_frame':int(robot['frames'][t]),
                                      'metrics':retrieval_metrics(similarity[t],targetmask),
                                      'spatial_prior':retrieval_metrics(spatial,targetmask),
                                      'global_query':retrieval_metrics(global_similarity[t],targetmask),
                                      'random64':random_metrics,'random_exact':random_exact})
                if source_label!=target_label:
                    visual.append((source_label,target_label,role,human,robot,similarity,spatial))
    vlow,vhigh=np.quantile(np.concatenate([item[5].ravel() for item in visual]),[.02,.98])
    fig,axs=plt.subplots(len(visual),6,figsize=(15,10),constrained_layout=True)
    heat=None
    for row,(source_label,target_label,role,human,robot,similarity,spatial) in enumerate(visual):
        axes_image(axs[row,0],human['rgb'][0],f'ORACLE query {source_label} human: '+['mouse','mat'][role])
        contour(axs[row,0],human['masks'][0,role],'lime')
        axes_image(axs[row,1],robot['rgb'][0],f'Target {target_label} robot | frame 0')
        for col,t in [(2,0),(5,len(robot['frames'])-1)]:
            ax=axs[row,col]
            heat=ax.imshow(similarity[t].reshape(16,16),cmap='viridis',vmin=vlow,vmax=vhigh,extent=(0,16,16,0))
            contour(ax,robot['masks'][t,role],'red')
            maximum=xy[np.argmax(similarity[t])]*16
            ax.scatter(*maximum,c='white',marker='x',s=50)
            ax.set_title(f'DINO cosine | frame {robot["frames"][t]}',fontsize=9)
            ax.set_xticks([]);ax.set_yticks([])
        ax=axs[row,3]
        ax.imshow(spatial.reshape(16,16),vmin=0,vmax=1,cmap='viridis',extent=(0,16,16,0))
        contour(ax,robot['masks'][0,role],'red');ax.set_xticks([]);ax.set_yticks([])
        ax.set_title('Spatial prior from HUMAN location',fontsize=8)
        axes_image(axs[row,4],robot['rgb'][-1],f'Target {target_label} | frame {robot["frames"][-1]}')
    fig.colorbar(heat,ax=axs.ravel().tolist(),shrink=.55,label='DINO cosine similarity, shared color scale')
    fig.suptitle('Cross-layout human-to-robot region retrieval: all four role/direction combinations\n'
                 'Human query uses reviewed SAM2 masks (ORACLE selection); red = target weak region, white x = argmax',fontsize=12)
    fig.savefig(args.output_dir/'03_cross_domain_retrieval.png',dpi=150);plt.close(fig)

    groups={}
    for mode in ['all','cross_layout','same_pair']:
        selected=[r for r in retrieval if mode=='all' or r['cross_layout']==(mode=='cross_layout')]
        groups[mode]={'evaluated_role_frames':len(selected),**{field:aggregate(selected,field)
                        for field in ['metrics','spatial_prior','global_query','random64']}}
    role_groups={}
    for role in ['mouse','mat']:
        selected=[r for r in retrieval if r['cross_layout'] and r['role']==role]
        role_groups[role]={'evaluated_role_frames':len(selected),
                          'pointing_hits':int(sum(r['metrics']['pointing_hit_at_coverage_005'] for r in selected)),
                          'spatial_prior_pointing_hits':int(sum(r['spatial_prior']['pointing_hit_at_coverage_005'] for r in selected)),
                          'global_query_pointing_hits':int(sum(r['global_query']['pointing_hit_at_coverage_005'] for r in selected)),
                          'random_expected_pointing_hits':float(sum(r['random_exact']['pointing_hit_at_coverage_005'] for r in selected)),
                          **{field:aggregate(selected,field) for field in ['metrics','spatial_prior','global_query','random64']}}
    failures=[{k:r[k] for k in ['human_query','robot_target','role','robot_frame','metrics']}
              for r in retrieval if r['cross_layout'] and not r['metrics']['pointing_hit_at_coverage_005']]
    change_groups={}
    for domain in ['human','robot']:
        for role in ['mouse','mat']:
            selected=[r for r in changes if r['domain']==domain and r['role']==role]
            change_groups[f'{domain}_{role}']={'valid_nonanchor_role_frames':len(selected),
                                             **{k:float(np.mean([r[k] for r in selected])) for k in
                                                ['inside_delta','outside_delta','enrichment','top20_role_mass_fraction']}}
    report={'status':'completed','scope':'frozen_encoder_diagnostics_only','seed':args.seed,
            'task_success_evaluated':False,'policy_checkpoint_used':False,'language_encoder_used':False,
            'selection':'Both pre-existing reviewed place_mouse_pad pairs; all valid robot frames; no success-based selection',
            'mask_source':'Official SAM2 weak labels plus Codex visual review, NOT human ground truth',
            'query_source':'Human frame-0 reviewed region masks; ORACLE selection, not autonomously inferred roles',
            'feature_preprocessing':'Non-affine per-patch layer_norm eps=1e-6; cosine uses L2 normalization',
            'pca':{'fitted_patches':len(x),'shared_across_all_streams':True,'explained_variance_ratio':
                    (values[-3:][::-1]/values.sum()).tolist(),'color_quantiles':[.02,.98]},
            'retrieval_protocol':{'soft_auc_random_expectation':.5,'random_draws_per_frame':64,
                                  'soft_auc_definition':'sum_{p,q} m_p*(1-m_q)*(I[s_p>s_q]+0.5*I[s_p=s_q]) / (sum_p m_p * sum_q (1-m_q)); m is patch mask coverage in [0,1]',
                                  'aggregation':'Compute each role/frame metric separately, then arithmetic mean across valid role-frames; not classification accuracy',
                                  'spatial_prior_sigma_in_normalized_image':.15,
                                  'top_fraction_actual':math.ceil(.05*256)/256,
                                  'pointing_positive_patch_coverage_threshold':.05},
            'retrieval_summary':groups,'cross_layout_by_role':role_groups,
            'cross_layout_pointing_failures':failures,'change_summary':change_groups,
            'retrieval_all_role_frames':retrieval,'change_all_role_frames':changes,
            'provenance':provenance,
            'limitations':['Two examples from one training task; not unseen-task evaluation.',
                           'Masks select the query and score retrieval, so these plots do not show autonomous task-role discovery.',
                           'PCA colors are unsupervised appearance axes, not task labels.',
                           'Feature change also responds to actors, lighting, occlusion and position; no hand/arm labels are available.',
                           'Retrieval/readability does not establish that Zero-WAM or RGB adapters causally use these features.',
                           'No trajectory from training, no language-following measurement, no policy-success claim.']}
    (args.output_dir/'feature_analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    rows=''.join(f'<tr><td>{mode}</td><td>{g["evaluated_role_frames"]}</td>'+
                 ''.join(f'<td>{g[m]["soft_auc"]:.3f}</td>' for m in ['metrics','spatial_prior','global_query','random64'])+'</tr>'
                 for mode,g in groups.items())
    role_rows=''.join(f'<tr><td>{role}</td><td>{g["pointing_hits"]}/{g["evaluated_role_frames"]}</td>'
                      f'<td>{g["spatial_prior_pointing_hits"]}/{g["evaluated_role_frames"]}</td>'
                      f'<td>{g["random_expected_pointing_hits"]:.3f}/{g["evaluated_role_frames"]}（期望）</td>'
                      f'<td>{g["metrics"]["soft_auc"]:.3f}</td></tr>' for role,g in role_groups.items())
    failure_rows=''.join(f'<tr><td>{r["human_query"]}→{r["robot_target"]}</td><td>{r["role"]}</td>'
                         f'<td>{r["robot_frame"]}</td><td>{r["metrics"]["argmax_mask_coverage"]:.3f}</td></tr>' for r in failures)
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>冻结 DINO 特征诊断</title>
<style>body{max-width:1300px;margin:32px auto;padding:0 20px;font:17px/1.6 system-ui;background:#f7f8fa;color:#18202d}img{width:100%;background:white;border:1px solid #ddd}table{border-collapse:collapse;background:white}th,td{padding:9px 18px;border:1px solid #ddd}h1{font-size:30px}.note{background:#fff3d7;padding:16px;border-left:5px solid #d59c21}a{color:#2455b5}</style>
<h1>冻结 DINOv2：真实人类／机器人视频的特征诊断</h1>
<p class="note">这是 encoder 可读性分析，不是策略成功率、训练收敛或语言理解证明。仅使用两条已有 place_mouse_pad 配对；区域 query 使用 SAM2＋Codex 视审弱标注，属于 oracle 选区。两个模型分支共用同一冻结 DINO cache。</p>
<h2>1. 统一 PCA：颜色轴对所有视频相同</h2><p>统一使用全部84个缓存帧拟合，所有图共用投影及颜色范围；每段固定展示开始、中间、结尾。</p><img src="01_shared_pca.png">
<h2>2. 特征变化：不等于物体动作</h2><p>显示相对于首帧的余弦距离；机械臂、手、遮挡、光照也会造成变化。绿色／青色仅画有效的鼠标／垫子弱标注区域；没有可靠手臂标注，因此不能计算“纯物体vs执行者”准确率。</p><img src="02_feature_change.png">
<h2>3. 跨布局区域对应：包含完整对照与失败</h2><p>用A人类的鼠标／垫子query检索B机器人，以及B→A；可见的所有机器人帧进入统计。空间先验是人类query原位置的高斯图；global query使用人类整幅特征均值；随机基线每帧64次。</p><img src="03_cross_domain_retrieval.png">
<p>指标：coverage-weighted AUROC，随机期望0.5。m是每个patch被弱mask覆盖的比例，s是检索分数；AUC=Σm<sub>p</sub>(1−m<sub>q</sub>)[I(s<sub>p</sub>&gt;s<sub>q</sub>)+0.5I(s<sub>p</sub>=s<sub>q</sub>)] / [Σm<sub>p</sub>Σ(1−m<sub>q</sub>)]。先逐角色、逐帧计算，再平均。它是区域排序指标，<strong>不是分类准确率或任务成功率</strong>。</p><table><tr><th>范围</th><th>有效角色-帧数</th><th>DINO区域query</th><th>空间先验</th><th>全图query</th><th>随机</th></tr>'''+rows+'''</table>
<p>AUROC较高不等于精确定位。以下按角色列出argmax落在目标patch内的次数（coverage&gt;0.05）。随机期望按正patch数量/256精确计算，非一次随机抽样的整数命中：</p><table><tr><th>角色</th><th>DINO命中/有效帧</th><th>空间先验命中</th><th>均匀随机期望命中</th><th>soft AUROC</th></tr>'''+role_rows+'''</table>
<p>全部跨布局argmax失败记录如下；图中B-human→A-robot鼠标末帧是其中一个失败，峰值落到机械臂附近：</p><table><tr><th>配对</th><th>角色</th><th>机器人帧</th><th>峰值patch目标覆盖</th></tr>'''+failure_rows+'''</table>
<p><a href="feature_analysis.json">全部数值、逐帧结果、来源与限制</a> · <a href="shared_pca.npz">固定PCA投影</a></p>
<p>下一步需要策略层面的干预：替换／打乱DINO特征、交换示范及语言，在相同观测与随机噪声下比较动作和任务结果，才能检验策略是否真正利用这些信息。</p></html>'''
    (args.output_dir/'feature_analysis.html').write_text(html)
    print(json.dumps({'output':str(args.output_dir),'pca':report['pca'],'retrieval':groups,'change':change_groups},indent=2))


if __name__=='__main__':
    main()
