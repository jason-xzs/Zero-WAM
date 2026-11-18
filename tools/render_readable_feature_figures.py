#!/usr/bin/env python3
"""Large-label PNG explanations of the existing frozen-DINO diagnostics.

Reuses the previously fitted shared PCA; never loads a policy checkpoint or
trains a model. All region contours are existing reviewed weak labels.
"""
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

from tools.analyze_rgb_features import decode, normalized, pool
from wan_va.dataset.role_feature_cache import file_sha256


def clear_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#aab4bf')


def photo(ax, rgb):
    ax.imshow(rgb, extent=(0, 16, 16, 0))
    clear_axis(ax)


def region(ax, mask, color):
    ax.contour(np.arange(16)+.5, np.arange(16)+.5, mask.reshape(16,16),
               levels=[.05], colors=[color], linewidths=2.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, default=Path('local/role_effect_smoke/records.json'))
    parser.add_argument('--role-manifest', type=Path,
                        default=Path('local/role_effect_smoke/reviewed_v2/reviewed_manifest_v2.json'))
    parser.add_argument('--pca', type=Path, default=Path('/share/personal/xiangzishang/compare_zero/role_effect_analysis/shared_pca.npz'))
    parser.add_argument('--font', type=Path, default=Path('/tmp/role_svg_fonts/NotoSansCJKsc-Regular.otf'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    font_manager.fontManager.addfont(str(args.font))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=args.font).get_name(),
                         'font.size':21, 'axes.unicode_minus':False})
    torch.set_num_threads(4)
    records = json.loads(args.records.read_text())['records']
    manifest = json.loads(args.role_manifest.read_text())
    annotations = {r['pair_key']:r for r in manifest['records']}
    pca = np.load(args.pca)
    streams, provenance = {}, []
    for record in records:
        ann_record = annotations[record['pair_key']]
        ann_path = args.role_manifest.parent/ann_record['annotation_file']
        assert file_sha256(ann_path) == ann_record['annotation_sha256']
        ann = torch.load(ann_path, map_location='cpu', weights_only=True)
        streams[record['label']] = {}
        for domain in ['human','robot']:
            path = Path(record[f'{domain}_cache_path'])
            assert file_sha256(path) == ann_record[f'{domain}_cache_sha256']
            cache = torch.load(path,map_location='cpu',weights_only=True)
            assert torch.equal(cache['frame_indices'],ann[f'{domain}_frame_indices'])
            features = cache['features'].flatten(0,1) if domain=='human' else cache['features']
            features, unit = normalized(features)
            frames = cache['frame_indices'].flatten().numpy()
            video = cache['metadata']['source_video_path']
            colors = ((features-pca['mean'])@pca['projection']-pca['low'])/(pca['high']-pca['low'])
            streams[record['label']][domain] = dict(features=features,unit=unit,frames=frames,
                rgb=decode(video,frames),pca=np.clip(colors,0,1).reshape(-1,16,16,3),
                masks=ann[f'{domain}_masks'].reshape(-1,2,256).numpy(),
                valid=ann[f'{domain}_valid'].reshape(-1,2).numpy(),
                delta=np.maximum(0,1-(unit*unit[:1]).sum(-1)))
            provenance.append({'pair':record['label'],'domain':domain,'cache':str(path),
                               'cache_sha256':file_sha256(path),'video':video})

    # 4: the same existing PCA transform for both domains, with large panels.
    fig, axes = plt.subplots(2,2,figsize=(10,10.5),layout='constrained')
    selections = [('human',16,'人类视频'),('robot',9,'机器人视频')]
    for row,(domain,index,label) in enumerate(selections):
        stream=streams['A'][domain]
        photo(axes[row,0],stream['rgb'][index])
        axes[row,1].imshow(stream['pca'][index],interpolation='nearest')
        clear_axis(axes[row,1])
        axes[row,0].set_ylabel(label,fontsize=25)
        axes[row,0].set_title(f'原始画面：第 {stream["frames"][index]} 帧',fontsize=23,pad=14)
        axes[row,1].set_title('同一投影得到的颜色',fontsize=23,pad=14)
    fig.suptitle('冻结 DINO：768 维特征 → 3 个颜色分量',fontsize=27)
    fig.savefig(args.output_dir/'4_DINO颜色可视化.png',dpi=160)
    plt.close(fig)

    # 5: one real robot sequence, fixed locations compared with its first frame.
    robot=streams['A']['robot']
    later=len(robot['frames'])-1
    all_delta=np.concatenate([s['delta'][1:].ravel() for pair in streams.values() for s in pair.values()])
    change_vmax=float(np.quantile(all_delta,.98))
    fig,axes=plt.subplots(1,3,figsize=(15,6.4),layout='constrained')
    photo(axes[0],robot['rgb'][0]);photo(axes[1],robot['rgb'][later])
    heat=axes[2].imshow(robot['delta'][later].reshape(16,16),vmin=0,vmax=change_vmax,
                       cmap='magma',extent=(0,16,16,0),interpolation='nearest')
    clear_axis(axes[2])
    for ax,index in [(axes[0],0),(axes[1],later),(axes[2],later)]:
        assert robot['valid'][index,1]
        region(ax,robot['masks'][index,1],'#00b8ff')
    for ax,title in zip(axes,['初始画面：第 0 帧',f'后续画面：第 {robot["frames"][later]} 帧','同一位置的特征变化']):
        ax.set_title(title,fontsize=23,pad=16)
    bar=fig.colorbar(heat,ax=axes[2],shrink=.78,pad=.035,ticks=[0,.2,.4])
    bar.ax.tick_params(labelsize=19)
    bar.set_label('变化幅度',fontsize=22)
    fig.suptitle('冻结 DINO：颜色越亮，变化越大\n蓝圈标出静止的垫子',fontsize=27)
    fig.savefig(args.output_dir/'5_变化图.png',dpi=140)
    plt.close(fig)

    # 6: hold the oracle human query fixed, show one hit and one failure.
    human=streams['B']['human']
    role=0
    assert human['valid'][0,role]
    query=pool(human['features'][0],human['masks'][0,role])
    similarity=robot['unit']@query
    cross_scores=[]
    for source,target in [('A','B'),('B','A')]:
        for r in range(2):
            h=streams[source]['human'];rob=streams[target]['robot']
            cross_scores.append((rob['unit']@pool(h['features'][0],h['masks'][0,r])).ravel())
    low,high=np.quantile(np.concatenate(cross_scores),[.02,.98])
    fig,axes=plt.subplots(2,3,figsize=(15,10),layout='constrained')
    selected_results=[]
    for row,index in enumerate([0,later]):
        assert robot['valid'][index,role]
        photo(axes[row,0],human['rgb'][0]);region(axes[row,0],human['masks'][0,role],'#00d72f')
        photo(axes[row,1],robot['rgb'][index]);region(axes[row,1],robot['masks'][index,role],'#f21c32')
        heat=axes[row,2].imshow(similarity[index].reshape(16,16),vmin=low,vmax=high,
                              cmap='viridis',extent=(0,16,16,0),interpolation='nearest')
        clear_axis(axes[row,2]);region(axes[row,2],robot['masks'][index,role],'#f21c32')
        peak=int(np.argmax(similarity[index]));y,x=divmod(peak,16)
        axes[row,2].scatter(x+.5,y+.5,marker='x',color='white',s=180,linewidths=3)
        hit=bool(robot['masks'][index,role,peak]>.05)
        for col,title in enumerate(['同一人类示范',f'机器人：第 {robot["frames"][index]} 帧',
                                    '最高响应落在鼠标' if hit else '最高响应偏到机械臂']):
            axes[row,col].set_title(title,fontsize=23,pad=15)
        axes[row,0].set_ylabel('匹配命中' if hit else '匹配失败',fontsize=25)
        selected_results.append({'human_query_pair':'B','robot_pair':'A','robot_frame':int(robot['frames'][index]),
                                 'role':'mouse','argmax_patch':peak,'target_coverage_at_peak':float(robot['masks'][index,role,peak]),
                                 'hit_at_coverage_005':hit})
    assert selected_results[0]['hit_at_coverage_005'] and not selected_results[1]['hit_at_coverage_005']
    bar=fig.colorbar(heat,ax=axes[:,2],shrink=.8,pad=.03,ticks=[0,.1,.2])
    bar.set_label('特征相似度',fontsize=22);bar.ax.tick_params(labelsize=19)
    fig.suptitle('冻结 DINO：给定人类图中的鼠标，寻找机器人图中的对应区域\n'
                 '绿圈：指定查询区域　红圈：目标区域　白叉：最高相似度',fontsize=25)
    fig.savefig(args.output_dir/'6_跨域对应.png',dpi=140)
    plt.close(fig)

    report={'policy_checkpoint_used':False,'training_performed':False,'features':'frozen DINOv2-base',
            'pca_file':str(args.pca),'pca_sha256':file_sha256(args.pca),
            'pca_selected_frames':{'A_human':int(streams['A']['human']['frames'][16]),
                                   'A_robot':int(streams['A']['robot']['frames'][9])},
            'pca_same_transform_for_all_images':True,'change_grid':'same pixel/patch locations, not object tracks',
            'change_formula':'max(0, 1-cosine(F_t,F_0))','change_shared_vmax':change_vmax,
            'retrieval_selected_examples':selected_results,'retrieval_shared_vmin_vmax':[float(low),float(high)],
            'query_source':'Existing reviewed SAM2 weak region selected by task role (oracle region), not model prediction',
            'all_frame_metrics':'../feature_analysis.json','provenance':provenance}
    (args.output_dir/'4-6_图像来源.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'output':str(args.output_dir),'retrieval':selected_results},ensure_ascii=False))


if __name__=='__main__':
    main()
