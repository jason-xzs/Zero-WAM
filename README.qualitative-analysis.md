# 角色、效果与语言诊断（2026-11-18）

本次不修改 policy、训练目标或评测默认设置。产物总入口：

`/share/personal/xiangzishang/compare_zero/role_effect_analysis/index.html`

## 实验与结论

| 诊断 | 实际输入与结果 | 能支持的结论 / 限制 |
| --- | --- | --- |
| 冻结 DINO 的统一 PCA | 两个人机配对、84 个缓存帧，共用投影与色阶；前三轴解释 29.18% 方差 | 显示背景、执行者和跨域结构；颜色不是语义分割。DINO 冻结，不会随训练变化。 |
| 跨布局区域检索 | Human A→Robot B 与 B→A；鼠标峰值命中 7/14，垫子 20/20；空间先验两者均 0；随机期望 2.20% / 4.80% | Query 是 SAM2＋Codex 视审区域均值，属于 oracle 选区。证明对象信息可读取，不能证明模型自主识别角色或策略实际利用它。全部有效帧与七个鼠标失败都公开。 |
| 区域排序 | 跨布局 coverage-weighted AUROC：DINO .950，空间先验 .445，随机 .500 | 公式与逐帧平均口径在 HTML/JSON；不是分类准确率、成功率。 |
| 特征变化图 | 变化最高 20% patches 覆盖机器人静态垫子 mask mass 约 2.5%，随机约 20.3% | 变化强度不等于任务相关性；原 WALA 还读当前特征，不能断言其丢失静态对象。 |
| C1 六步保存权重 | 严格加载实际 RGB head 和 Wan patch embedding，使用真实缓存；区域质量富集度 .896–1.001，均匀预测为 1；gate −6.795e−6 | CPU FP32 头部诊断，无 teacher 输入；两个配对曾参加训练，不是留出集。尚无可靠角色定位，不是完整训练效果。 |
| Base 语言干预 | 默认 target text off 时原指令／改写／相反指令的真实文本输入和 denoiser 输出逐元素相等；on 时 action 相对 RMS 差 2.24% / 4.62% | 单样本、单 seed、单 timestep；仅入口和敏感性，未测正确跟随、环境动作或成功率。与远端 RGB checkpoint 无关。 |

C1 曲线使用原六步日志，minibatch 与标签覆盖不同；辅助损失为 0 可能表示缺少监督，不能据曲线声明收敛。没有匹配的 step-0 权重，不报告伪造的训练前后对照。图中所有角色概率统一使用 `[0,1]` 色阶。

## 架构与语言审计

`RGB-Transition-ICL`：8 clips × 4 WALA tokens = 32，融合 human Wan 前缀。

`RGB-Role-Effect-ICL`：每 clip 保留 4 WALA tokens，加 2 binding 和 2 effect tokens，合计 64 memory；当前机器人观测进行绑定/readout，输出 32 条件 tokens，融合 robot video tokens。原 human Wan 前缀保留。角色绑定仅 cam_high，基础 readout 三相机。新 effect 对固定首帧后的未来帧没有独立时间编码，不能称已实现有序算子或 C2；原 WALA 与 clip 顺序仍保留。当前没有 3DGS、NeRF。

主干支持 UMT5 text cross-attention，但当前角色/效果头没有直接语言输入。`evaluation/robotwin/evaluate.py` 会设置 `TARGET_TEXT_CFG=-1`、`ICL_CFG=5`；单纯在该脚本外 export 不会开启目标文本。直接 client launcher 用 `TARGET_TEXT_CFG=1 ICL_CFG=5` 可以保留 ICL guidance 并让 robot target text 生效；本次未改默认。参见 [Zero-WAM §4.1](https://arxiv.org/html/2608.26103v1#S4.SS1) 的 ICL 推理说明。

Human prefix 缓存文本独立于 target text 开关；实测缓存文本 embedding 与 empty embedding 不同。默认 target off 不是所有语言清空。未来比较必须分别控制 human caption 与 robot instruction，且 RGB/C1 的文本设置相同。完整代码行号、caption 来源、开启方法见产物中的 `language_audit.txt`。

建议后续候选：分别用人域文本 L_h 与机器人文本 L_r 条件化角色 query、绑定和关系/效果 query，使用可训练投影或 cross-attention。DINO 与 UMT5 不是天然对齐空间。该方向尚未实现，不能把“语言改变输出”当作“语言被正确理解”。

## SAM 与方法主张

当前 SAM2 是离线弱标签教师，不参与部署，也没有独立的抓取点/IK 控制器；WAM 仍生成视频/动作。但 mask、角色指定和 donor 等价性审查都是额外训练先验，必须披露。Codex 视觉审查不是人工 GT，SAM2 也不会自动识别操作角色。

若主线要求仅从原有视频/轨迹/文本学习，当前 C1 应作为带对象监督的对照或诊断分支。无 mask 版本可以尝试语言条件化软区域和未来预测一致性，SAM 只用于评价；去掉 mask 后是否形成可靠角色仍待实测。本次没有移除或新增任何模型监督。

## 复跑

在本仓库根目录，使用现有环境 `/share/personal/xiangzishang/envs/zero-wam/bin/python`。输出到新目录以保留本次结果。

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 /share/personal/xiangzishang/envs/zero-wam/bin/python tools/analyze_rgb_features.py \
  --records local/role_effect_smoke/records.json \
  --role-manifest local/role_effect_smoke/reviewed_v2/reviewed_manifest_v2.json \
  --output-dir /share/personal/xiangzishang/compare_zero/role_effect_analysis_repeat

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 /share/personal/xiangzishang/envs/zero-wam/bin/python tools/analyze_role_checkpoint.py \
  --checkpoint /workspace/zero-wam/runs/role-effect-smoke-20261118/checkpoints/checkpoint_step_6/transformer \
  --records local/role_effect_smoke/records.json \
  --manifest local/role_effect_smoke/reviewed_v2/reviewed_manifest_v2.json \
  --metrics /workspace/zero-wam/runs/role-effect-smoke-20261118/metrics.jsonl \
  --output-dir /share/personal/xiangzishang/compare_zero/role_effect_analysis_repeat

CC=/share/personal/xiangzishang/envs/lingbot-toolchain/bin/x86_64-conda-linux-gnu-gcc \
CXX=/share/personal/xiangzishang/envs/lingbot-toolchain/bin/x86_64-conda-linux-gnu-g++ \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
/share/personal/xiangzishang/envs/zero-wam/bin/python tools/probe_language_conditioning.py \
  --output-dir /share/personal/xiangzishang/compare_zero/role_effect_analysis_repeat/language
```

前两个工具 CPU；最后一个实际加载 base pretrained，占一张 GPU。本次六条件含加载约 276 秒。Video 指标包括 CFG 两支原始输出；Action 指标包括全部 30 输出通道，非执行动作。没有 environment rollout，没有生成 future 写入 cache。

框架图为手写可编辑 SVG/PNG，绘制源码和代码来源 JSON 在报告目录，未覆盖原 `架构对比.svg`。大权重、图表和本机数据不加入 Git。
