# Role / Effect ICL：可运行的二维 C1 原型

本分支 `RGB-Role-Effect-ICL` 将 human demo 的对象绑定与变化表示分开，并在机器人当前观测中重新绑定对象。它是待检验的方法实现，不是已经成立的泛化结论。3DGS / NeRF 与 C2 有序算子组合均未在本版实现。

代码基于 `RGB-Readout-State` 的因果观测接入；正式训练默认从原始 `zero-wam-pretrain` 初始化，**不要求已有 RGB checkpoint**。保留原始视频、动作、MCP 和 WALA RGB 辅助目标。源码/论文依据及近邻重叠见 [README.role-effect-sources.md](README.role-effect-sources.md)。

## 方法与实际接口

1. 冻结 DINO patch 特征输入共享 human role head，预测每帧两个 soft role masks。标签来自显式对象提示与 SAM2 跟踪，只用于监督，主条件始终使用预测结果。
2. Binding tokens 编码 clip 起点的角色区域特征与二维几何；effect tokens 编码角色特征、区域几何、相对中心的变化。它们不是已被证明完全解耦的变量；DINO 的区域特征也不保证完全排除执行者。
3. 机器人 role binder 从**当前实际观察**的 Wan patch embeddings 中选择角色区域，和示范的 binding/effect 一起形成策略条件。当前只有 cam_high 有角色教师，因此角色 query/pooling 限定该视图；原策略与基础 readout 仍使用三路相机。
4. 共享辅助 decoder 预测该机器人对象的未来 DINO 区域变化与二维相对中心变化。未来帧、未来 masks、实体 ID 只进入目标和有效性判断。
5. 显式 reviewed donor 只替换 effect tokens，保留接收样本的当前机器人场景、binding、未来目标。microbatch=1 时也确实额外加载另一段 human video；不使用 batch 内自交换。

每 clip 保留 4 个原 WALA tokens，增加 2 个 binding 与 2 个 effect tokens；8 clips 共 64 个 memory tokens，readout 输出 32 tokens。原始 human Wan context 仍保留。因此需要 **RGB、RGB+State、相同数据/监督但无 factorization 或 donor 的消融**，不能把相对于 RGB 的全部变化归因于角色分路。

部署入口沿用 `VA_Server`。只需原来的人类视频/DINO cache与当前机器人观测；无需 role annotation、donor、robot future、测试时优化或重规划。这沿用原策略的观测条件，并不宣称机器人从头到尾完全不读传感器。

## 监督边界

角色顺序为 `[source, reference]`。单次放置示例为 `[mouse, mat]`；source/reference 是任务语义，SAM2 本身不提供这些语义。

独立 role sidecar 包含：

```text
human_masks [S,T,2,P], human_valid [S,T,2]
robot_masks [Tr,2,P], robot_valid [Tr,2]
robot_entity_ids [Tr,2]（角色更换实体时必须显式给出）
human_frame_indices, robot_frame_indices
metadata: schema, pair_key, 原 RGB 文件 SHA256, teacher revision, provenance
```

全程稳定角色可以在 loader 中生成固定实体 IDs。角色切换时，跨不同实体的未来差分被屏蔽，不把“前面的鼠标”减去“后面的篮子”。未标注样本使用 false validity，不能作为背景负标签。

首版 donor 限定到明确审核的**单一关系效果且角色稳定**的视频。必须记录 relation、direction、source/reference 类型、affordance、不同视频来源和批准理由。任务名一致不能自动建立 donor。多阶段任务仍可参与原策略与有效区间的局部监督，但不自动交换整段效果。

这些几何量是二维可见区域的统计量，不是真实物体中心、接触、inside/on 的三维真值。加入 3D 时应单独对照可靠深度/点云与物体局部 feature field，不能直接把动态人视频当静态多视图重建。

## 准备数据

`tools/prepare_role_records.py` 从正式12任务 manifest导出记录、原始帧和空 seed 模板。填写明确对象/part提示后，用官方 SAM2 环境运行：

```bash
/share/personal/xiangzishang/envs/vace22-inference/bin/python tools/prepare_role_masks_sam2.py \
  --records-json /path/to/records.json \
  --seed-json /path/to/role_seeds.json \
  --output-dir /path/to/role_candidates
```

候选工具不会自动批准 `reviewed=True`。检查 overlays、遮挡、跟踪身份和角色对应后，单独保存 reviewed manifest；保留原候选、审核者、mask模型版本及修改记录。跨样本 donor 要另行核验。

SAM2 工具同时接受固定两个角色的旧 `boxes_xyxy` 格式，以及多实体 `objects` + 明确原始帧区间 `role_events`。例如三块物体先 A→B、再 C→A，先为 A/B/C 各给一个种子，再逐段指定 source/reference；工具传播所有实体并保留 IDs。未覆盖区间不提供监督，事件区间不能重叠。这个接口支持多阶段标注，不意味着当前已经审核了真实多阶段样本。

完整训练默认要求全部选中配对有有效角色/未来监督；还会验证真实训练集合成员关系、文件 hash、精确帧索引、有效角色覆盖和启用的 donor。只写600条空记录不能通过。`ZERO_WAM_ROLE_ALLOW_PARTIAL=1` 是明确的部分监督实验，预检和 checkpoint 记录真实覆盖率。

## 训练

本机4×A800 80GB可使用原四卡配方；配置保留全参数主干训练与15,000步预算。初始 loss 系数均是实验超参，尚未调优；每项单独记录，统一经原 `rgb_loss_weight` 加到策略目标。

Triton 需要主机 C 编译器。脚本尊重现有 `CC`/系统编译器；本机若 PATH 中缺少编译器，会仅为该进程使用现成的 `lingbot-toolchain/bin` 中 GCC/G++12.4，不修改 Python 环境或 CUDA。

```bash
export ZERO_WAM_ROLE_MANIFEST=/path/to/full_reviewed_manifest.json
export ZERO_WAM_SAVE_ROOT=/workspace/zero-wam/runs/role-effect-subset12
bash script/train_role_effect.sh
```

先做有界实模型检查，默认6个 optimizer steps，保留12任务、加入reviewed样本，关闭 ICL dropout 以确保新分支被调用：

```bash
export ZERO_WAM_ROLE_MANIFEST=/path/to/smoke_reviewed_manifest.json
export ZERO_WAM_SAVE_ROOT=/workspace/zero-wam/runs/role-effect-smoke
bash script/smoke_role_effect.sh
```

smoke 显式允许部分监督、禁用 W&B；并不等价于正式训练。运行指标在 `$ZERO_WAM_SAVE_ROOT/metrics.jsonl`，保存的 `COMPLETE.json` 包含 smoke 标记和 role manifest hash/覆盖率。

## 本机验证与后续实验

实际执行状态及产物见 [README.role-effect-validation.md](README.role-effect-validation.md)。少量真实样本、梯度检查、保存加载和离线两 chunk 推理仅证明代码链路，不能证明未见任务成功率提升。

研究验证应首先固定原 pretrained、12任务、训练预算、测试种子与 human demo；增加 RGB+State 和相同标签资源的控制组，再分别去掉角色监督、局部未来、donor交换。用错误绑定/错误效果/不同视角的示范干预检验模型是否真的使用了预期变量；任务评测仍按每任务100次并保留逐seed结果。

C2 尚未实施。它要检验的是同一个 `PLACE_ON` 参数化操作在新的对象绑定和已改变状态下能否重复使用，例如 `PLACE_ON(A,B)` 后 `PLACE_ON(C,A)`；不能仅因加入时间embedding或把动作序列重复播放就声称获得操作组合泛化。
