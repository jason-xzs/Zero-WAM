# RGB transition：12 任务实验版本

本分支 `RGB-Transition-ICL` 基于 `Full-Robotwin` 的 `53999f2`，沿用 [README.robotwin.md](README.robotwin.md) 的 12 任务训练、7 个未见任务测评、4 卡全局 batch 8、IFP/MCP 和 15,000 步预算。新增配置为 `robotwin_subset12_rgb_train`。原 `robotwin_subset12_train` 可继续运行基线。

这是可检验的第一版假设：对 human/robot RGB 使用相同冻结表征，以“当前特征 → 未来特征变化”的预测目标训练 human 示范 token，再把它接入原 Zero-WAM 条件。短程检查验证实现和训练链路，不能证明未见任务成功率提升。

## 源码来源与新增部分

| 部分 | 实际来源 / 改动 |
| --- | --- |
| Transition bottleneck encoder/decoder | 原样保留 [WALA 的 `latent_action_model.py`](https://github.com/liujiahao2077/WALA/blob/4a944f07afe09e24743f20d3d9ab809db672dc63/wala/model/modules/latent_action_model/latent_action_model.py)，commit `4a944f07afe09e24743f20d3d9ab809db672dc63`；[许可证和说明](wan_va/modules/third_party/wala/NOTICE.md) 随代码保存 |
| 冻结 RGB backbone | 官方 [`facebook/dinov2-base`](https://huggingface.co/facebook/dinov2-base/tree/f9e44c814b77203eaa57a6bdbbd535f21ede1415)，固定 revision；直接使用 Transformers 实现 |
| 视频/动作扩散、MCP、训练与推理 | 本仓库 Full-Robotwin 基线，保留原任务划分、采样器和损失 |
| 本分支新增 | RGB cache/对齐校验、逐 patch 特征归一化、共享 tokenizer 的 human→robot 辅助预测、clip 顺序 embedding、零门控残差融合，以及 FSDP/保存/推理接入 |

WALA 源文件 SHA256 为 `9d9097e30309ac7d61ab2f0b5fd4cb313632eb2b85104fe29ab675d65a5b49b0`。测试在 `feature_normalization='none'` 下将包装器的编码、损失、梯度与原模块直接对照；默认启用的逐 patch 归一化是本分支的新增选择。没有使用 WALA 发布权重，也不是完整复现 WALA；新增 tokenizer 从头训练。GIFT 只提供设计启发，本分支没有借用其未核实的实现。全流程不需要 depth。

## 张量与监督

1. **离线冻结特征。** 同一个 DINOv2-base 编码 human 和 robot RGB，完整画面 resize 到 `224×224`，不做中心裁剪；去掉 CLS，得到每帧 `256×768` patch 特征。缓存 FP16；模块内逐 patch 非仿射 LayerNorm 在 FP32 计算，之后转换为模型计算类型。
2. **Human 条件。** 整段示范均匀采样 32 帧，组成 8 个有序 clip，每 clip 4 帧。WALA tokenizer 以 clip 第一帧及后续变化编码出 4 个 token；加入 clip 顺序 embedding，共 32 个 256 维 token。没有人为对象标签、深度、分割或阶段标签。
3. **Robot 辅助监督。** 缓存主相机 RGB，严格使用原 robot latent 的 `frame_ids[::4]`，而非用视频长度猜时间。相同 tokenizer 编码 robot 起点与未来，并重建未来特征变化。Human token 也输入同一个 decoder，固定以该配对 robot 训练片段的第一帧特征为条件，预测片段内后续变化。
4. **接入 Zero-WAM。** 原 human Wan latent 完成 patch embedding 后，增加低维 cross-attention 残差：`H_h + tanh(gate) * Attention(H_h, human_tokens)`。门初始为零；原 token 长度、RoPE、attention mask 不变。门由原视频/动作/MCP 训练目标学习打开。Robot future 只进入辅助损失，绝不进入 human 条件或推理。

每个辅助目标采用原 WALA RGB 的 `L1(delta_pred, delta_target) + 0.1 * cosine_loss`；三个目标为 human 自重建、robot 自重建、human→robot 预测，权重均为 1，整体乘 `rgb_loss_weight=0.1`。原视频/动作/MCP 损失不变，所有损失仅除一次梯度累积次数。

这里的“当前状态条件化”首先体现为辅助 decoder 由所选训练片段的起点预测未来；执行时仍由原 Zero-WAM 对 robot 当前观测和 human 条件联合建模。尚未实现每个执行 chunk 重新查询 transition decoder、显式完成进度或剩余任务 memory；没有声称已解决这些问题。原 human latent 仍保留，因此也不声称总上下文长度已压缩。

当前配置为 `max_future_steps=64`：每个序列最多 64 个后续特征时刻，加起点最多 65 个。这是与 Wan latent 对齐后的时间轴容量，原始视频帧号仍由 `frame_ids` 决定。Human 的默认 8×4 采样不变。预检会拒绝超长样本，不会自动截断或丢弃；正式训练前须对全部 600 组配对检查这一上限。

## 数据与环境

代码和环境留在 `/share/personal/xiangzishang`，数据、冻结特征、权重和训练 checkpoint 放 `/workspace`。

```bash
cd /share/personal/xiangzishang/code/Zero-WAM-RGB-Transition
export PYTHON_BIN=/share/personal/xiangzishang/envs/zero-wam/bin/python
export MODEL_PATH=/workspace/zero-wam/checkpoints/zero-wam-pretrain
export HUMAN_GEN_ROOT=/workspace/zero-wam/datasets/HumanGen
export ZERO_WAM_RGB_FEATURE_ROOT=/workspace/zero-wam/datasets/rgb_transition_dinov2_base
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1

# 完整 12 任务的 600 组训练配对。已有文件会复用。
"$PYTHON_BIN" tools/prepare_rgb_transition_features.py \
  --human-gen-root "$HUMAN_GEN_ROOT" --feature-root "$ZERO_WAM_RGB_FEATURE_ROOT" \
  --download-human --download-encoder --device cuda:0

# CPU 严格校验实际训练的全部配对；不加载大模型。
"$PYTHON_BIN" -m wan_va.train \
  --config-name robotwin_subset12_rgb_train --check-config
```

HumanGen 原始 human 视频发布在分片 `tar.zst` 内。工具固定官方 revision `971d12e942bf101c0d47716f928b1c044d02ac71`，流式解包，只保存所选训练视频；需要 `zstd` 可执行文件。冻结模型下载包括权重、processor、config 和 provenance 文件。工具不解码 Wan latent 冒充真实 RGB。

Human cache 与原 human latent 的相对路径一致；robot cache 保留任务、相机和片段路径。每项记录 encoder ID/revision、预处理、原视频和准确帧号。`sample_id` 在官方 manifest 内并非唯一，短测使用 `sample_id:pair_hash` 区分配对；同一 human 视频可以被多条 robot 配对复用。

训练在分配大模型前检查所有选中配对：任务数、缺失文件、有限值、human/robot 特征空间一致、所有样本预处理一致、RGB 与 latent 帧对齐、维度和长度上限。正式入口不会因缺少 cache 而静默缩小数据集。实际特征 provenance 写入 checkpoint，推理严格核对。全量准备状态以 600 配对预检为准。

## 有界训练与推理检查

```bash
# 仅准备短测数据：12 个训练任务各 1 个真实配对。
"$PYTHON_BIN" tools/prepare_rgb_transition_features.py \
  --human-gen-root "$HUMAN_GEN_ROOT" --feature-root "$ZERO_WAM_RGB_FEATURE_ROOT" \
  --download-human --download-encoder --smoke-one-per-task

export ZERO_WAM_SAVE_ROOT=/workspace/zero-wam/runs/rgb-transition-smoke
export ZERO_WAM_METRICS_PATH=/share/personal/xiangzishang/logs/rgb-transition-smoke.metrics.jsonl
CUDA_VISIBLE_DEVICES=0,1,2,3 bash script/smoke_rgb_transition.sh
```

短测入口默认四卡、每卡 batch 1、累积 2、6 个 optimizer step，关闭 W&B，最后保存 checkpoint。CLI 强制显式样本白名单，且 `--smoke-test` 仅允许 1–20 步，不会误跑 15,000 步。正式训练仍保留 W&B `zero-wam` 和每 5,000 步保存配置。

保存仍沿用基线的 SHA256 回读和 `COMPLETE.json`。这是 BF16 **weights-only** checkpoint，不包含 Adam/scheduler/RNG 状态，不能作为严格断点续训。新增 RGB 参数和实际特征配置随 transformer 保存；从原 checkpoint 加载时显式初始化新模块，已有 RGB checkpoint 则严格检查配置和参数完整性。

`tools/smoke_rgb_inference.py` 使用真实三相机观测、真实 human 示范和保存后的 checkpoint，检查：动作形状/有限值、实际 observation KV 更新、更换示范与 reset、CFG negative 隔离，以及推理不访问 robot future。默认保留官方 `50/50` 采样步数。

```bash
# INFERENCE_MODEL_ROOT 应含 transformer/vae/tokenizer/text_encoder；
# transformer 指向本次 checkpoint，其余组件复用原模型。
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" tools/smoke_rgb_inference.py \
  --model-path "$INFERENCE_MODEL_ROOT" --second-chunk \
  --output-json /share/personal/xiangzishang/logs/rgb-transition-inference.json
```

正式 7 任务测评也需要对应 human RGB 特征。使用基线评测实际采用的固定示范映射，避免取成 manifest 中任意首条示范：

```bash
"$PYTHON_BIN" tools/prepare_rgb_transition_features.py \
  --human-only \
  --eval-human-map evaluation/robotwin/robotwin_icl_human_videos.py \
  --download-human --device cuda:0
```

此路径只读取 human 示范，不读取 robot 视频或未来 latent。评测示范只作为测试时输入，不能进入上述 600 组训练。推理服务设置 `ZERO_WAM_RGB_FEATURE_ROOT`（也兼容 `RGB_FEATURE_ROOT`），或传 `--rgb-feature-root`；原 7 任务评测入口及其示范选择不变。功能推理短测不是 RoboTwin 闭环成功率测评。

## 正式对照实验

保持初始化 checkpoint、12/7 划分、全局 batch 8、15,000 步预算和原 7 任务测评参数一致，比较 `robotwin_subset12_train` 与 `robotwin_subset12_rgb_train`。不要使用 43 任务后训练权重初始化“12 任务”对照。后续应增加等 token/参数预算的无 transition 监督融合对照，并做 human 时序打乱与错误示范干预，区分结构、辅助目标和新增容量的影响。

当前门控、损失权重、clip 数量都是明确的初始实验设定，尚未通过最终成功率选优。先以短测和全数据预检排除实现问题，再决定是否投入长训练。

确认全量预检后，正式训练使用独立输出目录，并沿用基线文档的 W&B 登录/团队变量：

```bash
export ZERO_WAM_SAVE_ROOT=/workspace/zero-wam/runs/robotwin-subset12-rgb-b8
export WANDB_PROJECT=zero-wam
export WANDB_NAME=robotwin-subset12-rgb-b8
CONFIG_NAME=robotwin_subset12_rgb_train NGPU=4 MASTER_PORT=29683 \
  bash script/train.sh
```

上面是后续实验命令，不属于本次短测执行范围。

## 本机实测：2026-11-03

使用 `/share/personal/xiangzishang/envs/zero-wam/bin/python` 和 4 张 A800 80GB，日志位于 `/share/personal/xiangzishang/logs/rgb-transition-20261103/`。

| 检查 | 实测结果 |
| --- | --- |
| 完整测试集 | `151 passed`，包含 WALA 原实现数值/梯度对照、梯度累积、dropICL、缓存对齐、checkpoint 和推理隔离测试 |
| 正式训练数据 | 12 任务 / 600 配对完整预检通过；239 个独立 human 原视频；human 特征均为 `[8,4,256,768]`，robot 时间长度 7–35 |
| 固定测评示范 | 原评测映射的 7 个 human 原视频及 RGB 特征齐全；没有提取 heldout robot future |
| 四卡短训练 | 6 步，全局 batch 8、累积 2；所有损失/梯度有限，无跳过更新；gate 从 0 更新到 `6.495914e-6` |
| 保存与重载 | 8 个权重分片 SHA256 回读通过；重载无 missing/unexpected/mismatched keys，BF16 gate 与保存值精确一致 |
| 最长样本压力测试 | 从 6 步权重重载，12 个任务各选最长配对再跑 3 步，每条配对见到两次；强制保留 ICL，最大 35 帧，无 OOM/NaN/跳过更新，gate 继续更新 |
| 实际服务推理 | 官方 video/action `50/50`，ICL CFG 5、target text CFG -1；两个 chunk 均返回有限的 `[16,2,16]` 动作；真实三相机观测更新、human KV 保留、切换示范和 reset 均通过 |

短训练的编译后 step 为 11.3–13.4 秒；最长样本测试后两步为 12.5、15.7 秒。首步含编译，checkpoint 保存另计。这些是短测时间，不作为完整训练耗时或收敛速度承诺。

推理隔离额外做了同 checkpoint 的 RGB 开关对照：相同 packing 下，human A/B、保留/移除 human cache、启用/关闭 RGB 的 **CFG negative 最大差异均为 0**；关闭 RGB 会改变条件预测，表明新增条件路径确实被使用。另发现 packed CFG negative 与单 stream 的 BF16 预测不逐元素等价，最大差异 `0.37890625`；关闭 RGB 的原骨干路径逐元素复现同一差值。该跨形状差异单独记录，没有通过放宽容差把它当作隔离通过。CPU FP32 小模型的两种形式在 `1e-5` 容差内一致。

检查过程中已修复：FSDP 不支持标量参数、新分支在 Diffusers 缺失权重加载时未初始化、各 rank 初始化可能不同、dropICL 零损失归约类型不一致、时间长度上限不足、manifest ID 重复，以及测试入口 `inference_mode` 与 FSDP 版本计数不兼容。推理短测使用与正式服务相同的 `no_grad`。

主要本机证据：

- 数据：`/workspace/zero-wam/datasets/rgb_transition_dinov2_base/preparation_report.json`、`eval_human_feature_report.json`、`max_length_feature_report.json`。
- 训练：日志目录中的 `train-v2.log`、`train.metrics.jsonl`、`training_checks.json`；最长样本为 `max-length.log`、`max-length.metrics.jsonl`、`max_length_checks.json`。
- 推理：日志目录中的 `inference.json`，包含原始误差统计、RGB 开关对照和动作/缓存证据；早期失败记录也保留。
- 短测 checkpoint：`/workspace/zero-wam/runs/rgb-transition-smoke-20261103/checkpoints/checkpoint_step_6/`。
- 推理组件入口：`/share/personal/xiangzishang/tmp/rgb-transition-smoke-model-20261103/`，仅存放指向 `/workspace` 权重的链接。

本次只执行上述 6+3 步有界训练，没有启动 15,000 步训练，也未测量新方法的 RoboTwin 闭环成功率。
