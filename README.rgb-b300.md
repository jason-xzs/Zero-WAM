# RoboTwin Subset12 RGB：B300 单卡训练

配置名：`robotwin_subset12_rgb_b300`。启动器：`script/train_rgb_b300.sh`。

本配置根据用户提供的 `zero-wam-subset12-single-gpu-training-config.md`（2026-11-03）对齐已完成的单卡 Subset12 baseline。继承本分支的 RGB 模型、任务划分和辅助损失，只切换运行配方；四卡 `robotwin_subset12_rgb_train` 保持原配置。

## 参数对照

| 项目 | B300 单卡 |
| --- | --- |
| 进程数 / world size | 1 |
| 每卡 micro-batch | 1 |
| 梯度累积 | **8** |
| 全局有效 batch | **1 × 1 × 8 = 8** |
| 训练预算 | 15,000 个累积窗口，共 120,000 次样本消费 |
| 训练数据 | 相同 12 任务 × 50 = 600 配对，约 200 轮 |
| 学习率 | `1e-4`，200 步线性 warmup 后 constant |
| 优化器 | Fused AdamW，`foreach=False`，betas `(0.9, 0.95)`，epsilon `1e-8`，weight decay `0.01` |
| 梯度 | norm 裁剪 `1.0`，异常或超过 `20.0` 时跳过更新 |
| 精度 | FP32 主参数、BF16 计算、FP32 梯度归约 |
| 更新范围 | 全参数训练，保留 4 个 MCP 模块与 RGB 分支 |
| Activation checkpointing | 开启 |
| CPU / activation offload | 均关闭，启动器清除旧环境覆盖 |
| Seed | 42 |
| DataLoader workers | 4 |
| W&B | **默认关闭**，与此次提供的单卡 baseline 记录一致；可用 `--enable-wandb` 开启，project 为 `zero-wam` |
| 权重保存 | 每 5,000 步，SHA256 回读校验后生成 `COMPLETE.json` |

梯度累计期间逐 micro-batch 同步，optimizer/scheduler 只在第 8 次 micro-batch 后更新。首个窗口的学习率为 0，沿用 baseline warmup 行为。配置会拒绝四进程误启动及错误的全局 batch。

本分支的唯一参数总数为 `10,796,352,223`，其中 RGB 分支 `7,265,793`，原骨干 `10,789,086,430` 与单卡 baseline 记录相同。Checkpoint 清单中的 tensor 元素总数包含共享参数别名，不能直接作为唯一参数量比较。

**恢复策略的边界：** baseline 文档中每 1,000 步滚动保存完整恢复点，来自另一套 `train_formal_personal_v8.py`。本分支仍使用已验证的 **BF16 weights-only checkpoint**，没有迁移那套 optimizer/scheduler/RNG/sampler 恢复逻辑。本次配置对齐训练超参数和 5,000 步评测权重保存，不宣称具备相同的完整断点恢复能力。

## 迁移输入

在 B300 服务器使用已跑通 baseline 的 Python 环境。启动器默认调用当前环境的 `python`，也可通过 `PYTHON_BIN` 指定。B300 环境的 PyTorch/CUDA 和编译缓存应由该服务器管理。

训练需要：

- 原始 `zero-wam-pretrain` 模型，保持与 baseline 相同初始化；不要用已经完成 12 任务后训练的 baseline 权重初始化本实验。
- baseline 已用的 HumanGen 数据：任务 meta/actions、robot latents、human latents、ICL manifest。
- 本分支准备的完整 RGB 特征目录 `rgb_transition_dinov2_base/`，包括 `human/`、`robot/` 和用于短测的 `smoke_report.json`。原服务器位置为 `/workspace/zero-wam/datasets/rgb_transition_dinov2_base/`，600 组训练配对和固定 7 个测评 human 缓存均已准备。

只移动数据根目录时，保留任务/variant、chunk、camera、episode 区间目录结构即可复用缓存。元数据保留原始来源路径，但校验使用完整相对身份，并继续核对原 latent 帧号、encoder revision 和预处理；更换任务、相机或片段仍会报错。训练直接读取冻结特征，不需要重新下载 DINOv2 或重新解码 RGB。数据分开存放时，可向启动器传递公共入口的 `--dataset-path`、`--icl-manifest-path`、`--human-latent-path`、`--robot-latent-path` 覆盖参数。重新提取特征的方法见 [README.rgb-transition.md](README.rgb-transition.md)。

## 核对配置与启动

在 B300 上 checkout 本分支后，从仓库根目录执行。下面是目录约定示例，请将变量替换为该服务器的实际路径；启动器要求显式指定输入和输出，避免误用旧服务器目录。

```bash
set -euo pipefail
export PYTHON_BIN=python
export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH=/workspace/zero-wam/checkpoints/zero-wam-pretrain
export HUMAN_GEN_ROOT=/workspace/zero-wam/datasets/HumanGen
export ZERO_WAM_RGB_FEATURE_ROOT=/workspace/zero-wam/datasets/rgb_transition_dinov2_base
export ZERO_WAM_SAVE_ROOT=/workspace/zero-wam/runs/robotwin-subset12-rgb-b300-b8
export LOG_ROOT=/share/personal/xiangzishang/logs/zero-wam-rgb-b300
export ZERO_WAM_METRICS_PATH="$LOG_ROOT/train.metrics.jsonl"
mkdir -p "$LOG_ROOT"

# CPU 检查全部 600 组缓存及实际配置，不初始化训练进程组。
# 网络文件系统上扫描全量特征可能需要几分钟。
bash script/train_rgb_b300.sh --check-config | tee "$LOG_ROOT/config.json"
```

应显示 `world_size=1`、`batch_size=1`、`gradient_accumulation_steps=8`、`effective_batch_size=8`、`num_steps=15000`、`save_interval=5000`、`load_worker=4`、`enable_wandb=false`，实际数据源为 12 任务 / 600 配对。

先在目标 B300 上做 2 步有界短测，验证该机软件栈、显存和 checkpoint 写入：

```bash
SMOKE_IDS=$("$PYTHON_BIN" -c \
  'import json,sys; print(",".join(json.load(open(sys.argv[1]))["sample_ids"]))' \
  "$ZERO_WAM_RGB_FEATURE_ROOT/smoke_report.json")

ZERO_WAM_SAVE_ROOT=/workspace/zero-wam/runs/rgb-b300-smoke \
ZERO_WAM_METRICS_PATH="$LOG_ROOT/smoke.metrics.jsonl" \
bash script/train_rgb_b300.sh \
  --smoke-test --rgb-smoke-sample-ids "$SMOKE_IDS" \
  --num-steps 2 --save-interval 2 --load-worker 0 \
  2>&1 | tee "$LOG_ROOT/smoke.log"
```

短测仍为单卡累积 8；只有 12 个明确选中的真实配对，最多允许 20 步。原 `script/smoke_rgb_transition.sh` 是四卡入口，单卡使用上面的命令。

正式运行使用此前设置的独立 `ZERO_WAM_SAVE_ROOT`：

```bash
bash script/train_rgb_b300.sh 2>&1 | tee "$LOG_ROOT/train.log"
```

需要 W&B 时，在启动前设置自己的登录/团队环境变量，并添加 `--enable-wandb`：

```bash
export WANDB_PROJECT=zero-wam
export WANDB_NAME=robotwin-subset12-rgb-b300-b8
export WANDB_DIR="$LOG_ROOT/wandb"
mkdir -p "$WANDB_DIR"
bash script/train_rgb_b300.sh --enable-wandb 2>&1 | tee "$LOG_ROOT/train.log"
```

`CUDA_VISIBLE_DEVICES=4` 表示使用物理卡 4，仍只有一个训练进程；启动器不会继承旧的 `NGPU=4`。

本次代码回归 **177 项通过**，覆盖配置、8 次梯度累积、单进程 CLI 路由与迁移后的缓存身份。使用新数据根别名、保留原缓存元数据的 12 组真实配对预检及三相机推理输入读取也已通过。当前开发服务器是 A800，未在目标 B300 上执行整模型训练，也没有自动启动正式长训练。
