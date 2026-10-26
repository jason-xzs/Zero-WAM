# Full-Robotwin：四卡训练与子集实验

本分支统一使用 `wan_va.train`。43 任务和 12 任务只通过配置文件切换，没有独立的子集训练脚本，也不需要 `local/subset12/` 或迁移包。

本文假设环境、数据和初始化权重已经准备好，只说明训练与推理配置。

## 配置选择

| 配置名 | 配置文件 | 训练任务 | 训练步数 |
| --- | --- | --- | --- |
| `robotwin_train` | `wan_va/configs/va_robotwin_full_train_cfg.py` | 官方 43 个训练任务 | 50,000 |
| `robotwin_subset12_train` | `wan_va/configs/va_robotwin_subset12_train_cfg.py` | 指定 12 个任务 | 15,000 |

子集配置深拷贝 43 任务配置，只改任务白名单、任务数和总步数。两者都使用 4 GPU、每卡 batch 1、梯度累积 2，**全局 batch = 8**，每 5,000 步保存 checkpoint，W&B project 为 `zero-wam`。

原 `va_robotwin_train_cfg.py` 保留作为其他公开数据集也会继承的基础配置；四卡实验参数放在 `va_robotwin_full_train_cfg.py`，通过上表的注册名选择，避免改变其他数据集的训练默认值。

## Clone 与启动子集训练

```bash
git clone --branch Full-Robotwin --single-branch https://github.com/jason-xzs/Zero-WAM.git
cd Zero-WAM
```

在已配置好的 Python 环境中，从仓库根目录执行。下面的路径指向已有资源；输出目录应为本次实验的独立目录。

```bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MODEL_PATH=/workspace/zero-wam/checkpoints/zero-wam-pretrain
export HUMAN_GEN_ROOT=/workspace/zero-wam/datasets/HumanGen
export ZERO_WAM_SAVE_ROOT=/workspace/zero-wam/runs/robotwin-subset12-b8
export LOG_ROOT=/share/personal/xiangzishang/logs/zero-wam-subset12
export WANDB_PROJECT=zero-wam
export WANDB_MODE=online
export WANDB_NAME="robotwin-subset12-b8-$(date +%Y%m%d_%H%M%S)"
export WANDB_DIR="$LOG_ROOT/wandb"
export ZERO_WAM_METRICS_PATH="$LOG_ROOT/$WANDB_NAME.metrics.jsonl"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export ZERO_WAM_CPU_OFFLOAD=0
export ZERO_WAM_ACTIVATION_OFFLOAD=0
# 同步策略由选定配置控制，清除旧启动器可能留下的环境覆盖。
unset ZERO_WAM_SYNC_EACH_MICROBATCH ZERO_WAM_GPU_MEMORY_FRACTION
mkdir -p "$LOG_ROOT" "$WANDB_DIR" "$ZERO_WAM_SAVE_ROOT"

# CPU 检查实际任务选择与 batch；不加载模型、不初始化 CUDA/W&B。
python -m wan_va.train --config-name robotwin_subset12_train --check-config

# 同一个公共训练入口；默认 NGPU=4。
CONFIG_NAME=robotwin_subset12_train NGPU=4 MASTER_PORT=29641 \
  bash script/train.sh 2>&1 | tee "$LOG_ROOT/$WANDB_NAME.log"
```

W&B 使用服务器已有登录或 `WANDB_API_KEY`。如需要指定同一团队，设置 `WANDB_ENTITY=1661825351-beijing-institute-of-technology`；也兼容原 `WANDB_TEAM_NAME`。密钥不写入仓库或命令行。

CPU 检查应显示 `world_size=4`、`effective_batch_size=8`、`gradient_accumulation_steps=2`、`num_steps=15000`、`save_interval=5000`、`sync_each_microbatch=true`。数据源中应为 `selected_task_count=12`；完整 50 任务数据入口下，`excluded_roots=38`。正式训练加载完成后，当前发布数据应显示 `600 samples from 12 datasets`。

如已有数据位置不同，修改上面的路径变量即可；也保留 `--dataset-path`、`--icl-manifest-path`、`--human-latent-path` 等原 CLI 覆盖参数。`--check-config` 验证路径下的任务清单与 action stats，并不代替所有 latent/权重文件的完整校验。

启动 43 任务时，设置独立输出和日志目录，然后仅切换配置：

```bash
export ZERO_WAM_SAVE_ROOT=/workspace/zero-wam/runs/robotwin43-b8
export WANDB_NAME="robotwin43-b8-$(date +%Y%m%d_%H%M%S)"
export ZERO_WAM_METRICS_PATH="$LOG_ROOT/$WANDB_NAME.metrics.jsonl"
CONFIG_NAME=robotwin_train NGPU=4 MASTER_PORT=29641 \
  bash script/train.sh 2>&1 | tee "$LOG_ROOT/$WANDB_NAME.log"
```

两条命令是两个实验的启动方式；使用同一组 GPU 时择一运行。

## 训练参数与任务

共同配置：fused AdamW，学习率 `1e-4`，betas `(0.9, 0.95)`，weight decay `0.01`；warmup `200` 步后 constant；梯度 norm 裁剪为 `1.0`，异常或超过 `20.0` 时跳过更新。FP32 主参数、BF16 计算、FP32 梯度归约；开启 activation checkpointing，关闭 CPU/activation offload；每个 microbatch 同步分片梯度，每两个 microbatch 更新一次参数。

ICL/MCP 沿用当前训练：`drop_icl=0.1`、`droptext_target=0.4`、`icl_rope_h=24`；4 个 MCP 模块，采集层 `[3,11,19,29]`，loss weights `[0.5,0.25,0.15,0.1]`，future stride `2`，MCP SNR shift `10`。训练 chunk 随机 `1–4`、attention window 随机 `4–64`；video/action SNR shift 为 `5/1`。

12 个训练任务在 `wan_va/configs/robotwin_tasks.py` 中定义：

```text
move_pillbottle_pad    place_mouse_pad       place_container_plate
place_bread_skillet   place_can_basket      pick_dual_bottles
open_laptop          put_object_cabinet    beat_block_hammer
stack_blocks_two     stack_bowls_two       stack_bowls_three
```

43 任务配置排除官方 7 个测评任务；12 任务配置在此基础上应用白名单。同一公共数据加载器负责筛选，缺少所选任务、出现重复数据变体、白名单与测评任务重叠时拒绝训练。主配置中的任务设置会传入实际数据源，配置注册表不会被运行时改写。

官方 7 个测评任务，两种配置相同：

```text
place_object_scale   stamp_seal         open_microwave
move_stapler_pad     place_bread_basket place_empty_cup
stack_blocks_three
```

子集 600 组配对，43 任务 2,149 组；按相近样本遍历次数折算，`50000 × 600 / 2149 ≈ 13960`，取整为 15,000 步，约 200 轮。它是子集实验预算，不是已验证的最优收敛点。所有子集对照组使用相同初始化、划分和预算；从已训练 43 任务的权重开始，不能称为只用 12 任务后训练。

## 保存与评估

子集保存点为 5,000、10,000、15,000；43 任务保存到 50,000。路径为 `$ZERO_WAM_SAVE_ROOT/checkpoints/checkpoint_step_N/`，全部文件写入并完成 SHA256 回读校验后产生 `COMPLETE.json`。写入失败时各 rank 同步重试；额外保存不会改变正常保存间隔。

checkpoint 仍是 **BF16 模型权重**，不包含 optimizer、FP32 主参数、scheduler、RNG 和数据迭代位置，不能严格恢复训练。输出模型实体放在 `/workspace`。

两种训练配置共用 `evaluation.robotwin.evaluate`，四张卡各启动一个独立推理服务，执行相同 7 任务、3 seeds、每任务每 seed 100 次 rollout，共 2,100 次。推理仍是 `robotwin` 配置：video/action 采样步数 `50/50`，chunk `2`，window `64`，ICL CFG `5`，target text CFG `-1`，action CFG `1`，`icl_rope_h=24`。

调用方提供 `INFERENCE_MODEL_ROOT`、`ROBOTWIN_ROOT`、`ICL_LATENT_ROOT`、`EVAL_ROOT`。推理模型入口应包含完整模型组件，其中 transformer 指向待评估 checkpoint；只有 transformer 的训练目录不能直接作为完整推理根目录。

```bash
MODEL_PATH="$INFERENCE_MODEL_ROOT" python -m evaluation.robotwin.evaluate \
  --robotwin-root "$ROBOTWIN_ROOT" --icl-latent-root "$ICL_LATENT_ROOT" \
  --save-root "$EVAL_ROOT/checkpoint_step_15000" \
  --gpus 0 1 2 3 --seeds 0 1 2 --test-num 100
```

追加 `--dry-run` 可仅检查调度；每个 checkpoint 使用独立结果目录。成功判据使用本仓库官方版本：`stamp_seal` 为 3 cm 容差，`move_stapler_pad` 放在垫子上并释放、不限制 yaw。汇总使用任务等权平均，再报告三个 seed 的均值和总体标准差，输出单位 `[0,1]`。

## 相对官方代码的改动与验证边界

- 保留当前四卡训练已使用的梯度同步、日志/W&B 和 checkpoint 写入修复，以及关闭状态的可选 CPU offload 支持。
- 把旧 `local/subset12` 包装入口的任务筛选收进配置和公共数据加载器；增加公共 `--check-config`。四卡评估调度移到 `evaluation/robotwin/evaluate.py`，两种实验共用。
- `script/train.sh` 默认四卡，保留原训练入口并正确传递 CLI 参数；任务数和训练预算由配置决定。
- 未修改模型结构、MCP 实现、损失公式、视频/动作预处理、推理采样算法或成功判据。
- 梯度同步顺序保留数学上的全局 batch，但浮点归约顺序变化可能影响数值轨迹。既有四卡 FP32 小模型梯度/AdamW 对照不能证明完整 BF16 模型的最终成功率完全一致。
- 配置/任务选择和公共入口可用 CPU 检查；真实子集训练、完整模型训练轨迹对照及 RoboTwin 闭环评估仍需实际运行。环境安装、本机临时脚本、日志、数据和权重不属于本分支新增内容。
