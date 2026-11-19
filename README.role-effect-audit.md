# 正式训练前审查：2026-11-19

本次查出了实现问题并作了修正。下面的检查验证代码、梯度、数据边界和短程优化；不代表已完成15,000步训练，也不代表未见任务成功率提升。

**本机完整训练数据尚未就绪：600个配对中只有2个有审核后的角色弱标注。** 完整训练预检会明确报告 `2/600`、缺失598个配对，并在模型分配前停止。不能把 smoke 配置的 `ZERO_WAM_ROLE_ALLOW_PARTIAL=1` 当作完整角色监督实验。

## 修正了什么

| 问题 | 原行为 | 修正与边界 |
| --- | --- | --- |
| 新增变化特征丢失片段内顺序 | 保持首帧与帧集合不变，交换后续帧，effect 注意力汇总在数学上不变。原 WALA 分支有时间信息，不能代替这条分支的时间信息。 | 新训练配置显式加入可学习时间编码，交换中间帧的独立回归测试通过。旧 checkpoint 无该配置时保留原行为；不把增加时间编码称为已经实现操作组合。 |
| human 对象身份检查缺失 | 只检查 robot IDs；同一 human clip 的 source 从A变成C时，仍可能把不同对象特征相减。 | 校验 human IDs，每个 clip 内每个角色只允许一个已知实体；跨 clips 可以更换。当前模型不支持 clip 内换对象，遇到这种标注直接报错，需要保持对象一致或重新切分。 |
| donor 声明与实际 IDs 可矛盾 | `stable_roles=True` 只是一条声明，未检查是否与实际身份变化冲突。 | 同时核验 human/robot IDs，拒绝矛盾声明、同一实体占两个角色、错误角色顺序；未选入 smoke 的 donor 也要核验 sidecar。 |
| 有效性计数边界 | 独立计数函数可能把两个未知 ID `-1` 算作相同对象。 | 未知身份不能形成未来监督对；与模型实际损失的有效性规则一致。生产 loader 本来就禁止可见标签使用未知 ID。 |
| 无 donor 时多做一次编码 | 未标注样本用自身特征占位，也会执行第二次 human 编码。 | `donor_valid` 全为false时跳过第二次编码，保留用于分布式训练的零梯度连接；不影响策略输入。 |

时间编码沿用本仓库已有 [WALA tokenizer](wan_va/modules/third_party/wala/latent_action_model.py) 使用可学习时间 embedding 的做法。其他修正属于身份与数据契约检查，没有加入新的几何模型、语言模块、SAM推理依赖或测试时优化。

## 已执行的检查

- 全仓库测试：**425 passed**，81.10秒。初次运行有一个混合数据集测试因未设置本机 `HUMAN_GEN_ROOT` 失败；指向真实数据根目录后全量通过，没有跳过该测试。
- 补充动作路径检查后，Role集成测试 **6 passed**，包含一个新增用例；累计覆盖426个不同用例。该用例只对动作预测损失反传，不使用视频/辅助损失，确认初始对象特征、变化特征、时间编码与机器人区域匹配都收到有限、非零梯度。初始零gate的基线保持行为由另一个测试覆盖。
- 复现缺陷的新增用例在修正前失败，修正后通过；没有仅检查函数“能运行”。
- 检查 teacher masks、robot future、donor 和实体IDs只用于辅助目标/有效性，不改变同输入下的主策略输出；未标注帧不当作背景负例。当前观测来自已观测的干净latent，未来观测和生成视频不能替代它。
- 200步真实样本拟合：原始预训练Wan patch projection冻结，新RGB/角色模块随机初始化；使用2个审核配对、FP32、AdamW学习率`1e-4`、无warmup、chunk size 2。各辅助头有梯度，并能优于零变化基线。它只检验训练样本的可拟合性，不是200步完整策略训练。

| 辅助损失 | 随机初始化 | 200步后 | 始终预测无变化 |
| --- | ---: | ---: | ---: |
| human区域 BCE+Dice | 1.6604 | 0.4822 | — |
| robot区域 BCE+Dice | 1.6606 | 0.7140 | — |
| 对象未来特征 Smooth L1 | 0.17577 | 0.04081 | 0.15934 |
| 二维关系变化 Smooth L1 | 0.015427 | 0.00004684 | 0.00079941 |
| donor预测 | 0.19113 | 0.04085 | — |

- 4×A800 80GB完整模型：从原始`zero-wam-pretrain`初始化，Wan/MCP全参数训练，20个优化器窗口，全局batch 8，warmup 200，保留正式配置的ICL dropout 0.1。14配对覆盖12任务，其中2个有角色监督；五项角色相关损失均在15个窗口中实际出现。所有记录指标有限，优化器跳步0。首步后中位耗时12.67秒；rank0峰值allocated显存51.62 GiB，不冒充所有rank的峰值。
- 最长序列检查：检查600个配对的导出记录，最大为35个robot latent帧，长于上述短测的28帧。该配对的human视频7.04秒，同时处于全量human时长上限。再用包含该配对的15样本/12任务白名单跑2个优化器窗口（DDP补齐为16个样本，覆盖完整一轮），ICL dropout=0；两步完成，无OOM、无跳步。此检查不额外保存权重。
- 20步checkpoint：8个分片保存后逐片SHA256校验；保存配置包含`role_effect_temporal_encoding="learned"`，新增参数确实写入权重。以该checkpoint做的独立FP32/BF16时间顺序干预均产生不同输出；这不能解释为已学会动作顺序语义。
- 完整重载与两chunk、50/50去噪推理通过：missing/unexpected/mismatched keys及加载错误均为空；两段动作均为`[16,2,16]`且有限，分别耗时14.26、17.32秒（不含加载和诊断探针）。推理不使用角色标签、donor或robot future；观测更新、换示范、清空缓存正常。包含加载与全部探针共285.77秒，不作为正式速度基准。
- 当前观测的受控干预会改变新readout与正条件输出；同一输入packing下，改示范、改readout观测、关闭RGB分支的CFG-negative最大差异均为0。另有基线已知差异：packed与single-stream的BF16负分支最大差异0.4921875，关闭RGB后原样复现；未把它报告为跨packing等价。

## 仍需区分的限制

1. 目前只有2个同任务配对的SAM2弱标签和Codex视觉审查，不是人工GT，也不足以说明12任务角色识别或未见任务迁移。先补齐需要的标注，或明确设计部分监督实验。
2. 两个角色、二维区域中心/尺度与相对中心变化，不包含3D接触、inside/on真值，也不支持一个human clip内部切换实体。多阶段标注接口不能替代模型的这些限制。
3. 正donor监督不能单独排除模型忽略变化特征、依赖场景外观的捷径。需要多任务上的错误示范、移除变化特征、相同监督资源等消融，才能判断分离与迁移是否成立。
4. checkpoint沿用原仓库的 **BF16 weights-only** 保存，只支持权重重载；没有Adam、scheduler、RNG和采样器状态，不支持精确断点续训。此次没有把重载推理报告为断点续训验证。

## 复现与产物

真实样本拟合诊断：

```bash
python tools/check_role_effect_learning.py \
  --base-transformer /workspace/zero-wam/checkpoints/zero-wam-pretrain/transformer \
  --records local/role_effect_smoke/records.json \
  --manifest local/role_effect_smoke/reviewed_v2/reviewed_manifest_v2.json \
  --output-dir local/role_effect_audit/head_fit --device cuda:0 --steps 200
```

脚本拒绝覆盖已有报告；换一个output目录即可复跑。原始视频、feature cache、weak-label sidecar与checkpoint均不加入Git；数据准备方式见[方法说明](README.role-effect.md)。

20步完整模型检查：

```bash
export ZERO_WAM_ROLE_MANIFEST=/path/to/reviewed_manifest.json
export ZERO_WAM_SAVE_ROOT=/path/to/a/new/run
ROLE_SMOKE_STEPS=20 bash script/smoke_role_effect.sh --drop-icl 0.1
```

本机20步运行目录：`/workspace/zero-wam/runs/role-effect-audit-20261119`；最长序列检查：`/workspace/zero-wam/runs/role-effect-longest-audit-20261119`。原始日志与诊断报告位于`local/role_effect_audit/`。推理bundle在`/tmp/zero-wam-role-effect-audit-20261119-bundle`，只通过符号链接组合新transformer和原始模型组件。

推理检查：

```bash
python tools/smoke_rgb_inference.py \
  --model-path /tmp/zero-wam-role-effect-audit-20261119-bundle \
  --steps 50 --second-chunk --output-json local/role_effect_audit/inference.json
```

可移植数值摘要及受测源码SHA256见 [validation/role-effect-audit-20261119.json](validation/role-effect-audit-20261119.json)。该摘要明确标记 `full_training_ready=false`，不能替代缺失的角色数据或任务评测。
