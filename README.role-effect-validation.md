# C1 原型验证记录（2026-11-18）

这是训练/推理链路验证记录，不是任务成功率报告。未进行15,000步正式训练或7任务×100次评测。当前角色标注仅覆盖2/600训练配对，来自SAM2弱标签和Codex视觉审查，不是人工GT。

## 已完成的代码与数据检查

- 最终模型/数据/原RGB回归测试：251 passed，44.21秒。
- SAM2预处理的独立CPU回归：25 passed，7.69秒；三实体/两阶段、gap、乱序预测、可见性与来源校验均覆盖。合计276项，无需在测试中运行SAM2或GPU。
- 验证teacher、robot future、donor、实体ID不会改变主策略条件；有梯度的donor只进入effect分支；不同实体的未来差分被屏蔽。
- 新增/原始checkpoint加载和BF16小模型测试通过；真实部署的角色绑定读取当前cam_high，原readout仍读取三视角。
- 全量record导出验证600配对/12任务，含239独立human视频与600 robot视频；完整cache来源、帧锚点和视频首帧可解码。导出的seed模板没有自动填写或批准角色。
- 真实mask候选覆盖2human×32帧、2robot×10帧。审查后，human有效角色共75/128，robot有效角色共34/40；遮挡段不作负标签。

## 真实四卡训练

初始化：`/workspace/zero-wam/checkpoints/zero-wam-pretrain`，没有复用RGB训练checkpoint。

配方：4×A800 80GB，全参数Wan主干与MCP；microbatch1×accum2×4=8；6 optimizer steps，ICL dropout=0，禁用W&B。保留12任务的14个真实配对，其中2个有角色标签和双向合法donor。

运行结果：

- 6步完成，优化器跳步0，所有记录指标有限。
- `role_human / role_robot / object_future / relation_future / donor_effect` 均实际产生非零损失。
- 首步包含编译耗时170.98秒；后续步骤11.20–15.30秒，中位数12.72秒。
- rank0峰值allocated显存51.668GiB。该值不是所有rank的实测最大值。
- 最后记录的RGB gate为`-6.79379e-6`；有限步loss或gate变化不证明泛化提升。
- checkpoint共8分片，逐片SHA256写后校验通过；`COMPLETE.json`记录smoke状态、角色manifest hash及覆盖率。

运行目录：`/workspace/zero-wam/runs/role-effect-smoke-20261118`。

checkpoint：`/workspace/zero-wam/runs/role-effect-smoke-20261118/checkpoints/checkpoint_step_6`。

指标：同运行目录的`metrics.jsonl`；短测摘要：`local/role_effect_smoke/training_summary.json`。

训练中修正过两项问题，均发生在完成任何optimizer步骤之前：预检元数据的整数key不能写入EasyDict，已转字符串；容器PATH缺少C编译器，改用已有GCC12.4。独立CUDA Inductor编译测试通过，未安装依赖或更改全局环境。

## 真实模型推理

真实checkpoint重载及两chunk离线推理已通过，报告为`local/role_effect_smoke/inference_pass.json`，`status=passed`、`task_success_evaluated=false`。

- 严格重载：missing / unexpected / mismatched keys均为空，无加载错误。
- 配置：video/action各2步去噪，seed42；不是正式50步采样或机器人rollout。
- 使用真实`beat_block_hammer`观测完成两chunk，动作输出均为`[16,2,16]`且有限；第二段读取新真实观测。
- 无角色标签、donor或robot future作为推理输入；当前状态干预确实影响最终role readout，CFG-negative保持逐元素不变。
- 切换到另一段真实的瓶子放置示范、重置缓存以及恢复示范检查通过。
- BF16 checkpoint中的gate为`-6.7949295e-6`，与载入值一致。

仅推理两chunk的耗时分别9.77秒、6.96秒；完整验证含模型加载、编译及多次探针共288.87秒。这些数值不作为正式推理速度基准。

推理bundle使用`/tmp/zero-wam-role-effect-smoke-20261118-bundle`：transformer指向上述新checkpoint，其余组件指向原pretrained。`/workspace`是不能创建symlink的FUSE存储，因此没有在那里复制额外模型。

## 可复查产物

- `local/role_effect_smoke/tests_final.log`：251项测试输出。
- `local/role_effect_smoke/train_attempt3.log`：成功训练的完整日志。
- `local/role_effect_smoke/reviewed_v2/reviewed_manifest_v2.json`：两个配对的审查范围、弱标注来源、donor语义和文件hash。
- `local/role_effect_smoke/sam2_candidate_v1/`：未改动的原始候选、全部采样帧overlay及置信度/面积记录。
- `local/role_effect_smoke/all600_records.json`、`all600_seed_template.json`：全量准备记录与未批准提示模板。

这些本机产物和大权重没有加入Git；代码、测试、配置、准备工具及来源说明在新分支中。
