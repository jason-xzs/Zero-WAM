# RGB Role–Effect：方法依据、代码来源与实现边界

核查日期：2026-11-18。本文记录新分支的设计依据与来源，不是完成报告；不据此声称训练、推理或任务成功率检查已经通过，也不声称新颖性已经证明。

## 本轮范围

第一版选择 **2D role/event 监督 + recipient binding + local-future prediction + donor 对照**，沿用 RGB-Transition 的冻结 DINOv2 与原 Zero-WAM 主干。3DGS、NeRF、预训练几何模型及下文 C2 均为后续候选，本文不表示它们已实现。

- **Role / event**：区分本次操作的被操作对象或部件、参照对象或部件，以及它们之间发生的变化。角色是一次事件中的关系，不是永久对象类别；容器也可能在下一事件中成为被操作对象。
- **Recipient binding**：把 donor 示范中的角色映射到 recipient 机器人当前场景的对应区域。先保留可检查的区域/part 权重与可见性，不能把任意 learnable query 自动解释为语义角色。
- **Local future**：以当前 recipient 区域为条件，监督任务相关区域的未来特征/关系变化，避免仅靠全画面重建奖励背景和主体外观。真实 robot future 只作训练目标，不进入部署条件。
- **Donor 对照**：用另一个有效示范提供 effect、由 recipient 提供对象绑定和当前状态，检查预测是否保留操作效果并适应新布局。正样本需要可核验的相同事件语义；同 task ID 不自动意味着相同步骤、对象角色或可交换效果。无有效 donor 时应记录缺失，不能静默把自身当作跨示范证据。
- 这些术语指定要验证的结构和监督，不意味着分离、跨对象迁移或组合泛化已经成立。分组质量、错误角色、错误 donor、顺序干预和任务成功率需要分别检查。

## 已有直接复用：以仓库 NOTICE 为准

已阅读 [WALA NOTICE](wan_va/modules/third_party/wala/NOTICE.md)、其 [MIT LICENSE](wan_va/modules/third_party/wala/LICENSE)，以及 [OpenFlamingo NOTICE](wan_va/modules/third_party/open_flamingo/NOTICE.md)。下表区分继承的代码与本轮设计。

| 组件 | 精确来源与实际边界 |
| --- | --- |
| WALA tokenizer / decoder | [WALA 论文](https://arxiv.org/abs/2607.11397)；[官方源文件](https://github.com/liujiahao2077/WALA/blob/4a944f07afe09e24743f20d3d9ab809db672dc63/wala/model/modules/latent_action_model/latent_action_model.py)，commit `4a944f07afe09e24743f20d3d9ab809db672dc63`。本地 vendor 为原样复制，MIT。 |
| WALA 集成边界 | 现有 RGB 包装器使用 `use_depth=False`，没有使用 WALA 发布权重；保留 RGB delta 的 L1 + cosine 目标。原参数 `smooth_l1_weight` 在该 revision 实际加权 L1，本地称 `l1_weight`。归一化、human→robot 监督、clip 顺序与门控融合属于本仓库集成。 |
| DINOv2 | [论文](https://arxiv.org/abs/2304.07193)、[官方代码与 Apache-2.0 许可证](https://github.com/facebookresearch/dinov2)。既有缓存使用 `facebook/dinov2-base`、revision `f9e44c814b77203eaa57a6bdbbd535f21ede1415`，通过 Transformers 加载。DINO patch features 是语义特征，不等于 metric 3D 坐标。 |
| Perceiver building blocks | 已有 [OpenFlamingo 源码](https://github.com/mlfoundations/open_flamingo/blob/655f693fbfa04cd6e9a987d960654624d48917cf/open_flamingo/src/helpers.py)，commit `655f693fbfa04cd6e9a987d960654624d48917cf`，MIT。vendor 提取 `FeedForward` / `PerceiverAttention`，仅等价替换 `rearrange_many` 依赖；不代表复现 Flamingo。 |

WALA 原方法使用 DINOv3 与深度监督，当前 RGB 分支使用 DINOv2、无深度；不能把该分支的结果报告成完整 WALA 复现。已有方法、数据与配置细节见 [RGB-Transition 说明](README.rgb-transition.md)。

## 与现有 State Readout 的区别

[现有 readout](README.rgb-readout.md)研究“当前观测如何查询同一个 human memory”，包括 static、state、global-local 和 future-teacher。其状态来自机器人当前可用的 Wan 观测；普通状态 query 本身没有对象角色、关系效果或 donor 可交换性监督。

本轮只借用其**因果观测接口**：训练与部署均只让新条件路径读取已经可用的机器人观测。新的研究对象是角色/事件表示、recipient 绑定及局部效果监督；不能仅把 state readout 重命名为 role-effect，也不把它的旧 smoke 记录当成本分支验证结果。

原 human Wan 条件是否保留、融合位置与 token 数，以新分支实际配置为准。比较新监督时应匹配这些条件，避免把接入位置或新增容量的影响全归因于角色分离。

## 2D 对象/part 表示：参考与候选依赖

以下官方实现已经核实；除上节已有 vendor 外，列出来源不等于已经复制代码或采用权重。若新增复制，须在相应文件保留原署名并记录 commit、修改与许可证。

| 来源 | 已核实机制、许可证与本分支用途 |
| --- | --- |
| [SAM 2](https://arxiv.org/abs/2408.00714)；[官方实现](https://github.com/facebookresearch/sam2) | Promptable image/video segmentation；可由首帧点/框/mask传播对象区域。官方 [LICENSE](https://github.com/facebookresearch/sam2/blob/main/LICENSE) 为 Apache-2.0，README 明确模型 checkpoint 和训练代码同许可证。拟作离线 mask/track 来源，不复制其模型架构；提示来源、遮挡与失效必须记录。SAM2 不负责识别操作角色或跨 human/robot 身份。 |
| [DINOSAUR，ICLR 2023](https://arxiv.org/abs/2209.14860)；[官方框架](https://github.com/amazon-science/object-centric-learning-framework) | 冻结 DINO + slot attention，通过重建 DINO 特征学习对象分组；[LICENSE](https://github.com/amazon-science/object-centric-learning-framework/blob/main/LICENSE) 为 Apache-2.0。参考对象分组与重建监督，不能称其已经提供 task roles。 |
| [VideoSAUR，NeurIPS 2023](https://arxiv.org/abs/2306.04829)；[官方代码](https://github.com/martius-lab/videosaur) | 在特征重建之外预测跨帧 feature similarity；官方还提供 DINOv2 DINOSAUR / VideoSAUR 配置与 checkpoint。[LICENSE](https://github.com/martius-lab/videosaur/blob/main/LICENSE) 为 MIT，继承部分保留各自许可证。参考时序分组目标；首版不要求加载其模型。 |
| [Relational NDF，CoRL 2022](https://arxiv.org/abs/2211.09786)；[官方实现](https://github.com/anthonysimeonov/relational_ndf) | 两个对象的3D descriptor fields描述局部关系并迁移 rearrangement；[LICENSE](https://github.com/anthonysimeonov/relational_ndf/blob/master/LICENSE.md) 为 MIT。借鉴对象局部关系表示，不复用其点云优化控制流程；其3D输入与物体模型假设不属于当前RGB设置。 |

首版可以从可信 mask 区域聚合 DINO 特征；没有可靠 mask 时，learned grouping 仍是可实验的替代，但需要独立报告分组质量。完整对象 mask 不一定分辨微波炉把手、铰链等部件，part 分辨率是需要检查的数据问题。

## 3DGS / NeRF：可复用来源与新增条件

| 来源 | 真实作用与许可证 | 对当前任务的边界 |
| --- | --- | --- |
| [Feature3DGS，CVPR 2024](https://arxiv.org/abs/2312.03203)；[官方代码](https://github.com/ShijieZhou-UCLA/feature-3dgs) | 任意维 feature Gaussian rasterizer；原实验教师为 SAM / CLIP-LSeg。[LICENSE](https://github.com/ShijieZhou-UCLA/feature-3dgs/blob/main/LICENSE.md) 为 Gaussian-Splatting License，研究/评估用途。 | 每场景优化；官方流程需要 COLMAP/SfM 相机和点云。换为 DINO 教师需要适配，不是现成通用视频编码器。 |
| [LangSplat，CVPR 2024](https://arxiv.org/abs/2312.16084)；[官方代码](https://github.com/minghanqin/LangSplat) | SAM层级区域 + CLIP语言特征、scene-wise autoencoder与3D Gaussian语言场。[LICENSE](https://github.com/minghanqin/LangSplat/blob/main/LICENSE.md) 同为 Gaussian-Splatting License。 | 原方法不是“DINO 3DGS”；自定义场景需相机/SfM数据及先训练的RGB Gaussian场，不宜仅因有公开scene checkpoint就当作通用预训练特征。 |
| [FeatureNeRF，ICCV 2023](https://arxiv.org/abs/2303.12786)；[官方代码](https://github.com/jianglongye/featurenerf) | 蒸馏 DINO/扩散特征的 generalizable NeRF；可从单图预测3D语义特征体、迁移keypoint/part。[LICENSE](https://github.com/jianglongye/featurenerf/blob/main/LICENSE) 为 MIT。 | 单图推理确实存在，不能笼统说所有NeRF都要求测试时多视图。官方训练/模型围绕六类ShapeNet对象；适配动态操作场景仍需对象、相机、坐标和类别泛化验证。 |
| [VGGT，CVPR 2025](https://arxiv.org/abs/2503.11651)；[官方代码](https://github.com/facebookresearch/vggt) | 前馈预测相机、depth、point maps、tracks；不是可微渲染场。[LICENSE](https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt) 为自定义 VGGT License；README 区分原 checkpoint 与 Commercial checkpoint 的使用许可。 | 可作为预训练几何候选教师，无须先拟合3DGS；预测尺度、坐标和动态遮挡误差仍需检查，不能视为已标定机器人坐标真值。 |

当前可直接复用的训练链路是RGB特征与Wan latent；本分支尚未建立可核验的**human metric depth、跨视角同步与相机到机器人基坐标标定**。这表示当前接口/证据不足，不断言原始数据绝无相关字段。

把不同时间的人类操作帧直接视作静态场景多视图，会把对象运动混入相机/几何估计。动态GS能扩展这一条件，但需要运动/形变/遮挡建模；它不会自动解决任务角色或跨示范对应。

后续可分开对照：①纯2D区域与关系；②同样语义特征加置信度筛选的预训练几何；③若有同步RGB-D和标定，仅用robot侧3D作为训练教师；④具备重建条件后再增加GS/NeRF一致特征。各项单独报告输入、监督、预处理成本与重建失效，不能把几何方法的额外信息隐藏在“同RGB输入”中。

## 必须承认的近邻先例

| 论文 | 全文已核实的重叠与区别 |
| --- | --- |
| [ViVLA / See Once, Then Act](https://arxiv.org/html/2512.07582v1)，2025-12-08，§III-A / III-C | DINOv2当前/未来特征→transition latent action tokens→VQ；跨embodiment共享tokenizer并做action cycle consistency，用于测试时单视频示范策略。还用手/对象pose、object meshes和robot URDF等构建3DGS配对数据；“加DINO”“transition tokens”“结合3DGS”都不能单独声称首次。本次未找到可核验作者代码/代码许可证，不直接复用。 |
| [JEDI](https://arxiv.org/html/2610.40356v1)，2026-10-29，§3.3 / 3.4 / 3.7 | 示范slots与query当前状态预测冻结V-JEPA空间的下一stage keyframe，并用task/process negatives。Human阶段明确由human slots预测robot future；另加手/对象keypoints。因此“robot future监督human context”也有先例。本次未核验到官方发布代码/许可证，不将论文公式描述为已复用源码。 |
| [PEWAM / Keep the Effect, Drop the Actor](https://arxiv.org/html/2610.02398v1)，2026-10-01，§3 / Appendix A / L | Actor-removed对象3D轨迹、contact points、终态构成effect program，多执行共享effect已有训练先例。纯RGB示例只画图像坐标；真实机器人四任务用校准RGB-D和首帧对象点击。全文称接受后释放代码，尚未核验官方实现。我们不复用其闭环pointer/replanning；也不能仅以“效果与执行分离”宣称新颖。 |
| [SAKI](https://arxiv.org/html/2609.36031v1)，2026-09-28，§III-A–D | 已有角色重绑定、重复技能调用、跨示范排序，以及继承上一技能后的场景/机器人状态。其实现由 agent 准备显式 roles/phases/contact/adaptation rules，再进行几何绑定和全身运动优化；所读方法没有训练 human→robot 共享 latent transition operator 的目标。不能将“同操作、不同绑定、变化后状态”本身作为 C2 新颖性，也不能因其包含闭环执行就忽略概念重叠。 |
| [Compose by Focus](https://arxiv.org/html/2509.16053v2)，2026-03-08 revision，ICRA 2026，§III-A–D；[官方项目](https://computationalrobotics.seas.harvard.edu/SkillComposition/) | 任务相关对象/关系子图直接进入单个多技能策略，点云 encoder、GAT 和 diffusion action policy 以 BC 目标联合训练；VLM 在测试时分解任务，假定所需原子技能均已训练。因此不是仅做技能规划，但其设定也不是从未见 human video 提取操作的 ICL。 |
| [ORION](https://link.springer.com/article/10.1007/s10514-026-10253-8)，2026-05-08，§4.1–4.3 | 单人类视频→对象关系关键帧图；显式 target/reference 对象、跨场景对应、轨迹归一化/变形及动作优化已有先例。当前关系检索下一关键帧，并以新起终状态生成运动。不能将对象角色或对象中心跨人机转移本身作为 C1 首创；它所述机制与本分支的 learned role masks、recipient-local future、合法 donor effect-only 训练干预不同。 |

本分支可检验的差异在于：RGB输入下显式监督角色/part绑定、recipient局部未来，以及跨示范donor干预是否保持效果并改变绑定。它是否构成足够方法贡献，仍需要更完整近邻比较与对照实验，不能由模块组合或单次成功率上涨直接推出。

## 后续 C2 例子：共享操作，而非复用动作

三块堆叠可以表示为有序事件 `PLACE_ON(B, A)`、`PLACE_ON(C, B)`。两个事件共享 `PLACE_ON` 的效果算子，但绑定对象不同；第二次目标支撑面更高，输入状态已改变，抓取/运动也须重新由策略生成。

因此“可重复操作表示”不是把第一次的控制序列复制第二遍，也不是给token简单加一个repeat计数。需要保留事件顺序、参数绑定和应用后的状态条件；可用两块→三块、交换底座、重复次数与顺序干预区分算子复用和动作记忆。本轮C1实现不等于已经实现该C2。

**近邻核查后的收窄**：上述复用示例本身已被 SAKI 等工作覆盖。后续若研究 `z_next = F(z_current, effect, binding)`，差异应落在如何从视频学习共享状态变化算子、怎样约束多次组合后的状态、以及如何排除记住技能标签或顺序的替代解释；写出这个公式本身仍不证明新颖。C1 当前实现没有递归状态组合，不能将它提前报告为 C2 已完成。

## 证据边界

完整训练、受控表示干预和同seed/同human示范的任务评测承担不同证据责任。有限步 smoke 只能检查数据对齐、梯度、保存加载及因果输入隔离；低辅助loss或attention热图不证明角色语义、效果可交换或未见任务泛化。运行状态和结果应另由本分支实际日志记录，本文不预填通过结论。
