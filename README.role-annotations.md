# 在训练服务器准备完整角色标注

本分支：`RGB-Role-Effect-ICL`。本文件补齐“原视频 → 对象提示 → SAM2 候选 → PNG 审核 → reviewed manifest → 正式训练预检”的操作入口。**仓库包含准备工具，不包含已经完成的 600 对标注。** 原有两对标签仅用于有界验证，不能作为完整数据集。

`reviewed` 是看过具体视频/候选掩码之后的审核记录，不是把一个开关设为 true。可以由有视觉能力的 Codex 检查 PNG 并填写，但必须标明实际审核者，不能写成“人工真值”。SAM2 只提供训练辅助目标，部署时不输入 SAM2 标签。

## 可直接交给另一台服务器的任务

> 更新到 jason-xzs/Zero-WAM 的 RGB-Role-Effect-ICL 分支，按 README.role-annotations.md 在本机准备完整 600 个配对的角色标注。先定位本机 HumanGen、RGB 特征缓存、训练 Python 和 SAM2 Python；导出本机 records，再使用 tools/manage_role_annotations.py 建立准备目录。检查原始视频 PNG、填写对象身份和首帧框，分批跑 SAM2，逐一检查全部采样帧的掩码，把遮挡/跟踪错误标为无效，保留审核证据。多阶段示范保持每个 human clip 内对象身份固定；donor 必须另行核实语义一致，不能按任务名自动配对。最终合并并通过 check 的 600/600、12 任务完整预检。不要用 partial 模式、复制 reviewed=true 或关闭损失绕过缺失标注。预览用 PNG/视频，不用 HTML。报告最终 manifest 路径、各任务有效标签覆盖、donor 覆盖及未解决的问题；数据检查完成后再做短程训练验证。

## 1. 确认服务器自己的路径

在本仓库根目录操作。以下 `/path/to/...` 均需换成本机真实路径，不要直接复制另一台机器的绝对路径：

```bash
export PYTHON_BIN=/path/to/zero-wam-env/bin/python
export HUMAN_GEN_ROOT=/path/to/HumanGen
export ZERO_WAM_RGB_FEATURE_ROOT=/path/to/rgb_transition_dinov2_base
export MODEL_PATH=/path/to/zero-wam-pretrain
export SAM_PYTHON=/path/to/sam2-env/bin/python
export SAM_CHECKPOINT=/path/to/sam2.1_hiera_tiny.pt
export ROLE_ROOT=/path/to/new-role-annotation-run
mkdir -p "$ROLE_ROOT"
```

需要原始 human/robot 视频、原 Wan latent、正式 ICL manifest 和冻结 DINOv2-base 缓存。**不需要训练好的 RGB policy checkpoint。** RGB 缓存准备见 [README.rgb-transition.md](README.rgb-transition.md)。缓存必须对应正式配方的帧索引和 224×224 全图缩放，不能换成中心裁剪。

SAM2 独立环境需可用的 PyTorch/CUDA、官方 SAM2、`av`、`Pillow`。使用与目标 GPU 兼容的 PyTorch；尤其不要把 A800 的 CUDA 环境直接照搬到 B300。当前工具读取 SAM2 源码 Git commit，因此使用官方 Git checkout 安装，不用丢失源码信息的拷贝包。

已验证的官方源码 commit 为 `2b90b9f5ceec907a1c18123530e92e794ad901a4`，安装说明见 [该版本官方 README](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/README.md)。如果本机没有 SAM2，在独立环境安装该 checkout 与视频解码依赖。官方 tiny 权重下载地址是 [sam2.1_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt)，本机验证文件 SHA256 为 `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`。工具会记录实际源码版本、权重 hash 和 PyTorch 版本。

```bash
"$SAM_PYTHON" -c 'import torch, sam2, av; from PIL import Image; print(torch.__version__, torch.cuda.is_available(), sam2.__file__)'
```

## 2. 导出 600 个配对并建立可继续处理的目录

```bash
"$PYTHON_BIN" tools/prepare_role_records.py \
  --human-gen-root "$HUMAN_GEN_ROOT" \
  --feature-root "$ZERO_WAM_RGB_FEATURE_ROOT" \
  --output-json "$ROLE_ROOT/records.json"

"$PYTHON_BIN" tools/manage_role_annotations.py init \
  --records-json "$ROLE_ROOT/records.json" --work-dir "$ROLE_ROOT/work"
```

`init` 不覆盖旧目录。`records.json` 应为 **600 对、12 任务、每任务 50 对**。本次数据检查有 **239 段不同的 human video、600 段 robot video**。`work/index.json` 列出任务、pair keys 和可共享 human 提示的分组；每对位于 `work/jobs/<pair_key冒号改下划线>/`，内有 `records.json` 与空白 `seeds.json`。

先挑每个任务至少一个配对跑通下述流程，特别是多阶段和活动部件任务，再扩展到全部 600 对。可用 `--pair-key` 选择具体配对，用 `--task` 选择一个训练任务。准备进度可以随时查询：

```bash
"$PYTHON_BIN" tools/manage_role_annotations.py status --work-dir "$ROLE_ROOT/work"
```

## 3. 看原始帧，填写两个角色的身份与框

```bash
"$PYTHON_BIN" tools/manage_role_annotations.py preview \
  --work-dir "$ROLE_ROOT/work" --task place_mouse_pad
```

每对的 `previews/human/` 和 `previews/robot/` 包含全部采样帧 PNG、原分辨率 `frame_000000.png` 和拼图 `overlays_*.png`。需要看清细节时打开单帧；拼图缩放后的坐标不能用作原图框坐标。确认整个操作过程后再决定角色，不能只看首帧猜任务。

编辑对应 `seeds.json` 的两条 video 记录，保留模板中的 `pair_key`、`video_path`：

- `boxes_xyxy.source/reference`：首帧原图像素坐标 `[左,上,右,下]`，仅包围实际对象/部件，不框手或机械臂。
- `role_identity.source/reference`：说明哪个物体是被操作对象，哪个是关系参照对象；用可核验的外观/位置描述区别同类物体。
- `identity_review`：实际查看者与身份判断依据；完成这个检查后才填 `role_identity_approved: true`。
- `invalid_intervals`：已知遮挡/跟踪失败的原始视频帧区间，可先留空，掩码审核时补充。

同一段 human 视频可共享提示，不需要重复填写它对应的每个配对：

```bash
"$PYTHON_BIN" tools/manage_role_annotations.py share-human \
  --work-dir "$ROLE_ROOT/work" --from-pair '已填写human提示的真实pair_key'
```

它只复制**相同 human 视频路径、缓存和帧索引**的提示，不复制 robot 提示，也不把候选掩码标为已审核。已有编辑的目标会报错，避免覆盖其他标注。当前 runner 仍逐对执行 SAM2，human 提示复用不等同于 SAM2 推理去重。

以下只是看视频时的角色线索，不是自动标注规则：

| 任务 | 需要从实际视频确认 |
| --- | --- |
| beat_block_hammer | 锤子与被敲击的块 |
| move_pillbottle_pad | 药瓶与垫子 |
| open_laptop | 活动屏幕部件与底座；检查部件边界和遮挡 |
| pick_dual_bottles | 两个参与操作的瓶子；参照对象也可以运动 |
| place_bread_skillet | 面包与锅 |
| place_can_basket | 罐子与篮子 |
| place_container_plate | 容器与盘子 |
| place_mouse_pad | 鼠标与垫子 |
| put_object_cabinet | 开门/放物阶段涉及不同对象或部件，显式分段 |
| stack_blocks_two | 被放置的块与支撑块 |
| stack_bowls_three | 不同阶段的活动碗与支撑碗，保持实体 ID |
| stack_bowls_two | 活动碗与支撑碗 |

### 多阶段视频的标注格式

用 `objects` 替代 `boxes_xyxy`，每个实体给固定 ID、`description` 和首帧 `box_xyxy`；另填 `role_events` 指定每个原始帧区间使用哪两个实体。具体格式见 [prepare_role_masks_sam2.py](tools/prepare_role_masks_sam2.py) 开头的示例。

**每个 human clip 内角色必须保持同一实体，不同 clips 可以更换。** 以缓存 clip 帧 `[0,4,8,12]` 和 `[15,19,23,27]` 为例，可以用事件 `[0,12]: source=A, reference=B`、`[15,27]: source=C, reference=A`；不能在第一个 clip 的第 8 帧从 A 换到 C。读取每对 `human_frame_indices` 决定区间，不能假设所有视频的帧号都一样。当前 600 对的 human clip 帧区间互不重叠。

runner 在 SAM2 前检查片段内身份变化。robot 可以显式分阶段选择角色，原实体 ID 保持不变，跨实体的未来差分会被训练代码排除。事件未覆盖的区间是未知，不是背景负标签。

所有种子目前都在原始第 0 帧；不能为首帧不可见的实体伪造框，也不能把手的区域当成被遮挡物体。确实不满足此接口的配对应列出视频、原因和需要的修正，再处理标注接口；本工具不保证现有首帧提示接口对 600 对都足够。不能靠写入空标签宣称完成。

## 4. 分批生成 SAM2 候选

```bash
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" tools/manage_role_annotations.py run \
  --work-dir "$ROLE_ROOT/work" \
  --sam-python "$SAM_PYTHON" --checkpoint "$SAM_CHECKPOINT" \
  --device cuda:0 --limit 10
```

每次最多新尝试 10 对，未填写提示的配对会列出原因。重复同一命令会跳过输入、版本与候选 hash 都未变的已完成配对；坏文件、改过的种子或缓存、改变的教师版本会触发新 attempt。错误日志在各 job 的 `attempts/<时间与随机ID>/sam2.log`，失败不丢失其他已完成的配对。原始候选和旧审核不覆盖。

要用多张 GPU，可分别执行不同 `--shard-index 0/1/2/3 --num-shards 4`，各进程指定自己的 `CUDA_VISIBLE_DEVICES`；分片按固定 index 分配，job 另有锁。`--limit` 是每个进程的本次上限。请保留审图进度，避免一次生成大量无人检查的候选。

完整解码的 JPEG 放在各 attempt 中，600 对可能占用明显磁盘空间。完成审核与归档之前不要删除 PNG、原候选和 `completed.json`；这些文件用于继续处理及核验。

## 5. 审核每个采样帧，保留坏区域的排除记录

```bash
"$PYTHON_BIN" tools/manage_role_annotations.py review-template --work-dir "$ROLE_ROOT/work"
```

每个成功 job 会生成 `reviews/<候选annotation的SHA256>.json`。其中 `evidence` 链接到该候选的所有 human/robot PNG 拼图。预填图片路径只是方便查看，**不是已经审核**。

逐一看全部采样帧；确认 source/reference 对象、边界、身份持续性、遮挡和跟踪漂移。颜色：红色 source、青色 reference、黄色是自动判无效的候选。黄色部分仍画在图上便于检查，但不作为有效标签。按实际结果填写：

- `decision: "accept"` 或 `"reject"`；写明 `reviewer` 与具体 `notes`。Codex 审查就如实写 Codex，不能冒充人工精标。
- `reviewed_frames.human/robot`：实际检查过的**原始帧号**。必须覆盖对应 `expected_sampled_frames`，不能未看图就复制列表。
- `invalidate_intervals.<human或robot>.<source或reference>`：需额外剔除的闭区间，例如 `[[48,96]]`。合并工具同步清零 mask 和 validity，保留对象 ID，避免遮挡标签混进监督。
- `stable_roles`：human 和 robot 各自是否始终使用固定的两个实体；`single_effect`：整段是否是一个明确的关系效果。均需明确填 true/false。

自动判无效的区域不能靠审核字段恢复；框错了、身份错了或需要修复边界时，改 seeds，重新 `run`，再审核新候选。现有工具的面积/置信度过滤不证明掩码语义正确。旧审核不能复用到新候选。若排除后某对没有足够有效帧或未来监督，合并会失败，需要修正标签来源，不能直接降低完整性要求。

### donor 单独审核

不适用 donor 的配对保持 `donor_pair_key: null`、`donor_review: null`，多阶段任务也可以提供局部监督。适用时，donor 与接收者都必须是明确审核的**全程稳定角色、单一效果**，human 视频不同；两者 `effect_signature` 的 `relation`、`direction`、`source_type`、`reference_type`、`affordance` 完全相同，且不能遗漏关键方向/对象限制。

填写真实 `donor_pair_key` 与 `donor_review: {"approved": true, "reviewer": "实际审核者", "reason": "具体语义等价依据"}`。只因任务名相同不能批准。正式配方启用了 donor loss，完整预检会拒绝完全没有合法可用 donor 的 manifest。报告实际 donor 覆盖率；“存在一个 donor”只满足接口下限，不证明交换监督已充分覆盖数据。

## 6. 合并、完整预检、再进行训练

全部 job 通过审核后，合并到新目录：

```bash
"$PYTHON_BIN" tools/manage_role_annotations.py finalize \
  --work-dir "$ROLE_ROOT/work" --output-dir "$ROLE_ROOT/final-v1"

"$PYTHON_BIN" tools/manage_role_annotations.py check \
  --manifest "$ROLE_ROOT/final-v1/reviewed_manifest.json" \
  --human-gen-root "$HUMAN_GEN_ROOT" \
  --feature-root "$ZERO_WAM_RGB_FEATURE_ROOT" \
  --report "$ROLE_ROOT/full-preflight-v1.json"
```

合并结果含 `labels/*.pth`、审核 JSON、PNG 证据及 manifest。`finalize` 用实际训练 loader 验证已选配对，不能仅凭数量宣称正式训练就绪；失败时写 `INCOMPLETE.json`，不会发布 `reviewed_manifest.json`。`check` **在 CPU 上调用正式 RGB 和角色预检**，必须满足真实训练集合的 600/600、12 任务、逐文件 SHA256、帧索引、片段内身份、有效未来区间和合法 donor。不加载大模型，不需要 4 张 GPU。

检查成功的报告包含 `formal_training_preflight_passed: true`。这证明接口与覆盖通过，不能取代视觉语义审核或保证成功率。应先运行已有的短程训练检查，观察各辅助损失/有效样本覆盖，再启动正式配方：

```bash
export ZERO_WAM_ROLE_MANIFEST="$ROLE_ROOT/final-v1/reviewed_manifest.json"
unset ZERO_WAM_ROLE_ALLOW_PARTIAL
export ZERO_WAM_SAVE_ROOT=/path/to/new-role-effect-smoke
ROLE_SMOKE_PAIR_KEYS=$("$PYTHON_BIN" - "$ZERO_WAM_ROLE_MANIFEST" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))['records']
first_per_task = {}
for row in rows:
    first_per_task.setdefault(row['task_name'], row['pair_key'])
assert len(first_per_task) == 12
recipient = next(row for row in rows if row.get('donor_pair_key'))
keys = list(first_per_task.values()) + [recipient['pair_key'], recipient['donor_pair_key']]
print(','.join(dict.fromkeys(keys)))
PY
)
bash script/train_role_effect.sh --smoke-test \
  --rgb-smoke-sample-ids "$ROLE_SMOKE_PAIR_KEYS" \
  --num-steps 20 --save-interval 20 --disable-wandb --load-worker 0 --drop-icl 0

# 短程检查通过后，给正式训练设置另一个新输出目录。
export ZERO_WAM_SAVE_ROOT=/path/to/new-role-effect-full
bash script/train_role_effect.sh
```

这里显式选出 12 任务和一个合法 donor 配对做 20 步检查，保持所选样本标签完整，并记录 smoke 标记；不依赖另一台服务器可能没有的 `smoke_report.json`。完整性必须由上面的 `check` 先验收，短程训练成功不能证明 600 对标签齐全。四卡训练配方与原始 pretrained 设置见 [README.role-effect.md](README.role-effect.md)。

跨服务器搬运应复制完成目录及**对应 RGB 缓存的原始文件**，或在目标服务器用本机缓存重新生成标签。RGB 缓存重新保存后，即使张量看起来相同，文件 hash 也可能改变；不要直接修改 manifest 的 SHA256 来消除报错。以上流程在训练服务器原地生成，能避免这类路径/来源混淆。

## 本次准备工具的验证记录（2026-11-20，北京时间）

59 项不同的相关自动测试通过，覆盖继续处理、失败隔离、环境路径、片段内身份变化、旧审核拒绝、坏区域排除、donor 检查及失败时不发布 manifest。真实 `place_mouse_pad` 配对重新完成 SAM2、42 个采样帧的 Codex 视觉审核、遮挡排除、合并与 loader 验证；重复运行确认跳过未改变的已完成配对。正式 `check` 正确拒绝该单样本结果，报告 `1/600`、缺少 599 对。

这次验证没有完成全量标签，也没有重新训练策略。机器可读记录见 [validation/role-annotations-20261120.json](validation/role-annotations-20261120.json)。
