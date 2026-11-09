# 第一阶段数据：2D框、审查和单机8卡DLC

第一阶段保留原图、简短定位文本，以及由 NAVSIM GT 3D 框投影得到的2D框和patch soft mask。画框图片只用于检查。本文中的DLC命令是离线数据生成命令；两阶段模型训练及其数据读取接口仍未接入。

## 标签位置和格式

已有样例的坐标原本就在 `outputs/stage1_grounding_pilot_20261107/grounding/projections.json`，每个相机包含 `objects[].bbox` 和 `selected_gt_indices`。本次新增明确的训练标签导出，检查后的样例在：

- `outputs/stage1_grounding_pilot_20261107/grounding_checked/scenes/29af16ce1c435102.json`
- `outputs/stage1_grounding_pilot_20261107/grounding_checked/masks/29af16ce1c435102.npz`
- `outputs/stage1_grounding_pilot_20261107/boxes_2d_checked/index.html`
- `outputs/stage1_grounding_pilot_20261107/boxes_2d_checked/selected_boxes.json`

NPZ可用 `np.load(path, allow_pickle=False)` 读取：

| 字段 | 内容 |
| --- | --- |
| `boxes_2d` | `(M,4)`，只包含实际选中用于监督的框，`xmin,ymin,xmax,ymax` |
| `box_camera_index` | `(M,)`，0/1/2/3对应f0/b0/l0/r0 |
| `box_gt_index` | `(M,)`，当前帧GT标注行号 |
| `box_category` | `(M,)`，对应类别 |
| `image_size` | `[1148,672]`，坐标所属图像尺寸 |
| `soft_mask` | `(4,3936)`，按框并集覆盖14×14patch后归一化 |
| `valid` | `(4,)`，是否有有效接地监督；无框时false、mask全零 |
| `camera_order` | 明确记录相机顺序 |
| `input_hash` | 绑定本场景图像身份、文本、GT、标定和投影规则 |

当前仍是类别并集弱监督，不是按颜色/短语精确匹配单个实例。实际接地损失使用 `soft_mask`；2D坐标用于复查、重建目标或将来调整监督策略。

## 本次审查与修改

- 批量模式按原生NAVSIM scene filter遍历日志中的全部有效场景，修正了旧试验脚本每日志只选一个场景的限制。当前帧固定索引3，VLM不读取未来帧或GT。
- `SUBSET=train/val`复用DrivoR实际的 `default_train_val_test_log_split.yaml`，避免将验证日志混入阶段1训练数据。`SPLIT=navtest`使用test数据目录及相应原生filter。
- 沿用参考Qwen缓存启动器的 `torchrun` 模式，每卡一个冻结模型，不做DDP梯度训练；明确绑定 `cuda:LOCAL_RANK`。
- 日志先确定性打乱，再按rank分片；不同rank写不同目录，不共写一个不断增长的JSON。
- 文本和审计按场景原子保存；最终文本JSON只在结束时聚合一次，避免逐视图重写所有历史审计的平方级I/O。
- 同目录续跑只复用输入及版本匹配、已通过规则检查的场景。提示词、模型、相关源码或分片设置改变时拒绝混用旧缓存；使用新输出目录。
- 第一阶段显式限制五类接地对象，不传导航。未修正的运动/方位问题、生成失败或阶段类别/长度不合格进入 `review_queue.json`，不进入训练索引。
- 批量模式只导出数值标签，不给全数据集生成大量JPG/PNG。需要看图时可以单独渲染选定场景。
- 标签校验逐场景检查原始图像/GT/当前指令、标定、文本重放、2D边界、选择类别、NPZ坐标及相机索引、mask形状及归一化。每rank默认均匀抽取20场景，从GT重新投影并重建mask核对；`DEEP_SCENES=0`对所有场景执行这项深度检查。
- 最后合并要求所有rank完成并通过校验，检测跨rank重复token，只将规则通过且有有效接地目标的场景写入 `training_index.jsonl`。

自动规则通过不代表事实完全正确；颜色、左右、车道归属、遗漏对象等仍需分层抽查图像。当前环境只有一张PPU，实跑验证使用单卡小样本；8卡需要在DLC资源上实际执行。

最终验证记录：28项检查通过；`outputs/stage1_dlc_verified_20261107/` 使用最终代码完成单卡2场景、8视图的端到端导出和深度校验。再次执行相同命令后，2个场景文本全部复用、重新生成数量为0，2份框/mask也全部复用，再次校验和合并通过。详细结果见该目录的 `review_report.json`。原始单场景新版NPZ包含38个选中2D框，按f0/b0/l0/r0分别为6/14/14/4个。

## DLC命令

以下均在单机8卡DLC任务的主进程执行一次。

先做每卡1场景、合计最多8场景的验证：

```bash
bash -lc '
cd /mnt/zhaozc_workspace/project/LQH/Driv-calude_data/DrivoR-main
NPROC_PER_NODE=8 \
SCENES_PER_RANK=1 \
SPLIT=navtrain SUBSET=train \
OUTPUT_DIR=/mnt/zhaozc_workspace/project/LQH/Driv-calude_data/DrivoR-main/outputs/stage1_navtrain_8gpu_pilot \
bash scripts/data_generation/run_stage1_data_8gpu_dlc.sh
'
```

全量训练划分生成，使用独立输出目录：

```bash
bash -lc '
cd /mnt/zhaozc_workspace/project/LQH/Driv-calude_data/DrivoR-main
NPROC_PER_NODE=8 \
SCENES_PER_RANK=0 \
SPLIT=navtrain SUBSET=train \
OUTPUT_DIR=/mnt/zhaozc_workspace/project/LQH/Driv-calude_data/DrivoR-main/outputs/stage1_navtrain_8gpu_full \
bash scripts/data_generation/run_stage1_data_8gpu_dlc.sh
'
```

`SCENES_PER_RANK=0`代表每个分片的全部场景，不是batch size。每次VLM前向仍是单图。中断后用相同指令、相同输出目录重启，复用通过检查的缓存。首次生成与续跑不需要切换模式。

验证集使用 `SPLIT=navtrain SUBSET=val` 和独立输出目录；navtest使用 `SPLIT=navtest SUBSET=all`。数据根目录默认复用 `DrivoR-main-copy/dataset`，可用 `DATA_ROOT`覆盖。环境默认分别为已有 `mot_vega`（Qwen）和 `drivorpython`（NAVSIM/投影）。

仅检查8卡启动命令，不启动任务：

```bash
bash scripts/data_generation/run_stage1_data_8gpu_dlc.sh --dry-run
```

## 产物和后续读取

输出根目录中：

- `scene_text.json`：通过规则检查且有目标的场景文本。
- `training_index.jsonl`：每行对应一个场景，包含文本审计缓存、2D框JSON、NPZ标签的路径。
- `review_queue.json`：规则失败及没有有效目标的场景，均不进入训练索引。
- `summary.json`：合并统计及各rank的生成/复用数量。
- `shards/rank_XXXX/`：该rank的prepared输入、逐场景文本审计、框和mask、验证报告。
- `launcher_logs/`：torchrun各rank日志。

批量模式的2D框图片可用已有渲染器按需生成，例如每rank取2个场景：

```bash
/mnt/zhaozc_workspace/envs/drivorpython/bin/python -B scripts/data_generation/render_2d_boxes.py \
  --prepared OUTPUT_DIR/shards/rank_0000/prepared \
  --captions OUTPUT_DIR/shards/rank_0000/captions \
  --grounding OUTPUT_DIR/shards/rank_0000/grounding \
  --scenes 2 --output OUTPUT_DIR/audit_rank0
```

其中 `OUTPUT_DIR` 替换成实际输出路径。为保证版本一致，正式开始全量生成后应固定代码和提示词。
