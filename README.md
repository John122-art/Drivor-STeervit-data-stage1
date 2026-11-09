# DrivoR + SteerViT：第一阶段数据生成工程

本仓库是实际工程的独立源码快照，包含第一阶段 Qwen-Drive 四视图文本、NAVSIM GT 投影及接地 mask 生成代码，并附 31 个真实场景的小样本。

## 从哪里开始

1. [第一阶段运行说明](docs/stage1_data_dlc.md)
2. [第一阶段完整启动脚本](scripts/data_generation/run_stage1_data_8gpu_dlc.sh)
3. [当前第一阶段 prompt](scripts/data_generation/prompts/stage1_grounding_v1.txt)
4. [31 个真实场景及读取示例](data_samples/stage1/README.md)
5. [原指导方案](docs/steer_drivor_qwen_gt_grounding_analysis.reference.md)
6. [原工程 README](README_DRIVOR_ORIGINAL.md)
7. [本机上传指令](docs/github_stage1_upload_command.txt)

原工程 navsim、nuplan-devkit、scripts、docs、assets 等源码已保留；完整 outputs、reports、权重、数据集和环境目录已排除。目录名包含 stage2 的现有代码也忠实保留，但本仓库提供的数据示例与启动说明聚焦第一阶段。

## 在原服务器运行全量第一阶段生成

```bash
bash -lc '
set -euo pipefail
cd /mnt/zhaozc_workspace/project/LQH/Driv-calude_data/DrivoR-main
NPROC_PER_NODE=8 \
SCENES_PER_RANK=0 \
DEEP_SCENES=80 \
SPLIT=navtrain \
SUBSET=train \
DATA_ROOT=/mnt/zhaozc_workspace/project/LQH/DrivoR-main-copy/dataset \
OUTPUT_DIR=/mnt/zhaozc_workspace/project/LQH/Driv-calude_data/DrivoR-main/outputs/stage1_navtrain_8gpu_full \
bash scripts/data_generation/run_stage1_data_8gpu_dlc.sh
'
```

上面是正在使用的原工程指令。独立导出副本不共享原 rank 缓存；不要从新路径指向旧 OUTPUT_DIR 续跑，配置/路径指纹不同会拒绝缓存。克隆到其他环境运行时，需要配置自己的数据、Qwen 模型、RoBERTa、Python 环境、SDK 和参考工程路径，使用新的 OUTPUT_DIR；完整图像、地图、模型及环境不在 Git 中。

## 关键数据约定和实现状态

相机顺序为 cam_f0、cam_b0、cam_l0、cam_r0；仅当前帧 index=3 的原图供 VLM 使用。第一阶段只生成五个接地类别的简短外观/图内位置短语，不输入导航。现有第一阶段代码仍会逐短语重述不合格内容；第二阶段禁用模型重述属于另一套策略。

投影代码对 3D 框边做 0.1m 近平面截断，并按提及类别选择 50m 内全部框；48×82 patch 形成 3936 维监督。空文本相机跳过接地。离线不使用 GridMask。

本仓库包含数据生成代码；四相机文本 Dataset 接入和 SteerViT 两阶段训练尚未实现。不要把方案说明当作训练实现已经完成。

`data_samples` 的原始缓存/图像/GT 是实际数据；其中 NPZ 与 2D 框图是导出时用原有投影代码新计算的 31 场景示例，不是全量任务已完成标签的证明。详情与来源哈希见 `export_manifest.json` 和样本索引。

## 外部源码依赖

`external_references/qwen_drive/` 和 `external_references/qwen_text_backend/` 保存了原生成脚本使用的外部源码副本，便于审核。原脚本中的 import/绝对路径未改成这些副本，以保持运行版本一致；在新环境落地时需显式配置。

## 空 GT 校验修复

已修复空标注 JSON 丢失二维形状导致的校验失败，保留已有文本缓存续跑的方法见 [修复说明](docs/stage1_empty_gt_fix.md)。

## 第二阶段 focus 试生成缓存

[Stage 2 数据与原图](data_samples/stage2_focus_v1_8gpu_pilot160/)：train、val、test 各 160 个场景，共 480 个场景，包含对应的 1,920 张四相机原图、文本、类别、导航指令和 GT 接地缓存。下载后通过目录内 `portable_scene_index.jsonl` 使用相对路径读取。

[数据说明](data_samples/stage2_focus_v1_8gpu_pilot160_README.md) · [源文件校验清单](data_samples/stage2_focus_v1_8gpu_pilot160_snapshot.json)


## Stage 2 v22 数据快照

[stage2_focus_v22_8gpu_pilot160](data_samples/stage2_focus_v22_8gpu_pilot160/)：本次生成输出和 1280 张对应四相机原图。各分区完成状态、文本缓存数量和便携索引见目录内 README；未生成的文本明确标记。
