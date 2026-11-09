# 第一阶段空 GT 校验修复

故障为 `validate_pilot.py` 的 `(0, 7)` 与 `(0,)` 形状比较失败。原始 NAVSIM 没有框时数组是 `(0,7)`，`.tolist()` 和 JSON 将其保存为 `[]`，读取后 NumPy 推断为 `(0,)`。这不是 VLM 空文本或显存错误；一个 rank 校验失败会由 torchrun 终止其余 rank。

`assert_annotation_boxes_match` 仅在原始值确为 `(0,7)` 且 JSON 值为空列表时还原空数组形状。非空框仍逐值精确比较；其他形状或内容差异仍报错。实现沿用了本工程第二阶段校验器已有的同名处理方式。

回归检查：

```bash
cd /mnt/zhaozc_workspace/project/LQH/Driv-calude_data/DrivoR-main
PYTHONDONTWRITEBYTECODE=1 /mnt/zhaozc_workspace/envs/drivorpython/bin/python -B scripts/data_audit/test_empty_gt_validation.py
```

已通过 5 项检查，覆盖空数组 JSON 往返、真实非空数值差异、错误形状、缓存版本兼容约束、文本载荷不变及迁移重复执行。另核对了 rank6 的 3 个真实空 GT 场景；这不等于全量验证已完成。

## 保留已有数据续跑

生成器的 `cache_version` 包含校验脚本源码哈希。直接修改校验脚本会使已有文本缓存版本不匹配。因此增加了独立的 `scripts/data_audit/migrate_validation_shape_cache.py`，并由第一、第二阶段启动脚本在启动 worker 前调用。

该工具只允许这次已审查的单个 validator 哈希变更。其他 Python 源码改动、文件集合变化或元数据指纹不一致仍会被拒绝。它只更新 `metadata.json` 内该校验器哈希和 `cache_version`，以及场景缓存最外层 `cache_version` 字段；不改文本、audit、GT、图片、投影或 NPZ。原 `metadata.json` 按字节备份为 `metadata.before_empty_gt_fix.json`，迁移状态和每个文件的前后 SHA-256 另存。元数据最后原子写入，中断后可以再次执行。

手动执行（不调用 Qwen、不使用 GPU）：

```bash
python3 -B scripts/data_audit/migrate_validation_shape_cache.py \
  --output /mnt/zhaozc_workspace/project/LQH/Driv-calude_data/DrivoR-main/outputs/stage1_navtrain_8gpu_full
```

保持原 `OUTPUT_DIR`，重新执行原 8 卡指令即可。已通过规则的文本缓存按原逻辑复用；尚未生成或需要复核的文本按原策略处理。已有匹配的 NPZ 可以恢复使用。不要把 `DEEP_SCENES` 或其他运行参数随意改变，否则原有 `run_config` 检查仍会拒绝。

本地修复不代表 DLC 任务已经重新启动。全量 `validation.json` 和 `COMPLETE.json` 应由续跑任务成功完成后生成。
