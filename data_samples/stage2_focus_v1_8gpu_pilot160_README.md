# stage2_focus_v1_8gpu_pilot160 完整生成缓存

原目录：`/mnt/zhaozc_workspace/project/LQH/Driv-calude_data/DrivoR-main/outputs/stage2_focus_v1_8gpu_pilot160`

最终同步时间：2026-11-09T12:41:58.909941+00:00。共 1452 个文件，约 25.35 MiB。

| 划分 | prepared | 文本缓存 | NPZ | 已合并文本 | COMPLETE |
|---|---:|---:|---:|---:|---|
| train | 160 | 160 | 160 | 160 | True |
| val | 160 | 160 | 160 | 160 | True |

共 320 个场景，train 和 val 均已完成全分片校验与合并。没有 test 分区。

保留原始 generation_contract.json、prepared、captions、grounding、summary、索引及日志结构；具体生成策略以 contract 和每个 rank 的 metadata.json 为准（stage2_focus_v1）。

所有复制文件核对 SHA-256，JSON/JSONL 解析和 NPZ 的 ZIP CRC 已检查。完整清单见 [快照清单](stage2_focus_v1_8gpu_pilot160_snapshot.json)。

这里上传的是生成输出目录。相机原图、Qwen/RoBERTa 权重和 NAVSIM 原始日志等仍由 JSON 中的外部路径引用，未额外复制；这是 pilot 数据，不是全量 NAVSIM 数据集。

索引中的缓存/NPZ 绝对路径指向原服务器；在克隆目录读取本次上传的文件时，应将上述原目录前缀替换为本仓库的 data_samples/stage2_focus_v1_8gpu_pilot160。

[打开数据目录](stage2_focus_v1_8gpu_pilot160/)
