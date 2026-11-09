# Stage 2：stage2_focus_v1_8gpu_pilot160

这是第二阶段（Stage 2）导航条件文本数据，生成策略为 `stage2_focus_v1`。仓库名称包含 stage1，但本目录为独立的 Stage 2 数据。

train、val、test 各 160 个场景，共 480 个场景。三个分区的 24 个 rank 校验均已通过并完成合并。pilot160 指每个分区各 160 个场景。

包含每个场景当前帧的四张未经缩放、未经修改的相机原图，按 `cam_f0、cam_b0、cam_l0、cam_r0` 排列：共 1,920 张，约 374.31 MiB。原图逐张核对 prepared manifest 中的 SHA-256。

- `images/<partition>/<scene_token>/cam_*.jpg`：对应的四张原图。
- `portable_scene_index.jsonl`：480 条样本索引，含 scene_token、四相机文本和类别、导航指令、相对图像路径、相对 caption/grounding 路径。所有路径相对于本目录。
- `image_manifest.json`：原图来源、相对路径、文件大小和 SHA-256。
- `train/`、`val/`、`test/`：原样保留的生成结果，包括 scene_text.json、prepared 标定/GT、原始 VLM 输出及审计、grounding 框/NPZ、校验结果、索引和日志。
- `generation_contract.json`：生成契约和配置；完整提示词及生成参数见各 rank 的 `captions/metadata.json`。

原始生成文件中的绝对路径保持原样以便追溯。下载后使用 `portable_scene_index.jsonl` 读取，可避免依赖原服务器图片路径。

```python
from pathlib import Path
import json
from PIL import Image

root = Path("data_samples/stage2_focus_v1_8gpu_pilot160")
with (root / "portable_scene_index.jsonl").open() as f:
    sample = json.loads(next(f))
for camera in sample["camera_order"]:
    image = Image.open(root / sample["images"][camera])
    text = sample["scene_text"][camera]["text"]
    categories = sample["scene_text"][camera]["categories"]
    print(camera, image.size, text, categories)
```

这里只包含本次 pilot 及对应原图；模型权重、完整 NAVSIM 原始日志和地图不在上传范围内。生成与划分来源以保留的 manifest、contract 和 metadata 为准。

同步完成：2026-11-09T12:49:24.088022+00:00。原始生成文件 1792 个，约 28.16 MiB；另加上述原图和便携索引。

[打开数据目录](stage2_focus_v1_8gpu_pilot160/) · [源文件快照清单](stage2_focus_v1_8gpu_pilot160_snapshot.json)
