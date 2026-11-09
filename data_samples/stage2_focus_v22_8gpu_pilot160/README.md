# Stage 2：stage2_focus_v22_8gpu_pilot160

本目录是第二阶段（Stage 2）数据。保留原始生成文件，并补充每个 prepared 场景的四张相机原图。

快照时间：2026-11-09T13:44:13.417744+00:00。生成任务仍可能继续；本次上传不代表尚未完成的分区已完成。

| 分区 | prepared 场景 | 已生成文本缓存 | grounding NPZ | 完成 |
|---|---:|---:|---:|---|
| train | 160 | 160 | 160 | 是 |
| val | 160 | 147 | 80 | 否（生成中快照） |

共 320 个已准备场景，1280 张未经缩放、未修改的原图（约 255.48 MiB）。每张图片均核对 prepared manifest 的 SHA-256。

- `images/<partition>/<scene_token>/cam_*.jpg`：f0、b0、l0、r0 四张原图。
- `portable_scene_index.jsonl`：每个 prepared 场景一条，图像及缓存路径均相对于本目录，含导航、四相机文本和类别。
- `caption_available=false`、`scene_text=null` 表示快照时文本尚未生成，不是模型返回空文本；相应 grounding 路径可能为 null。
- `image_manifest.json`：原图来源、大小和 SHA-256。
- `upload_snapshot.json`：源文件校验清单及各分区快照进度。
- `train/`、`val/` 等目录：原样保留的文本、VLM 输出/审计、prepared 标定和 GT、grounding 框/mask、日志及生成配置。

原始生成文件内的服务器绝对路径保留用于追溯；下载后建议使用便携索引读取。未包含模型权重、完整 NAVSIM 日志和地图。

```python
from pathlib import Path
import json
from PIL import Image
root = Path("data_samples/stage2_focus_v22_8gpu_pilot160")
with (root / "portable_scene_index.jsonl").open() as f:
    row = next(json.loads(line) for line in f if json.loads(line)["caption_available"])
for cam in row["camera_order"]:
    image = Image.open(root / row["images"][cam])
    print(cam, image.size, row["scene_text"][cam])
```
