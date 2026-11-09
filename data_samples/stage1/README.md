# 第一阶段小样本

这里附有限数量的实际 navtrain/train 场景，用于读代码、核对数据格式和检查投影。当前共 31 个场景、124 张原图，样本文件约 50.5 MiB；具体场景数见 index.json 的 scene_count。不是全量训练集，也不是可续跑的 rank 缓存。

- `index.json`：统一索引；其中相对路径均相对于本目录。
- `scene_text.json`：token → 四相机 text/categories；与真实缓存一致。
- `generation_metadata.original.json`：原 rank 的完整生成配置，scenes/views 是该原 rank 的规模，不是这里的样本数。
- `samples/<token>/caption.original.json`：原始缓存按字节复制，包含 raw_text 和审计。
- `manifest_record.original.json` / `ground_truth.original.json` / `calibration.original.json`：原路径、GT 与原始标定；原服务器绝对路径只作来源证据。
- `images/cam_*.jpg`：四相机原图，SHA-256 与原 manifest 一致；`*_2d.jpg` 只画此次监督选中的 2D 框。
- `grounding.npz` / `projection.json`：本次导出用工程原有 project_box、patch_distribution、pack_grounding_arrays 计算的小样本标签。全量任务当时尚未进入标签导出，不将这些标签冒充全量任务产物。

相机顺序为 f0、b0、l0、r0。NPZ 的 soft_mask 为 (4,3936)，valid 为 (4,)；有效 mask 和为 1，无效 mask 为零。类别级框选择、0.1m 近平面截断、50m 范围与原脚本一致。

读取示例（从工程根目录执行）：

```python
from pathlib import Path
import json
import numpy as np
from PIL import Image
root = Path("data_samples/stage1")
sample = json.loads((root / "index.json").read_text())["samples"][0]
images = [Image.open(root / sample["images"][cam]).convert("RGB") for cam in sample["camera_order"]]
texts = [sample["scene_text"][cam]["text"] for cam in sample["camera_order"]]
labels = np.load(root / sample["grounding_npz"], allow_pickle=False)
print(sample["scene_token"], texts, labels["soft_mask"].shape, labels["valid"])
```
