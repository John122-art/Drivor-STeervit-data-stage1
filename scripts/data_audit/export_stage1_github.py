#!/usr/bin/env python3
"""Export source plus a few authentic stage-1 scenes, without copying full caches.

Run with the existing drivorpython environment. The destination is separate from
the running data-generation project and is never used as a resume cache.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
SKIP_TOP = {'outputs', 'reports', 'dataset', 'datasets', 'weights', 'exp', 'checkpoints', 'envs', 'venv'}
SKIP_ANY = {'.git', '.cache', '__pycache__', '.pytest_cache', '.mypy_cache', 'node_modules'}
SKIP_SUFFIXES = {'.pyc', '.pyo', '.safetensors', '.ckpt', '.pth', '.pt', '.bundle'}
CAMERAS = ['cam_f0', 'cam_b0', 'cam_l0', 'cam_r0']


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hash_stream(handle)


def hash_stream(handle):
    value = hashlib.sha256()
    for block in iter(lambda: handle.read(1024*1024), b''):
        value.update(block)
    return value.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def copy_source(target):
    inventory = []
    for base, dirs, files in os.walk(ROOT, followlinks=False):
        relative = Path(base).relative_to(ROOT)
        dirs[:] = [d for d in dirs if d not in SKIP_ANY
                   and not (relative == Path('.') and d in SKIP_TOP)
                   and not (Path(base)/d).is_symlink()]
        for name in sorted(files):
            path = Path(base)/name
            if path.suffix in SKIP_SUFFIXES or name in {'.env', '.git'}:
                continue
            if path.is_symlink() and (not path.exists() or not path.resolve().is_relative_to(ROOT)):
                continue
            if path.stat().st_size >= 90*1024**2:
                raise ValueError(f'Unexpected large source file: {path}')
            rel = path.relative_to(ROOT)
            dest = target/rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path,dest)
            inventory.append({'path':str(rel),'sha256':digest(dest),'bytes':dest.stat().st_size})
    return inventory


def export_samples(target, count, seed):
    sys.path.insert(0,str(ROOT/'scripts/data_generation'))
    from common import fingerprint
    from geometry import project_box, patch_distribution
    from build_grounding import pack_grounding_arrays
    full = ROOT/'outputs/stage1_navtrain_8gpu_full'
    shard = full/'shards/rank_0000'
    manifest=json.loads((shard/'prepared/manifest.json').read_text())
    records={r['token']:r for r in manifest['records']}
    gt=json.loads((shard/'prepared/ground_truth.json').read_text())
    candidates=sorted((shard/'captions/scenes').glob('*.json'))
    random.Random(seed).shuffle(candidates)
    chosen=[]
    root=target/'data_samples/stage1'
    for path in candidates:
        raw_bytes=path.read_bytes()
        caption=json.loads(raw_bytes)
        if caption['rule_status'] != 'passed':
            continue
        token=caption['token']
        record, source=records[token],gt[token]
        projections, distributions, flags = {}, [], []
        for ci,camera in enumerate(CAMERAS):
            objects=[]
            for gi,(box,category) in enumerate(zip(source['boxes'],source['names'])):
                if category not in {'vehicle','pedestrian','bicycle','traffic_cone','barrier'} or np.linalg.norm(np.asarray(box)[:2])>50:
                    continue
                projected=project_box(box,source['cam_K'][ci],source['lidar_to_camera'][ci],near=0.1)
                if projected is not None:
                    objects.append(dict(projected,category=category,gt_index=gi))
            view=caption['scene_text'][camera]
            selected=[o for o in objects if o['category'] in view['categories']] if view['text'].strip() else []
            mask,valid=patch_distribution([o['bbox'] for o in selected])
            distributions.append(mask)
            flags.append(valid)
            projections[camera]={'objects':objects,'selected_gt_indices':[o['gt_index'] for o in selected],'valid':bool(valid)}
        if not any(flags):
            continue
        folder=root/'samples'/token
        (folder/'images').mkdir(parents=True)
        (folder/'caption.original.json').write_bytes(raw_bytes)
        dump(folder/'manifest_record.original.json',record)
        dump(folder/'ground_truth.original.json',source)
        hashes={}
        portable_images={}
        for camera in CAMERAS:
            dest=folder/'images'/f'{camera}.jpg'
            shutil.copyfile(record['images'][camera],dest)
            hashes[camera]=digest(dest)
            assert hashes[camera]==record['image_sha256'][camera]
            portable_images[camera]=str(dest.relative_to(root))
            image=Image.open(dest).convert('RGB').resize((1148,672))
            draw=ImageDraw.Draw(image)
            selected=set(projections[camera]['selected_gt_indices'])
            for obj in projections[camera]['objects']:
                if obj['gt_index'] in selected:
                    draw.rectangle(obj['bbox'],outline='#00ff80',width=3)
                    draw.text(tuple(obj['bbox'][:2]),f'{obj["category"]} #{obj["gt_index"]}',fill='yellow',stroke_width=1,stroke_fill='black')
            image.save(folder/'images'/f'{camera}_2d.jpg',quality=92)
        geometry_hash=digest(ROOT/'scripts/data_generation/geometry.py')
        label_hash=fingerprint({'record':record,'gt':source,'caption':caption['scene_text'],
            'near':0.1,'radius':50.0,'geometry_sha256':geometry_hash,'label_schema':2})
        arrays=pack_grounding_arrays(token,projections,distributions,flags,label_hash)
        np.savez_compressed(folder/'grounding.npz',**arrays)
        dump(folder/'projection.json',{'token':token,'input_hash':label_hash,'image_size':[1148,672],
             'coordinate_format':'xyxy','near_clipped_box_views':sum(o['near_clipped'] for v in projections.values() for o in v['objects']),
             'cameras':projections})
        log=Path(manifest['data_root'])/'navsim_logs'/manifest['data_split']/(record['log_name']+'.pkl')
        with log.open('rb') as handle:
            frames=pickle.load(handle)
        frame=next(f for f in frames if f['token']==token)
        raw={'scene_token':token,'timestamp':int(frame['timestamp']),'lidar2ego':np.asarray(frame['lidar2ego']).tolist(),'cameras':{}}
        for camera in CAMERAS:
            c=frame['cams'][camera.upper()]
            with Image.open(folder/'images'/f'{camera}.jpg') as image:
                size=list(image.size)
            raw['cameras'][camera]={'intrinsics':np.asarray(c['cam_intrinsic']).tolist(),
                'sensor2lidar_rotation':np.asarray(c['sensor2lidar_rotation']).tolist(),
                'sensor2lidar_translation':np.asarray(c['sensor2lidar_translation']).tolist(),
                'original_size':size}
        dump(folder/'calibration.original.json',raw)
        chosen.append({'scene_token':token,'log_name':record['log_name'],'camera_order':CAMERAS,
            'scene_text':caption['scene_text'],'images':portable_images,'image_sha256':hashes,
            'caption_original':str((folder/'caption.original.json').relative_to(root)),
            'manifest_record_original':str((folder/'manifest_record.original.json').relative_to(root)),
            'ground_truth_original':str((folder/'ground_truth.original.json').relative_to(root)),
            'grounding_npz':str((folder/'grounding.npz').relative_to(root)),
            'boxes_json':str((folder/'projection.json').relative_to(root)),'grounding_valid':list(map(bool,flags)),
            'source_cache':str(path),'source_cache_sha256':hashlib.sha256(raw_bytes).hexdigest()})
        print('Exported sample',token,flags,flush=True)
        if len(chosen)==count:
            break
    if len(chosen)!=count:
        raise RuntimeError(f'Only {len(chosen)} samples selected')
    dump(root/'index.json',{'schema_version':1,'camera_order':CAMERAS,'path_base':'directory containing this index.json',
        'split':'navtrain','subset':'train','seed':seed,'scene_count':count,'samples':chosen,
        'provenance':'Original captions/GT/images copied; grounding is newly derived for this small export using unchanged native geometry. Not a finished full-run cache or a resume directory.'})
    dump(root/'scene_text.json',{v['scene_token']:v['scene_text'] for v in chosen})
    shutil.copyfile(shard/'captions/metadata.json',root/'generation_metadata.original.json')
    (root/'README.md').write_text('''# 第一阶段小样本

这里仅附三个实际 navtrain/train 场景，用于读代码、核对数据格式和检查投影。不是全量训练集，也不是可续跑的 rank 缓存。

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
''')
    return chosen


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target',type=Path,required=True)
    parser.add_argument('--scenes',type=int,default=3)
    parser.add_argument('--seed',type=int,default=20261109)
    a=parser.parse_args()
    target=a.target.resolve()
    if target==ROOT or ROOT in target.parents:
        parser.error('Destination must be outside the running source project')
    if a.scenes<1:
        parser.error('scenes must be positive')
    target.mkdir(parents=True,exist_ok=False)
    inventory=copy_source(target)
    originals=target/'README_DRIVOR_ORIGINAL.md'
    (target/'README.md').rename(originals)
    gi=target/'.gitignore'
    gi.write_text(gi.read_text()+'\n# Export: never commit full runtime data or credentials\n/outputs/\n/reports/\n/.cache/\n/datasets/\n/checkpoints/\n*.ckpt\n*.safetensors\n*.pth\n*.pt\n*.bundle\n.env\n*.pem\n')
    samples=export_samples(target,a.scenes,a.seed)
    # Preserve the external source dependencies used by the existing generators.
    external=target/'external_references'
    qwen=Path('/mnt/zhaozc_workspace/project/LQH/Qwen-drive/Qwen-Drive-1.0-main/src/qwen_drive')
    backend=Path('/mnt/zhaozc_workspace/project/LQH/DrivoR-main-copy/navsim/agents/qwen_text')
    for origin,dest in [(qwen,external/'qwen_drive'),(backend,external/'qwen_text_backend')]:
        dest.mkdir(parents=True)
        for f in origin.rglob('*.py'):
            if '__pycache__' in f.parts:
                continue
            t=dest/f.relative_to(origin);t.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,t)
    guide=Path('/mnt/zhaozc_workspace/project/LQH/Drivor-orial/DrivoR-main/docs/steer_drivor_qwen_gt_grounding_analysis.md')
    shutil.copyfile(guide,target/'docs/steer_drivor_qwen_gt_grounding_analysis.reference.md')
    dump(target/'export_manifest.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'original_project':str(ROOT),'export_project':str(target),'original_source_file_count':len(inventory),
        'original_source_inventory':inventory,'sample_scenes':len(samples),'sample_tokens':[s['scene_token'] for s in samples],
        'excluded_root_directories':sorted(SKIP_TOP|SKIP_ANY),
        'note':'README renamed and gitignore extended in export only. Runtime generation code was not changed. External references are audit copies, not automatically substituted imports.'})
    (target/'README.md').write_text('''# DrivoR + SteerViT：第一阶段数据生成工程

本仓库是实际工程的独立源码快照，包含第一阶段 Qwen-Drive 四视图文本、NAVSIM GT 投影及接地 mask 生成代码，并附 3 个真实场景的小样本。

## 从哪里开始

1. [第一阶段运行说明](docs/stage1_data_dlc.md)
2. [第一阶段完整启动脚本](scripts/data_generation/run_stage1_data_8gpu_dlc.sh)
3. [当前第一阶段 prompt](scripts/data_generation/prompts/stage1_grounding_v1.txt)
4. [三个真实场景及读取示例](data_samples/stage1/README.md)
5. [原指导方案](docs/steer_drivor_qwen_gt_grounding_analysis.reference.md)
6. [原工程 README](README_DRIVOR_ORIGINAL.md)

原工程 navsim、nuplan-devkit、scripts、docs、assets 等源码已保留；完整 outputs、reports、权重、数据集和环境目录已排除。目录名包含 stage2 的现有代码也忠实保留，但本仓库提供的数据示例与启动说明聚焦第一阶段。

## 在原服务器运行全量第一阶段生成

```bash
bash -lc '
set -euo pipefail
cd /mnt/zhaozc_workspace/project/LQH/Driv-calude_data/DrivoR-main
NPROC_PER_NODE=8 \\
SCENES_PER_RANK=0 \\
DEEP_SCENES=80 \\
SPLIT=navtrain \\
SUBSET=train \\
DATA_ROOT=/mnt/zhaozc_workspace/project/LQH/DrivoR-main-copy/dataset \\
OUTPUT_DIR=/mnt/zhaozc_workspace/project/LQH/Driv-calude_data/DrivoR-main/outputs/stage1_navtrain_8gpu_full \\
bash scripts/data_generation/run_stage1_data_8gpu_dlc.sh
'
```

上面是正在使用的原工程指令。独立导出副本不共享原 rank 缓存；不要从新路径指向旧 OUTPUT_DIR 续跑，配置/路径指纹不同会拒绝缓存。克隆到其他环境运行时，需要配置自己的数据、Qwen 模型、RoBERTa、Python 环境、SDK 和参考工程路径，使用新的 OUTPUT_DIR；完整图像、地图、模型及环境不在 Git 中。

## 关键数据约定和实现状态

相机顺序为 cam_f0、cam_b0、cam_l0、cam_r0；仅当前帧 index=3 的原图供 VLM 使用。第一阶段只生成五个接地类别的简短外观/图内位置短语，不输入导航。现有第一阶段代码仍会逐短语重述不合格内容；第二阶段禁用模型重述属于另一套策略。

投影代码对 3D 框边做 0.1m 近平面截断，并按提及类别选择 50m 内全部框；48×82 patch 形成 3936 维监督。空文本相机跳过接地。离线不使用 GridMask。

本仓库包含数据生成代码；四相机文本 Dataset 接入和 SteerViT 两阶段训练尚未实现。不要把方案说明当作训练实现已经完成。

`data_samples` 的原始缓存/图像/GT 是实际数据；其中 NPZ 与 2D 框图是导出时用原有投影代码新计算的 3 场景示例，不是全量任务已完成标签的证明。详情与来源哈希见 `export_manifest.json` 和样本索引。

## 外部源码依赖

`external_references/qwen_drive/` 和 `external_references/qwen_text_backend/` 保存了原生成脚本使用的外部源码副本，便于审核。原脚本中的 import/绝对路径未改成这些副本，以保持运行版本一致；在新环境落地时需显式配置。
''')
    files=[f for f in target.rglob('*') if f.is_file()]
    print(json.dumps({'target':str(target),'files':len(files),'total_bytes':sum(f.stat().st_size for f in files),
                      'samples':[s['scene_token'] for s in samples]},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
