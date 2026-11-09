"""Read generated per-camera text alongside the reference DrivoR camera features.

This checks data compatibility and default_collate ordering, not planner inference.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CAMERAS, IMAGE_SIZE, read_json, write_json
from selection import scene_selection
from worker import PARTITIONS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True, help="Merged train, val or test directory")
    p.add_argument("--reference-repo", type=Path,
        default=Path("/mnt/zhaozc_workspace/project/LQH/DrivoR-main-copy"))
    p.add_argument("--data-root", type=Path,
        default=Path("/mnt/zhaozc_workspace/project/LQH/DrivoR-main-copy/dataset"))
    p.add_argument("--scenes", type=int, default=2)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.scenes < 1:
        p.error("--scenes must be positive")
    sys.path[:0] = [str(a.reference_repo), str(a.reference_repo / "nuplan-devkit")]
    import torch
    from torch.utils.data import default_collate
    from omegaconf import OmegaConf
    import navsim.agents.drivoR.drivor_features as native
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SensorConfig, SceneFilter
    native_path = Path(native.__file__).resolve()
    assert native_path.is_relative_to(a.reference_repo.resolve()), native_path
    summary = read_json(a.dataset / "summary.json")
    split, subset = PARTITIONS[summary["partition"]]
    config, _, _ = scene_selection(split, subset, 3407)
    sensors = SensorConfig.build_no_sensors()
    for c in CAMERAS:
        setattr(sensors, c, [3])
    builder = native.DrivoRFeatureBuilder(OmegaConf.create({"image_size": list(IMAGE_SIZE)}))
    samples, expected, tokens = [], [], []
    with (a.dataset / "dataset_index.jsonl").open() as index:
        for line in index:
            record = json.loads(line)
            one = dict(config, tokens=[record["scene_token"]], log_names=[record["log_name"]], max_scenes=1)
            data_split = "test" if split == "navtest" else "trainval"
            loader = SceneLoader(a.data_root / "navsim_logs" / data_split,
                a.data_root / "sensor_blobs" / data_split, SceneFilter(**one), sensors)
            features = builder._get_camera_feature(loader.get_agent_input_from_token(record["scene_token"]))
            assert features["image"].shape == (4, 3, IMAGE_SIZE[1], IMAGE_SIZE[0])
            texts = [record["scene_text"][c]["text"] for c in CAMERAS]
            samples.append({"image": features["image"], "scene_text": texts})
            expected.extend(texts)
            tokens.append(record["scene_token"])
            if len(samples) == a.scenes:
                break
    assert samples, "No dataset records"
    batch = default_collate(samples)
    flattened = [batch["scene_text"][n][b] for b in range(len(samples)) for n in range(4)]
    assert flattened == expected and len(flattened) == len(samples) * 4
    assert torch.isfinite(batch["image"]).all()
    write_json(a.output, {"status": "passed", "scenes": len(samples), "tokens": tokens,
        "native_feature_source": str(native_path), "image_batch_shape": list(batch["image"].shape),
        "camera_order": CAMERAS, "b_major_flattening_verified": True,
        "planner_forward_tested": False, "scope": "native image features + four-view text data contract"})
    print(f"PASS: {len(samples)} native DrivoR samples; four-view text/image ordering verified")


if __name__ == "__main__":
    main()
