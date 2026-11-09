"""Prepare current images and separate GT with native NAVSIM/DrivoR code."""
import argparse
from pathlib import Path
import sys
from common import CAMERAS, IMAGE_SIZE, ROOT, sha256, write_json
from navigation import navigation_text
from selection import scene_selection

sys.path[:0] = [str(ROOT), str(ROOT / "nuplan-devkit")]
import numpy as np
from omegaconf import OmegaConf
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import Scene, SceneFilter, SensorConfig
from navsim.agents.drivoR.drivor_features import DrivoRFeatureBuilder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=20, help="0 selects all eligible scenes")
    parser.add_argument("--all-scenes", action="store_true", help="Use every eligible scene per log")
    parser.add_argument("--split", choices=["navtrain", "navtest"], default="navtrain")
    parser.add_argument("--subset", choices=["train", "val", "all"], default="all")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/zhaozc_workspace/project/LQH/DrivoR-main-copy/dataset"))
    args = parser.parse_args()
    if args.scenes < 0 or (args.scenes == 0 and not args.all_scenes):
        parser.error("Use --all-scenes with --scenes 0 for a full shard")
    config, logs, split_hashes = scene_selection(args.split, args.subset, args.seed, args.rank, args.world_size)
    if config["num_history_frames"] != 4:
        raise ValueError("Expected current camera frame index 3")
    data_split = "trainval" if args.split == "navtrain" else "test"
    log_root = args.data_root / "navsim_logs" / data_split
    sensor_root = args.data_root / "sensor_blobs" / data_split
    if not log_root.is_dir() or not sensor_root.is_dir():
        raise FileNotFoundError(f"Missing dataset roots: {log_root}, {sensor_root}")
    missing_logs = {log for log in logs if not (log_root / f"{log}.pkl").is_file()}
    if missing_logs and args.scenes == 0:
        raise FileNotFoundError(f"Full shard missing {len(missing_logs)} logs, e.g. {sorted(missing_logs)[:3]}")
    args.output.mkdir(parents=True, exist_ok=False)
    sensors = SensorConfig.build_no_sensors()
    for camera in CAMERAS:
        setattr(sensors, camera, [3])
    builder = DrivoRFeatureBuilder(OmegaConf.create({"image_size": list(IMAGE_SIZE)}))
    records, ground_truth, skipped = [], {}, []
    for log in logs:
        if log in missing_logs:
            continue
        one_log = dict(config, log_names=[log], max_scenes=None if args.all_scenes else 1)
        loader = SceneLoader(log_root, sensor_root, SceneFilter(**one_log), sensors)
        for token in loader.tokens:
            current = loader.scene_frames_dicts[token][3]
            assert current["token"] == token
            if "lidar2ego" not in current or not np.allclose(current["lidar2ego"], np.eye(4), atol=1e-8):
                raise ValueError(f"{token}: review coordinate convention; lidar2ego is not identity")
            images = {cam: str((sensor_root / current["cams"][cam.upper()]["data_path"]).resolve()) for cam in CAMERAS}
            missing = [p for p in images.values() if not Path(p).is_file()]
            if missing:
                skipped.append({"token": token, "missing_images": missing})
                continue
            agent_input = loader.get_agent_input_from_token(token)
            command = np.asarray(agent_input.ego_statuses[-1].driving_command).tolist()
            np.testing.assert_array_equal(command, current["driving_command"])
            features = builder._get_camera_feature(agent_input)
            assert features["image"].shape == (4, 3, IMAGE_SIZE[1], IMAGE_SIZE[0])
            annotations = Scene._build_annotations(current)
            records.append({"token": token, "log_name": log, "frame_index": 3,
                "timestamp": int(current["timestamp"]), "images": images,
                "driving_command": command, "navigation_text": navigation_text(command),
                "navigation_source": "AgentInput.ego_statuses[-1].driving_command",
                "image_sha256": {k: sha256(v) for k, v in images.items()}})
            ground_truth[token] = {"boxes": annotations.boxes.tolist(), "names": list(annotations.names),
                "cam_K": features["cam_K"].tolist(), "lidar_to_camera": features["world_2_cam"].tolist()}
            print(f"Prepared {len(records)}/{args.scenes or 'all'}: {token} ({log})", flush=True)
            if args.scenes and len(records) == args.scenes:
                break
        if args.scenes and len(records) == args.scenes:
            break
    if args.scenes and len(records) != args.scenes:
        raise RuntimeError(f"Only found {len(records)} scenes; expected {args.scenes}")
    write_json(args.output / "manifest.json", {"schema_version": 3,
        "camera_order": CAMERAS, "image_size": IMAGE_SIZE, "split": args.split,
        "data_split": data_split, "subset": args.subset, "seed": args.seed,
        "rank": args.rank, "world_size": args.world_size,
        "all_scenes_per_log": args.all_scenes, "selected_log_count": len(logs),
        "split_hashes": split_hashes, "grid_mask": False,
        "vlm_input": "one current image; navigation only when explicitly enabled",
        "navigation_one_hot_order": ["turn left", "go straight", "turn right", "unknown"],
        "data_root": str(args.data_root.resolve()), "records": records, "skipped": skipped,
        "missing_logs": sorted(missing_logs),
        "source_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in [
            Path(__file__), Path(__file__).with_name("selection.py"),
            ROOT / "navsim/common/dataloader.py", ROOT / "navsim/common/dataclasses.py",
            ROOT / "navsim/agents/drivoR/drivor_features.py"]}})
    write_json(args.output / "ground_truth.json", ground_truth)


if __name__ == "__main__":
    main()

