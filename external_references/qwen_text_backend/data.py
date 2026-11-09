"""Reuse local NAVSIM scene filtering and AgentInput conversion; never load future sensors/maps."""

from pathlib import Path

from .cache import file_identity, fingerprint
from .prompts import CAMERAS, build_qwen_questions


def records_from_loader(loader, num_history_frames):
    records = []
    for token in sorted(loader.tokens):
        # Explicitly isolate past/current input. Future frames only participate in NAVSIM's
        # scene eligibility filter, never in model conditioning.
        frames = loader.scene_frames_dicts[token][:num_history_frames]
        agent_input = loader.get_agent_input_from_token(token)
        current = frames[-1]
        times = [(int(f["timestamp"]) - int(current["timestamp"])) / 1e6 for f in frames]
        history = [status.ego_pose.tolist() for status in agent_input.ego_statuses]
        command = [float(v) for v in agent_input.ego_statuses[-1].driving_command]
        camera_files = {
            name: str((loader._sensor_blobs_path / current["cams"][name]["data_path"]).resolve())
            for name, _ in CAMERAS
        }
        image_identity = [file_identity(camera_files[name]) for name, _ in CAMERAS]
        record = {
            "log_name": current["log_name"], "token": token,
            "timestamp_us": int(current["timestamp"]),
            "camera_files": camera_files, "image_identity": image_identity,
            "history_trajectory": history, "relative_times_s": times,
            "driving_command": command,
            "question": build_qwen_questions([history], [command], [times])[0],
        }
        record["input_hash"] = fingerprint(record)
        records.append(record)
    return records


def build_records(log_path, sensor_path, filter_config, log_names=None, tokens=None, max_scenes=None):
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SceneFilter, SensorConfig

    config = {k: v for k, v in filter_config.items() if not k.startswith("_")}
    allowed_logs = config.pop("log_names", None)
    available = {p.stem for p in Path(log_path).glob("*.pkl")}
    selected = available if allowed_logs is None else set(allowed_logs)
    if log_names is not None:
        if allowed_logs is not None and not set(log_names).issubset(selected):
            raise ValueError("Requested log is outside the selected NAVSIM split")
        selected = set(log_names)
    missing = selected - available
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} split logs, e.g. {sorted(missing)[:3]}")
    split_tokens = config.get("tokens")
    if tokens is not None:
        if split_tokens is not None and not set(tokens).issubset(split_tokens):
            raise ValueError("Requested token is outside the selected NAVSIM split")
        config["tokens"] = tokens
    limit = max_scenes if max_scenes is not None else config.get("max_scenes")
    records = []
    # Read one log at a time: avoid retaining full annotations for an entire train split.
    for log_name in sorted(selected):
        cfg = dict(config, log_names=[log_name], max_scenes=None if limit is None else limit - len(records))
        loader = SceneLoader(
            data_path=Path(log_path), sensor_blobs_path=Path(sensor_path),
            scene_filter=SceneFilter(**cfg), sensor_config=SensorConfig.build_no_sensors(),
        )
        records.extend(records_from_loader(loader, cfg["num_history_frames"]))
        if limit is not None and len(records) >= limit:
            break
    if not records:
        raise ValueError("No eligible scenes found for the requested split/logs/tokens")
    if tokens is not None and limit is None:
        missing_tokens = set(tokens) - {record["token"] for record in records}
        if missing_tokens:
            raise ValueError(f"Requested tokens not found/eligible: {sorted(missing_tokens)}")
    return sorted(records, key=lambda record: (record["log_name"], record["token"]))
