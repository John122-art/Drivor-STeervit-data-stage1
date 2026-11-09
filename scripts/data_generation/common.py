"""Shared contracts for the offline, current-frame-only data pilot."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAMERAS = ("cam_f0", "cam_b0", "cam_l0", "cam_r0")
CAMERA_NAMES = ("front", "rear", "front-left", "front-right")
CATEGORIES = {"vehicle", "pedestrian", "bicycle", "traffic_cone", "barrier",
              "traffic_light", "traffic_sign", "other"}
GROUND_CATEGORIES = {"vehicle", "pedestrian", "bicycle", "traffic_cone", "barrier"}
GROUND_CATEGORY_ORDER = ("vehicle", "pedestrian", "bicycle", "traffic_cone", "barrier")
IMAGE_SIZE = (1148, 672)
PATCH_SIZE = 14


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def shard_items(items, rank, world_size):
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid rank/world size")
    return list(items)[rank::world_size]


def write_npz(path, **arrays):
    """Atomic numeric cache writes; readers never observe a partial ZIP file."""
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
