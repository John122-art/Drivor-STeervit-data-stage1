"""Reuse DrivoR scene filters and its actual training/validation log split."""
import random
import yaml
from common import ROOT, sha256, shard_items


def scene_selection(split, subset, seed, rank=0, world_size=1):
    if split not in {"navtrain", "navtest"} or subset not in {"train", "val", "all"}:
        raise ValueError("Unsupported split/subset")
    if split == "navtest" and subset != "all":
        raise ValueError("navtest uses subset=all")
    path = ROOT / "navsim/planning/script/config/common/train_test_split/scene_filter" / f"{split}.yaml"
    config = {k: v for k, v in yaml.safe_load(path.read_text()).items() if not k.startswith("_")}
    logs = sorted(config["log_names"])
    hashes = {str(path.relative_to(ROOT)): sha256(path)}
    if split == "navtrain" and subset != "all":
        log_path = ROOT / "navsim/planning/script/config/training/default_train_val_test_log_split.yaml"
        native = yaml.safe_load(log_path.read_text())
        if set(native["train_logs"]) & set(native["val_logs"]):
            raise ValueError("Native training/validation logs overlap")
        logs = [log for log in logs if log in set(native[f"{subset}_logs"])]
        hashes[str(log_path.relative_to(ROOT))] = sha256(log_path)
    random.Random(seed).shuffle(logs)
    return config, shard_items(logs, rank, world_size), hashes
