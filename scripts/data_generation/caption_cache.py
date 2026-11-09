"""Atomic, per-scene caption caches, following the reference Qwen cache pattern."""
from pathlib import Path
from common import CAMERAS, GROUND_CATEGORIES, fingerprint, read_json


def rule_status(audit, stage1=False):
    """Rule checks only; 'passed' does not assert visual factuality."""
    if set(audit) != set(CAMERAS) or any(v["status"] == "generation_failed" for v in audit.values()):
        return "generation_failed"
    for view in audit.values():
        if view["status"] not in {"ok", "empty_scene"}:
            return "needs_review"
        if any(not r["accepted"] for r in view.get("repairs", [])):
            return "needs_review"
        if any(r["reason"] not in {"duplicate", "token_budget"} for r in view["removed"]):
            return "needs_review"
        if stage1 and (not set(view["categories"]) <= GROUND_CATEGORIES
                       or sum(len(o["phrase"].split()) for o in view["objects"]) > 60):
            return "needs_review"
    return "passed"


def load_scene_cache(path, version, record):
    path = Path(path)
    if not path.is_file():
        return None
    data = read_json(path)
    if data.get("cache_version") != version or data.get("input_hash") != fingerprint(record):
        raise ValueError(f"Caption cache version/input mismatch: {path}; use a new output directory")
    if data.get("token") != record["token"] or set(data.get("scene_text", {})) != set(CAMERAS):
        raise ValueError(f"Incomplete caption cache: {path}")
    for camera in CAMERAS:
        if data["scene_text"][camera] != {k: data["audit"][camera][k] for k in ("text", "categories")}:
            raise ValueError(f"Caption/audit mismatch: {path}")
    return data


def load_audit(directory, token, legacy=None):
    if legacy is not None:
        return legacy[token]
    return read_json(Path(directory) / "scenes" / f"{token}.json")["audit"]
