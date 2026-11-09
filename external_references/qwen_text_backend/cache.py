"""Versioned text cache; gzip/pickle layout follows ReCogDrive training/dataset.py."""

import gzip
import hashlib
import json
import os
import pickle
import re
import tempfile
import zlib
from pathlib import Path

SCHEMA_VERSION = 1


def description_errors(text):
    """Reject known wrong-task outputs; this is a format check, not visual fact verification."""
    errors = []
    if not isinstance(text, str) or not text.strip():
        return ["empty_text"]
    coordinate = r"\(\s*[+-]?\d+(?:\.\d+)?\s*,\s*[+-]?\d+(?:\.\d+)?\s*,\s*[+-]?\d+(?:\.\d+)?\s*\)"
    if re.search(r"\[\s*PT\b", text, flags=re.I) or len(re.findall(coordinate, text)) >= 2:
        errors.append("trajectory_instead_of_description")
    if "<think>" in text or "</think>" in text:
        errors.append("unexpected_thinking_trace")
    return errors


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def dump_feature_target_to_pickle(path, data_dict):
    """ReCogDrive gzip(level=1)/pickle writer with atomic replacement for resumability."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=1) as stream:
                pickle.dump(data_dict, stream, protocol=pickle.HIGHEST_PROTOCOL)
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_feature_target_from_pickle(path):
    # Local trusted cache files only, as in the existing NAVSIM cache reader.
    with gzip.open(path, "rb") as stream:
        return pickle.load(stream)


def cache_file(root, record):
    return Path(root) / record["log_name"] / record["token"] / "qwen_text.gz"


def load_qwen_text(path, expected_version=None, expected_input_hash=None):
    """Public reader for later planner integration. Incomplete/mismatched entries are rejected."""
    record = load_feature_target_from_pickle(path)
    if not isinstance(record, dict) or record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported cache schema: {path}")
    if not record.get("complete") or not record.get("usable") or description_errors(record.get("text")):
        raise ValueError(f"Incomplete text cache: {path}")
    if expected_version is not None and record.get("cache_version") != expected_version:
        raise ValueError(f"Cache version mismatch: {path}")
    if expected_input_hash is not None and record.get("input_hash") != expected_input_hash:
        raise ValueError(f"Input fingerprint mismatch: {path}")
    return record


def is_valid_cache(path, version, input_hash):
    try:
        load_qwen_text(path, version, input_hash)
        return True
    except (OSError, ValueError, EOFError, pickle.UnpicklingError, AttributeError, TypeError, zlib.error):
        return False


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def model_identity(path):
    """Hash configs/tokenizer contents and checkpoint file metadata (not all multi-GB weights)."""
    path = Path(path).resolve()
    files = sorted(p for p in path.iterdir() if p.is_file() and (
        p.suffix in (".json", ".jinja", ".safetensors")
    ) and p.name not in ("args.json", "trainer_state.json"))
    if not any(p.suffix == ".safetensors" for p in files):
        raise FileNotFoundError(f"No safetensors checkpoint in {path}")
    result = []
    for item in files:
        identity = file_identity(item)
        if item.suffix != ".safetensors":
            identity["sha256"] = hashlib.sha256(item.read_bytes()).hexdigest()
        result.append(identity)
    return result
