"""Keep every selected planning scene; grounding validity never selects eval samples."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CAMERAS, fingerprint, read_json, write_json
from caption_cache import load_scene_cache
from text_policy import POLICY_VERSION, scene_status, view_needs_review, empty_cameras, fallback_cameras
from worker import PARTITIONS


def generation_contract(meta):
    keys = ("model", "model_weight_bytes", "model_config_sha256", "model_weight_mtime_ns",
            "source_hashes", "pipeline_source_hashes", "prompt", "prompt_sha256",
            "torch", "transformers", "dtype", "attention", "current_image_pixels",
            "roberta_tokenizer", "max_text_tokens", "decoding", "phrase_repair",
            "stage1", "navigation_conditioned", "backend_source_hashes", "text_policy", "model_calls_per_view")
    return {k: meta[k] for k in keys}


def merge_partition(output, partition, world_size):
    if partition not in PARTITIONS or world_size < 1:
        raise ValueError("Invalid partition/world size")
    bundle = Path(output).resolve()
    root = bundle / partition
    root.mkdir(parents=True, exist_ok=True)
    (root / "COMPLETE.json").unlink(missing_ok=True)
    texts, seen, reviews, counts = {}, set(), [], []
    settings, contract = None, None
    no_grounding = empty_count = fallback_count = fallback_scene_count = all_empty = 0
    temporary = root / f"dataset_index.jsonl.{os.getpid()}.tmp"
    try:
        with temporary.open("w", encoding="utf-8") as index:
            for rank in range(world_size):
                shard = root / "shards" / f"rank_{rank:04d}"
                config = read_json(shard / "run_config.json")
                complete = read_json(shard / "COMPLETE.json")
                assert complete["configuration_hash"] == fingerprint(config)
                assert complete["rank"] == config["rank"] == rank
                assert complete["world_size"] == config["world_size"] == world_size
                assert config["partition"] == partition and not config["stage1"]
                assert complete["validation"] == read_json(shard / "validation.json")["status"] == "passed"
                shared = {k: v for k, v in config.items() if k != "rank"}
                if settings is None:
                    settings = shared
                assert settings == shared, "Rank settings differ"
                manifest = read_json(shard / "prepared/manifest.json")
                records = manifest["records"]
                tokens = {r["token"] for r in records}
                assert len(tokens) == len(records), "Duplicate token within a rank"
                assert not seen.intersection(tokens), "Rank selections overlap"
                assert not manifest.get("skipped"), "Missing images cannot silently shrink the dataset"
                seen.update(tokens)
                meta = read_json(shard / "captions/metadata.json")
                assert not meta["stage1"] and meta["navigation_conditioned"]
                assert meta["text_policy"] == POLICY_VERSION and meta["model_calls_per_view"] == 1
                assert not meta["phrase_repair"]["enabled"]
                current_contract = generation_contract(meta)
                if contract is None:
                    contract = current_contract
                assert contract == current_contract, "Rank generators differ"
                captions = read_json(shard / "captions/scene_text.json")
                flagged = set(read_json(shard / "captions/review_queue.json")["tokens"])
                assert set(captions) == tokens, "Incomplete caption coverage"
                assert flagged <= tokens
                for record in records:
                    token = record["token"]
                    cache_path = shard / "captions/scenes" / f"{token}.json"
                    cache = load_scene_cache(cache_path, meta["cache_version"], record)
                    audit = cache["audit"]
                    status = scene_status(audit)
                    assert cache["rule_status"] == status
                    assert (status != "passed") == (token in flagged)
                    assert captions[token] == cache["scene_text"]
                    text = captions[token]
                    mask = shard / "grounding/masks" / f"{token}.npz"
                    boxes = shard / "grounding/scenes" / f"{token}.json"
                    assert boxes.is_file()
                    with np.load(mask, allow_pickle=False) as labels:
                        assert str(labels["scene_token"]) == token
                        assert labels["camera_order"].tolist() == list(CAMERAS)
                        assert labels["valid"].shape == (4,)
                        valid = labels["valid"].tolist()
                    empty = empty_cameras(audit)
                    fallback = fallback_cameras(audit)
                    empty_count += len(empty)
                    fallback_count += len(fallback)
                    fallback_scene_count += bool(fallback)
                    all_empty += len(empty) == 4
                    for ci, camera in enumerate(CAMERAS):
                        if camera in empty:
                            assert not valid[ci], "Empty camera must not receive grounding supervision"
                        if view_needs_review(audit[camera]):
                            reviews.append({"token": token, "rank": rank, "camera": camera,
                                "status": audit[camera]["status"], "has_text": bool(text[camera]["text"]),
                                "empty_text_fallback": camera in fallback, "audit_cache": str(cache_path)})
                    texts[token] = text
                    no_grounding += not any(valid)
                    index.write(json.dumps({"scene_token": token, "log_name": record["log_name"],
                        "partition": partition, "camera_order": CAMERAS, "images": record["images"],
                        "driving_command": record["driving_command"], "navigation_text": record["navigation_text"],
                        "scene_text": text, "caption_cache": str(cache_path), "caption_status": status,
                        "text_fallback": bool(fallback), "text_fallback_cameras": fallback,
                        "empty_text_cameras": empty, "view_status": {c: audit[c]["status"] for c in CAMERAS},
                        "grounding_npz": str(mask), "boxes_json": str(boxes),
                        "grounding_valid": valid,
                        "grounding_role": "audit_only" if partition == "test" else "auxiliary_supervision"},
                        ensure_ascii=False, allow_nan=False) + "\n")
                counts.append(read_json(shard / "captions/summary.json"))
        # This contract is shared across train/val/test, independent of rank or scene lists.
        contract_path = bundle / "generation_contract.json"
        if contract_path.exists():
            assert read_json(contract_path) == contract, "Train/val/test must use identical generators/prompts"
        for other in PARTITIONS:
            if other == partition or not (bundle / other / "COMPLETE.json").is_file():
                continue
            other_tokens = set(read_json(bundle / other / "scene_text.json"))
            assert not seen.intersection(other_tokens), f"Scene overlap with {other}"
            other_config = read_json(bundle / other / "summary.json")["settings"]
            assert {k: v for k, v in settings.items() if k not in {"partition", "split", "subset"}} == {
                k: v for k, v in other_config.items() if k not in {"partition", "split", "subset"}}
        write_json(contract_path, contract)
        os.replace(temporary, root / "dataset_index.jsonl")
    finally:
        temporary.unlink(missing_ok=True)
    write_json(root / "scene_text.json", texts)
    write_json(root / "review_queue.json", {"rule_failures": reviews,
        "fallback_policy": "Per camera: keep usable phrases and all other views; never clear the whole scene"})
    summary = {"stage": 2, "partition": partition, "world_size": world_size,
        "prepared_scenes": len(seen), "dataset_scenes": len(texts), "empty_text_fallback_scenes": fallback_scene_count,
        "empty_text_views": empty_count, "empty_text_fallback_views": fallback_count,
        "all_texts_empty_scenes": all_empty, "review_views": len(reviews), "text_policy": POLICY_VERSION,
        "scenes_without_grounding": no_grounding, "navigation_conditioned": True,
        "settings": settings, "generation_contract_hash": fingerprint(contract), "rank_counts": counts,
        "grounding_role": "audit_only" if partition == "test" else "auxiliary_supervision",
        "status": "complete_with_review_items" if reviews else "complete_rule_checks"}
    write_json(root / "summary.json", summary)
    write_json(root / "COMPLETE.json", {"dataset_scenes": len(texts), "summary_hash": fingerprint(summary)})
    print(f"Merged {partition}: {len(texts)} scenes, {fallback_count} camera fallbacks, {no_grounding} without grounding", flush=True)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--partition", choices=PARTITIONS, required=True)
    p.add_argument("--world-size", type=int, required=True)
    a = p.parse_args()
    merge_partition(a.output, a.partition, a.world_size)
