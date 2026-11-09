"""Merge completed disjoint shards; quarantine rule failures from training index."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from common import CAMERAS, fingerprint, read_json, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--world-size", type=int, required=True)
    a = p.parse_args()
    root = a.output.resolve()
    seen, texts, review, no_labels, counts = set(), {}, [], [], []
    temporary = root / f"training_index.jsonl.{os.getpid()}.tmp"
    settings = None
    try:
        with temporary.open("w", encoding="utf-8") as index:
            for rank in range(a.world_size):
                shard = root / "shards" / f"rank_{rank:04d}"
                config = read_json(shard / "run_config.json")
                complete = read_json(shard / "COMPLETE.json")
                assert complete["configuration_hash"] == fingerprint(config)
                assert complete["rank"] == rank and complete["world_size"] == a.world_size
                assert complete["validation"] == read_json(shard / "validation.json")["status"] == "passed"
                shared = {k: v for k, v in config.items() if k != "rank"}
                if settings is None:
                    settings = shared
                assert settings == shared
                manifest = read_json(shard / "prepared/manifest.json")
                records = {r["token"]: r for r in manifest["records"]}
                assert not seen.intersection(records), "Rank selections overlap"
                seen.update(records)
                meta = read_json(shard / "captions/metadata.json")
                assert meta["stage1"] and not meta["navigation_conditioned"]
                captions = read_json(shard / "captions/scene_text.json")
                for token, caption in captions.items():
                    mask = shard / "grounding/masks" / f"{token}.npz"
                    with np.load(mask, allow_pickle=False) as labels:
                        assert str(labels["scene_token"]) == token and "boxes_2d" in labels
                        if not labels["valid"].any():
                            no_labels.append(token)
                            continue
                    texts[token] = caption
                    index.write(json.dumps({"scene_token": token, "log_name": records[token]["log_name"],
                        "camera_order": CAMERAS, "caption_cache": str(shard / "captions/scenes" / f"{token}.json"),
                        "grounding_npz": str(mask), "boxes_json": str(shard / "grounding/scenes" / f"{token}.json")}) + "\n")
                for token in read_json(shard / "captions/review_queue.json")["tokens"]:
                    review.append({"token": token, "rank": rank,
                        "audit_cache": str(shard / "captions/scenes" / f"{token}.json")})
                counts.append(read_json(shard / "captions/summary.json"))
        os.replace(temporary, root / "training_index.jsonl")
    finally:
        temporary.unlink(missing_ok=True)
    write_json(root / "scene_text.json", texts)
    write_json(root / "review_queue.json", {"rule_failures": review, "no_valid_grounding": no_labels,
        "meaning": "These scenes are excluded from training_index.jsonl; passed rules do not prove semantic accuracy"})
    write_json(root / "summary.json", {"world_size": a.world_size, "prepared_scenes": len(seen),
        "training_scenes": len(texts), "rule_review_scenes": len(review), "no_label_scenes": len(no_labels),
        "navigation_conditioned": False, "stage1": True, "rank_counts": counts,
        "settings": settings, "status": "completed_with_review_items" if review else "completed_rule_checks"})
    print(f"Merged {len(texts)} training scenes; {len(review)} rule review, {len(no_labels)} without targets: {root}")


if __name__ == "__main__":
    main()
