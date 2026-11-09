"""One torchrun worker per device; native preparation -> Qwen -> boxes/masks."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import time
from common import ROOT, fingerprint, read_json, sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--split", choices=["navtrain", "navtest"], default="navtrain")
    p.add_argument("--subset", choices=["train", "val", "all"], default="train")
    p.add_argument("--scenes", type=int, default=1, help="Per rank; 0 means full shard")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--deep-scenes", type=int, default=20)
    p.add_argument("--prompt", type=Path, required=True)
    p.add_argument("--drivor-python", type=Path, required=True)
    p.add_argument("--qwen-python", type=Path, required=True)
    a = p.parse_args()
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if a.scenes < 0 or not 0 <= rank < world:
        p.error("Invalid scene/rank settings")
    shard = a.output.resolve() / "shards" / f"rank_{rank:04d}"
    shard.mkdir(parents=True, exist_ok=True)
    config = {"rank": rank, "world_size": world, "split": a.split, "subset": a.subset,
        "scenes": a.scenes, "seed": a.seed, "data_root": str(a.data_root.resolve()),
        "prompt": str(a.prompt.resolve()), "prompt_sha256": sha256(a.prompt),
        "deep_scenes": a.deep_scenes, "stage1": True}
    config_path = shard / "run_config.json"
    if config_path.is_file() and read_json(config_path) != config:
        raise ValueError("Shard configuration changed; use a new OUTPUT_DIR")
    write_json(config_path, config)
    (shard / "COMPLETE.json").unlink(missing_ok=True)
    scripts = ROOT / "scripts/data_generation"

    def run(python, script, *args):
        command = [str(python), "-B", str(scripts / script), *map(str, args)]
        print(f"[rank {rank}] {shlex.join(command)}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)

    prepared = shard / "prepared"
    manifest_path = prepared / "manifest.json"
    if manifest_path.is_file() and (prepared / "ground_truth.json").is_file():
        old = read_json(manifest_path)
        expected = {"rank": rank, "world_size": world, "split": a.split, "subset": a.subset,
                    "seed": a.seed, "data_root": str(a.data_root.resolve()), "all_scenes_per_log": True}
        if any(old.get(k) != v for k, v in expected.items()):
            raise ValueError("Prepared shard no longer matches selection")
        for relative, digest in {**old["source_hashes"], **old["split_hashes"]}.items():
            if sha256(ROOT / relative) != digest:
                raise ValueError(f"Preparation code/split changed: {relative}; use a new OUTPUT_DIR")
        print(f"[rank {rank}] reusing prepared current-frame inputs", flush=True)
    else:
        if prepared.exists():
            prepared.rename(shard / f"prepared.incomplete.{time.time_ns()}")
        run(a.drivor_python, "prepare_navsim.py", "--output", prepared,
            "--data-root", a.data_root, "--split", a.split, "--subset", a.subset,
            "--seed", a.seed, "--rank", rank, "--world-size", world,
            "--all-scenes", "--scenes", a.scenes)
    run(a.qwen_python, "generate_captions.py", "--prepared", prepared,
        "--output", shard / "captions", "--scenes", 0, "--stage1", "--stream-cache", "--resume",
        "--device", f"cuda:{local}", "--prompt-file", a.prompt)
    run(a.drivor_python, "build_grounding.py", "--prepared", prepared,
        "--captions", shard / "captions", "--output", shard / "grounding",
        "--no-render", "--stream", "--resume")
    run(a.drivor_python, "validate_pilot.py", "--prepared", prepared,
        "--captions", shard / "captions", "--grounding", shard / "grounding",
        "--deep-scenes", a.deep_scenes, "--output", shard / "validation.json")
    write_json(shard / "COMPLETE.json", {"rank": rank, "world_size": world,
        "configuration_hash": fingerprint(config), "validation": read_json(shard / "validation.json")["status"]})
    print(f"[rank {rank}] finished: {shard}", flush=True)


if __name__ == "__main__":
    main()
