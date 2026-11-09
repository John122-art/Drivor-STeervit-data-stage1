"""Independent navigation-caption worker; reuse the existing data-generation stages."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from common import ROOT, fingerprint, read_json, sha256, write_json

PARTITIONS = {"train": ("navtrain", "train"), "val": ("navtrain", "val"),
              "test": ("navtest", "all")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True, help="Bundle root; each partition gets its own directory")
    p.add_argument("--partition", choices=PARTITIONS, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--scenes", type=int, default=1, help="Per rank; 0 selects the full shard")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--deep-scenes", type=int, default=20)
    p.add_argument("--prompt", type=Path, required=True)
    p.add_argument("--drivor-python", type=Path, required=True)
    p.add_argument("--qwen-python", type=Path, required=True)
    a = p.parse_args()
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if min(a.scenes, a.seed, a.deep_scenes) < 0 or not 0 <= rank < world:
        p.error("Invalid scene/rank settings")
    prompt = a.prompt.read_text()
    if "{camera_name}" not in prompt or "{navigation_command}" not in prompt:
        p.error("Stage 2 requires the camera and navigation prompt placeholders")
    split, subset = PARTITIONS[a.partition]
    shard = a.output.resolve() / a.partition / "shards" / f"rank_{rank:04d}"
    shard.mkdir(parents=True, exist_ok=True)
    config = {"rank": rank, "world_size": world, "partition": a.partition,
        "split": split, "subset": subset, "scenes": a.scenes, "seed": a.seed,
        "data_root": str(a.data_root.resolve()), "prompt": str(a.prompt.resolve()),
        "prompt_sha256": sha256(a.prompt), "deep_scenes": a.deep_scenes,
        "stage1": False, "navigation_conditioned": True,
        "orchestration_hashes": {f.name: sha256(f) for f in sorted(Path(__file__).parent.glob("*.py"))}}
    config_path = shard / "run_config.json"
    if config_path.exists() and read_json(config_path) != config:
        raise ValueError("Stage-2 shard settings/code changed; use a new OUTPUT_DIR")
    write_json(config_path, config)
    (shard / "COMPLETE.json").unlink(missing_ok=True)

    def run(python, script, *args):
        cmd = [str(python), "-B", str(SCRIPTS / script), *map(str, args)]
        print(f"[{a.partition} rank {rank}] {shlex.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)

    prepared = shard / "prepared"
    manifest_path = prepared / "manifest.json"
    if manifest_path.is_file() and (prepared / "ground_truth.json").is_file():
        old = read_json(manifest_path)
        expected = {"rank": rank, "world_size": world, "split": split, "subset": subset,
                    "seed": a.seed, "data_root": str(a.data_root.resolve()), "all_scenes_per_log": True}
        if any(old.get(k) != v for k, v in expected.items()):
            raise ValueError("Prepared inputs differ from requested stage-2 partition")
        for relative, digest in {**old["source_hashes"], **old["split_hashes"]}.items():
            if sha256(ROOT / relative) != digest:
                raise ValueError(f"Preparation code/split changed: {relative}; use a new OUTPUT_DIR")
    else:
        if prepared.exists():
            prepared.rename(shard / f"prepared.incomplete.{time.time_ns()}")
        run(a.drivor_python, "prepare_navsim.py", "--output", prepared, "--data-root", a.data_root,
            "--split", split, "--subset", subset, "--seed", a.seed, "--rank", rank,
            "--world-size", world, "--all-scenes", "--scenes", a.scenes)
    if read_json(manifest_path)["skipped"]:
        raise ValueError("Missing images would change the scene list; inspect prepared/manifest.json")
    run(a.qwen_python, "stage2/generate.py", "--prepared", prepared,
        "--output", shard / "captions", "--scenes", 0, "--with-navigation", "--stream-cache", "--resume",
        "--device", f"cuda:{local}", "--prompt-file", a.prompt)
    run(a.drivor_python, "build_grounding.py", "--prepared", prepared,
        "--captions", shard / "captions", "--output", shard / "grounding", "--no-render", "--stream", "--resume")
    run(a.drivor_python, "stage2/validate.py", "--prepared", prepared,
        "--captions", shard / "captions", "--grounding", shard / "grounding",
        "--deep-scenes", a.deep_scenes, "--output", shard / "validation.json")
    write_json(shard / "COMPLETE.json", {"rank": rank, "world_size": world,
        "configuration_hash": fingerprint(config), "validation": read_json(shard / "validation.json")["status"]})
    print(f"[{a.partition} rank {rank}] complete: {shard}", flush=True)


if __name__ == "__main__":
    main()
