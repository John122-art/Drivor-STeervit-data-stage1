"""One original-image VQA call per camera; keep valid text from every other view."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import sys
import time

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from common import CAMERAS, CAMERA_NAMES, fingerprint, read_json, sha256, write_json
from caption_cache import load_scene_cache
from navigation import build_question, navigation_text, qwen_navigation_text
from text_policy import POLICY_VERSION, clean_response, scene_status, view_needs_review, empty_cameras, fallback_cameras


def process_scene(record, prompt, vocabulary, tokenizer, generate_answer):
    """The callable sees only this camera image, its prompt and current command."""
    command = qwen_navigation_text(record["driving_command"], vocabulary)
    audit = {}
    for camera, name in zip(CAMERAS, CAMERA_NAMES):
        question = build_question(prompt, name, command)
        call = generate_answer(record["images"][camera], question)
        cleaned = clean_response(call, tokenizer)
        call["error"] = cleaned["error"]
        audit[camera] = dict(cleaned, attempts=[call], repairs=[], navigation_command=command)
    texts = {c: {k: audit[c][k] for k in ("text", "categories")} for c in CAMERAS}
    return texts, audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prompt-file", type=Path, required=True)
    p.add_argument("--model", type=Path, default=Path("/mnt/zhaozc_workspace/project/LQH/Qwen-drive/checkpoint"))
    p.add_argument("--source", type=Path, default=Path("/mnt/zhaozc_workspace/project/LQH/Qwen-drive/Qwen-Drive-1.0-main/src"))
    p.add_argument("--reference-repo", type=Path, default=Path("/mnt/zhaozc_workspace/project/LQH/DrivoR-main-copy"))
    p.add_argument("--roberta", type=Path, default=Path("/mnt/zhaozc_workspace/project/LQH/RoBERTa"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--scenes", type=int, default=0)
    p.add_argument("--with-navigation", action="store_true", required=True)
    p.add_argument("--stream-cache", action="store_true", required=True)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    prompt = a.prompt_file.read_text(encoding="utf-8").strip()
    manifest_path = a.prepared / "manifest.json"
    manifest = read_json(manifest_path)
    assert manifest["camera_order"] == list(CAMERAS)
    if not 0 <= a.scenes <= len(manifest["records"]) or a.max_new_tokens < 1:
        p.error("Invalid scene/token limit")
    records = manifest["records"][:a.scenes] if a.scenes else manifest["records"]
    for record in records:
        assert navigation_text(record["driving_command"]) == record["navigation_text"]
    a.output.mkdir(parents=True, exist_ok=a.resume)
    (a.output / "scenes").mkdir(exist_ok=True)
    # Reuse the reference's strict VLM loader and the official single-image VQA processor.
    sys.path.insert(0, str(a.reference_repo.resolve()))
    import torch
    import transformers
    from transformers import RobertaTokenizer
    from navsim.agents.qwen_text import backend
    assert Path(backend.__file__).resolve().is_relative_to(a.reference_repo.resolve())
    if a.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Device unavailable; use the documented PPU launcher")
        torch.cuda.set_device(torch.device(a.device))
    processor = backend.load_processor(a.model, source_path=a.source)
    package = backend.import_qwen_drive(a.source)
    from qwen_drive.modeling_qwen_drive import VQA_DECODE_DEFAULTS
    tokenizer = RobertaTokenizer.from_pretrained(a.roberta, local_files_only=True)
    meta = {"manifest_sha256": sha256(manifest_path), "camera_order": CAMERAS,
        "model": str(a.model.resolve()), "model_weight_bytes": (a.model/"model.safetensors").stat().st_size,
        "model_config_sha256": sha256(a.model/"config.json"),
        "model_weight_mtime_ns": (a.model/"model.safetensors").stat().st_mtime_ns,
        "source_hashes": backend.source_identity(a.source),
        "backend_source_hashes": {f.name: sha256(f) for f in Path(backend.__file__).parent.glob("*.py")},
        "pipeline_source_hashes": {str(f.relative_to(SCRIPTS)): sha256(f)
            for f in sorted([*SCRIPTS.glob("*.py"), *Path(__file__).parent.glob("*.py")])},
        "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_file": str(a.prompt_file.resolve()), "python": sys.executable,
        "torch": torch.__version__, "transformers": transformers.__version__, "dtype": "bfloat16",
        "attention": "eager", "current_image_pixels": processor.config.current_image_pixels,
        "device": a.device, "roberta_tokenizer": str(a.roberta), "max_text_tokens": 128,
        "scenes": len(records), "views": 4*len(records), "batch_size": 1,
        "navigation_conditioned": True, "stage1": False, "stream_cache": True,
        "qwen_nav_commands": list(package.NAV_COMMANDS),
        "navigation_by_scene": {r["token"]: {"driving_command": r["driving_command"],
            "text": qwen_navigation_text(r["driving_command"], package.NAV_COMMANDS),
            "source": r["navigation_source"]} for r in records},
        "phrase_repair": {"enabled": False, "max_attempts_per_phrase": 0},
        "text_policy": POLICY_VERSION, "model_calls_per_view": 1,
        "decoding": dict(VQA_DECODE_DEFAULTS, max_new_tokens=a.max_new_tokens),
        "loading_errors": {}, "planner_instantiated": False}
    version = fingerprint(meta)
    meta["cache_version"] = version
    if (a.output/"metadata.json").exists() and read_json(a.output/"metadata.json").get("cache_version") != version:
        raise ValueError("Caption policy/input version changed; use a new output directory")
    cached_scenes, pending = {}, []
    for record in records:
        for camera, path in record["images"].items():
            if sha256(path) != record["image_sha256"][camera]:
                raise ValueError(f"Source image changed: {path}")
        path = a.output/"scenes"/(record["token"]+".json")
        cache = load_scene_cache(path, version, record) if a.resume else None
        if cache is None:
            pending.append(record)
        else:
            assert cache["rule_status"] == scene_status(cache["audit"])
            cached_scenes[record["token"]] = cache
    generator = backend.FrozenQwenTextGenerator(a.model, processor, a.device, "eager") if pending else None
    if generator:
        assert not generator.loading_report["loading_errors"]
    write_json(a.output/"metadata.json", meta)

    def generate_answer(image, question):
        start = time.monotonic()
        inputs = processor.encode_batch([[image]], [question], system=None)
        assert inputs["image_grid_thw"].shape[0] == 1
        grid = inputs["image_grid_thw"].tolist()
        result = generator.generate(inputs, a.max_new_tokens)[0]
        return {"answer": result["text"], "raw_text": result["raw_text"],
            "finish_reason": result["finish_reason"], "generated_tokens": result["generated_tokens"],
            "seconds": time.monotonic()-start, "question": question, "image": image, "image_grid_thw": grid}

    texts, reviews, statuses = {}, [], Counter()
    empty_count = fallback_count = all_empty = 0
    for record in records:
        token = record["token"]
        cache = cached_scenes.get(token)
        if cache is None:
            text, audit = process_scene(record, prompt, package.NAV_COMMANDS, tokenizer, generate_answer)
            cache = {"token": token, "cache_version": version, "input_hash": fingerprint(record),
                "rule_status": scene_status(audit), "scene_text": text, "audit": audit}
            write_json(a.output/"scenes"/(token+".json"), cache)
        else:
            print(f"Reused scene {token}", flush=True)
        texts[token] = cache["scene_text"]
        audit = cache["audit"]
        empty_count += len(empty_cameras(audit))
        fallback_count += len(fallback_cameras(audit))
        all_empty += len(empty_cameras(audit)) == 4
        for camera in CAMERAS:
            view = audit[camera]
            statuses[view["status"]] += 1
            if view_needs_review(view):
                reviews.append({"token": token, "camera": camera, "status": view["status"],
                    "has_text": bool(view["text"]), "removed": view["removed"], "local_edits": view["local_edits"]})
            print(f'{token} {camera}: {view["status"]}, {view["token_count"]} tokens; {view["text"]}', flush=True)
    write_json(a.output/"scene_text.json", texts)
    write_json(a.output/"review_queue.json", {"tokens": sorted({v["token"] for v in reviews}),
        "views": reviews, "meaning": "Review only; retain usable phrases and all other cameras"})
    write_json(a.output/"summary.json", {"scenes": len(records), "views": 4*len(records), "statuses": dict(statuses),
        "resumed_scenes": len(cached_scenes), "written_scenes": len(pending), "model_calls": 4*len(pending),
        "phrase_repairs_attempted": 0, "phrase_repairs_accepted": 0,
        "empty_text_views": empty_count, "empty_text_fallback_views": fallback_count,
        "all_texts_empty_scenes": all_empty, "review_views": len(reviews), "text_policy": POLICY_VERSION})
    # Even a fully empty shard is a complete, auditable planning dataset.


if __name__ == "__main__":
    main()
