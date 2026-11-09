"""Single-view Qwen-Drive captions, reusing official encode_vqa and VQA defaults.

VLM-only loading is adapted from DrivoR-main-copy/navsim/agents/qwen_text/backend.py.
Optional navigation uses the current AgentInput command and official Qwen vocabulary.
No history, GT annotations or future frames are supplied to the VLM.
"""
import argparse
import hashlib
from pathlib import Path
import sys
import time
from collections import Counter

from common import CAMERAS, CAMERA_NAMES, GROUND_CATEGORIES, fingerprint, read_json, sha256, write_json
from caption_cache import load_scene_cache, rule_status
from text_processing import PROMPT, clean_caption
from caption_repair import repair_caption
from navigation import build_question, navigation_text, qwen_navigation_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=2)
    parser.add_argument("--model", type=Path, default=Path("/mnt/zhaozc_workspace/project/LQH/Qwen-drive/checkpoint"))
    parser.add_argument("--source", type=Path, default=Path("/mnt/zhaozc_workspace/project/LQH/Qwen-drive/Qwen-Drive-1.0-main/src"))
    parser.add_argument("--roberta", type=Path, default=Path("/mnt/zhaozc_workspace/project/LQH/RoBERTa"))
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt-file", type=Path, help="Literal prompt with {camera_name} and optional {navigation_command}")
    parser.add_argument("--with-navigation", action="store_true", help="Use current NAVSIM input command with Qwen's official vocabulary")
    parser.add_argument("--stage1", action="store_true", help="Enforce five grounding classes and omit navigation")
    parser.add_argument("--stream-cache", action="store_true", help="Atomic per-scene audit files for bulk generation")
    parser.add_argument("--resume", action="store_true", help="Reuse matching completed stream caches")
    args = parser.parse_args()
    prompt = args.prompt_file.read_text(encoding="utf-8").strip() if args.prompt_file else PROMPT
    if "{camera_name}" not in prompt:
        parser.error("Prompt must contain {camera_name}")
    if ("{navigation_command}" in prompt) != args.with_navigation:
        parser.error("--with-navigation requires a prompt containing {navigation_command}, and vice versa")
    if args.stage1 and args.with_navigation:
        parser.error("Stage 1 does not take navigation")
    if args.resume and not args.stream_cache:
        parser.error("--resume requires --stream-cache")
    manifest_path = args.prepared / "manifest.json"
    manifest = read_json(manifest_path)
    assert manifest["camera_order"] == list(CAMERAS)
    if not 0 <= args.scenes <= len(manifest["records"]):
        parser.error("--scenes must select a nonempty subset of the prepared manifest")
    records = manifest["records"][:args.scenes] if args.scenes else manifest["records"]
    if args.with_navigation:
        for record in records:
            if "driving_command" not in record:
                parser.error("Prepared manifest has no current navigation command; prepare a new manifest")
            if record.get("navigation_text") != navigation_text(record["driving_command"]):
                parser.error("Prepared command vector and text disagree")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    sys.path.insert(0, str(args.source.resolve()))
    import torch
    import transformers
    from safetensors import safe_open
    import qwen_drive
    from qwen_drive import NAV_COMMANDS, QwenDriveConfig, QwenDriveProcessor
    from qwen_drive.modeling_qwen_drive import VQA_DECODE_DEFAULTS, _strip_thinking
    from transformers import AutoTokenizer, RobertaTokenizer, Qwen3_5ForConditionalGeneration
    assert Path(qwen_drive.__file__).resolve().parent == args.source.resolve() / "qwen_drive"
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("PPU/GPU not available; run via the documented device launcher")
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
    config = QwenDriveConfig.from_pretrained(args.model, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    roberta = RobertaTokenizer.from_pretrained(args.roberta, local_files_only=True)
    processor = QwenDriveProcessor(tokenizer, config)
    with safe_open(str(args.model / "model.safetensors"), framework="pt", device="cpu") as weights:
        keys = list(weights.keys())
    if not keys or not all(k.startswith("vlm.") for k in keys):
        raise ValueError("Expected root Qwen-Drive VLM-only checkpoint")
    print("Loading Qwen-Drive VLM only (BF16/eager), using existing reference loader", flush=True)
    model, info = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, config=config.vlm_config, key_mapping={r"^vlm\.": ""},
        dtype=torch.bfloat16, device_map=args.device, attn_implementation="eager",
        local_files_only=True, output_loading_info=True)
    errors = {k: [str(x) for x in v] for k, v in info.items() if v}
    if errors:
        raise RuntimeError(f"Incomplete VLM load: {errors}")
    model.eval().requires_grad_(False)
    if config.vlm_config.tie_word_embeddings:
        assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    metadata = {"manifest_sha256": sha256(manifest_path), "camera_order": CAMERAS,
                "model": str(args.model.resolve()), "model_weight_bytes": (args.model / "model.safetensors").stat().st_size,
                "model_config_sha256": sha256(args.model / "config.json"),
                "model_weight_mtime_ns": (args.model / "model.safetensors").stat().st_mtime_ns,
                "source_hashes": {str(p): sha256(p) for p in sorted((args.source / "qwen_drive").glob("*.py"))},
                "pipeline_source_hashes": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "prompt_file": str(args.prompt_file.resolve()) if args.prompt_file else None,
                "python": sys.executable, "torch": torch.__version__, "transformers": transformers.__version__,
                "dtype": "bfloat16", "attention": "eager", "current_image_pixels": config.current_image_pixels,
                "device": args.device,
                "checkpoint_tensors": len(keys), "loading_errors": errors, "planner_instantiated": False,
                "roberta_tokenizer": str(args.roberta), "max_text_tokens": 128,
                "scenes": len(records), "views": 4 * len(records), "batch_size": 1,
                "navigation_conditioned": args.with_navigation,
                "stage1": args.stage1, "stream_cache": args.stream_cache,
                "qwen_nav_commands": list(NAV_COMMANDS),
                "navigation_by_scene": {r["token"]: {
                    "driving_command": r["driving_command"],
                    "text": qwen_navigation_text(r["driving_command"], NAV_COMMANDS),
                    "source": r["navigation_source"]} for r in records} if args.with_navigation else {},
                "phrase_repair": {"enabled": True, "max_attempts_per_phrase": 2,
                                  "same_current_image": True, "category_preserved": True},
                "decoding": dict(VQA_DECODE_DEFAULTS, max_new_tokens=args.max_new_tokens)}
    version = fingerprint(metadata)
    metadata["cache_version"] = version
    if args.resume and (args.output / "metadata.json").is_file():
        if read_json(args.output / "metadata.json").get("cache_version") != version:
            raise ValueError("Output version changed; use a new output directory")
    write_json(args.output / "metadata.json", metadata)
    if args.stream_cache:
        (args.output / "scenes").mkdir(exist_ok=True)
    texts, audits = {}, {}
    statuses, scene_statuses, review_tokens = [], Counter(), []
    repair_count = repaired_count = reused_count = written_count = 0
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, (tuple, list)) else [eos])

    def generate_answer(image_path, question, token_budget):
        start = time.monotonic()
        inputs = processor.encode_vqa([image_path], question, device=args.device)
        assert inputs["image_grid_thw"].shape[0] == 1
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
        inputs["mm_token_type_ids"] = (inputs["input_ids"] == config.vlm_config.image_token_id).long()
        width = inputs["input_ids"].shape[1]
        params = dict(VQA_DECODE_DEFAULTS)
        torch.manual_seed(params.pop("seed"))
        params.pop("presence_penalty", None)
        params.update(max_new_tokens=token_budget, pad_token_id=tokenizer.pad_token_id, use_cache=True)
        with torch.inference_mode():
            generated = model.generate(**inputs, **params)
        ids = generated[0, width:].tolist()
        raw = tokenizer.decode(ids, skip_special_tokens=True)
        return {"raw_text": raw, "answer": _strip_thinking(raw),
                "generated_tokens": len(ids), "seconds": time.monotonic() - start,
                "finish_reason": "eos" if any(i in eos for i in ids) else "length",
                "question": question, "image": image_path,
                "image_grid_thw": inputs["image_grid_thw"].tolist()}

    for record in records:
        token = record["token"]
        cache_path = args.output / "scenes" / f"{token}.json"
        if args.resume and cache_path.is_file():
            for camera, path in record["images"].items():
                if sha256(path) != record["image_sha256"][camera]:
                    raise ValueError(f"Source image changed: {path}")
            cached = load_scene_cache(cache_path, version, record)
            if cached["rule_status"] == "passed" and rule_status(cached["audit"], args.stage1) == "passed":
                texts[token] = cached["scene_text"]
                for view in cached["audit"].values():
                    statuses.append(view["status"])
                    repair_count += len(view.get("repairs", []))
                    repaired_count += sum(r["accepted"] for r in view.get("repairs", []))
                reused_count += 1
                scene_statuses["passed"] += 1
                print(f"Reused scene {token}", flush=True)
                continue
            write_json(args.output / "review_history" / f'{token}.{fingerprint(cached)[:16]}.json', cached)
        texts[token], audits[token] = {}, {}
        for camera, camera_name in zip(CAMERAS, CAMERA_NAMES):
            path = record["images"][camera]
            if sha256(path) != record["image_sha256"][camera]:
                raise RuntimeError(f"Image changed since preparation: {path}")
            command_text = qwen_navigation_text(record["driving_command"], NAV_COMMANDS) if args.with_navigation else None
            question = (build_question(prompt, camera_name, command_text) if args.prompt_file
                        else prompt.format(camera_name=camera_name))
            attempts, cleaned = [], None
            for attempt in range(2):
                attempt_question = question
                if attempt:
                    attempt_question += ("\nYour previous answer failed JSON/schema validation. Return exactly the specified JSON object. "
                                         "Use only these category values: vehicle, pedestrian, bicycle, traffic_cone, barrier, "
                                         "traffic_light, traffic_sign, other. The category traffic_conditions is NOT allowed; "
                                         "traffic conditions, road markings and buildings must use other. "
                                         "Include image position in every phrase. No markdown fences or additional text.")
                    if args.stage1:
                        attempt_question += " For this stage use ONLY vehicle, pedestrian, bicycle, traffic_cone, barrier."
                result = generate_answer(path, attempt_question, args.max_new_tokens * (attempt + 1))
                answer = result["answer"]
                error = None
                try:
                    if result["finish_reason"] != "eos":
                        raise ValueError("generation_truncated")
                    cleaned = clean_caption(answer, roberta)
                    if args.stage1 and not set(cleaned["categories"]) <= GROUND_CATEGORIES:
                        raise ValueError("Stage 1 only supports the five grounding categories")
                except (ValueError, TypeError) as exc:
                    error = str(exc)
                    cleaned = None
                attempts.append(dict(result, error=error))
                if cleaned is not None:
                    break
            repairs, initial_cleaning = [], cleaned
            if cleaned is not None:
                cleaned, repairs = repair_caption(answer, roberta,
                    lambda repair_question: generate_answer(path, repair_question, 512), camera_name,
                    navigation_command=command_text)
                for repair in repairs:
                    print(f'{token} {camera}: repair {repair["accepted"]}: {repair["original"]["phrase"]} -> {repair["replacement"]}', flush=True)
            else:
                cleaned = {"text": "", "categories": [], "objects": [], "removed": [],
                           "token_count": len(roberta.encode("")), "status": "generation_failed"}
            texts[token][camera] = {k: cleaned[k] for k in ("text", "categories")}
            audits[token][camera] = dict(cleaned, attempts=attempts, initial_cleaning=initial_cleaning,
                                        repairs=repairs, navigation_command=command_text)
            if not args.stream_cache:
                write_json(args.output / "scene_text.json", texts)
                write_json(args.output / "audit.json", audits)
            print(f'{token} {camera}: {cleaned["status"]}, {cleaned["token_count"]} tokens; {cleaned["text"]}', flush=True)
        status = rule_status(audits[token], args.stage1)
        scene_statuses[status] += 1
        for view in audits[token].values():
            statuses.append(view["status"])
            repair_count += len(view.get("repairs", []))
            repaired_count += sum(r["accepted"] for r in view.get("repairs", []))
        if status != "passed":
            review_tokens.append(token)
        if args.stream_cache:
            write_json(cache_path, {"token": token, "cache_version": version,
                "input_hash": fingerprint(record), "rule_status": status,
                "scene_text": texts[token], "audit": audits[token]})
            written_count += 1
            if status != "passed":
                texts.pop(token)
            audits.pop(token)
    if args.stream_cache:
        write_json(args.output / "scene_text.json", texts)
    write_json(args.output / "review_queue.json", {"tokens": review_tokens,
        "meaning": "rule failures; semantic image review is still needed for passed entries"})
    write_json(args.output / "summary.json", {"scenes": len(records), "views": len(statuses),
               "statuses": {s: statuses.count(s) for s in sorted(set(statuses))},
               "scene_rule_statuses": dict(scene_statuses), "resumed_scenes": reused_count,
               "written_scenes": written_count, "phrase_repairs_attempted": repair_count,
               "phrase_repairs_accepted": repaired_count,
               "peak_device_memory_bytes": torch.cuda.max_memory_allocated(torch.device(args.device)) if args.device.startswith("cuda") else None})
    if "generation_failed" in statuses and not args.stream_cache:
        raise SystemExit("Pilot contains failed generations; empty fallbacks saved for inspection, not approved for bulk use")
    if args.stream_cache and records and not texts:
        raise SystemExit("No scenes passed rule checks; inspect per-scene caches and review_queue.json")


if __name__ == "__main__":
    main()
