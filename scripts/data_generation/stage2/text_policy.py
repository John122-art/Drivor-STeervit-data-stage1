"""Camera-local cleaning without asking a VLM to re-describe an object."""
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CAMERAS, CATEGORIES
from text_processing import clean_caption

POLICY_VERSION = "camera_local_no_model_rewrite_v2"
# Remove an action only when immediately followed by a static location phrase.
# Do not alter color, object type, location, future claims, or ego-relative words.
STATIC_ACTION = re.compile(
    r"\b(?:(?:is|are|was|were)\s+)?"
    r"(?:driving|moving|walking|running|stopped|parked|turning|crossing|ascending|descending|climbing)"
    r"\s+(?=(?:in|on|at|near|along|beside|by|under|next\s+to)\b)", re.I)


def reject_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def empty_result(tokenizer, error):
    return {"text": "", "categories": [], "objects": [], "removed": [], "local_edits": [],
        "token_count": len(tokenizer.encode("", add_special_tokens=True)), "status": "generation_failed",
        "format_normalization": "none", "error": error}


def clean_response(call, tokenizer):
    """Keep all valid objects in this view; malformed objects do not discard peers."""
    if call["finish_reason"] != "eos":
        return empty_result(tokenizer, "generation_truncated")
    answer = call["answer"].strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", answer, re.I)
    try:
        payload = json.loads(fenced.group(1) if fenced else answer, parse_constant=reject_constant)
        if not isinstance(payload, dict) or set(payload) != {"objects"} or not isinstance(payload["objects"], list):
            raise ValueError("Expected exactly {'objects': [...]} JSON")
    except (ValueError, TypeError) as exc:
        return empty_result(tokenizer, str(exc))
    objects, invalid, edits = [], [], []
    for index, item in enumerate(payload["objects"]):
        if isinstance(item, str):
            normalized = {"phrase": item, "category": "other"}
        elif isinstance(item, dict) and isinstance(item.get("phrase"), str):
            category = item.get("category")
            category = category if isinstance(category, str) and category in CATEGORIES else "other"
            normalized = {"phrase": item["phrase"], "category": category}
        else:
            invalid.append({"object_index": index, "original": item, "reason": "invalid_object_schema"})
            continue
        if normalized != item:
            edits.append({"object_index": index, "original": item, "replacement": dict(normalized),
                "reason": "preserve_phrase_default_unknown_category_to_other"})
        obj = {"phrase": " ".join(normalized["phrase"].split()), "category": normalized["category"]}
        if obj["category"] in {"vehicle", "pedestrian", "bicycle"}:
            phrase = STATIC_ACTION.sub("", obj["phrase"])
            if phrase != obj["phrase"]:
                edits.append({"object_index": index, "original": dict(obj),
                    "replacement": {"phrase": phrase, "category": obj["category"]},
                    "reason": "remove_action_before_static_location"})
                obj["phrase"] = phrase
        objects.append(obj)
    result = clean_caption(json.dumps({"objects": objects}), tokenizer)
    result["removed"] = invalid + result["removed"]
    result["local_edits"] = edits
    result["error"] = None
    result["format_normalization"] = "json_fence_removed" if fenced else "none"
    if not result["objects"] and payload["objects"]:
        result["status"] = "filtered_empty"
    return result


def view_needs_review(entry):
    return (entry["status"] not in {"ok", "empty_scene"}
        or bool(entry.get("local_edits"))
        or any(r["reason"] not in {"duplicate", "token_budget"} for r in entry["removed"]))


def scene_status(audit):
    if set(audit) != set(CAMERAS):
        raise ValueError("Missing camera audit")
    return "needs_review" if any(view_needs_review(v) for v in audit.values()) else "passed"


def empty_cameras(audit):
    return [c for c in CAMERAS if not audit[c]["text"].strip()]


def fallback_cameras(audit):
    return [c for c in empty_cameras(audit) if audit[c]["status"] != "empty_scene"]
