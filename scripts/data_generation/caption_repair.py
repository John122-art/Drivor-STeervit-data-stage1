"""Ask the same VLM to repair individual phrases against the same current image."""
import json

from text_processing import clean_caption, parse_caption

REPAIR_REASONS = {"unverified_motion", "future_or_advice", "ambiguous_ego_relative_position"}


def apply_repairs(answer, tokenizer, repairs):
    """Rebuild and validate final text from the auditable original object indices."""
    objects, _ = parse_caption(answer)
    seen = set()
    for repair in repairs:
        index = repair["object_index"]
        if index in seen or not 0 <= index < len(objects) or repair["original"] != objects[index]:
            raise ValueError("Invalid or duplicate repair source")
        seen.add(index)
        if repair["accepted"]:
            replacement = repair["replacement"]
            probe = clean_caption(json.dumps({"objects": [replacement]}), tokenizer)
            if (replacement["category"] != objects[index]["category"] or probe["removed"]
                    or probe["objects"] != [replacement]):
                raise ValueError("Invalid repaired phrase")
            objects[index] = replacement
    return clean_caption(json.dumps({"objects": objects}), tokenizer)


def repair_caption(answer, tokenizer, generate, camera_name, max_attempts=2, navigation_command=None):
    """Repair only unsafe phrases; generate(question) must see the same image.

    Valid initial objects retain their order and wording. Failed repairs remain
    subject to the existing cleaner and are explicitly recorded as unresolved.
    """
    objects, _ = parse_caption(answer)
    repairs = []
    for index, obj in enumerate(objects):
        probe = clean_caption(json.dumps({"objects": [obj]}), tokenizer)
        if not probe["removed"] or probe["removed"][0]["reason"] not in REPAIR_REASONS:
            continue
        reason = probe["removed"][0]["reason"]
        question = (
            f"You are looking at the {camera_name} camera image of a self-driving car.\n"
            + (f"Active navigation command: [{navigation_command}]. This is route intent, not evidence of objects or road geometry.\n"
               if navigation_command is not None else "")
            +
            "An earlier draft description needs a factual rewrite:\n"
            + json.dumps(obj) + "\n"
            "Look at this same image again. Describe ONLY this object's visible appearance and its "
            "current position IN THE IMAGE, including a relationship to the curb, lane, or crosswalk "
            "only if clearly visible. The draft is not ground truth. Do not invent details. "
            "Keep this object's category unchanged and do not describe other objects or buildings. "
            "Use one short phrase, at most 20 English words. Do not describe movement, future events, "
            "intentions, or driving advice. Do not use driving, walking, crossing, moving, stopped, "
            "parked, ascending, descending, or climbing as actions. Use image-relative locations, "
            "including for the rear camera. State where the object appears using left, right, center, "
            "upper, lower, foreground, or background OF THE IMAGE. Never use ego, our car, ahead, "
            "directly in front, or directly behind. "
            "If this object cannot be identified reliably in the image, return {\"objects\": []}. "
            "Otherwise return JSON only with exactly one object:\n"
            + json.dumps({"objects": [{"phrase": "...", "category": obj["category"]}]})
        )
        record = {"object_index": index, "original": obj, "reason": reason,
                  "accepted": False, "replacement": None, "attempts": []}
        for attempt in range(max_attempts):
            extra = "" if attempt == 0 else (
                "\nYour previous rewrite failed validation: " + str(record["attempts"][-1]["error"])
                + "\nPrevious reply: " + record["attempts"][-1]["answer"]
                + "\nCorrect that failure. Replace action/ego-relative wording with the object's appearance "
                  "and its position within the image (left, center, right, upper, lower). "
                  "Keep the exact original category; do not repeat the invalid wording.")
            result = dict(generate(question + extra))
            result["error"] = None
            try:
                if result["finish_reason"] != "eos":
                    raise ValueError("generation_truncated")
                proposed, _ = parse_caption(result["answer"])
                if not proposed:
                    result["error"] = "object_not_reliably_identified"
                    record["attempts"].append(result)
                    break
                if len(proposed) != 1 or proposed[0]["category"] != obj["category"]:
                    raise ValueError("Rewrite must preserve exactly one object and its category")
                checked = clean_caption(json.dumps({"objects": proposed}), tokenizer)
                if checked["removed"] or not checked["objects"]:
                    raise ValueError("Rewrite still contains unsafe wording or exceeds token budget")
                if len(proposed[0]["phrase"].split()) > 20:
                    raise ValueError("Rewrite exceeds 20 words")
                record.update(accepted=True, replacement=proposed[0])
            except (ValueError, TypeError) as exc:
                result["error"] = str(exc)
            record["attempts"].append(result)
            if record["accepted"]:
                break
        repairs.append(record)
    final = apply_repairs(answer, tokenizer, repairs)
    for record in repairs:
        record["retained_in_final"] = record["accepted"] and record["replacement"] in final["objects"]
    return final, repairs
