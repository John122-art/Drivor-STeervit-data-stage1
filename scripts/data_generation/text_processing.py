"""Structured caption validation; never insert GT-derived phrases or boxes."""
import json
import re
import unicodedata
from common import CATEGORIES

PROMPT = '''You are looking at the {camera_name} camera image of a self-driving car.
Describe the visible driving scene using concise, factual English phrases.
Include relevant road layout, markings, traffic controls, road users, obstacles,
and surrounding structures when clearly visible. There is no fixed number of
items. Prioritize informative details, avoid repetition, and keep the combined
phrases under 80 English words. EVERY phrase must include appearance and position
IN THE IMAGE, for example "a white van near the center of the image", "buildings
across the upper image", or "a crosswalk across the lower image".
Do not infer intentions, future events, or motion that cannot be established
from this single image. Do not give driving advice. Do not output bounding
boxes or coordinates. Use "other" for road layout, markings, and buildings.
Return JSON only, without explanations or markdown:
{{"objects": [{{"phrase": "...", "category": "..."}}]}}
Categories: vehicle, pedestrian, bicycle, traffic_cone, barrier, traffic_light,
traffic_sign, other. These are the ONLY category values; do not invent categories
such as road_marking or building. Describe cars by color/type/location, without
motion words such as driving, moving, turning, following, walking, or stopped.
Use image coordinates even for the rear camera. Do not say "ahead" or "directly
in front". For each item use appearance + object + image location, e.g.
{{"objects": [{{"phrase": "a white car in the center of the image", "category": "vehicle"}},
{{"phrase": "buildings across the upper image", "category": "other"}}]}}.
This example specifies the format only, not what is in the input image.
If nothing can be described reliably, return {{"objects": []}}.'''

FORBIDDEN = re.compile(r"\b(will|going\s+to|about\s+to|should|plans?|intend\w*|must|ought)\b", re.I)
MOTION = re.compile(r"\b(driving|drives|moving|moves|turning|turns|following|follows|merging|merges|approaching|approaches|overtaking|stopped|parked|accelerating|braking|walking|walks|running|runs|crossing|crosses|ascending|ascends|descending|descends|climbing|climbs)\b", re.I)
AMBIGUOUS_POSITION = re.compile(r"\b(ahead|directly\s+in\s+front|behind\s+(?:the\s+)?(?:ego|our)\s+(?:car|vehicle))\b", re.I)
# "near a pedestrian crossing" names a location, not observed movement.
STATIC_CROSSING = re.compile(r"\b(?:near|at|beside|by|along|around|next\s+to)\s+(?:(?:the|a|an)\s+)?(?:(?:pedestrian|zebra)\s+)?crossings?\b", re.I)


def parse_caption(answer):
    """Validate schema and normalize phrases without deleting any objects."""
    answer = answer.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", answer, re.I)
    if fenced:
        answer = fenced.group(1)
    payload = json.loads(answer)
    if not isinstance(payload, dict) or set(payload) != {"objects"} or not isinstance(payload["objects"], list):
        raise ValueError("Expected exactly {'objects': [...]} JSON")
    objects = []
    for obj in payload["objects"]:
        if not isinstance(obj, dict) or set(obj) != {"phrase", "category"}:
            raise ValueError("Each object must contain phrase and category only")
        phrase, category = obj["phrase"], obj["category"]
        if not isinstance(phrase, str) or not isinstance(category, str) or category not in CATEGORIES:
            raise ValueError("Invalid phrase or category")
        objects.append({"phrase": " ".join(phrase.split()), "category": category})
    return objects, bool(fenced)


def clean_caption(answer, tokenizer, max_tokens=128):
    """Return phrase-aligned cleaned data; callers can repair semantic removals."""
    objects, fenced = parse_caption(answer)
    kept, removed, seen = [], [], set()
    for obj in objects:
        phrase, category = obj["phrase"], obj["category"]
        reason = None
        if not phrase:
            reason = "empty_phrase"
        elif not re.search(r"[A-Za-z]", phrase) or any(ch.isalpha() and "LATIN" not in unicodedata.name(ch, "") for ch in phrase):
            reason = "non_english_script"
        elif FORBIDDEN.search(phrase):
            reason = "future_or_advice"
        elif category in {"vehicle", "pedestrian", "bicycle"} and MOTION.search(STATIC_CROSSING.sub("crosswalk", phrase)):
            reason = "unverified_motion"
        elif AMBIGUOUS_POSITION.search(phrase):
            reason = "ambiguous_ego_relative_position"
        elif phrase.casefold() in seen:
            reason = "duplicate"
        else:
            candidate = ", ".join([o["phrase"] for o in kept] + [phrase])
            if len(tokenizer.encode(candidate, add_special_tokens=True)) > max_tokens:
                reason = "token_budget"
        if reason:
            removed.append({"phrase": phrase, "category": category, "reason": reason})
        else:
            kept.append({"phrase": phrase, "category": category})
            seen.add(phrase.casefold())
    text = ", ".join(o["phrase"] for o in kept)
    return {"text": text, "categories": [o["category"] for o in kept], "objects": kept,
            "token_count": len(tokenizer.encode(text, add_special_tokens=True)),
            "removed": removed,
            "format_normalization": "json_fence_removed" if fenced else "none",
            "status": "ok" if kept else ("filtered_empty" if objects else "empty_scene")}
