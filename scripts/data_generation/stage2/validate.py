"""Reuse the original geometry checks and replay camera-local, no-rewrite cleaning."""
import argparse
from pathlib import Path
import pickle
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from transformers import RobertaTokenizer
from common import CAMERAS, GROUND_CATEGORIES, IMAGE_SIZE, read_json, sha256, write_json
from text_processing import clean_caption
from caption_cache import load_audit
from text_policy import POLICY_VERSION, clean_response, scene_status
from navigation import navigation_text, qwen_navigation_text
from geometry import patch_distribution, project_box
from build_grounding import load_projection


def assert_annotation_boxes_match(raw_boxes, saved_boxes):
    """JSON turns an empty (0, 7) array into []; restore only that lost shape."""
    raw = np.asarray(raw_boxes)
    saved = np.asarray(saved_boxes)
    if saved.shape == (0,) and raw.shape == (0, 7):
        saved = saved.reshape(0, 7)
    np.testing.assert_array_equal(raw, saved)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--captions", type=Path, required=True)
    p.add_argument("--grounding", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--roberta", type=Path, default=Path("/mnt/zhaozc_workspace/project/LQH/RoBERTa"))
    p.add_argument("--deep-scenes", type=int, default=0, help="0 reprojects every scene; otherwise evenly sample N scenes")
    a = p.parse_args()
    if a.deep_scenes < 0:
        p.error("--deep-scenes must be nonnegative")
    manifest = read_json(a.prepared / "manifest.json")
    gt = read_json(a.prepared / "ground_truth.json")
    texts = read_json(a.captions / "scene_text.json")
    legacy_audit = read_json(a.captions / "audit.json") if (a.captions / "audit.json").is_file() else None
    meta = read_json(a.captions / "metadata.json")
    assert meta["text_policy"] == POLICY_VERSION and meta["model_calls_per_view"] == 1
    assert not meta["phrase_repair"]["enabled"]
    legacy_proj = read_json(a.grounding / "projections.json") if (a.grounding / "projections.json").is_file() else None
    ground_meta = read_json(a.grounding / "summary.json")
    assert meta["manifest_sha256"] == sha256(a.prepared / "manifest.json")
    assert ground_meta["manifest_sha256"] == meta["manifest_sha256"]
    assert ground_meta["caption_sha256"] == sha256(a.captions / "scene_text.json")
    assert ground_meta["gt_sha256"] == sha256(a.prepared / "ground_truth.json")
    assert not meta["loading_errors"] and not meta["planner_instantiated"]
    tokenizer = RobertaTokenizer.from_pretrained(a.roberta, local_files_only=True)
    records = {r["token"]: r for r in manifest["records"]}
    assert set(texts) == set(records), "Every scene must be retained, including empty-camera fallbacks"
    tokens = list(texts)
    deep = set(tokens if not a.deep_scenes else [tokens[i] for i in np.linspace(
        0, max(0, len(tokens)-1), min(a.deep_scenes, len(tokens)), dtype=int)])
    data_split = manifest.get("data_split", "test" if manifest["split"] == "navtest" else "trainval")
    identities, views, valid_views, max_tokens = [], 0, 0, 0
    last_log, frames_by_token = None, {}
    for token, record in records.items():
        log_path = Path(manifest["data_root"]) / "navsim_logs" / data_split / (record["log_name"] + ".pkl")
        if log_path != last_log:
            with log_path.open("rb") as f:
                frames_by_token = {frame["token"]: frame for frame in pickle.load(f)}
            last_log = log_path
        frame = frames_by_token[token]
        assert int(frame["timestamp"]) == record["timestamp"]
        if "driving_command" in record:
            np.testing.assert_array_equal(frame["driving_command"], record["driving_command"])
            assert navigation_text(record["driving_command"]) == record["navigation_text"]
        assert_annotation_boxes_match(frame["anns"]["gt_boxes"], gt[token]["boxes"])
        assert list(frame["anns"]["gt_names"]) == gt[token]["names"]
        np.testing.assert_allclose(frame["lidar2ego"], np.eye(4), atol=1e-8)
        if token in deep:
            identities.append({"token": token, "lidar2ego": np.asarray(frame["lidar2ego"]).tolist(),
                               "raw_log": str(log_path)})
        for ci, camera in enumerate(CAMERAS):
            raw = frame["cams"][camera.upper()]
            path = (Path(manifest["data_root"]) / "sensor_blobs" / data_split / raw["data_path"]).resolve()
            assert str(path) == record["images"][camera]
            assert sha256(path) == record["image_sha256"][camera]
            with Image.open(path) as image:
                width, height = image.size
            k = np.array(raw["cam_intrinsic"], dtype=float)
            k[0, [0, 2]] *= IMAGE_SIZE[0] / width
            k[1, [1, 2]] *= IMAGE_SIZE[1] / height
            np.testing.assert_allclose(k, gt[token]["cam_K"][ci])
            t = np.eye(4)
            t[:3, :3] = raw["sensor2lidar_rotation"]
            t[:3, 3] = raw["sensor2lidar_translation"]
            np.testing.assert_allclose(np.linalg.inv(t), gt[token]["lidar_to_camera"][ci])
        if token not in texts:
            continue
        audits = load_audit(a.captions, token, legacy_audit)
        if meta.get("stream_cache"):
            cache = read_json(a.captions / "scenes" / f"{token}.json")
            assert cache["rule_status"] == scene_status(audits)
        projections = load_projection(a.grounding, token, legacy_proj)
        assert set(texts[token]) == set(CAMERAS)
        with np.load(a.grounding / "masks" / f"{token}.npz", allow_pickle=False) as target:
            assert target["camera_order"].tolist() == list(CAMERAS)
            assert str(target["scene_token"]) == token
            masks, valid = target["soft_mask"], target["valid"]
            assert masks.shape == (4, 3936) and valid.shape == (4,)
            assert np.isfinite(masks).all() and np.all(masks >= 0)
            np.testing.assert_allclose(masks.sum(axis=1), valid.astype(float), atol=1e-6)
            valid_views += int(valid.sum())
            expected_boxes, expected_ci, expected_ids, expected_cats = [], [], [], []
            for ci, camera in enumerate(CAMERAS):
                entry = audits[camera]
                assert not entry.get("repairs") and len(entry["attempts"]) == 1
                calls = entry["attempts"]
                for call in calls:
                    assert call["image"] == record["images"][camera] and len(call["image_grid_thw"]) == 1
                    if meta.get("navigation_conditioned"):
                        cmd = qwen_navigation_text(record["driving_command"], meta["qwen_nav_commands"])
                        assert entry["navigation_command"] == cmd
                        assert meta["navigation_by_scene"][token]["driving_command"] == record["driving_command"]
                        assert f"Active navigation command: [{cmd}]" in call["question"]
                    elif meta.get("stage1"):
                        assert entry["navigation_command"] is None
                        assert "Active navigation command:" not in call["question"]
                cleaned = clean_response(calls[0], tokenizer)
                for key, value in cleaned.items():
                    assert entry[key] == value, (token, camera, key)
                assert texts[token][camera] == {k: cleaned[k] for k in ("text", "categories")}
                assert cleaned["token_count"] <= 128
                max_tokens = max(max_tokens, cleaned["token_count"])
                if meta.get("stage1"):
                    assert set(cleaned["categories"]) <= GROUND_CATEGORIES
                source = projections[camera]
                objects = source["objects"]
                by_id = {o["gt_index"]: o for o in objects}
                assert len(by_id) == len(objects)
                selected = [o for o in objects if cleaned["text"].strip() and o["category"] in cleaned["categories"]]
                assert source["selected_gt_indices"] == [o["gt_index"] for o in selected]
                assert bool(valid[ci]) == source["valid"] == bool(selected)
                for obj in objects:
                    box = np.asarray(obj["bbox"], dtype=float)
                    assert box.shape == (4,) and np.isfinite(box).all()
                    assert 0 <= box[0] < box[2] <= IMAGE_SIZE[0]
                    assert 0 <= box[1] < box[3] <= IMAGE_SIZE[1]
                for obj in selected:
                    expected_boxes.append(obj["bbox"])
                    expected_ci.append(ci)
                    expected_ids.append(obj["gt_index"])
                    expected_cats.append(obj["category"])
                if token in deep:
                    fresh = {}
                    for gi, (box, cat) in enumerate(zip(gt[token]["boxes"], gt[token]["names"])):
                        if cat not in GROUND_CATEGORIES or np.linalg.norm(np.asarray(box)[:2]) > ground_meta["radius_m"]:
                            continue
                        projected = project_box(box, gt[token]["cam_K"][ci],
                                                gt[token]["lidar_to_camera"][ci], ground_meta["near_plane_m"])
                        if projected is not None:
                            fresh[gi] = projected["bbox"]
                    assert set(fresh) == set(by_id)
                    for gi, bbox in fresh.items():
                        np.testing.assert_allclose(bbox, by_id[gi]["bbox"], atol=1e-5)
                        assert by_id[gi]["category"] == gt[token]["names"][gi]
                    mask, fresh_valid = patch_distribution([fresh[o["gt_index"]] for o in selected])
                    assert fresh_valid == valid[ci]
                    np.testing.assert_allclose(mask, masks[ci], atol=1e-6)
                views += 1
            if ground_meta.get("label_schema", 1) >= 2:
                assert str(target["coordinate_format"]) == "xyxy"
                np.testing.assert_array_equal(target["image_size"], IMAGE_SIZE)
                np.testing.assert_allclose(target["boxes_2d"], np.asarray(expected_boxes).reshape(-1, 4), atol=1e-4)
                np.testing.assert_array_equal(target["box_camera_index"], expected_ci)
                np.testing.assert_array_equal(target["box_gt_index"], expected_ids)
                np.testing.assert_array_equal(target["box_category"], expected_cats)
                scene = read_json(a.grounding / "scenes" / f"{token}.json")
                assert str(target["input_hash"]) == scene["input_hash"]
    html_path = a.grounding / "index.html"
    if html_path.is_file():
        for relative in set(re.findall(r'(?:src|href)="([^"]+)"', html_path.read_text())):
            with Image.open(a.grounding / relative) as image:
                image.verify()
    write_json(a.output, {"status": "passed", "prepared_scenes": len(records),
        "current_images_verified": len(records)*4, "caption_scenes": len(texts), "caption_views": views,
        "valid_grounding_views": valid_views, "deep_reprojected_scenes": len(deep),
        "max_roberta_tokens": max_tokens, "camera_order": CAMERAS,
        "all_lidar2ego_identity": True, "coordinate_checks": identities,
        "navigation_conditioned": meta.get("navigation_conditioned", False),
        "text_policy": POLICY_VERSION, "model_rewrites": 0,
        "checks": ["raw frame/images/GT and current command", "native intrinsics/inverse extrinsics",
                   "camera-local cleaning replay, one model call and same camera/command", "selected 2D bounds/category/GT indices",
                   "NPZ boxes and camera indices", "all mask shapes/normalization",
                   "sampled or full GT reprojection and mask reconstruction", "gallery files"],
        "visual_review": "numeric checks do not establish visual caption factuality"})
    print(f"PASS: {len(records)} scenes, {views} captions, {valid_views} valid targets, {len(deep)} deep checks", flush=True)


if __name__ == "__main__":
    main()

