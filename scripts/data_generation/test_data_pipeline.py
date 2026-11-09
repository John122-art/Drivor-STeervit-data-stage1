"""Geometry and text contracts that would silently corrupt offline labels."""
import json
import unittest
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
from common import CAMERAS
from geometry import box_corners, clip_edges, patch_distribution, project_box
from text_processing import clean_caption
from caption_repair import apply_repairs, repair_caption
from navigation import build_question, navigation_text, qwen_navigation_text
from common import fingerprint, shard_items, write_json, write_npz
from caption_cache import load_scene_cache, rule_status
from build_grounding import pack_grounding_arrays
from selection import scene_selection
from common import sha256, read_json


class WordTokenizer:
    def encode(self, text, add_special_tokens=True):
        return text.split() + (["<s>", "</s>"] if add_special_tokens else [])


class DataContracts(unittest.TestCase):
    def test_native_corner_dimensions_and_heading(self):
        corners = box_corners([0, 0, 0, 4, 2, 6, 0])
        np.testing.assert_allclose(np.ptp(corners, axis=0), [4, 2, 6])
        rotated = box_corners([0, 0, 0, 4, 2, 6, np.pi / 2])
        np.testing.assert_allclose(np.ptp(rotated, axis=0), [2, 4, 6])

    def test_near_plane_intersections(self):
        corners = box_corners([0, 0, 0, 2, 2, 2, 0])
        edges = clip_edges(corners, 0.1)
        self.assertTrue(np.all(edges[:, :, 2] >= 0.1))
        self.assertTrue(np.any(edges[:, :, 2] == 0.1))
        self.assertEqual(len(clip_edges(corners - [0, 0, 5])), 0)

    def test_pinhole_projection_and_camera_translation(self):
        k = np.array([[100, 0, 500], [0, 100, 300], [0, 0, 1]])
        result = project_box([0, 0, 10, 2, 2, 2, 0], k, np.eye(4))
        np.testing.assert_allclose(result["bbox"], [500-100/9, 300-100/9, 500+100/9, 300+100/9])
        t = np.eye(4); t[2, 3] = -20
        self.assertIsNone(project_box([0, 0, 10, 2, 2, 2, 0], k, t))

    def test_union_not_double_counted(self):
        y, valid = patch_distribution([[0, 0, 14, 14], [0, 0, 14, 14], [14, 0, 28, 14]])
        self.assertTrue(valid)
        np.testing.assert_allclose(y[:2], [0.5, 0.5])
        self.assertEqual(np.count_nonzero(y), 2)

    def test_partial_patch_and_empty(self):
        y, _ = patch_distribution([[7, 0, 28, 14]])
        np.testing.assert_allclose(y[:2], [1/3, 2/3])
        empty, valid = patch_distribution([])
        self.assertFalse(valid); self.assertEqual(empty.sum(), 0)

    def test_text_filters_keep_categories_aligned(self):
        data = {"objects": [{"phrase": "black car", "category": "vehicle"},
                            {"phrase": "should turn left", "category": "other"},
                            {"phrase": "中文", "category": "other"},
                            {"phrase": "white sign at the right", "category": "traffic_sign"}]}
        result = clean_caption(json.dumps(data), WordTokenizer(), max_tokens=5)
        self.assertEqual(result["text"], "black car")
        self.assertEqual(result["categories"], ["vehicle"])
        self.assertEqual(len(result["removed"]), 3)

    def test_json_schema_and_empty(self):
        for value in ['[]', '{"objects": "car"}', '{"objects": [{"phrase":"car","category":"unknown"}]}', 'not json']:
            with self.assertRaises(ValueError): clean_caption(value, WordTokenizer())
        self.assertEqual(clean_caption('{"objects":[]}', WordTokenizer())["status"], "empty_scene")

    def test_near_plane_clipped_projection_stays_in_image(self):
        k = np.array([[100, 0, 500], [0, 100, 300], [0, 0, 1]])
        result = project_box([0, 0, 0, 2, 2, 2, 0], k, np.eye(4))
        self.assertTrue(result["near_clipped"])
        np.testing.assert_allclose(result["bbox"], [0, 0, 1148, 672])

    def test_fenced_json_only_no_surrounding_prose(self):
        result = clean_caption('```json\n{"objects":[]}\n```', WordTokenizer())
        self.assertEqual(result["format_normalization"], "json_fence_removed")
        with self.assertRaises(ValueError):
            clean_caption('Here is JSON: ```json\n{"objects":[]}\n```', WordTokenizer())

    def test_motion_claim_removes_aligned_category(self):
        result = clean_caption('{"objects":[{"phrase":"moving car on the right","category":"vehicle"}]}', WordTokenizer())
        self.assertEqual(result["text"], "")
        self.assertEqual(result["categories"], [])
        self.assertEqual(result["removed"][0]["reason"], "unverified_motion")

    def test_direction_wording_and_sign_are_distinct(self):
        result = clean_caption('{"objects":[{"phrase":"black car directly in front","category":"vehicle"}, {"phrase":"pedestrian crossing sign on the right","category":"traffic_sign"}]}', WordTokenizer())
        self.assertEqual(result["categories"], ["traffic_sign"])
        self.assertEqual(result["removed"][0]["reason"], "ambiguous_ego_relative_position")

    def test_repair_preserves_other_objects_and_original_order(self):
        answer = json.dumps({"objects": [
            {"phrase": "green light at top center", "category": "traffic_light"},
            {"phrase": "white van driving in center lane", "category": "vehicle"}]})
        questions = []
        def generate(question):
            questions.append(question)
            return {"answer": '{"objects":[{"phrase":"white van in the center lane of the image","category":"vehicle"}]}',
                    "finish_reason": "eos"}
        result, repairs = repair_caption(answer, WordTokenizer(), generate, "front")
        self.assertEqual(result["categories"], ["traffic_light", "vehicle"])
        self.assertEqual(result["objects"][0]["phrase"], "green light at top center")
        self.assertEqual(repairs[0]["object_index"], 1)
        self.assertTrue(repairs[0]["accepted"] and repairs[0]["retained_in_final"])
        self.assertEqual(len(questions), 1)
        self.assertIn("same image", questions[0])
        self.assertEqual(result, apply_repairs(answer, WordTokenizer(), repairs))

    def test_repair_retries_motion_and_keeps_static_pedestrian(self):
        answer = '{"objects":[{"phrase":"pedestrians walking near the right curb","category":"pedestrian"}]}'
        replies = iter(["pedestrians walking near right curb", "pedestrians near the right curb of the image"])
        def generate(question):
            return {"answer": json.dumps({"objects": [{"phrase": next(replies), "category": "pedestrian"}]}),
                    "finish_reason": "eos"}
        result, repairs = repair_caption(answer, WordTokenizer(), generate, "front-right")
        self.assertEqual(result["categories"], ["pedestrian"])
        self.assertEqual(len(repairs[0]["attempts"]), 2)
        self.assertIsNotNone(repairs[0]["attempts"][0]["error"])
        self.assertTrue(repairs[0]["accepted"])

    def test_unidentifiable_repair_does_not_invent_object(self):
        answer = '{"objects":[{"phrase":"moving car on left","category":"vehicle"}]}'
        result, repairs = repair_caption(answer, WordTokenizer(),
            lambda q: {"answer": '{"objects":[]}', "finish_reason": "eos"}, "rear")
        self.assertFalse(repairs[0]["accepted"])
        self.assertEqual(result["objects"], [])
        self.assertEqual(result["removed"][0]["reason"], "unverified_motion")

    def test_repair_rejects_wrong_category_or_multiple_objects(self):
        answer = '{"objects":[{"phrase":"moving car on left","category":"vehicle"}]}'
        replies = iter([
            [{"phrase": "car on left", "category": "other"}],
            [{"phrase": "car on left", "category": "vehicle"}, {"phrase": "car on right", "category": "vehicle"}]])
        result, repairs = repair_caption(answer, WordTokenizer(),
            lambda q: {"answer": json.dumps({"objects": next(replies)}), "finish_reason": "eos"}, "rear")
        self.assertFalse(repairs[0]["accepted"])
        self.assertEqual(len(repairs[0]["attempts"]), 2)
        self.assertEqual(result["categories"], [])

    def test_static_crossing_location_and_ascending_action(self):
        result = clean_caption(json.dumps({"objects": [
            {"phrase": "pedestrians near the pedestrian crossing on the left", "category": "pedestrian"},
            {"phrase": "pedestrians ascending escalator on right", "category": "pedestrian"},
            {"phrase": "pedestrians crossing the road", "category": "pedestrian"}]}), WordTokenizer())
        self.assertEqual(len(result["objects"]), 1)
        self.assertEqual(len(result["removed"]), 2)

    def test_repair_replay_rejects_tampered_source(self):
        answer = '{"objects":[{"phrase":"moving car on left","category":"vehicle"}]}'
        with self.assertRaises(ValueError):
            apply_repairs(answer, WordTokenizer(), [{"object_index": 0,
                "original": {"phrase": "different car", "category": "vehicle"}, "accepted": False}])

    def test_navigation_mapping_does_not_reuse_navsim_index(self):
        qwen_commands = ("GO STRAIGHT", "TURN LEFT", "TURN RIGHT")
        self.assertEqual(qwen_navigation_text([1, 0, 0, 0], qwen_commands), "TURN LEFT")
        self.assertEqual(qwen_navigation_text([0, 1, 0, 0], qwen_commands), "GO STRAIGHT")
        self.assertEqual(qwen_navigation_text([0, 0, 1], qwen_commands), "TURN RIGHT")
        self.assertEqual(navigation_text([0, 0, 0, 1]), "unknown")
        self.assertEqual(qwen_navigation_text([0, 0, 0], qwen_commands), "UNKNOWN")

    def test_navigation_rejects_invalid_vectors(self):
        for command in ([1, 1, 0], [0, 0.5, 0.5], [float("nan"), 0, 0], [1, 0], [1, 0, 0, 0, 0]):
            with self.assertRaises(ValueError):
                navigation_text(command)

    def test_navigation_template_preserves_json_and_requires_command(self):
        template = '{camera_name}: [{navigation_command}] {"objects": []}'
        self.assertEqual(build_question(template, "rear", "TURN LEFT"), 'rear: [TURN LEFT] {"objects": []}')
        with self.assertRaises(ValueError):
            build_question(template, "front")
        with self.assertRaises(ValueError):
            build_question('{camera_name}: {"objects": []}', "front", "GO STRAIGHT")

    def test_phrase_repair_keeps_navigation_context(self):
        questions = []
        def generate(question):
            questions.append(question)
            return {"answer": '{"objects":[{"phrase":"white van at center of image","category":"vehicle"}]}',
                    "finish_reason": "eos"}
        result, repairs = repair_caption('{"objects":[{"phrase":"moving white van","category":"vehicle"}]}',
            WordTokenizer(), generate, "front", navigation_command="GO STRAIGHT")
        self.assertEqual(result["categories"], ["vehicle"])
        self.assertTrue(repairs[0]["accepted"])
        self.assertIn("Active navigation command: [GO STRAIGHT]", questions[0])

    def test_eight_shards_are_disjoint_and_complete(self):
        items = list(range(101))
        shards = [shard_items(items, i, 8) for i in range(8)]
        self.assertEqual(sorted(x for shard in shards for x in shard), items)
        self.assertEqual(sum(map(len, shards)), len(set(x for shard in shards for x in shard)))
        with self.assertRaises(ValueError):
            shard_items(items, 8, 8)

    def test_native_train_and_val_log_selection_are_disjoint(self):
        _, train, _ = scene_selection("navtrain", "train", 3407)
        _, val, _ = scene_selection("navtrain", "val", 3407)
        self.assertTrue(train and val)
        self.assertFalse(set(train) & set(val))
        shards = [scene_selection("navtrain", "train", 3407, r, 8)[1] for r in range(8)]
        self.assertEqual(set(train), {log for shard in shards for log in shard})
        self.assertEqual(len(train), sum(map(len, shards)))

    def test_npz_stores_only_selected_boxes_with_camera_and_gt_indices(self):
        projections = {camera: {"objects": [], "selected_gt_indices": [], "valid": False} for camera in CAMERAS}
        projections[CAMERAS[0]] = {"objects": [
            {"bbox": [0, 0, 14, 14], "gt_index": 4, "category": "vehicle"},
            {"bbox": [28, 0, 42, 14], "gt_index": 5, "category": "pedestrian"}],
            "selected_gt_indices": [4], "valid": True}
        projections[CAMERAS[3]] = {"objects": [
            {"bbox": [14, 0, 28, 14], "gt_index": 7, "category": "barrier"}],
            "selected_gt_indices": [7], "valid": True}
        distributions = [patch_distribution([[0, 0, 14, 14]])[0],
                         np.zeros(3936), np.zeros(3936), patch_distribution([[14, 0, 28, 14]])[0]]
        data = pack_grounding_arrays("example", projections, distributions, [1, 0, 0, 1], "hash")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "target.npz"
            write_npz(path, **data)
            with np.load(path, allow_pickle=False) as loaded:
                np.testing.assert_array_equal(loaded["boxes_2d"], [[0, 0, 14, 14], [14, 0, 28, 14]])
                np.testing.assert_array_equal(loaded["box_camera_index"], [0, 3])
                np.testing.assert_array_equal(loaded["box_gt_index"], [4, 7])
                self.assertEqual(loaded["box_category"].tolist(), ["vehicle", "barrier"])
                self.assertEqual(loaded["soft_mask"].shape, (4, 3936))

    def test_empty_box_export_has_consistent_numeric_shapes(self):
        projections = {camera: {"objects": [], "selected_gt_indices": [], "valid": False} for camera in CAMERAS}
        data = pack_grounding_arrays("empty", projections, [np.zeros(3936)]*4, [False]*4, "hash")
        self.assertEqual(data["boxes_2d"].shape, (0, 4))
        self.assertEqual(data["box_camera_index"].shape, (0,))
        self.assertFalse(data["valid"].any())

    def test_stream_cache_rejects_input_version_and_alignment_mismatch(self):
        view = {"text": "car on left", "categories": ["vehicle"], "objects": [{"phrase": "car on left", "category": "vehicle"}],
                "status": "ok", "removed": [], "repairs": []}
        record = {"token": "example", "images": {"cam_f0": "original.jpg"}}
        payload = {"token": "example", "cache_version": "v1", "input_hash": fingerprint(record),
                   "scene_text": {c: {k: view[k] for k in ("text", "categories")} for c in CAMERAS},
                   "audit": {c: view for c in CAMERAS}, "rule_status": "passed"}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scene.json"
            write_json(path, payload)
            self.assertEqual(load_scene_cache(path, "v1", record)["token"], "example")
            with self.assertRaises(ValueError):
                load_scene_cache(path, "v2", record)
            with self.assertRaises(ValueError):
                load_scene_cache(path, "v1", {**record, "images": {"cam_f0": "different.jpg"}})
            payload["scene_text"][CAMERAS[0]]["text"] = "mismatched"
            write_json(path, payload)
            with self.assertRaises(ValueError):
                load_scene_cache(path, "v1", record)

    def test_failed_repair_and_unsupported_stage1_category_need_review(self):
        view = {"status": "ok", "removed": [], "repairs": [], "categories": ["vehicle"],
                "objects": [{"phrase": "car on left", "category": "vehicle"}]}
        audit = {camera: dict(view) for camera in CAMERAS}
        self.assertEqual(rule_status(audit, True), "passed")
        audit[CAMERAS[0]]["repairs"] = [{"accepted": False}]
        self.assertEqual(rule_status(audit, True), "needs_review")
        audit[CAMERAS[0]]["repairs"] = []
        audit[CAMERAS[0]]["categories"] = ["traffic_light"]
        self.assertEqual(rule_status(audit, True), "needs_review")

    def test_bulk_grounding_exports_boxes_and_resumes_complete_scene(self):
        from build_grounding import main as build
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared, captions, output = root/"prepared", root/"captions", root/"grounding"
            record = {"token": "synthetic", "log_name": "example", "images": {}}
            write_json(prepared/"manifest.json", {"camera_order": CAMERAS, "records": [record]})
            k = [[100, 0, 574], [0, 100, 336], [0, 0, 1]]
            write_json(prepared/"ground_truth.json", {"synthetic": {"boxes": [[0, 0, 10, 2, 2, 2, 0]],
                "names": ["vehicle"], "cam_K": [k]*4, "lidar_to_camera": [np.eye(4).tolist()]*4}})
            write_json(captions/"metadata.json", {"manifest_sha256": sha256(prepared/"manifest.json")})
            write_json(captions/"scene_text.json", {"synthetic": {c: {"text": "car at center", "categories": ["vehicle"]} for c in CAMERAS}})
            argv = ["build", "--prepared", str(prepared), "--captions", str(captions),
                    "--output", str(output), "--stream", "--no-render", "--resume"]
            with patch.object(sys, "argv", argv):
                build()
            self.assertIn("synthetic", read_json(output/"index.json"))
            with np.load(output/"masks/synthetic.npz", allow_pickle=False) as labels:
                self.assertEqual(labels["boxes_2d"].shape, (4, 4))
                np.testing.assert_array_equal(labels["box_camera_index"], [0, 1, 2, 3])
            with patch.object(sys, "argv", argv):
                build()
            self.assertEqual(read_json(output/"summary.json")["resumed_scenes"], 1)


if __name__ == "__main__":
    unittest.main()
