"""Tests for keeping planning/evaluation scenes and separating stage-2 caches."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CAMERAS, ROOT, fingerprint, read_json, write_json, write_npz
from merge import generation_contract, merge_partition
from worker import PARTITIONS
from text_policy import POLICY_VERSION, clean_response, scene_status
from generate import process_scene


def fixture(root, partition, token, passed=True, has_grounding=False, failed_cameras=(), rank=0, world_size=1):
    shard = Path(root) / partition / f"shards/rank_{rank:04d}"
    split, subset = PARTITIONS[partition]
    config = {"rank": rank, "world_size": world_size, "partition": partition, "split": split,
              "subset": subset, "stage1": False, "navigation_conditioned": True}
    write_json(shard/"run_config.json", config)
    write_json(shard/"COMPLETE.json", {"rank": rank, "world_size": world_size,
        "configuration_hash": fingerprint(config), "validation": "passed"})
    write_json(shard/"validation.json", {"status": "passed"})
    record = {"token": token, "log_name": "log_" + token, "images": {c: c+".jpg" for c in CAMERAS},
              "driving_command": [0, 1, 0, 0], "navigation_text": "go straight"}
    write_json(shard/"prepared/manifest.json", {"records": [record], "skipped": []})
    keys = ("model", "model_weight_bytes", "model_config_sha256", "model_weight_mtime_ns",
        "source_hashes", "pipeline_source_hashes", "prompt", "prompt_sha256", "torch", "transformers",
        "dtype", "attention", "current_image_pixels", "roberta_tokenizer", "max_text_tokens",
        "decoding", "phrase_repair")
    meta = dict.fromkeys(keys, "fixed")
    meta.update(stage1=False, navigation_conditioned=True, cache_version="version",
        phrase_repair={"enabled": False}, text_policy=POLICY_VERSION, model_calls_per_view=1,
        backend_source_hashes={})
    write_json(shard/"captions/metadata.json", meta)
    view = {"text": "green traffic light at the top", "categories": ["traffic_light"],
            "objects": [{"phrase": "green traffic light at the top", "category": "traffic_light"}],
            "status": "ok", "removed": [], "repairs": [], "local_edits": []}
    failed_cameras = set(CAMERAS if not passed else failed_cameras)
    audit = {c: deepcopy(view) for c in CAMERAS}
    for c in failed_cameras:
        audit[c].update(text="", categories=[], objects=[], status="generation_failed")
    text = {c: {k: audit[c][k] for k in ("text", "categories")} for c in CAMERAS}
    write_json(shard/"captions/scenes"/(token+".json"), {"token": token, "cache_version": "version",
        "input_hash": fingerprint(record), "audit": audit, "scene_text": text,
        "rule_status": scene_status(audit)})
    write_json(shard/"captions/scene_text.json", {token: text})
    write_json(shard/"captions/review_queue.json", {"tokens": [token] if failed_cameras else []})
    write_json(shard/"captions/summary.json", {"scenes": 1})
    write_npz(shard/"grounding/masks"/(token+".npz"), scene_token=np.asarray(token),
        camera_order=np.asarray(CAMERAS), valid=np.asarray([has_grounding and c not in failed_cameras for c in CAMERAS]))
    write_json(shard/"grounding/scenes"/(token+".json"), {})
    return shard


class Stage2Contracts(unittest.TestCase):
    def test_eight_shards_keep_every_scene_and_camera_local_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            for rank in range(8):
                fixture(root, "train", f"scene_{rank}", has_grounding=True,
                    failed_cameras=("cam_b0",) if rank == 5 else (), rank=rank, world_size=8)
            report = merge_partition(root, "train", 8)
            self.assertEqual(report["dataset_scenes"], 8)
            self.assertEqual(report["empty_text_views"], 1)
            rows = [json.loads(line) for line in (Path(root)/"train/dataset_index.jsonl").read_text().splitlines()]
            self.assertEqual(len({r["scene_token"] for r in rows}), 8)
            self.assertEqual(rows[5]["grounding_valid"], [True, False, True, True])

    def test_eight_shards_reject_duplicate_scene(self):
        with tempfile.TemporaryDirectory() as root:
            for rank in range(8):
                fixture(root, "train", "duplicate" if rank >= 6 else f"scene_{rank}", rank=rank, world_size=8)
            with self.assertRaisesRegex(AssertionError, "Rank selections overlap"):
                merge_partition(root, "train", 8)
            self.assertFalse((Path(root)/"train/COMPLETE.json").exists())

    def test_no_grounding_never_removes_planning_or_eval_scenes(self):
        for partition in PARTITIONS:
            with self.subTest(partition=partition), tempfile.TemporaryDirectory() as root:
                fixture(root, partition, "scene")
                report = merge_partition(root, partition, 1)
                self.assertEqual(report["dataset_scenes"], 1)
                self.assertEqual(report["scenes_without_grounding"], 1)
                row = json.loads((Path(root)/partition/"dataset_index.jsonl").read_text())
                self.assertFalse(row["text_fallback"])
                self.assertEqual(row["scene_text"]["cam_f0"]["categories"], ["traffic_light"])
                self.assertEqual(row["grounding_role"], "audit_only" if partition=="test" else "auxiliary_supervision")

    def test_failed_text_keeps_scene_with_explicit_empty_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            fixture(root, "test", "bad", passed=False)
            report = merge_partition(root, "test", 1)
            self.assertEqual(report["dataset_scenes"], 1)
            self.assertEqual(report["empty_text_fallback_scenes"], 1)
            row = json.loads((Path(root)/"test/dataset_index.jsonl").read_text())
            self.assertTrue(row["text_fallback"])
            self.assertTrue(Path(row["grounding_npz"]).is_file())
            self.assertFalse(any(row["grounding_valid"]))
            self.assertTrue(all(v["text"]=="" for v in row["scene_text"].values()))
            self.assertEqual(report["empty_text_fallback_views"], 4)

    def test_one_bad_camera_preserves_the_other_three_and_their_grounding(self):
        with tempfile.TemporaryDirectory() as root:
            fixture(root,"train","partial",has_grounding=True,failed_cameras=["cam_b0"])
            report=merge_partition(root,"train",1)
            row=json.loads((Path(root)/"train/dataset_index.jsonl").read_text())
            self.assertEqual(row["text_fallback_cameras"],["cam_b0"])
            self.assertEqual(row["grounding_valid"],[True,False,True,True])
            self.assertEqual(row["scene_text"]["cam_b0"]["text"],"")
            self.assertTrue(all(row["scene_text"][c]["text"] for c in CAMERAS if c!="cam_b0"))
            self.assertEqual(report["empty_text_fallback_views"],1)
            self.assertEqual(report["all_texts_empty_scenes"],0)

    def test_cross_split_generator_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            fixture(root, "train", "a")
            merge_partition(root, "train", 1)
            shard = fixture(root, "val", "b")
            meta = read_json(shard/"captions/metadata.json")
            meta["prompt"] = "changed"
            write_json(shard/"captions/metadata.json", meta)
            with self.assertRaises(AssertionError):
                merge_partition(root, "val", 1)
            self.assertFalse((Path(root)/"val/COMPLETE.json").exists())

    def test_cross_split_duplicate_scene_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            fixture(root, "train", "same")
            merge_partition(root, "train", 1)
            fixture(root, "test", "same")
            with self.assertRaises(AssertionError):
                merge_partition(root, "test", 1)

    def test_missing_caption_coverage_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            shard=fixture(root,"val","a")
            write_json(shard/"captions/scene_text.json",{})
            with self.assertRaises(AssertionError):
                merge_partition(root,"val",1)

    def test_stage1_generator_not_accepted(self):
        with tempfile.TemporaryDirectory() as root:
            shard=fixture(root,"train","a")
            meta=read_json(shard/"captions/metadata.json")
            meta["stage1"]=True
            write_json(shard/"captions/metadata.json",meta)
            with self.assertRaises(AssertionError):
                merge_partition(root,"train",1)

    def test_full_dry_run_covers_all_three_partitions(self):
        import os
        env=dict(os.environ, NPROC_PER_NODE="8",SCENES_PER_RANK="0",DATA_SPLITS="train,val,test")
        r=subprocess.run(["bash","scripts/data_generation/run_stage2_data_8gpu_dlc.sh","--dry-run"],
            cwd=ROOT,env=env,check=True,capture_output=True,text=True)
        self.assertEqual(len(r.stdout.splitlines()),6)
        for part in PARTITIONS:
            self.assertIn("--partition "+part,r.stdout)
        self.assertIn("--scenes 0",r.stdout)
        self.assertIn("scene_navigation_v2.txt",r.stdout)
        self.assertNotIn("stage1_worker",r.stdout)


class WordTokenizer:
    def encode(self,text,add_special_tokens=True):
        return text.split()+(["<s>","</s>"] if add_special_tokens else [])


class EmptyAnnotations(unittest.TestCase):
    def test_empty_gt_json_roundtrip(self):
        from validate import assert_annotation_boxes_match
        raw = np.empty((0, 7))
        assert_annotation_boxes_match(raw, json.loads(json.dumps(raw.tolist())))

    def test_nonempty_gt_still_requires_exact_match(self):
        from validate import assert_annotation_boxes_match
        raw = np.arange(14).reshape(2, 7)
        assert_annotation_boxes_match(raw, raw.tolist())
        for saved in [[], raw.ravel().tolist(), (raw + 1).tolist()]:
            with self.assertRaises(AssertionError):
                assert_annotation_boxes_match(raw, saved)


class CameraLocalCleaning(unittest.TestCase):
    def clean(self,objects):
        return clean_response({"answer":json.dumps({"objects":objects}),"finish_reason":"eos"},WordTokenizer())

    def test_local_action_removal_preserves_white_car_and_location(self):
        result=self.clean([{"phrase":"white sedan driving in the right lane","category":"vehicle"}])
        self.assertEqual(result["text"],"white sedan in the right lane")
        self.assertEqual(result["categories"],["vehicle"])
        self.assertEqual(len(result["local_edits"]),1)

    def test_unsafe_phrase_does_not_delete_other_objects(self):
        result=self.clean([{"phrase":"a car will turn left","category":"vehicle"},
            {"phrase":"a pedestrian on the right sidewalk","category":"pedestrian"}])
        self.assertEqual(result["text"],"a pedestrian on the right sidewalk")
        self.assertEqual(result["categories"],["pedestrian"])
        self.assertEqual(result["removed"][0]["reason"],"future_or_advice")

    def test_invalid_category_keeps_language_without_inventing_supported_class(self):
        result=self.clean([{"phrase":"black car on the left","category":"vehicle"},
            {"phrase":"a blue sign","category":"unsupported"}])
        self.assertEqual(result["text"],"black car on the left, a blue sign")
        self.assertEqual(result["categories"],["vehicle","other"])
        self.assertEqual(len(result["local_edits"]),1)

    def test_string_objects_keep_original_phrases_with_no_supported_grounding_class(self):
        result=self.clean(["white pickup truck in the left lane","white lane markings at the bottom"])
        self.assertEqual(result["text"],"white pickup truck in the left lane, white lane markings at the bottom")
        self.assertEqual(result["categories"],["other","other"])
        self.assertEqual(len(result["local_edits"]),2)

    def test_invalid_object_does_not_erase_peer_objects(self):
        result=self.clean([{"phrase":"black car on the left","category":"vehicle"},42])
        self.assertEqual(result["text"],"black car on the left")
        self.assertEqual(result["removed"][0]["reason"],"invalid_object_schema")

    def test_motion_relations_without_static_location_are_not_blindly_rewritten(self):
        result=self.clean([{"phrase":"car following a truck","category":"vehicle"}])
        self.assertEqual(result["text"],"")
        self.assertEqual(result["local_edits"],[])
        self.assertEqual(result["status"],"filtered_empty")

    def test_empty_failed_truncated_and_nonfinite_responses(self):
        self.assertEqual(self.clean([])["status"],"empty_scene")
        for call in [{"answer":"not JSON","finish_reason":"eos"},
                     {"answer":'{"objects":[]}',"finish_reason":"length"},
                     {"answer":'{"objects":[NaN]}',"finish_reason":"eos"}]:
            self.assertEqual(clean_response(call,WordTokenizer())["status"],"generation_failed")

    def test_one_model_call_per_camera_no_redescription(self):
        record={"driving_command":[0,1,0,0],"images":{c:c+".jpg" for c in CAMERAS}}
        calls=[]
        def generate(image,question):
            calls.append((image,question))
            answer="not JSON" if image=="cam_b0.jpg" else '{"objects":[{"phrase":"white sedan driving in the right lane","category":"vehicle"}]}'
            return {"answer":answer,"finish_reason":"eos"}
        text,audit=process_scene(record,"{camera_name}: {navigation_command}",["GO STRAIGHT","TURN LEFT","TURN RIGHT"],WordTokenizer(),generate)
        self.assertEqual(len(calls),4)
        self.assertEqual(text["cam_b0"]["text"],"")
        self.assertTrue(all(text[c]["text"]=="white sedan in the right lane" for c in CAMERAS if c!="cam_b0"))
        self.assertTrue(all(not v["repairs"] and len(v["attempts"])==1 for v in audit.values()))

    def test_entire_scene_failure_returns_four_explicit_empty_views(self):
        record={"driving_command":[0,1,0,0],"images":{c:c+".jpg" for c in CAMERAS}}
        text,audit=process_scene(record,"{camera_name}: {navigation_command}",["GO STRAIGHT"],WordTokenizer(),
            lambda image,question:{"answer":"not JSON","finish_reason":"eos"})
        self.assertEqual(set(text),set(CAMERAS))
        self.assertTrue(all(not v["text"] for v in text.values()))
        self.assertEqual(scene_status(audit),"needs_review")


if __name__ == "__main__":
    unittest.main()
