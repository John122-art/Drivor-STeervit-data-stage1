"""Regression checks for JSON empty annotations and cache compatibility."""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('cache_fix', Path(__file__).with_name('migrate_validation_shape_cache.py'))
cache_fix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache_fix)
tree = ast.parse((ROOT / 'scripts/data_generation/validate_pilot.py').read_text())
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'assert_annotation_boxes_match')
namespace = {'np': np}
exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual validator helper>', 'exec'), namespace)
compare = namespace['assert_annotation_boxes_match']


class EmptyGTTests(unittest.TestCase):
    def test_empty_json_round_trip(self):
        raw = np.empty((0, 7))
        saved = json.loads(json.dumps(raw.tolist()))
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(raw, saved)
        compare(raw, saved)

    def test_real_values_remain_exact(self):
        raw = np.asarray([[1.5, 2, 3, 4, 5, 6, 0.75]], dtype=np.float32)
        compare(raw, json.loads(json.dumps(raw.tolist())))
        changed = raw.tolist()
        changed[0][0] += 0.01
        with self.assertRaises(AssertionError):
            compare(raw, changed)

    def test_wrong_shapes_and_missing_boxes_still_fail(self):
        for raw, saved in [(np.empty((0, 8)), []), (np.zeros((1, 7)), []),
                           (np.zeros((1, 7)), [0]*7), (np.zeros((1, 7)), [[0]*8])]:
            with self.assertRaises(AssertionError):
                compare(raw, saved)

    def metadata(self):
        old = {'model': 'unchanged', 'stage1': True,
               'pipeline_source_hashes': {'validate_pilot.py': cache_fix.OLD_VALIDATOR, 'generate_captions.py': 'same'}}
        old['cache_version'] = cache_fix.fingerprint(old)
        return old

    def test_only_reviewed_validator_transition_is_accepted(self):
        old = self.metadata()
        sources = dict(old['pipeline_source_hashes'], **{'validate_pilot.py': cache_fix.FIXED_VALIDATOR})
        new = cache_fix.plan_metadata(old, sources)
        self.assertNotEqual(old['cache_version'], new['cache_version'])
        self.assertEqual(old['model'], new['model'])
        self.assertIsNone(cache_fix.plan_metadata(new, sources))
        for name in ['validate_pilot.py', 'generate_captions.py', 'new_file.py']:
            changed = dict(sources, **{name: 'unreviewed_change'})
            with self.assertRaises(ValueError):
                cache_fix.plan_metadata(old, changed)

    def test_caption_payload_preserved_and_retry_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'token.json'
            old, new = 'a'*64, 'b'*64
            value = {'token': 'token', 'cache_version': old,
                     'scene_text': {'cam_f0': {'text': '', 'categories': []}},
                     'audit': {'raw_text': '\n```json\n{"objects":[]}\n```'}}
            before = (json.dumps(value, indent=2) + '\n').encode()
            path.write_bytes(before)
            cache_fix.rewrite_cache(path, old, new)
            after = path.read_bytes()
            self.assertEqual(after, before.replace(old.encode(), new.encode(), 1))
            self.assertEqual(cache_fix.rewrite_cache(path, old, new)[1], 'already_updated')
            self.assertEqual(path.read_bytes(), after)


if __name__ == '__main__':
    unittest.main()
