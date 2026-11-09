#!/usr/bin/env python3
"""Read-only, CPU audit of the existing stage-1/stage-2 generation caches.

Never imports or modifies the generation pipeline; captures a fixed filename
inventory and hashes each file actually read. Run with system Python + PIL,
numpy and PyYAML. Reports are separate from resumable generation directories.
"""
import base64
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import pickle
import random
import re
import shutil
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('/mnt/zhaozc_workspace/project/LQH/DrivoR-main-copy/dataset')
GUIDE = Path('/mnt/zhaozc_workspace/project/LQH/Drivor-orial/DrivoR-main/docs/steer_drivor_qwen_gt_grounding_analysis.md')
CAMERAS = ['cam_f0', 'cam_b0', 'cam_l0', 'cam_r0']
SEED = 20261109
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'reports/data_audit_20261109'
GROUPS = {
    'stage1': ROOT / 'outputs/stage1_navtrain_8gpu_full',
    'stage2': ROOT / 'outputs/stage2_navigation_no_rewrite_v2_8gpu_full_rerun_20261109/train',
}


def now():
    return datetime.now(timezone.utc).isoformat()


def jread(path):
    return json.loads(Path(path).read_text())


def dump(path, obj):
    path = OUT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=serial) + '\n')


def serial(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def copy(source, relative):
    target = OUT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target


def raw_kind(call):
    answer = call.get('answer', '').strip()
    fence = re.fullmatch(r'```(?:json)?\s*(\{[\s\S]*\})\s*```', answer, re.I)
    try:
        v = json.loads(fence.group(1) if fence else answer,
                       parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
    except (ValueError, TypeError):
        return 'truncated' if call.get('finish_reason') != 'eos' else 'json_syntax_error'
    if call.get('finish_reason') != 'eos':
        return 'truncated'
    if not isinstance(v, dict) or set(v) != {'objects'} or not isinstance(v['objects'], list):
        return 'root_schema_error'
    if not v['objects']:
        return 'vlm_empty_list'
    if call.get('error'):
        return 'recorded_validation_error'
    return 'nonempty_json'


def read_cache(path):
    try:
        data = path.read_bytes()
        return path, data, json.loads(data), None
    except Exception as exc:
        return path, b'', None, repr(exc)


def counter_view():
    return dict(total=0, empty=0, status=Counter(), empty_causes=Counter(),
                phrase_count=Counter(), token_count=Counter(), categories=Counter(),
                categories_per_view=Counter(), removed=Counter(), raw_calls=Counter(),
                grids=Counter(), repairs=0, accepted_repairs=0, local_edits=0)


def audit_group(name, root):
    start = now()
    paths = sorted(root.glob('shards/rank_*/captions/scenes/*.json'))
    rng = random.Random(SEED + (0 if name == 'stage1' else 1))
    examples = rng.sample(paths, 3)
    visuals = rng.sample(paths, 5)
    wanted = {p.stem for p in examples + visuals}
    records, tokens, logs, token_logs = {}, set(), set(), {}
    metadata, ranks, manifest_info, source_checks = {}, [], [], []
    for shard in sorted((root / 'shards').glob('rank_*')):
        manifest = jread(shard / 'prepared/manifest.json')
        meta = jread(shard / 'captions/metadata.json')
        metadata[shard.name] = meta
        copy(shard / 'captions/metadata.json', f'metadata/{name}/{shard.name}/metadata.json')
        copy(shard / 'run_config.json', f'metadata/{name}/{shard.name}/run_config.json')
        for record in manifest['records']:
            token = record['token']
            if token in tokens:
                raise ValueError(f'duplicate prepared token: {name} {token}')
            tokens.add(token)
            logs.add(record['log_name'])
            token_logs[token] = record['log_name']
            if token in wanted:
                records[token] = record
        mi = {k: v for k, v in manifest.items() if k != 'records'}
        mi['record_count'] = len(manifest['records'])
        mi['manifest_sha256_actual'] = sha((shard / 'prepared/manifest.json').read_bytes())
        mi['metadata_manifest_hash_matches'] = mi['manifest_sha256_actual'] == meta['manifest_sha256']
        manifest_info.append(mi)
        for rel, digest in meta.get('pipeline_source_hashes', {}).items():
            source = ROOT / 'scripts/data_generation' / rel
            source_checks.append({'rank': shard.name, 'source': rel,
                'recorded': digest, 'matches_current': source.is_file() and sha(source.read_bytes()) == digest})
        ranks.append({'rank': shard.name, 'prepared': len(manifest['records']),
                      'complete': (shard / 'COMPLETE.json').is_file(),
                      'grounding_npz': len(list((shard / 'grounding/masks').glob('*.npz'))),
                      'grounding_json': len(list((shard / 'grounding/scenes').glob('*.json')))})
    stats = {c: counter_view() for c in CAMERAS}
    passed_stats = {c: counter_view() for c in CAMERAS}
    rule_counts, rank_counts = Counter(), Counter()
    reservoirs, bucket_counts = defaultdict(list), Counter()
    seen, cached_logs = set(), set()
    consistency = Counter()
    anomalies = []
    capture = {}
    with (OUT / f'{name}_snapshot.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['scene_token', 'rank', 'log_name', 'source_file', 'sha256', 'bytes', 'rule_status'])
        with ThreadPoolExecutor(max_workers=8) as pool:
            for offset in range(0, len(paths), 256):
                for path, data, value, error in pool.map(read_cache, paths[offset:offset + 256]):
                    if error:
                        anomalies.append({'file': str(path), 'error': error})
                        continue
                    token = value['token']
                    rank = path.parents[2].name
                    log = token_logs.get(token, '')
                    rule = value['rule_status']
                    writer.writerow([token, rank, log, str(path), sha(data), len(data), rule])
                    consistency['files_read'] += 1
                    consistency['duplicate_tokens'] += token in seen
                    consistency['unknown_tokens'] += token not in tokens
                    consistency['filename_mismatch'] += token != path.stem
                    consistency['cache_version_mismatch'] += value['cache_version'] != metadata[rank]['cache_version']
                    seen.add(token)
                    cached_logs.add(log)
                    rank_counts[rank] += 1
                    rule_counts[rule] += 1
                    if token in wanted:
                        capture[token] = value
                        (OUT / 'samples' / name).mkdir(parents=True, exist_ok=True)
                        (OUT / 'samples' / name / path.name).write_bytes(data)
                        dump(f'samples/{name}/{token}.manifest_record.json', records[token])
                    for camera in CAMERAS:
                        view = value['audit'][camera]
                        caption = value['scene_text'][camera]
                        objects = view.get('objects', [])
                        consistency['text_audit_mismatch'] += caption['text'] != view['text']
                        consistency['category_audit_mismatch'] += caption['categories'] != view['categories']
                        consistency['phrase_text_mismatch'] += caption['text'] != ', '.join(o['phrase'] for o in objects)
                        consistency['token_over_128'] += view['token_count'] > 128
                        empty = not caption['text'].strip()
                        cause = None
                        attempts = view.get('attempts', [])
                        if empty:
                            if view['status'] == 'empty_scene':
                                cause = 'vlm_empty_list'
                            elif view['status'] == 'generation_failed':
                                cause = raw_kind(attempts[-1]) if attempts else 'no_attempt_record'
                            else:
                                cause = 'postprocessing_removed_all'
                        for target in [stats[camera]] + ([passed_stats[camera]] if rule == 'passed' else []):
                            target['total'] += 1
                            target['empty'] += empty
                            target['status'][view['status']] += 1
                            if cause:
                                target['empty_causes'][cause] += 1
                            target['phrase_count'][len(objects)] += 1
                            target['token_count'][view['token_count']] += 1
                            target['categories'].update(caption['categories'])
                            target['categories_per_view'].update(set(caption['categories']))
                            target['removed'].update(x['reason'] for x in view.get('removed', []))
                            target['repairs'] += len(view.get('repairs', []))
                            target['accepted_repairs'] += sum(x['accepted'] for x in view.get('repairs', []))
                            target['local_edits'] += len(view.get('local_edits', []))
                            for call in attempts:
                                target['raw_calls'][raw_kind(call)] += 1
                                target['grids'][str(call.get('image_grid_thw'))] += 1
                        for ai, call in enumerate(attempts):
                            kind = raw_kind(call)
                            bucket = (camera, kind)
                            bucket_counts[bucket] += 1
                            item = {'group': name, 'scene_token': token, 'camera': camera,
                                    'attempt_index': ai, 'kind': kind, 'source_file': str(path),
                                    'source_sha256': sha(data), 'call': call}
                            reservoir = reservoirs[bucket]
                            if len(reservoir) < 8:
                                reservoir.append(item)
                            else:
                                index = rng.randrange(bucket_counts[bucket])
                                if index < 8:
                                    reservoir[index] = item
                if offset % 8192 == 0:
                    print(f'{name}: scanned {min(offset+256, len(paths))}/{len(paths)}', flush=True)
    for rank in ranks:
        rank['cached'] = rank_counts[rank['rank']]
    for c in stats:
        for target in (stats[c], passed_stats[c]):
            target['empty_fraction'] = target['empty'] / target['total'] if target['total'] else None
    result = {'group': name, 'path': str(root), 'snapshot_start': start, 'snapshot_end': now(),
              'prepared_scenes': len(tokens), 'selected_log_count': len(logs),
              'filename_snapshot_count': len(paths), 'read_scene_count': len(seen),
              'cached_log_count': len(cached_logs), 'rules': rule_counts, 'ranks': ranks,
              'grounding_npz_count': sum(x['grounding_npz'] for x in ranks),
              'merged_scene_text_exists': (root / 'scene_text.json').is_file(),
              'training_index_exists': (root / 'training_index.jsonl').is_file(),
              'dataset_index_exists': (root / 'dataset_index.jsonl').is_file(),
              'per_camera_all_cached': stats, 'per_camera_rule_passed_scenes': passed_stats,
              'consistency': consistency, 'read_errors': anomalies,
              'sample_tokens': [p.stem for p in examples], 'visual_tokens': [p.stem for p in visuals],
              'raw_bucket_counts': {f'{c}/{k}': n for (c,k),n in bucket_counts.items()},
              'source_hash_mismatches': [x for x in source_checks if not x['matches_current']]}
    dump(f'{name}_statistics.json', result)
    dump(f'metadata/{name}/manifest_headers.json', manifest_info)
    dump(f'metadata/{name}/pipeline_source_hash_checks.json', source_checks)
    for suffix, names in [('selected_logs', logs), ('cached_logs', cached_logs)]:
        (OUT / f'{name}_{suffix}.txt').write_text('\n'.join(sorted(names)) + '\n')
    return result, records, capture, metadata, tokens, logs, reservoirs


def montage(name, record, cache):
    width = 720
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 21)
    title_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 25)
    dummy = ImageDraw.Draw(Image.new('RGB', (1,1)))
    lines = []
    for camera in CAMERAS:
        words = (cache['scene_text'][camera]['text'] or '[EMPTY TEXT]').split()
        wrapped, line = [], ''
        for word in words:
            candidate = (line + ' ' + word).strip()
            if line and dummy.textlength(candidate, font=font) > width - 24:
                wrapped.append(line)
                line = word
            else:
                line = candidate
        wrapped.append(line)
        lines.append(wrapped)
    header = 96 + max(map(len, lines)) * 27
    sheet = Image.new('RGB', (width*4, header + 450), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), f'{name} | scene {record["token"]} | f0, b0, l0, r0', font=title_font, fill='black')
    dims = {}
    for i, camera in enumerate(CAMERAS):
        x = i*width
        draw.text((x+12, 48), camera, font=title_font, fill='#174378')
        for j,line in enumerate(lines[i]):
            draw.text((x+12, 85+j*27), line, font=font, fill='black')
        with Image.open(record['images'][camera]) as image:
            dims[camera] = list(image.size)
            image = image.convert('RGB')
            image.thumbnail((width, 450))
            sheet.paste(image, (x+(width-image.width)//2, header))
        draw.line((x,45,x,sheet.height), fill='#bbbbbb')
    relative = f'images/{name}_{record["token"]}.png'
    sheet.save(OUT / relative)
    return {'group': name, 'token': record['token'], 'file': relative, 'original_dimensions': dims,
            'images': record['images'], 'scene_text': cache['scene_text']}


def calibration(record):
    log_path = DATA / 'navsim_logs/trainval' / (record['log_name'] + '.pkl')
    with log_path.open('rb') as f:
        frames = pickle.load(f)
    matches = [(i,v) for i,v in enumerate(frames) if v['token'] == record['token']]
    assert len(matches) == 1
    index, frame = matches[0]
    annotations = frame['anns']
    result = {'scene_token': record['token'], 'log_name': record['log_name'],
              'source_log_pickle': str(log_path), 'index_in_log_pickle': index,
              'scene_current_frame_index': 3, 'timestamp': frame['timestamp'],
              'annotations.boxes_first_10': annotations['gt_boxes'][:10],
              'annotations.names': annotations['gt_names'],
              'annotations.names_first_10': annotations['gt_names'][:10],
              'box_count': len(annotations['gt_boxes']),
              'box_columns': ['x','y','z','length','width','height','heading'],
              'lidar2ego': frame.get('lidar2ego'), 'cameras': {}}
    for c in CAMERAS:
        raw = frame['cams'][c.upper()]
        with Image.open(record['images'][c]) as image:
            size = list(image.size)
        result['cameras'][c] = {'image_path': record['images'][c], 'original_size_width_height': size,
            'intrinsics': raw['cam_intrinsic'], 'sensor2lidar_rotation': raw['sensor2lidar_rotation'],
            'sensor2lidar_translation': raw['sensor2lidar_translation'], 'distortion': raw.get('distortion')}
    dump('calibration/raw_values.json', result)
    np.savez(OUT / 'calibration/annotations_original_arrays.npz', boxes=annotations['gt_boxes'],
             names=np.asarray(annotations['gt_names']))
    copy(record['images']['cam_f0'], 'calibration/front_original.jpg')
    for shard in sorted((GROUPS['stage1'] / 'shards').glob('rank_*')):
        gt = jread(shard / 'prepared/ground_truth.json')
        if record['token'] in gt:
            dump('calibration/prepared_ground_truth_full.json', gt[record['token']])
            assert np.array_equal(np.asarray(gt[record['token']]['boxes']), annotations['gt_boxes'])
            break
    return result


class Report:
    def __init__(self):
        self.md, self.htm = [], []

    def heading(self, text, level=2):
        self.md.append('#'*level + ' ' + text)
        self.htm.append(f'<h{level}>{html.escape(text)}</h{level}>')

    def para(self, text):
        self.md.append(text)
        self.htm.append('<p>' + html.escape(text) + '</p>')

    def code(self, text, language='text', title=None):
        # Longer fence preserves literal triple-backticks in unmodified samples.
        fence = '`' * (max([len(x) for x in re.findall(r'`+', text)] + [3]) + 1)
        if title:
            self.md.append('**' + title + '**')
        self.md.append(f'{fence}{language}\n{text}\n{fence}')
        block = '<pre><code>' + html.escape(text) + '</code></pre>'
        self.htm.append(('<details><summary>' + html.escape(title) + '</summary>' + block + '</details>') if title else block)

    def table(self, header, rows):
        self.md.append('| ' + ' | '.join(header) + ' |\n|' + '|'.join(['---']*len(header)) + '|\n' +
                       '\n'.join('| ' + ' | '.join(str(v).replace('|','/') for v in row) + ' |' for row in rows))
        self.htm.append('<table><thead><tr>' + ''.join('<th>'+html.escape(h)+'</th>' for h in header) +
                        '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>')

    def image(self, relative, caption):
        self.md.append(f'![{caption}]({relative})')
        data = (OUT / relative).read_bytes()
        mime = 'image/png' if relative.endswith('.png') else 'image/jpeg'
        self.htm.append('<figure><img src="data:'+mime+';base64,'+base64.b64encode(data).decode()+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>')

    def save(self):
        (OUT / 'report.md').write_text('\n\n'.join(self.md)+'\n')
        css = 'body{font:16px/1.65 sans-serif;margin:30px auto;max-width:1500px;padding:0 20px;color:#17212d}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f4f7;padding:16px;font:13px/1.5 monospace}table{border-collapse:collapse;display:block;overflow:auto}td,th{border:1px solid #cdd6df;padding:7px 12px;text-align:left}img{max-width:100%;height:auto}details{border:1px solid #ccd4de;padding:12px;margin:10px 0}summary{cursor:pointer;font-weight:bold}h2{border-bottom:2px solid #9eb6d2;padding-top:24px}'
        (OUT / 'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Steer-DrivoR 数据审计</title><style>'+css+'</style><body>'+''.join(self.htm)+'</body></html>')


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'images').mkdir()
    started = now()
    results = {name: audit_group(name, root) for name,root in GROUPS.items()}
    config = ROOT / 'navsim/planning/script/config'
    navtest = yaml.safe_load((config / 'common/train_test_split/scene_filter/navtest.yaml').read_text())
    navtrain = yaml.safe_load((config / 'common/train_test_split/scene_filter/navtrain.yaml').read_text())
    native = yaml.safe_load((config / 'training/default_train_val_test_log_split.yaml').read_text())
    testlogs, testtokens = set(navtest['log_names']), set(navtest['tokens'])
    vallogs = set(navtrain['log_names']) & set(native['val_logs'])
    overlap = {'stage1_stage2_prepared_tokens': len(results['stage1'][4] & results['stage2'][4]),
               'stage1_stage2_selected_logs': len(results['stage1'][5] & results['stage2'][5]),
               'navtest_reference_logs': len(testlogs), 'navtest_reference_tokens': len(testtokens),
               'native_val_logs_intersect_navtrain': len(vallogs), 'groups': {}}
    for name, (_,_,_,_,tokens,logs,_) in results.items():
        overlap['groups'][name] = {'logs_overlap_navtest': sorted(logs & testlogs),
                                  'tokens_overlap_navtest': sorted(tokens & testtokens),
                                  'logs_overlap_native_val': sorted(logs & vallogs)}
    dump('split_overlap.json', overlap)
    for n,logs in [('reference_navtest_logs',testlogs),('reference_navtrain_val_logs',vallogs)]:
        (OUT / f'{n}.txt').write_text('\n'.join(sorted(logs))+'\n')
    raw_choices = []
    for ci,camera in enumerate(CAMERAS):
        candidates = [v for result in results.values() for (c,k),items in result[6].items() if c == camera for v in items]
        rng = random.Random(SEED+10+ci)
        chosen = []
        # A seeded stratified sample: ensure empty/failure examples when observed.
        for kinds in [{'json_syntax_error','root_schema_error','recorded_validation_error','truncated'}, {'vlm_empty_list'}]:
            pool = [v for v in candidates if v['kind'] in kinds]
            if pool:
                chosen.append(rng.choice(pool))
        for group in GROUPS:
            pool = [v for v in candidates if v['group'] == group and v not in chosen and v['kind'] == 'nonempty_json']
            if pool and len(chosen) < 5:
                chosen.append(rng.choice(pool))
        chosen += rng.sample([v for v in candidates if v not in chosen], 5-len(chosen))
        raw_choices += chosen
    for i,item in enumerate(raw_choices,1):
        path = OUT / 'raw_outputs' / f'{i:02d}_{item["group"]}_{item["scene_token"]}_{item["camera"]}.txt'
        path.parent.mkdir(exist_ok=True)
        path.write_text(item['call']['raw_text'])
        item['raw_output_file'] = str(path.relative_to(OUT))
    dump('raw_outputs/index.json', raw_choices)
    visuals = []
    for name,(stats,records,caches,*_) in results.items():
        for token in stats['visual_tokens']:
            visuals.append(montage(name, records[token], caches[token]))
    dump('images/index.json', visuals)
    first = results['stage1'][0]['sample_tokens'][0]
    cal = calibration(results['stage1'][1][first])
    code_index = []
    sources = sorted((ROOT / 'scripts/data_generation').rglob('*.py')) + sorted((ROOT / 'scripts/data_generation').glob('*.sh')) + sorted((ROOT / 'scripts/data_generation/prompts').glob('*.txt'))
    sources += [ROOT / 'navsim/planning/training/dataset.py', ROOT / 'navsim/common/dataclasses.py',
                ROOT / 'navsim/common/dataloader.py', ROOT / 'navsim/agents/drivoR/drivor_features.py',
                config / 'common/train_test_split/scene_filter/navtrain.yaml', config / 'common/train_test_split/scene_filter/navtest.yaml',
                config / 'training/default_train_val_test_log_split.yaml']
    for path in sources:
        rel = str(path.relative_to(ROOT))
        copy(path, 'code/' + rel)
        code_index.append({'source': str(path), 'copy': 'code/'+rel, 'sha256': sha(path.read_bytes())})
    qwen = Path('/mnt/zhaozc_workspace/project/LQH/Qwen-drive/Qwen-Drive-1.0-main/src/qwen_drive')
    for path in [qwen/'scene.py',qwen/'images.py',qwen/'modeling_qwen_drive.py',Path('/mnt/zhaozc_workspace/project/LQH/DrivoR-main-copy/navsim/agents/qwen_text/backend.py')]:
        relative = 'code/reference/' + ('qwen_' if path.parent==qwen else '') + path.name
        copy(path,relative)
        code_index.append({'source':str(path),'copy':relative,'sha256':sha(path.read_bytes())})
    copy(GUIDE, 'code/reference/steer_drivor_qwen_gt_grounding_analysis.md')
    copy(Path(__file__), 'code/audit_generated_data.py')
    dump('code/index.json',code_index)
    pilot_root = ROOT / 'outputs/stage2_navigation_no_rewrite_v2_final_20261107'
    partition_status = {}
    for split in ['train','val','test']:
        directory = GROUPS['stage2'].parent / split
        pilot = pilot_root / split
        pilot_meta_path = pilot / 'shards/rank_0000/captions/metadata.json'
        pilot_meta = jread(pilot_meta_path) if pilot_meta_path.exists() else {}
        current = results['stage2'][3]['rank_0000']
        partition_status[split] = {'full_directory_exists':directory.is_dir(),
            'full_complete':(directory/'COMPLETE.json').exists(),
            'full_merged_text_exists':(directory/'scene_text.json').exists(),
            'pilot_path':str(pilot), 'pilot_complete':(pilot/'COMPLETE.json').exists(),
            'pilot_scenes':len(jread(pilot/'scene_text.json')) if (pilot/'scene_text.json').exists() else 0,
            'pilot_prompt_matches_full':pilot_meta.get('prompt') == current['prompt'],
            'pilot_model_matches_full':all(pilot_meta.get(k)==current.get(k) for k in ['model','model_config_sha256','model_weight_bytes']),
            'pilot_policy_matches_full':pilot_meta.get('text_policy')==current.get('text_policy'),
            'pilot_decoding_matches_full':pilot_meta.get('decoding')==current.get('decoding')}
    dump('partition_status.json',partition_status)
    report = Report()
    build_report(report, results, overlap, raw_choices, visuals, cal, code_index, partition_status, started)
    report.save()
    dump('audit_manifest.json',{'started':started,'finished':now(),'seed':SEED,'groups':GROUPS,
         'snapshot_definition':'Per-group fixed filename inventory; each original cache read once and hashed. Generation may continue after inventory capture.',
         'samples_per_group':3,'montages':10,'raw_calls':20,'modified_generation_sources':False})
    archive = OUT.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(OUT.rglob('*')):
            if path.is_file():
                z.write(path, str(path.relative_to(OUT.parent)))
    print(json.dumps({'report':str(OUT/'report.html'),'markdown':str(OUT/'report.md'),'archive':str(archive),
                      'counts':{k:v[0]['read_scene_count'] for k,v in results.items()}},ensure_ascii=False),flush=True)


def build_report(r, results, overlap, raw, visuals, cal, code_index, partition_status, started):
    r.heading('Steer-DrivoR 数据检查报告（实际缓存快照）',1)
    r.para(f'检查开始时间 UTC：{started}。随机种子：{SEED}。本报告暂按“两组＝第一阶段定位文本、第二阶段最新 rerun 带导航文本”解释。只读取已有数据，不调用 VLM、不修改生成脚本、不重写任何训练缓存。HTML 内嵌全部展示图片，可单文件打开；ZIP 同时包含原始样本、统计、完整 log 列表、标定、代码副本。')
    r.para('重要：以下全量数据仍处于生成中。prepared 数量不是已完成文本数量，已缓存文本也不是已完成接地训练样本数量。文件名列表在各组开始时固定，统计对应此次读取快照；正在生成的后续样本不纳入。第一阶段和第二阶段的原始场景集合相同，不能把它们当作互不重叠的 train/test。')
    r.heading('1. 数据结构、字段和每组 3 条完整原始样本')
    for name,root in GROUPS.items():
        r.para(f'{name} 实际目录：{root}')
    r.code('shards/rank_0000 ... rank_0007/\n  run_config.json\n  prepared/manifest.json       # 场景、原图路径、导航与来源\n  prepared/ground_truth.json   # GT 与变换，和 VLM 输入分离\n  captions/metadata.json       # 模型、prompt、采样、源码哈希\n  captions/scenes/<token>.json # 每场景原始输出、清洗结果和审计\n  captions/scene_text.json    # 本 rank 完成后导出\n  grounding/scenes/<token>.json # 投影结果，后续步骤生成\n  grounding/masks/<token>.npz   # 接地监督，后续步骤生成\n  COMPLETE.json                # 本 rank 全流程完成标记\nscene_text.json                 # 所有 rank 合并完成后导出\ntraining_index.jsonl            # stage1 最终可训练索引\ndataset_index.jsonl             # stage2 最终索引\nreview_queue.json, summary.json # 完成后汇总')
    r.para('JSON 为 UTF-8 文本，JSONL 每行一个样本，NPZ 为 NumPy 压缩数组。index.html 是人工查看页面，不是模型语言输入。模型应读取 scene_text.json 中 token→相机→text；类别和 GT mask 是监督构造辅助。尚未合并时，可从 captions/scenes/<token>.json 的 scene_text 字段读取同样内容。')
    fields = [
      ('manifest.records[].token','NAVSIM initial_token，当前帧 token；不是相机图像文件 token。'),
      ('log_name / frame_index / timestamp','日志名；场景历史窗口当前帧索引 3；原数据时间戳（微秒）。'),
      ('images / image_sha256','按 f0,b0,l0,r0 键保存原图绝对路径与 SHA-256；PIL.Image.open(path).convert("RGB") 读取。'),
      ('driving_command / navigation_text / navigation_source','当前帧导航 one-hot；对应文本；AgentInput.ego_statuses[-1].driving_command。原始 one-hot 顺序 left,straight,right,unknown；Qwen 命令映射由 navigation.py 完成。'),
      ('manifest.schema_version / camera_order / image_size','格式版本；固定四相机顺序；规划/投影分辨率 [1148,672]，不等于 VLM 输入分辨率。'),
      ('manifest.split / data_split / subset','逻辑 navtrain/navtest；物理 trainval/test；原生训练/验证 log 子集。'),
      ('manifest.seed / rank / world_size / all_scenes_per_log','日志洗牌种子、分片序号和总数；是否取该 log 全部符合 filter 的场景。'),
      ('manifest.selected_log_count / split_hashes / source_hashes','该分片选中日志数量、划分 YAML 哈希、准备数据源码哈希。'),
      ('manifest.grid_mask / vlm_input / navigation_one_hot_order / data_root','离线不使用 GridMask；输入约束说明；导航编码顺序；数据根目录。'),
      ('manifest.skipped / missing_logs','缺图跳过记录和缺失日志；必须检查，不能以选中数冒充可用数。'),
      ('cache.token / cache_version / input_hash / rule_status','样本键；生成配置指纹；manifest record 指纹；passed 或 needs_review 等规则状态，passed 不等于人工语义正确。'),
      ('cache.scene_text.<cam>.text','最终短语按逗号加空格连接的语言输入，可能为空字符串。'),
      ('cache.scene_text.<cam>.categories','与保留短语一一对应的类别列表，允许重复，不是去重集合。'),
      ('cache.audit.<cam>.objects / text / categories / token_count','最终保留的 phrase/category 对象及拼接结果；RoBERTa tokenizer token 数，含特殊 token，空串通常为 2。'),
      ('audit.status / removed / format_normalization','ok、empty_scene、filtered_empty、generation_failed；被删短语及理由；是否去掉 Markdown JSON 围栏。'),
      ('audit.attempts[].raw_text / answer','生成文本 decode 后的保存值；answer 为去掉 thinking 内容后的文本。raw_text 未做 JSON 解析，但不是原始 token ID。'),
      ('attempts[].question / image / image_grid_thw','本次实际完整 prompt；单张当前帧原图路径；视觉 patch 网格 [T,H,W]。'),
      ('attempts[].generated_tokens / seconds / finish_reason / error','本次生成 token 数、耗时、eos 或截断结束、校验错误。'),
      ('audit.initial_cleaning / repairs（stage1）','初次清洗结果；逐短语重述的原词、替换、是否接受和全部尝试。'),
      ('audit.local_edits / error（stage2）','无需再调用模型的局部修正及相机级错误。'),
      ('audit.navigation_command','该视图实际使用的导航命令；stage1 为 null。'),
      ('prepared.ground_truth.<token>.boxes / names','当前 annotations 的 Nx7 三维框与 N 个类别，[x,y,z,length,width,height,heading]，米/弧度。'),
      ('ground_truth.cam_K / lidar_to_camera','四相机缩放后的内参和 sensor2lidar 求逆的 4×4 变换；原始标定另见第 7 节。'),
      ('grounding JSON cameras.<cam>.objects','50 米范围、支持类别、可投影 GT 框；bbox=xyxy、category、gt_index、near_clipped；有渲染时另有 segments。'),
      ('selected_gt_indices / valid / near_clipped_box_views','根据文本类别选中的框下标、相机是否有有效监督、跨近平面的框-视图数量。'),
      ('NPZ soft_mask / valid','(4,3936) float32，48×82 行优先；有效行归一化和为 1，无效为零；valid 为 (4,) bool。'),
      ('NPZ boxes_2d / box_camera_index / box_gt_index / box_category','M×4 像素 xyxy；对应相机索引、原始 GT 下标和类别，只含被选择的监督框。'),
      ('NPZ camera_order / scene_token / grid_shape / image_size / coordinate_format / input_hash','相机排列、场景键、[48,82]、[1148,672]、xyxy、输入及标签配置指纹。'),
      ('stage1 training_index 每行','scene_token, log_name, camera_order, caption_cache, grounding_npz, boxes_json；图片路径从 manifest 读，不能从此行直接取得。'),
      ('stage2 dataset_index 每行','上述索引并包含 partition, images, driving_command, navigation_text, scene_text, caption_status, text_fallback/cameras, empty_text_cameras, view_status, grounding_valid 等；完整生成代码附后。'),
      ('metadata / run_config','模型文件位置/大小/时间戳/配置哈希，依赖版本、源码哈希、prompt 与哈希、解码、设备、导航与修复策略；run_config 另记录 split/rank/scenes/deep_scenes 及工作环境。每 rank 原文件完整附在 metadata/。')]
    r.table(['字段','含义'],fields)
    r.para('以下每组 3 个 captions/scenes 原文件完整贴出（包括 4 个相机、原始输出及审计，未删字段、未改文字）；另附其 manifest record，提供图像读取路径。samples/ 中的 .json 文件按字节复制，report 中 JSON 围栏仅为展示。注意这些是原始生成缓存样本，不宣称都已通过最终训练筛选。')
    for name,(stats,records,*_) in results.items():
        for i,token in enumerate(stats['sample_tokens'],1):
            r.heading(f'{name} 原始样本 {i}：{token}',3)
            r.code((OUT/f'samples/{name}/{token}.json').read_text(),'json',f'完整原始缓存 {token}.json（展开）')
            r.code(json.dumps(records[token],ensure_ascii=False,indent=2),'json','对应 manifest record（字段和值保持原样；展示时缩进）')
    r.heading('2. 规模、来源、划分与测试集重叠')
    rows=[]
    for name,(s,*_) in results.items():
        rows.append([name,s['prepared_scenes'],s['read_scene_count'],s['rules'].get('passed',0),s['rules'].get('needs_review',0),s['rules'].get('generation_failed',0),s['grounding_npz_count'],s['merged_scene_text_exists']])
    r.table(['组','prepared 场景','已读缓存场景','规则 passed','needs_review','generation_failed','全量目录 NPZ','已最终合并'],rows)
    r.para('generation_failed 是场景规则状态（该场景至少一个相机生成失败），不能与相机失败数量混为一谈。needs_review 也不等于整场景空文本：第二阶段只要有局部动作词删除等编辑就会标记 review，但保留其他合格描述并继续导出。第一阶段还会检查保留短语总词数是否超过 60，prompt 的词数约束不是由模型必然保证。')
    for name,(s,*_) in results.items():
        r.para(f'{name} 快照范围：{s["snapshot_start"]} ～ {s["snapshot_end"]}；选中 {s["selected_log_count"]} logs，已有缓存覆盖 {s["cached_log_count"]} logs。文件逐项 SHA-256 见 {name}_snapshot.csv。')
        r.table(['rank','选中场景','已缓存','NPZ','完成'],[[x['rank'],x['prepared'],x['cached'],x['grounding_npz'],x['complete']] for x in s['ranks']])
    r.para('两组均为 SPLIT=navtrain、SUBSET=train，来自物理 trainval。原生 navtrain.yaml 共指定 1192 个 log、103288 个 scene token；再与 default_train_val_test_log_split.yaml 的 train_logs 取交集得到 978 logs、85109 场景。val 是同一 navtrain filter 与原生 val_logs 的交集（214 logs）；本工程没有独立 navval.yaml，不能把这里的 val 自动等同于其他工程/版本的 navval。')
    r.para('SCENES_PER_RANK=0 配合脚本 --all-scenes 表示取分片中全部符合原生过滤条件的场景，不是每 log 一条。按 seed=3407 洗牌日志后分配给 8 个 rank；窗口 4 个历史帧 + 10 个未来帧、frame_interval=1、has_route=True，并受 YAML token 列表限制。未来帧用于 NAVSIM 样本窗口条件，不作为 VLM 输入。DEEP_SCENES=80 是深度检查抽样数，不是只生成 80 场景。')
    r.para('两组 prepared token 交集为 '+str(overlap['stage1_stage2_prepared_tokens'])+'，log 交集为 '+str(overlap['stage1_stage2_selected_logs'])+'。第一组生成简单实例外观/位置描述，不输入导航，保留逐短语重述；第二组加入当前导航，描述道路关系与交通控制，禁用模型重述，采用相机级局部清洗。二者不是按场景互斥划分，训练第二阶段可继续使用相同图像，但文本采用第二阶段版本。')
    r.code(json.dumps(overlap,ensure_ascii=False,indent=2),'json','与原生 navtest 及 val 的实际交集检查')
    r.para('无重叠结论仅针对本次提供的原生 YAML 划分和 manifest，不证明模型预训练数据不包含这些图像。旧 stage2_navigation_no_rewrite_v2_8gpu_full 是另一轮未完成生成，不能和 rerun 简单相加成新场景数；本报告未将旧 run 混入统计。')
    for name in ['stage1_selected_logs','stage2_selected_logs','reference_navtrain_val_logs','reference_navtest_logs']:
        r.code((OUT/f'{name}.txt').read_text(),'text',f'{name} 完整 log 名列表（展开）')
    r.para('各组“已生成缓存实际覆盖”的 log 列表另见 stage1_cached_logs.txt / stage2_cached_logs.txt。下表区分全量和先前各 1 场景的 pilot；最新全量脚本依次完成 train→val→test，当前 train 未完成，不能说全量 val/test 文本已经生成。第一阶段所给启动命令只生成 train。')
    r.code(json.dumps(partition_status,ensure_ascii=False,indent=2),'json','train / val / test 现状与 pilot 配置一致性')
    r.heading('3. 文本生成配置、完整 prompt、图像与后处理')
    r.para('模型为本地 Qwen-Drive-1.0-4B 的 VLM checkpoint：/mnt/zhaozc_workspace/project/LQH/Qwen-drive/checkpoint。使用其 Qwen3.5 VLM，不实例化规划 expert。权重文件 model.safetensors 为 9,078,630,512 字节；这里的“4B”来自本地模型/项目说明，不以文件字节数反推参数量。没有记录远端仓库 revision，不能给出未经证实的 commit；元数据已保存本地配置 SHA-256、权重大小/mtime、代码哈希。')
    for name,(_,_,_,metas,*_) in results.items():
        meta=metas['rank_0000']
        r.heading(name+' 实际 prompt 原文',3)
        r.code(meta['prompt'],'text')
        keys=['model','model_weight_bytes','model_config_sha256','model_weight_mtime_ns','prompt_sha256','python','torch','transformers','dtype','attention','current_image_pixels','roberta_tokenizer','max_text_tokens','batch_size','navigation_conditioned','stage1','phrase_repair','decoding','text_policy','model_calls_per_view']
        r.code(json.dumps({k:meta[k] for k in keys if k in meta},ensure_ascii=False,indent=2),'json',name+' 实际生成参数')
    r.para('相机变量 front、rear、front-left、front-right 分别对应 f0、b0、l0、r0。stage2 的 navigation_command 来自当前帧指令映射，而非 GT 未来轨迹；实际逐调用 prompt 见原样样本的 attempts.question。一次只输入一张当前帧（窗口 index=3）的原图；GT annotations 绝不拼入 VLM prompt。')
    r.para('传给 CameraFrame 的路径指向原图；模型内部按官方 smart_resize 进行等比例缩放并使尺寸对齐 patch_size×spatial_merge_size=32，像素预算 921600，RGB 归一化。例如 image_grid_thw=[[1,44,80]] 对应 H=704、W=1280（patch_size=16），不是 1148×672；实际网格分布见下面统计。原始文件尺寸见第 6、7 节逐图记录。官方单图 temporal patch 填充不代表使用额外未来帧。')
    r.para('do_sample=True，temperature=0.01、top_k=1、top_p=0.001、repetition_penalty=1，seed=3407。元数据 presence_penalty=0，后端在调用 generate 前将该字段移除。RoBERTa 的 128 token 上限是最终语言输入长度，不是 VLM 输出上限。stage1 首次 max_new_tokens=1024，JSON/schema/类别失败最多重试 1 次，上限 2048；不合格短语另可重述最多 2 次，每次 512。stage2 每相机只调用 1 次，无 JSON 重试、无模型重述。')
    r.para('共同后处理：接受纯 JSON 或完整 Markdown JSON 围栏；规范空白；过滤空短语、非拉丁字母脚本、未来/建议词、特定运动词（只针对 vehicle/pedestrian/bicycle）、特定自车相对方位词；去重；按生成顺序将完整短语拼接，超出 RoBERTa 128 token 的候选短语舍弃，继续考虑后面的短语。不是把句子截成半句。单条短语过长且没有其他合格短语时仍可能为空。英文脚本检查不能保证语法/语义一定正确。')
    r.para('stage1：严格校验对象只有 phrase/category 字段，且此阶段限制五个接地类别；JSON/schema 连续失败后该相机为空、记录 generation_failed。语义问题可触发逐短语同图重述；仍未通过的场景进入 review 队列，最终训练索引排除规则失败场景及四相机均无有效 mask 的场景。正常返回 objects=[] 是 empty_scene，不自动使整个场景不合格。')
    r.para('stage2：坏 JSON/根 schema/截断只使本相机为空；保留另外三个相机。合法 phrase 的缺失/未知类别转 other；特定“动作词+静态方位介词”局部删除动作词；其余只删有问题短语，记录 local_edits / removed / review。最终索引保留全部规划场景，即便四相机为空，也不能用文本/GT 可用性筛选评测场景。')
    for rel in ['scripts/data_generation/text_processing.py','scripts/data_generation/caption_repair.py','scripts/data_generation/stage2/text_policy.py']:
        r.code((ROOT/rel).read_text(),'python',rel+' 后处理完整源码')
    r.heading('4. 20 条未做 JSON 解析的 VLM 保存输出')
    r.para('从本次实际缓存中按相机和输出类型分层随机抽样（种子固定）；每个相机 5 条，共 20 条，兼顾两组，优先保留真实空列表及真实错误。不是均匀随机样本，因此不能用这 20 条估计错误率。不存在的错误类型不会编造。raw_text 是保存的 decode 输出（已 skip_special_tokens），不是 token IDs；以下内容不做 JSON 规范化、不删除 thinking/fence。原字符串另存 raw_outputs/*.txt；完整调用和来源哈希在 raw_outputs/index.json。')
    r.table(['组','相机','类型','token','attempt'],[[x['group'],x['camera'],x['kind'],x['scene_token'],x['attempt_index']] for x in raw])
    for i,item in enumerate(raw,1):
        r.code(item['call']['raw_text'],'text',f'{i:02d} {item["group"]} {item["scene_token"]} {item["camera"]} {item["kind"]}')
    r.heading('5. 基础统计与一致性检查')
    r.para('分母为快照中成功读取的场景数，每场景四视图。短语数按 audit.objects 条数计算，不按逗号拆句；类别次数按每条保留短语累计，另存“出现该类别的视图数”，避免把重复类别误当重复样本。所有统计包含空视图；passed 子集另存 JSON。原始调用错误率和最终空文本率不同：stage1 重试后可能恢复。这里 raw_calls 只统计整张相机描述的 attempts（包括 JSON 重试），不混入 repairs 中单条短语重述的调用；重述另列次数。')
    r.para('原因口径：vlm_empty_list=有效 objects=[]；json_syntax_error=输出正常结束但不是可解析 JSON；root_schema_error=JSON 可解析但根字段不合要求；truncated=未以 eos 结束（可能同时 JSON 不完整）；recorded_validation_error=JSON 可解析但字段/类别等生成器校验失败；postprocessing_removed_all=原来非空但后处理删掉全部短语。前表“解析/生成失败空”合并这些失败类型，下面分项给出，不能把所有失败称为坏 JSON。')
    for name,(s,*_) in results.items():
        r.heading(name,3)
        r.table(['相机','视图数','空文本','空比例','VLM空列表','解析/生成失败空','过滤后空'],[
            [c,v['total'],v['empty'],f'{v["empty_fraction"]:.4%}',v['empty_causes'].get('vlm_empty_list',0),
             sum(n for k,n in v['empty_causes'].items() if k not in {'vlm_empty_list','postprocessing_removed_all'}),
             v['empty_causes'].get('postprocessing_removed_all',0)] for c,v in s['per_camera_all_cached'].items()])
        for c,v in s['per_camera_all_cached'].items():
            r.para(f'{c} 短语数分布（短语数:视图数）：'+json.dumps(dict(sorted(v['phrase_count'].items())),ensure_ascii=False))
            r.para(f'{c} 类别出现次数：'+json.dumps(v['categories'],ensure_ascii=False))
            r.para(f'{c} 最终空原因：'+json.dumps(v['empty_causes'],ensure_ascii=False)+'；原始调用分类：'+json.dumps(v['raw_calls'],ensure_ascii=False))
            r.para(f'{c} 删除原因：'+json.dumps(v['removed'],ensure_ascii=False)+f'；重述短语数 {v["repairs"]}，接受 {v["accepted_repairs"]}；局部编辑数 {v["local_edits"]}。')
            r.para(f'{c} image_grid_thw 分布：'+json.dumps(v['grids'],ensure_ascii=False))
        r.code(json.dumps({'consistency':s['consistency'],'read_errors':s['read_errors'],'source_hash_mismatches':s['source_hash_mismatches']},ensure_ascii=False,indent=2),'json','文件结构及当前源码与生成元数据哈希检查')
    r.para('规则过滤有边界：运动词表不是语义判别模型，例如 traveling 等未在当前 MOTION 正则中；局部纠正也不能证明空间描述真实。生成文本仍需图文抽查，不能把 passed 当作“无幻觉”的保证。本报告保留原输出，未静默修复这些问题。')
    r.heading('6. 随机 10 场景的四视图文本拼图')
    r.para('每组从该组快照缓存均匀随机抽 5 场景，共 10 张；不按文本是否为空筛选。每张严格按 f0、b0、l0、r0 横排，上方为完整最终 text；空串显示 [EMPTY TEXT] 仅用于可视化，不写回模型输入。图片仅在报告中等比例缩小，无 GridMask、无 GT 框绘制；原图路径与尺寸见 images/index.json。')
    for v in visuals:
        r.image(v['file'],f'{v["group"]} / {v["token"]}')
    r.heading('7. 一个当前帧的原始标定、标注和前视原图')
    r.para('以下直接读取原始 NAVSIM log pickle 中 token 对应当前帧；boxes 映射来自 Scene._build_annotations：anns.gt_boxes，names 来自 anns.gt_names。完整 names 保留，另给前 10 个以对齐 boxes 前 10 行。calibration/annotations_original_arrays.npz 保存原始数值数组；prepared_ground_truth_full.json 保存该场景的全部框、缩放内参与逆变换，便于对比。')
    r.code(json.dumps(cal,ensure_ascii=False,indent=2,default=serial),'json','原始标定与标注完整数值（展开）')
    r.para('intrinsics 对应 raw cams.CAM_*.cam_intrinsic，sensor2lidar_rotation 为 3×3 矩阵，不是四元数。使用列向量 convention：p_lidar=R·p_camera+t；投影需其逆变换。原图尺寸与原始内参、缩放到 1148×672 的 cam_K 必须对应。数据准备代码要求 lidar2ego 为单位矩阵，否则停止准备，避免静默混用坐标系。distortion 原值一并给出，当前投影代码未额外做畸变校正；此报告没有据此宣称投影已通过人工核对。')
    r.image('calibration/front_original.jpg','同一场景前视相机原图（文件字节直接复制，HTML 显示缩放）')
    r.heading('8. 已写代码、接入状态与尚未完成部分')
    r.para('已复制实际生成代码到 code/scripts/data_generation/：stage1 启动/worker/generate_captions，stage2 启动/worker/generate，prepare_navsim/selection/navigation，caption_repair/text_processing/text_policy，geometry/build_grounding，校验与合并脚本及当前 prompts。code/index.json 给出每个文件的原路径、归档路径和 SHA-256。以下关键生成与投影源码也直接内嵌在报告，便于单文件查看。')
    r.para('Dataset 接入尚未实现：本工程 navsim/planning/training/dataset.py 没有读取这些 scene_text/grounding 文件的逻辑。已附该文件当前原版，不能把 stage2/check_feature_input.py 的兼容性测试当作训练 Dataset 已接入。也未实现本方案两阶段训练中的空文本特征置零、CA 绕过、text dropout/grounding-valid 联动；这些属于后续模型训练代码，而非本次离线生成。')
    r.para('GT 投影及 mask 代码已经写好：GT 三维框 12 条边在 z=0.1m 近平面截断，再投影、裁剪到图像；取水平距离≤50m 且文本提及类别的全部有效框。patch 覆盖按二维矩形并集计算，重叠不重复计数，按 14×14 patch 汇总归一化。当前仍是按类别取全部框，未实施“提及 n 次选最近 n 框”的实例近似。traffic_light/traffic_sign/other 无 GT 接地监督。空文本相机 valid=False。')
    r.para('全量生成的顺序是准备→该分片全部文本→该分片全部投影/mask→验证→全局合并。当前全量目录没有成品 NPZ/最终索引并不等于投影代码缺失；不能直接启动接地训练。报告的四视图拼图只审查文本与图像，不能替代投影框的人工几何验收。离线本身无 GridMask；后续训练关闭 GridMask 仍需在训练实现中落实。')
    for rel in ['scripts/data_generation/generate_captions.py','scripts/data_generation/stage2/generate.py',
                'scripts/data_generation/geometry.py','scripts/data_generation/build_grounding.py',
                'scripts/data_generation/merge_stage1_shards.py','scripts/data_generation/stage2/merge.py',
                'navsim/planning/training/dataset.py']:
        r.code((OUT/'code'/rel).read_text(),'python',rel+' 完整源码')
    r.code(json.dumps(code_index,ensure_ascii=False,indent=2),'json','全部代码附件索引与校验值')
    r.para('本次检查限于文件、统计、输出样本和图像可视化，没有替你重新生成数据或修改既有训练/生成逻辑。后续若指定“两组”为其他目录，应重新按指定目录统计，不能沿用本报告的分母。')


if __name__ == '__main__':
    main()
