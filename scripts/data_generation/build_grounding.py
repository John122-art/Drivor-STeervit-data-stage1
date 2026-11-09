"""Render all-view GT projection audit and optional caption-conditioned soft masks."""
import argparse
import html
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from common import CAMERAS, GROUND_CATEGORIES, IMAGE_SIZE, fingerprint, read_json, sha256, write_json, write_npz
from geometry import patch_distribution, project_box

COLORS = {"vehicle": "#00e78a", "pedestrian": "#ffda44", "bicycle": "#57b9ff",
          "traffic_cone": "#ff8b3d", "barrier": "#f278ee"}


def pack_grounding_arrays(token, projections, distributions, valid, input_hash):
    boxes, cameras, indices, categories = [], [], [], []
    for ci, camera in enumerate(CAMERAS):
        source = projections[camera]
        selected = set(source["selected_gt_indices"])
        for obj in source["objects"]:
            if obj["gt_index"] in selected:
                boxes.append(obj["bbox"])
                cameras.append(ci)
                indices.append(obj["gt_index"])
                categories.append(obj["category"])
    return {"soft_mask": np.stack(distributions).astype(np.float32),
        "valid": np.asarray(valid, dtype=bool), "camera_order": np.asarray(CAMERAS),
        "scene_token": np.asarray(token), "grid_shape": np.asarray([48, 82]),
        "image_size": np.asarray(IMAGE_SIZE), "coordinate_format": np.asarray("xyxy"),
        "boxes_2d": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        "box_camera_index": np.asarray(cameras, dtype=np.int64),
        "box_gt_index": np.asarray(indices, dtype=np.int64),
        "box_category": np.asarray(categories, dtype="U16"), "input_hash": np.asarray(input_hash)}


def load_projection(directory, token, legacy=None):
    if legacy is not None:
        return legacy[token]
    return read_json(Path(directory) / "scenes" / f"{token}.json")["cameras"]


def draw_projection(image_path, objects, camera, token):
    image = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    for obj in objects:
        color = COLORS[obj["category"]]
        for segment in obj["segments"]:
            draw.line([tuple(p) for p in segment], fill=color, width=2)
        draw.rectangle(obj["bbox"], outline=color, width=2)
        x, y = obj["bbox"][:2]
        label = f'{obj["category"]} #{obj["gt_index"]}'
        draw.text((x + 2, y + 2), label, font=font, fill=color, stroke_width=1, stroke_fill="black")
    draw.rectangle((0, 0, IMAGE_SIZE[0], 30), fill="black")
    draw.text((8, 5), f"{token}  {camera}  current frame=3  boxes={len(objects)}", font=font, fill="white")
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--captions", type=Path, help="Caption directory; omitted for geometric audit only")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--near", type=float, default=0.1)
    parser.add_argument("--radius", type=float, default=50.0)
    parser.add_argument("--no-render", action="store_true", help="Save training labels without millions of audit images")
    parser.add_argument("--stream", action="store_true", help="Write per-scene projection files instead of a large global JSON")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.near <= 0 or args.radius <= 0:
        parser.error("near and radius must be positive")
    if args.resume and not args.stream:
        parser.error("--resume requires --stream")
    if args.stream and not args.no_render:
        parser.error("Bulk --stream mode requires --no-render; render a selected audit subset separately")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    (args.output / "images").mkdir(exist_ok=args.resume)
    (args.output / "masks").mkdir(exist_ok=args.resume)
    (args.output / "scenes").mkdir(exist_ok=args.resume)
    manifest = read_json(args.prepared / "manifest.json")
    assert manifest["camera_order"] == list(CAMERAS)
    gt = read_json(args.prepared / "ground_truth.json")
    captions = {}
    if args.captions:
        metadata = read_json(args.captions / "metadata.json")
        if metadata["manifest_sha256"] != sha256(args.prepared / "manifest.json"):
            raise ValueError("Captions were generated for a different input manifest")
        captions = read_json(args.captions / "scene_text.json")
        unknown = set(captions) - {r["token"] for r in manifest["records"]}
        if unknown:
            raise ValueError(f"Unknown caption scene tokens: {unknown}")
        if any(set(v) != set(CAMERAS) for v in captions.values()):
            raise ValueError("Caption scene is incomplete")
    page = ['<!doctype html><meta charset="utf-8"><title>NAVSIM per-view data pilot</title>',
            '<style>body{font:16px sans-serif;max-width:1500px;margin:24px auto;background:#171b22;color:#eee} img{max-width:100%} .views{display:grid;grid-template-columns:1fr 1fr;gap:16px} article{background:#252b35;padding:12px} pre{white-space:pre-wrap}</style>',
            '<h1>Current-frame GT projection audit</h1><p>Order: f0, b0, l0, r0. GridMask OFF. GT boxes are for audit/targets only; VLM receives original images.</p>']
    projections, count_valid, clipped, resumed = {}, 0, 0, 0
    scene_index = {}
    records = [r for r in manifest["records"] if not args.stream or r["token"] in captions]
    geometry_hash = sha256(Path(__file__).with_name("geometry.py"))
    for record in records:
        token = record["token"]
        source = gt[token]
        input_hash = fingerprint({"record": record, "gt": source, "caption": captions.get(token),
            "near": args.near, "radius": args.radius, "geometry_sha256": geometry_hash, "label_schema": 2})
        scene_path = args.output / "scenes" / f"{token}.json"
        mask_path = args.output / "masks" / f"{token}.npz"
        if args.resume and scene_path.is_file() and mask_path.is_file():
            old = read_json(scene_path)
            with np.load(mask_path, allow_pickle=False) as target:
                matches = (old.get("input_hash") == input_hash and str(target["input_hash"]) == input_hash
                           and target["soft_mask"].shape == (4, 3936) and "boxes_2d" in target)
                if matches:
                    count_valid += int(target["valid"].sum())
            if matches:
                clipped += old["near_clipped_box_views"]
                scene_index[token] = {"boxes": f"scenes/{token}.json", "mask": f"masks/{token}.npz"}
                resumed += 1
                continue
        projections[token] = {}
        distributions, valid_flags = [], []
        scene_clipped = 0
        if not args.no_render:
            sheet = Image.new("RGB", (1148, 672))
            page.append(f'<h2>{html.escape(token)}</h2><p>{html.escape(record["log_name"])}</p><div class="views">')
        for index, camera in enumerate(CAMERAS):
            objects = []
            for gi, (box, category) in enumerate(zip(source["boxes"], source["names"])):
                if category not in GROUND_CATEGORIES or np.linalg.norm(np.asarray(box)[:2]) > args.radius:
                    continue
                projected = project_box(box, source["cam_K"][index], source["lidar_to_camera"][index], args.near)
                if projected is not None:
                    objects.append(dict(projected, category=category, gt_index=gi))
                    clipped += int(projected["near_clipped"])
                    scene_clipped += int(projected["near_clipped"])
            caption = captions.get(token, {}).get(camera, {"text": "", "categories": []})
            selected = [o for o in objects if o["category"] in caption["categories"]] if caption["text"].strip() else []
            distribution, valid = patch_distribution([o["bbox"] for o in selected])
            distributions.append(distribution)
            valid_flags.append(valid)
            count_valid += int(valid)
            projections[token][camera] = {"objects": objects, "selected_gt_indices": [o["gt_index"] for o in selected], "valid": valid}
            if args.no_render:
                for obj in objects:
                    obj.pop("segments", None)
                continue
            image = draw_projection(record["images"][camera], objects, camera, token)
            relative = f"images/{token}_{camera}.jpg"
            image.save(args.output / relative, quality=92)
            sheet.paste(image.resize((574, 336)), ((index % 2) * 574, (index // 2) * 336))
            page.append(f'<article><h3>{camera}</h3><a href="{relative}"><img src="{relative}"></a><pre>{html.escape(caption["text"])}</pre><p>Grounding valid: {valid}; selected GT: {[o["gt_index"] for o in selected]}</p></article>')
            if token in captions:
                base = Image.open(record["images"][camera]).convert("RGB").resize(IMAGE_SIZE).convert("RGBA")
                intensity = distribution.reshape(48, 82)
                intensity = intensity / intensity.max() if valid else intensity
                alpha = Image.fromarray((intensity * 110).astype(np.uint8)).resize(IMAGE_SIZE, Image.Resampling.NEAREST)
                overlay = Image.new("RGBA", IMAGE_SIZE, (255, 80, 30, 0))
                overlay.putalpha(alpha)
                heatmap = Image.alpha_composite(base, overlay).convert("RGB")
                heatmap_relative = f"images/{token}_{camera}_grounding.png"
                heatmap.save(args.output / heatmap_relative)
                page.append(f'<article><h3>{camera}: category grounding target</h3><a href="{heatmap_relative}"><img src="{heatmap_relative}"></a><p>Valid: {valid}; normalized patch target, not VLM-generated boxes.</p></article>')
        if not args.no_render:
            page.append('</div>')
            sheet.save(args.output / "images" / f"{token}_sheet.jpg", quality=93)
        if token in captions:
            masks = np.stack(distributions)
            assert masks.shape == (4, 3936) and np.isfinite(masks).all()
            write_npz(mask_path, **pack_grounding_arrays(token, projections[token], distributions, valid_flags, input_hash))
        write_json(scene_path, {"token": token, "input_hash": input_hash, "image_size": IMAGE_SIZE,
            "coordinate_format": "xyxy", "near_clipped_box_views": scene_clipped,
            "cameras": projections[token]})
        scene_index[token] = {"boxes": f"scenes/{token}.json", "mask": f"masks/{token}.npz"}
        print(f"Projected {token}: {[len(projections[token][c]['objects']) for c in CAMERAS]}", flush=True)
        if args.stream:
            projections.pop(token)
    (args.output / "index.html").write_text("\n".join(page), encoding="utf-8")
    if not args.stream:
        write_json(args.output / "projections.json", projections)
    write_json(args.output / "index.json", scene_index)
    write_json(args.output / "summary.json", {
        "scenes": len(records), "views": 4 * len(records), "resumed_scenes": resumed,
        "rendered": not args.no_render, "stream": args.stream, "label_schema": 2,
        "caption_scenes": len(captions), "valid_grounding_views": count_valid,
        "near_plane_m": args.near, "near_clipped_box_views": clipped,
        "radius_m": args.radius, "distance_reference": "annotation/lidar horizontal origin, per native NAVSIM convention",
        "camera_order": CAMERAS, "grid_mask": False, "selection": "all valid boxes of mentioned supported categories",
        "manifest_sha256": sha256(args.prepared / "manifest.json"),
        "gt_sha256": sha256(args.prepared / "ground_truth.json"),
        "caption_sha256": sha256(args.captions / "scene_text.json") if args.captions else None,
        "visual_review": "numeric labels saved; visual sampling is separate from automatic checks"})


if __name__ == "__main__":
    main()
