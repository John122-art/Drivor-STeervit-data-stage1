"""Show the exact selected 2D rectangles used for existing grounding masks."""
import argparse
import html
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import numpy as np

from common import CAMERAS, IMAGE_SIZE, read_json, write_json
from build_grounding import COLORS, load_projection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--grounding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=0, help="0 renders all captioned scenes; limit for bulk audits")
    args = parser.parse_args()
    records = {r["token"]: r for r in read_json(args.prepared / "manifest.json")["records"]}
    captions = read_json(args.captions / "scene_text.json")
    projections = read_json(args.grounding / "projections.json") if (args.grounding / "projections.json").is_file() else None
    if args.scenes:
        captions = dict(list(captions.items())[:args.scenes])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "images").mkdir()
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    page = ['<!doctype html><meta charset="utf-8"><title>Grounding 2D rectangles</title>',
            '<style>body{font:16px sans-serif;background:#171b22;color:#eee;max-width:1500px;margin:24px auto}img{max-width:100%}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}pre{white-space:pre-wrap}</style>',
            '<h1>Selected 2D rectangles and patch targets</h1>',
            '<p>Left: only the selected 2D rectangles used for grounding. Right: existing normalized patch target. Coordinates: [xmin, ymin, xmax, ymax], image size 1148 x 672. No 3D wireframes.</p>']
    exported = {}
    for token, views in captions.items():
        scene_projection = load_projection(args.grounding, token, projections)
        page.append(f'<h2>{html.escape(token)}</h2>')
        exported[token] = {}
        sheet = Image.new("RGB", IMAGE_SIZE)
        for ci, camera in enumerate(CAMERAS):
            source = scene_projection[camera]
            selected = set(source["selected_gt_indices"])
            objects = [o for o in source["objects"] if o["gt_index"] in selected]
            exported[token][camera] = {
                "image_size": list(IMAGE_SIZE), "coordinate_format": "xmin,ymin,xmax,ymax",
                "valid": source["valid"],
                "boxes": [{k: o[k] for k in ("gt_index", "category", "bbox")} for o in objects]}
            image = Image.open(records[token]["images"][camera]).convert("RGB").resize(IMAGE_SIZE)
            draw = ImageDraw.Draw(image)
            for obj in objects:
                color = COLORS[obj["category"]]
                draw.rectangle(obj["bbox"], outline=color, width=3)
                x, y = obj["bbox"][:2]
                draw.text((x + 2, max(32, y - 20)), f'{obj["category"]} #{obj["gt_index"]}',
                          fill=color, font=font, stroke_width=1, stroke_fill="black")
            draw.rectangle((0, 0, IMAGE_SIZE[0], 30), fill="black")
            draw.text((8, 5), f"{token}  {camera}  2D selected boxes={len(objects)}", fill="white", font=font)
            relative = f"images/{token}_{camera}_2d.png"
            image.save(args.output / relative)
            sheet.paste(image.resize((574, 336)), ((ci % 2) * 574, (ci // 2) * 336))
            mask_path = args.grounding / "images" / f"{token}_{camera}_grounding.png"
            if not mask_path.is_file():
                # Bulk jobs intentionally omit image rendering; rebuild this audit view from numeric labels.
                with np.load(args.grounding / "masks" / f"{token}.npz", allow_pickle=False) as target:
                    distribution = target["soft_mask"][ci].reshape(48, 82)
                    valid = bool(target["valid"][ci])
                intensity = distribution / distribution.max() if valid else distribution
                alpha = Image.fromarray((intensity * 110).astype(np.uint8)).resize(IMAGE_SIZE, Image.Resampling.NEAREST)
                overlay = Image.new("RGBA", IMAGE_SIZE, (255, 80, 30, 0))
                overlay.putalpha(alpha)
                with Image.open(records[token]["images"][camera]) as original:
                    base = original.convert("RGBA").resize(IMAGE_SIZE)
                mask_path = args.output / "images" / f"{token}_{camera}_grounding.png"
                Image.alpha_composite(base, overlay).convert("RGB").save(mask_path)
            mask_relative = os.path.relpath(mask_path.resolve(), args.output.resolve())
            if not mask_path.is_file():
                raise FileNotFoundError(mask_path)
            page.append(f'<h3>{camera}</h3><pre>{html.escape(views[camera]["text"])}</pre>'
                        f'<div class="pair"><a href="{relative}"><img src="{relative}"></a>'
                        f'<a href="{mask_relative}"><img src="{mask_relative}"></a></div>')
        sheet.save(args.output / "images" / f"{token}_2d_sheet.png")
    write_json(args.output / "selected_boxes.json", exported)
    (args.output / "index.html").write_text("\n".join(page), encoding="utf-8")
    print(f"Rendered {len(captions) * 4} views: {args.output / 'index.html'}")


if __name__ == "__main__":
    main()
