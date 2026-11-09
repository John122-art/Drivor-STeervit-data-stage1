"""Build a local gallery of unannotated current images and generated captions."""
import argparse
import html
import json
from pathlib import Path

from PIL import Image

from common import CAMERAS, IMAGE_SIZE, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-captions", type=Path, help="Optional previous caption directory")
    args = parser.parse_args()
    records = {r["token"]: r for r in read_json(args.prepared / "manifest.json")["records"]}
    captions = read_json(args.captions / "scene_text.json")
    audits = read_json(args.captions / "audit.json")
    metadata = read_json(args.captions / "metadata.json")
    previous = read_json(args.compare_captions / "scene_text.json") if args.compare_captions else {}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "images").mkdir()
    page = ['<!doctype html><meta charset="utf-8"><title>Scene caption preview</title>',
            '<style>body{font:17px sans-serif;max-width:1400px;margin:24px auto;background:#171b22;color:#eee}img{max-width:100%}pre{white-space:pre-wrap}article{background:#252b35;padding:16px;margin:16px 0}</style>',
            '<h1>Current-frame images and generated captions</h1>',
            '<p>Each caption was generated from one original current-frame image. No GT annotations enter generation.</p>',
            f'<details><summary>Prompt</summary><pre>{html.escape(metadata["prompt"])}</pre></details>']
    summary = {}
    for token, views in captions.items():
        page.append(f'<h2>{html.escape(token)}</h2>')
        nav = metadata.get("navigation_by_scene", {}).get(token)
        if nav is not None:
            page.append(f'<p><strong>Active navigation command: [{html.escape(nav["text"])}]</strong>'
                        f' — NAVSIM current input: {html.escape(str(nav["driving_command"]))}</p>')
        summary[token] = {}
        for camera in CAMERAS:
            entry = audits[token][camera]
            text = views[camera]["text"]
            words = sum(len(o["phrase"].split()) for o in entry["objects"])
            relative = f"images/{token}_{camera}.jpg"
            with Image.open(records[token]["images"][camera]) as original:
                original.convert("RGB").resize(IMAGE_SIZE).save(args.output / relative, quality=95)
            summary[token][camera] = {"status": entry["status"], "english_words": words,
                "roberta_tokens": entry["token_count"], "removed": entry["removed"],
                "navigation_command": entry.get("navigation_command"),
                "repairs": [{k: r[k] for k in ("original", "accepted", "replacement", "retained_in_final")}
                            for r in entry.get("repairs", [])]}
            comparison = ""
            if camera in previous.get(token, {}):
                comparison = (f'<p>Previous description:</p><pre>{html.escape(previous[token][camera]["text"])}</pre>'
                              '<p>New description:</p>')
            page.append(f'<article><h3>{camera}</h3><a href="{relative}"><img src="{relative}"></a>'
                        f'{comparison}<pre>{html.escape(text)}</pre><p>{words} English words; {entry["token_count"]} RoBERTa tokens; '
                        f'status: {html.escape(entry["status"])}</p>'
                        f'<details><summary>Filtered phrases</summary><pre>{html.escape(str(entry["removed"]))}</pre></details></article>')
            if entry.get("repairs"):
                page.append('<details><summary>Original phrases and static rewrites</summary><pre>'
                            + html.escape(json.dumps(entry["repairs"], ensure_ascii=False, indent=2))
                            + '</pre></details>')
    write_json(args.output / "summary.json", summary)
    (args.output / "index.html").write_text("\n".join(page), encoding="utf-8")
    print(f"Preview: {args.output / 'index.html'}")


if __name__ == "__main__":
    main()
