#!/usr/bin/env python3
"""Build a static side-by-side video comparison page for CAP arm inference runs."""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import os
import time
from pathlib import Path
from typing import Any


DEFAULT_VARIANTS = ("uniform", "e_only", "s_only", "s_max1", "current")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("ARM_INFER_OUTPUT", "outputs/arm-inference")),
        help="CAP arm inference root containing variant directories",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: ROOT/comparison)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_VARIANTS),
        help="Variant directory names in display order",
    )
    parser.add_argument(
        "--external-videos",
        action="store_true",
        help="Keep relative MP4 paths instead of embedding videos in index.html",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def relative_path(path: Path, root: Path) -> str:
    # Both paths originate under the already-resolved scan root. Avoid resolve()
    # here: on the shared filesystem it can turn a small metadata scan into a
    # very slow per-file network lookup.
    return path.relative_to(root).as_posix()


def selection_info(batch_dir: Path) -> dict[str, Any]:
    selection_path = batch_dir / "selected_ids.json"
    payload = read_json(selection_path) if selection_path.is_file() else None
    selected_ids = payload.get("selected_ids", []) if payload else []
    if not isinstance(selected_ids, list):
        selected_ids = []
    selected_ids = [int(value) for value in selected_ids if isinstance(value, int)]
    selected_ids_sha256 = hashlib.sha256(
        json.dumps(selected_ids, separators=(",", ":")).encode("ascii")
    ).hexdigest() if selected_ids else ""
    digest = ""
    if selection_path.is_file():
        digest = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    return {
        "batch": batch_dir.name,
        "selection_sha256": digest,
        "selected_ids_sha256": selected_ids_sha256,
        "selected_ids": selected_ids,
        "sample_count": payload.get("sample_count") if payload else None,
        "random_seed": payload.get("random_seed") if payload else None,
    }


def candidate_record(
    manifest_path: Path, root: Path, variant: str, batch_dir: Path
) -> dict[str, Any] | None:
    manifest = read_json(manifest_path)
    if not manifest:
        return None
    sample_id = manifest.get("sample_id")
    try:
        sample_id = int(sample_id)
    except (TypeError, ValueError):
        return None
    sample_dir = manifest_path.parent
    generated = sample_dir / "generated.mp4"
    target = sample_dir / "target_clip.mp4"
    if not (nonempty(generated) and nonempty(target)):
        return None
    manifest_variant = manifest.get("variant")
    if manifest_variant not in (None, variant):
        return None
    try:
        freshness = max(manifest_path.stat().st_mtime_ns, generated.stat().st_mtime_ns, target.stat().st_mtime_ns)
    except OSError:
        return None
    return {
        "sample_id": sample_id,
        "variant": variant,
        "batch": batch_dir.name,
        "generated": relative_path(generated, root),
        "target": relative_path(target, root),
        "checkpoint": manifest.get("checkpoint"),
        "inference_steps": manifest.get("inference_steps"),
        "seed": manifest.get("seed"),
        "fps": manifest.get("fps"),
        "frames": manifest.get("frames"),
        "width": manifest.get("width"),
        "height": manifest.get("height"),
        "freshness_ns": freshness,
    }


def discover_variant(
    root: Path, variant: str
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    variant_root = root / variant / "random-batches"
    candidates: dict[int, dict[str, Any]] = {}
    batches = sorted((path for path in variant_root.glob("*") if path.is_dir()), reverse=True)
    for batch_dir in batches:
        samples_root = batch_dir / "samples"
        for manifest_path in samples_root.glob("sample-*/manifest.json"):
            record = candidate_record(manifest_path, root, variant, batch_dir)
            if record is None:
                continue
            previous = candidates.get(record["sample_id"])
            if previous is None or record["freshness_ns"] > previous["freshness_ns"]:
                candidates[record["sample_id"]] = record

    source_counts: dict[str, int] = {}
    for record in candidates.values():
        source_counts[record["batch"]] = source_counts.get(record["batch"], 0) + 1
    preferred_batch = max(source_counts, key=lambda name: (name, source_counts[name])) if source_counts else None
    source = {
        "variant": variant,
        "complete_sample_count": len(candidates),
        "batch": preferred_batch,
        "batches": source_counts,
    }
    if preferred_batch:
        source.update(selection_info(variant_root / preferred_batch))
    return candidates, source


def build_data(root: Path, variants: list[str]) -> dict[str, Any]:
    discovered: dict[str, dict[int, dict[str, Any]]] = {}
    sources: dict[str, dict[str, Any]] = {}
    for variant in variants:
        discovered[variant], sources[variant] = discover_variant(root, variant)

    all_ids = set().union(*(records.keys() for records in discovered.values()))
    preferred_order: list[int] = []
    current_source = sources.get("current", {})
    current_batch = current_source.get("batch")
    current_selection = (
        selection_info(root / "current" / "random-batches" / current_batch)
        if current_batch
        else {}
    )
    for sample_id in current_selection.get("selected_ids", []):
        if sample_id in all_ids and sample_id not in preferred_order:
            preferred_order.append(sample_id)
    ordered_ids = preferred_order + sorted(all_ids.difference(preferred_order))

    rows: list[dict[str, Any]] = []
    complete_count = 0
    for sample_id in ordered_ids:
        variants_data = {
            variant: discovered[variant].get(sample_id) for variant in variants
        }
        target_record = variants_data.get("current")
        if target_record is None:
            target_record = next(
                (record for record in variants_data.values() if record is not None), None
            )
        complete = all(record is not None for record in variants_data.values()) and target_record is not None
        complete_count += int(complete)
        rows.append(
            {
                "sample_id": sample_id,
                "complete": complete,
                "target": {
                    "path": target_record["target"],
                    "source_variant": target_record["variant"],
                    "batch": target_record["batch"],
                }
                if target_record
                else None,
                "variants": {
                    variant: (
                        {
                            key: value
                            for key, value in record.items()
                            if key != "freshness_ns"
                        }
                        if record
                        else None
                    )
                    for variant, record in variants_data.items()
                },
            }
        )

    selection_hashes = {
        variant: source.get("selection_sha256") for variant, source in sources.items()
    }
    selection_id_hashes = {
        variant: source.get("selected_ids_sha256") for variant, source in sources.items()
    }
    return {
        "title": "CAP arm video comparison",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "variants": variants,
        "sources": sources,
        "selection_hashes": selection_hashes,
        "selection_id_hashes": selection_id_hashes,
        "selection_hashes_match": len(set(value for value in selection_id_hashes.values() if value)) == 1,
        "stats": {
            "total_sample_ids": len(rows),
            "complete_rows": complete_count,
            "incomplete_rows": len(rows) - complete_count,
        },
        "rows": rows,
    }


def js_data(data: dict[str, Any]) -> str:
    # Keep the embedded JSON safe when a prompt or metadata value contains HTML-like text.
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")


def make_cell(
    label: str,
    record: dict[str, Any] | None,
    root: Path,
    *,
    target: bool = False,
    embed_videos: bool = True,
) -> str:
    if record is None:
        return (
            f'<section class="video-cell missing-cell" data-column="{html.escape(label)}">'
            f'<h3>{html.escape(label)}</h3><div class="missing">Missing</div></section>'
        )
    path = record["path"] if target else record["generated"]
    src = path
    if embed_videos:
        video_path = root / path
        try:
            encoded = base64.b64encode(video_path.read_bytes()).decode("ascii")
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"cannot embed video {video_path}: {exc}") from exc
        src = f"data:video/mp4;base64,{encoded}"
    details = []
    if not target:
        details.append(f"step {record.get('inference_steps', '?')}")
        details.append(f"seed {record.get('seed', '?')}")
    details.append(str(record.get("batch", "")))
    detail_text = " · ".join(value for value in details if value)
    return (
        f'<section class="video-cell" data-column="{html.escape(label)}">'
        f'<h3>{html.escape(label)}</h3>'
        f'<video class="clip" data-kind="{("target" if target else "generated")}" '
        f'data-source="{html.escape(record.get("source_variant", record.get("variant", label)))}" '
        f'data-path="{html.escape(path)}" controls muted playsinline preload="metadata">'
        f'<source src="{html.escape(src)}" type="video/mp4">'
        "Your browser cannot play this video."
        "</video>"
        f'<div class="video-detail">{html.escape(detail_text)}</div>'
        "</section>"
    )


def render_html(data: dict[str, Any], root: Path, *, embed_videos: bool = True) -> str:
    variants = data["variants"]
    variant_json = json.dumps(variants, ensure_ascii=False)
    cells_per_row = len(variants) + 1
    rows_html: list[str] = []
    for row in data["rows"]:
        sample_id = row["sample_id"]
        row_class = "complete-row" if row["complete"] else "incomplete-row"
        target_record = row["target"]
        target_cell_record = None
        if target_record:
            source_variant = target_record["source_variant"]
            target_cell_record = row["variants"].get(source_variant)
            if target_cell_record:
                target_cell_record = dict(target_cell_record)
                target_cell_record["source_variant"] = source_variant
                target_cell_record["path"] = target_record["path"]
        cells = [make_cell("target_clip", target_cell_record, root, target=True, embed_videos=embed_videos)]
        for variant in variants:
            cells.append(
                make_cell(
                    variant,
                    row["variants"].get(variant),
                    root,
                    embed_videos=embed_videos,
                )
            )
        rows_html.append(
            f'<article class="sample-row {row_class}" data-sample-id="{sample_id}">'
            f'<aside class="sample-meta"><strong>sample {sample_id}</strong>'
            f'<span class="row-state">{"complete" if row["complete"] else "incomplete"}</span>'
            f'<div class="row-actions"><button type="button" class="row-play">Play row</button>'
            f'<button type="button" class="row-reset">Reset</button></div></aside>'
            f'<div class="video-grid" style="--columns:{cells_per_row}">' + "".join(cells) + "</div>"
            "</article>"
        )

    summary = data["stats"]
    embedded = js_data(data)
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CAP arm video comparison</title>
<style>
:root {{ color-scheme: dark; --bg:#111417; --panel:#191d22; --panel2:#20262d; --line:#35404a; --text:#e7edf2; --muted:#9eabb6; --accent:#6bb7ff; --good:#74d39b; --warn:#f4c76a; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.4 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
header {{ position:sticky; top:0; z-index:5; padding:18px 22px 14px; background:rgba(17,20,23,.97); border-bottom:1px solid var(--line); }}
h1 {{ margin:0 0 4px; font-size:22px; letter-spacing:0; }}
.subtitle {{ color:var(--muted); margin:0 0 14px; }}
.toolbar {{ display:flex; align-items:center; flex-wrap:wrap; gap:8px 12px; }}
.toolbar label, .toolbar button, .toolbar select, .toolbar input {{ font:inherit; }}
.toolbar label {{ display:inline-flex; align-items:center; gap:6px; color:var(--muted); }}
input[type=search] {{ width:150px; padding:7px 9px; border:1px solid var(--line); border-radius:4px; background:var(--panel); color:var(--text); }}
button, select {{ padding:7px 10px; border:1px solid var(--line); border-radius:4px; background:var(--panel2); color:var(--text); cursor:pointer; }}
button:hover, select:hover {{ border-color:var(--accent); }}
button:focus-visible, input:focus-visible, select:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}
.progress-wrap {{ min-width:180px; flex:1 1 260px; display:flex; align-items:center; gap:8px; color:var(--muted); }}
input[type=range] {{ width:100%; accent-color:var(--accent); }}
.column-controls {{ display:flex; flex-wrap:wrap; gap:6px; padding-top:10px; color:var(--muted); }}
.column-controls label {{ padding:5px 8px; border:1px solid var(--line); border-radius:4px; }}
main {{ padding:16px 22px 42px; }}
.stats {{ display:flex; flex-wrap:wrap; gap:10px 18px; color:var(--muted); margin:0 0 14px; }}
.stats strong {{ color:var(--text); }}
.grid-header, .sample-row {{ min-width:calc(150px + {cells_per_row} * 250px); }}
.grid-header {{ display:grid; grid-template-columns:150px repeat({cells_per_row}, minmax(220px,1fr)); border:1px solid var(--line); border-bottom:0; background:var(--panel2); position:sticky; top:136px; z-index:3; }}
.grid-header > div {{ padding:9px 10px; border-right:1px solid var(--line); font-weight:650; }}
.grid-header > div:last-child {{ border-right:0; }}
.sample-row {{ display:grid; grid-template-columns:150px minmax(0,1fr); border:1px solid var(--line); border-bottom:0; background:var(--panel); }}
.sample-row:last-child {{ border-bottom:1px solid var(--line); }}
.sample-row.incomplete-row {{ border-left:3px solid var(--warn); }}
.sample-meta {{ padding:12px 10px; border-right:1px solid var(--line); display:flex; flex-direction:column; gap:7px; position:sticky; left:0; z-index:2; background:var(--panel); }}
.sample-meta strong {{ font-size:13px; word-break:break-word; }}
.row-state {{ color:var(--good); font-size:12px; }}
.incomplete-row .row-state {{ color:var(--warn); }}
.row-actions {{ display:flex; flex-direction:column; gap:5px; margin-top:auto; }}
.row-actions button {{ padding:5px 6px; font-size:12px; }}
.video-grid {{ display:grid; grid-template-columns:repeat(var(--columns), minmax(220px,1fr)); min-width:0; }}
.video-cell {{ min-width:0; padding:8px; border-right:1px solid var(--line); }}
.video-cell:last-child {{ border-right:0; }}
.video-cell h3 {{ margin:0 0 6px; font-size:13px; font-weight:650; }}
.clip {{ display:block; width:100%; aspect-ratio:1280 / 704; object-fit:contain; background:#080a0c; border:1px solid #2c343c; }}
.video-detail {{ min-height:18px; margin-top:5px; color:var(--muted); font-size:11px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.missing {{ display:grid; place-items:center; aspect-ratio:1280 / 704; border:1px dashed var(--line); color:var(--muted); background:#14181c; }}
.hidden {{ display:none !important; }}
.notice {{ margin:0 0 14px; padding:9px 11px; border:1px solid var(--line); background:var(--panel); color:var(--muted); }}
.notice.good {{ border-color:#356b50; color:#b8e9ca; }}
@media (max-width:800px) {{ header {{ padding:14px; }} main {{ padding:12px 14px 28px; }} .grid-header {{ top:164px; }} }}
</style>
</head>
<body>
<header>
  <h1>CAP arm video comparison</h1>
  <p class="subtitle">Aligned by sample ID. Videos stay in their original batch directories.</p>
  <div class="toolbar">
    <label>Sample ID <input id="search" type="search" placeholder="e.g. 192834" autocomplete="off"></label>
    <label><input id="completeOnly" type="checkbox" checked> only complete rows</label>
    <button type="button" id="playVisible">Play visible</button>
    <button type="button" id="pauseVisible">Pause visible</button>
    <button type="button" id="resetVisible">Reset visible</button>
    <label>Speed <select id="speed"><option value="0.5">0.5x</option><option value="1" selected>1x</option><option value="1.5">1.5x</option><option value="2">2x</option></select></label>
    <label><input id="loop" type="checkbox"> loop</label>
    <div class="progress-wrap"><span>Progress</span><input id="progress" type="range" min="0" max="1" step="0.001" value="0" aria-label="Set progress for visible videos"><output id="progressValue">0%</output></div>
  </div>
  <div class="column-controls" id="columnControls"></div>
</header>
<main>
  <div class="stats" id="stats"></div>
  <div class="notice" id="notice"></div>
  <div class="grid-header"><div>Sample</div><div>target_clip</div>{''.join(f'<div data-header-column="{html.escape(v)}">{html.escape(v)}</div>' for v in variants)}</div>
  <div id="rows">{''.join(rows_html)}</div>
</main>
<script>
const DATA = {embedded};
const VARIANTS = {variant_json};
const rowsRoot = document.getElementById('rows');
const search = document.getElementById('search');
const completeOnly = document.getElementById('completeOnly');
const progress = document.getElementById('progress');
const progressValue = document.getElementById('progressValue');
const speed = document.getElementById('speed');
const loop = document.getElementById('loop');
const allRows = [...document.querySelectorAll('.sample-row')];
const allVideos = [...document.querySelectorAll('video.clip')];
const groupVideos = row => [...row.querySelectorAll('video.clip')];
const visibleRows = () => allRows.filter(row => !row.classList.contains('hidden'));
function setRows() {{
  const query = search.value.trim().toLowerCase();
  allRows.forEach(row => {{
    const matchesId = !query || row.dataset.sampleId.includes(query);
    const matchesComplete = !completeOnly.checked || row.classList.contains('complete-row');
    row.classList.toggle('hidden', !(matchesId && matchesComplete));
  }});
}}
function setPlayback(videos, playing) {{
  videos.forEach(video => {{
    if (playing) {{ const result = video.play(); if (result) result.catch(() => {{}}); }}
    else video.pause();
  }});
}}
function resetVideos(videos) {{ videos.forEach(video => {{ video.pause(); video.currentTime = 0; }}); }}
function syncGroup(source) {{
  const row = source.closest('.sample-row');
  if (!row || source.dataset.syncing === '1') return;
  row.querySelectorAll('video.clip').forEach(video => {{
    if (video === source || !Number.isFinite(video.duration)) return;
    if (Math.abs(video.currentTime - source.currentTime) > 0.06) {{
      video.dataset.syncing = '1';
      video.currentTime = source.currentTime;
      video.dataset.syncing = '0';
    }}
  }});
}}
allVideos.forEach(video => {{
  video.playbackRate = Number(speed.value);
  video.loop = loop.checked;
  video.addEventListener('timeupdate', () => syncGroup(video));
  video.addEventListener('play', () => setPlayback(groupVideos(video.closest('.sample-row')), true));
  video.addEventListener('pause', () => {{
    const row = video.closest('.sample-row');
    if (row && [...row.querySelectorAll('video.clip')].some(item => item !== video && !item.paused)) return;
    setPlayback(groupVideos(row), false);
  }});
}});
allRows.forEach(row => {{
  row.querySelector('.row-play').addEventListener('click', () => setPlayback(groupVideos(row), true));
  row.querySelector('.row-reset').addEventListener('click', () => resetVideos(groupVideos(row)));
}});
document.getElementById('playVisible').addEventListener('click', () => setPlayback(visibleRows().flatMap(groupVideos), true));
document.getElementById('pauseVisible').addEventListener('click', () => setPlayback(visibleRows().flatMap(groupVideos), false));
document.getElementById('resetVisible').addEventListener('click', () => resetVideos(visibleRows().flatMap(groupVideos)));
speed.addEventListener('change', () => allVideos.forEach(video => video.playbackRate = Number(speed.value)));
loop.addEventListener('change', () => allVideos.forEach(video => video.loop = loop.checked));
progress.addEventListener('input', () => {{
  const value = Number(progress.value);
  progressValue.value = `${{Math.round(value * 100)}}%`;
  progressValue.textContent = progressValue.value;
  visibleRows().flatMap(groupVideos).forEach(video => {{ if (Number.isFinite(video.duration)) video.currentTime = value * video.duration; }});
}});
search.addEventListener('input', setRows);
completeOnly.addEventListener('change', setRows);
const columnControls = document.getElementById('columnControls');
VARIANTS.forEach(variant => {{
  const label = document.createElement('label');
  label.innerHTML = `<input type="checkbox" checked data-toggle-column="${{variant}}"> ${{variant}}`;
  label.querySelector('input').addEventListener('change', event => {{
    const enabled = event.target.checked;
    document.querySelectorAll(`[data-column="${{variant}}"], [data-header-column="${{variant}}"]`).forEach(item => item.classList.toggle('hidden', !enabled));
  }});
  columnControls.appendChild(label);
}});
document.getElementById('stats').innerHTML = `<span><strong>${{DATA.stats.total_sample_ids}}</strong> sample IDs</span><span><strong>${{DATA.stats.complete_rows}}</strong> complete rows</span><span><strong>${{DATA.stats.incomplete_rows}}</strong> incomplete rows</span><span>generated ${{DATA.generated_utc}}</span>`;
const notice = document.getElementById('notice');
if (DATA.selection_hashes_match) {{ notice.classList.add('good'); notice.textContent = 'All variants use the same fixed sample selection.'; }}
else {{ notice.textContent = 'Selection manifests differ; compare sample IDs carefully.'; }}
setRows();
</script>
</body>
</html>
'''


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = (args.output or root / "comparison").resolve()
    if not root.is_dir():
        raise SystemExit(f"CAP inference root does not exist: {root}")
    data = build_data(root, args.variants)
    output.mkdir(parents=True, exist_ok=True)
    (output / "data.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "index.html").write_text(
        render_html(data, root, embed_videos=not args.external_videos), encoding="utf-8"
    )
    print(f"wrote {output / 'index.html'}")
    print(f"wrote {output / 'data.json'}")
    print(f"sample_ids={data['stats']['total_sample_ids']} complete={data['stats']['complete_rows']} incomplete={data['stats']['incomplete_rows']}")
    for variant in args.variants:
        source = data["sources"][variant]
        print(f"{variant}: complete={source['complete_sample_count']} preferred_batch={source.get('batch')}")


if __name__ == "__main__":
    main()
