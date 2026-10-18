# Visualizing the pipeline (`run_full_pipeline.py`)

`docs/pipeline_diagram.py` renders the pick-and-place pipeline as a plain SVG:

```bash
python3 docs/pipeline_diagram.py                 # writes docs/pipeline_pick_and_place_stage0b_options.svg
python3 docs/pipeline_diagram.py my_variant.svg  # custom output path
```

It has no dependencies. To make a variation, edit `rows_spec` in the script: each row is a list of
`(kind, [label lines], centre_x | None)` cells. Rows with several cells are drawn as parallel
alternatives joined by a bus; the input section (rows 0-2) is hand-routed.

## Rendering pitfalls

- **Mermaid SVG labels are invisible in many viewers.** `mermaid-cli` puts node labels in HTML
  `<foreignObject>` elements, which image viewers, Inkscape and some doc tools drop. Setting
  `htmlLabels: false` emits plain `<text>`, but its multi-line output still overlapped when
  rasterized. Hence the hand-written SVG generator. (Mermaid is fine inside a GitHub-rendered
  ```` ```mermaid ```` block.)
- **Verify with a renderer that lacks HTML-label support**, e.g. `cairosvg`
  (`python3 -m venv v && v/bin/pip install cairosvg`, then `cairosvg.svg2png(url=..., write_to=...)`).
  A browser will hide label problems that other viewers show.
- `npx`/Node was not installed on the dev machine; `docker run minlag/mermaid-cli` worked as a fallback.

## Pipeline facts the diagram encodes

Verified against `run_full_pipeline.py` (constants may change; re-check before relying on them).

- **`--mode` is the main branch.** `pick-and-place` runs the fast path P1-P4 and skips Stages 3-12;
  `full-scene` and `full-scene-with-robot` run Stages 3-11 (Stage 12 only with a robot target).
- **P0 (`--ai-scene-agent`) runs after Stage 2**, because it needs the generated meshes. The random
  pair and the `--mimic-robot` caption are resolved *before* Stage 1. The diagram groups all three
  in one "how is the target chosen" row, so it is not strict execution order.
- **Target selection options are mutually exclusive:** `--pick_place_target`, `--ai-scene-agent`,
  `--mimic-robot` (and random selection when none is given; it needs at least 2 classes).
  `--pick_place_target` also requires `--classes` to contain the label.
- **`--video` cannot be combined with `--stage-0-use-stereo`.** Stereo is pick-and-place only and
  also rejects `--skip-frames`, `--max-frames` and `--vggt-viewer`; it needs exactly one left/right
  pair plus a ZED `.conf`.
- **Stage 0b always runs in pick-and-place mode** (Qwen3-VL). `--classes` conditions it; without
  `--classes` it freely discovers objects. In full-scene modes `--classes` is required and Grounding
  DINO is used.
- **The "Full-scene: Grounding DINO on --classes" node is a simplification.** Grounding DINO isn't a
  separate stage: in full-scene modes it runs inside Stage 1 (every frame, best-confidence frame per
  class as the SAM2 seed). It's drawn beside Stage 0b as the alternative box source, but the diagram's
  downstream path (P1-P4) is pick-and-place only; full-scene continues to Stages 3-12 (not drawn). In
  pick-and-place mode Grounding DINO is also loaded lazily for empty-mask re-detection (not drawn).
- **Penetration handling (P3b / 9b)** is on by default: check, then fix (+z lift), then gravity
  settle. Each part has its own `--no-penetration-*` flag.
- **`RUN_GLB = False`, so Stage 10 (chunked GLB bake) is disabled**, and `USE_TRELLIS = True`, so
  Stage 2 always runs.
- `--start-from-stage` / `--stop-after-stage` skip or truncate stages; they are not drawn.

## What the current diagram does not show

- `--pick_place_target` as an explicit option in the target-selection row.
- `--room`, the penetration flags, `--skip_isaac`, and the full-scene branch (Stages 3-12).
