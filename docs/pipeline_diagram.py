"""Render the pick-and-place pipeline block diagram (run_full_pipeline.py) as a plain SVG.

Usage:
    python3 docs/pipeline_diagram.py [out.svg]     # default: docs/pipeline_pick_and_place_stage0b_options.svg

No dependencies. Labels are plain <text> elements (no HTML foreignObject), so the file renders
the same in browsers, image viewers and Inkscape. Edit `rows_spec` to add or change variations;
see docs/visualize_pipeline.md for the pipeline facts the diagram encodes and the rendering pitfalls.
"""
import sys
from pathlib import Path
from xml.sax.saxutils import escape

FS, LH, PADX, PADY, GAP, CW = 15, 21, 28, 14, 40, 8.6
W = 900
cx = W / 2
OFF = 195
XSTEREO, XVGGT = cx - OFF, cx + OFF
XVID = XVGGT + 70      # video column sits over the right half of the VGGT box
XJOIN = XVGGT - 80     # where the image-folder edge lands on the VGGT box

# row = list of (kind, lines, centre_x | None); a one-cell row with None is centred
rows_spec = [
    [("io",   ["Image folder + scene_name"], XSTEREO), ("io", ["Video file + scene_name"], XVID)],
    [("step", ["Extract video frames"], XVID)],
    [("step", ["Stage 0: FoundationStereo export"], XSTEREO), ("step", ["Stage 0: VGGT-Omega export"], XVGGT)],
    [("step", ["Stage 0b: Qwen3-VL", "free discovery"], None), ("step", ["Stage 0b: Qwen3-VL", "on --classes"], None), ("step", ["Full-scene: Grounding", "DINO on --classes"], None)],
    [("step", ["Random pick/place pair"], None), ("step", ["Stage 0c: --mimic-robot caption"], None), ("step", ["P0: --ai-scene-agent"], None)],
    [("step", ["Stage 1: SAM2 2D seg & tracking"], None)],
    [("step", ["Stage 1.5: 3D segmentation"], None)],
    [("step", ["Stage 2: MV-SAM3D reconstruction"], None)],
    [("step", ["P1: align meshes to scene scale"], None)],
    [("step", ["P2: convert_asset.py"], None)],
    [("step", ["P3: compose_isaac_scene.py"], None)],
    [("step", ["P3b: penetration check"], None)],
    [("step", ["P4: demo_franka_pickplace.py"], None)],
    [("io",   ["pick_place.mp4"], None)],
]

def size(kind, lines):
    return max(len(l) for l in lines) * CW + 2 * PADX, len(lines) * LH + 2 * PADY

rows, y = [], 20
for spec in rows_spec:
    ws = [size(k, ls)[0] for k, ls, _ in spec]
    h = max(size(k, ls)[1] for k, ls, _ in spec)
    if all(c is not None for _, _, c in spec):
        centres = [c for _, _, c in spec]
    elif len(spec) == 1:
        centres = [cx]
    else:
        tot = sum(ws) + 40 * (len(ws) - 1)
        centres, xx = [], cx - tot / 2
        for w_ in ws:
            centres.append(xx + w_ / 2)
            xx += w_ + 40
    row = [(k, ls, c - w / 2, y, w, h, c) for (k, ls, _), c, w in zip(spec, centres, ws)]
    rows.append(row)
    y += h + GAP
TH = y - GAP + 20

o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{TH:.0f}" viewBox="0 0 {W} {TH:.0f}" font-family="DejaVu Sans, Arial, Helvetica, sans-serif" font-size="{FS}">',
     '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker></defs>',
     f'<rect width="{W}" height="{TH:.0f}" fill="white"/>']

def line(pts, arrow=False):
    p = " ".join(f"{a:.1f},{b:.1f}" for a, b in pts)
    m = ' marker-end="url(#a)"' if arrow else ""
    o.append(f'<polyline points="{p}" fill="none" stroke="#333" stroke-width="1.5"{m}/>')

def bot(r): return r[0][3] + r[0][5]
def top(r): return r[0][3]

# --- hand-routed input section (rows 0-2) ---
img, vid = rows[0]
frames = rows[1][0]
stereo, vggt = rows[2]
line([(vid[6], bot(rows[0])), (frames[6], top(rows[1]) - 1)], True)           # video -> frames
line([(frames[6], bot(rows[1])), (XVID, top(rows[2]) - 1)], True)  # frames -> VGGT
line([(img[6], bot(rows[0])), (stereo[6], top(rows[2]) - 1)], True)           # image -> stereo
ybr = bot(rows[1]) + GAP / 2
line([(img[6], ybr), (XJOIN, ybr), (XJOIN, top(rows[2]) - 1)], True)          # image -> VGGT
o.append(f'<circle cx="{img[6]:.1f}" cy="{ybr:.1f}" r="3.5" fill="#333"/>')

# --- automatic connectors for rows 3.. ---
for i in range(3, len(rows)):
    prev, cur = rows[i - 1], rows[i]
    ybot, ytop = bot(prev), top(cur)
    mid = (ybot + ytop) / 2
    if len(prev) == 1 and len(cur) == 1:
        line([(prev[0][6], ybot), (cur[0][6], ytop - 1)], True)
        continue
    xs = [q[6] for q in prev + cur]
    line([(min(xs), mid), (max(xs), mid)])
    for p in prev:
        line([(p[6], p[3] + p[5]), (p[6], mid)])
    for c in cur:
        line([(c[6], mid), (c[6], ytop - 1)], True)

# --- nodes ---
for row in rows:
    for k, ls, x, y0, w, h, c in row:
        if k == "io":
            s = 14
            o.append(f'<polygon points="{x+s:.1f},{y0:.1f} {x+w:.1f},{y0:.1f} {x+w-s:.1f},{y0+h:.1f} {x:.1f},{y0+h:.1f}" fill="#eef3ff" stroke="#333" stroke-width="1.5"/>')
        else:
            o.append(f'<rect x="{x:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}" rx="4" fill="#f4f4f4" stroke="#333" stroke-width="1.5"/>')
        for j, t in enumerate(ls):
            ty = y0 + PADY + LH * j + FS * 0.85
            bold = ' font-weight="bold"' if k == "step" else ""
            o.append(f'<text x="{c:.1f}" y="{ty:.1f}" text-anchor="middle" fill="#111"{bold}>{escape(t)}</text>')
o.append("</svg>")
out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("pipeline_pick_and_place_stage0b_options.svg")
out_path.write_text("\n".join(o))
