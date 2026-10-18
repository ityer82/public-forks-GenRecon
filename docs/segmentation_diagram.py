"""Render the Stage 1 / 1.5 segmentation block diagram (segmentation/main_light.py) as a plain SVG.

Usage:
    python3 docs/segmentation_diagram.py [out.svg]     # default: docs/segmentation_pipeline.svg

No dependencies. Same conventions as docs/pipeline_diagram.py (plain <text>, no foreignObject).
Main flow is the centre column; each row may carry a dashed `note` on the right holding the
condition / threshold that gates it. Edit `rows_spec` to change the diagram.
"""
import sys
from pathlib import Path
from xml.sax.saxutils import escape

FS, LH, PADX, PADY, GAP, CW = 14, 20, 24, 12, 38, 8.0
XMAIN, XNOTE0 = 400, 760   # centre of the main column; left edge of the notes
W = 1340

# kind: io | step | cond (diamond-ish hexagon: a decision) | alt (parallel alternatives, `lines` is a list of cells)
# row = dict(kind, lines, note=[...] | None, band=str | None)
rows_spec = [
    dict(kind="io", lines=["Stage 0 export", "rgb/  depth/ + depth_conf  camera poses  sparse points"]),
    dict(kind="alt", cells=[
        ["Box source A (pick-and-place)", "Qwen3-VL boxes on 1 reference image", "padded +5% (--detection-box-padding-frac)"],
        ["Box source B (full-scene)", "Grounding DINO on every frame", "best-confidence frame per class"],
    ], band="Stage 1: per-object 2D masks (detect_and_segment.py)"),
    dict(kind="step", lines=["Seed SAM2 video predictor", "1 obj_id per object, box prompt on its seed frame"],
         note=["Same class may be seeded more than once", "(different frames) -> several obj_ids per class"]),
    dict(kind="step", lines=["Propagate through all frames", "forward from frame 0, backward from last frame"],
         note=["Seeded from several frames -> always cover the", "full video in both directions"]),
    dict(kind="cond", lines=["Object mask empty in a frame?"],
         note=["yes -> re-detect with Grounding DINO ONLY IF the object's 3D",
               "frustum still reprojects to >= 2% of the frame (--reproj_area_threshold)",
               "otherwise: assume out of view, leave empty",
               "no COLMAP model / too few points for a frustum ->",
               "re-detect unconditionally. Only missing obj_ids are re-seeded."]),
    dict(kind="io", lines=["Per-object masks", "<label>/mask_bin/<frame>.png  +  labels.json"]),
    dict(kind="step", lines=["Lift mask pixels to 3D", "VGGT depth + pose, per frame, per object"],
         note=["Pixel dropped if it:", "- is claimed by >1 class in that frame (ambiguous)",
               "- sits on a depth edge (rel. jump > 0.03, --depth_edge_rtol)",
               "- has depth_conf below the percentile of candidate pixels (--depth_conf_thres)",
               "Then concat all frames, voxel dedup 5 mm (one real point per cell)"],
         band="Stage 1.5: lifting to a per-object 3D point cloud (segment_pointcloud.py)"),
    dict(kind="cond", lines=["Hull consensus (RANSAC over frames)"],
         note=["Each frame's 3D convex hull is a seed; reproject into all other frames.",
               "Inlier if >= 50% of that frame's mask falls inside the silhouette",
               "(--hull_overlap_threshold). Best seed's inliers fuse into one hull;",
               "every frame is re-scored. Fail -> frame rejected: its points dropped,",
               "masks quarantined + rejection manifest. Frames w/o a detection skipped."]),
    dict(kind="step", lines=["Per-point cross-frame validation", "reproject each point into every camera (z-buffer)"],
         note=["Reject if fg vote ratio <= 0.6 (--point_consistency_ratio_thres)",
               "Points seen with a detection in < 3 frames are kept unconditionally",
               "(--point_consistency_min_frames: too little evidence)"]),
    dict(kind="step", lines=["Background assignment", "sparse points > 2 cm from every object point"],
         note=["--bg_assign_radius 0.02, KD-tree nearest neighbour"]),
    dict(kind="io", lines=["Per-object clouds", "<label>/point_cloud/<label>.ply  +  background.ply"]),
    dict(kind="alt", cells=[
        ["Stage 2: MV-SAM3D", "per-view masks + view selection"],
        ["Stage P1 / 5", "mesh-to-scene alignment"],
    ]),
]


def size(lines):
    return max(len(l) for l in lines) * CW + 2 * PADX, len(lines) * LH + 2 * PADY


def note_size(note):
    return max(len(l) for l in note) * 7.0 + 24, len(note) * 18 + 14


o = []


def text(x, y, t, bold=False, anchor="middle", size_=FS, fill="#111", italic=False):
    st = ' font-weight="bold"' if bold else ""
    st += ' font-style="italic"' if italic else ""
    o.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size_}" fill="{fill}"{st}>{escape(t)}</text>')


def line(pts, arrow=False, dash=False, color="#333"):
    p = " ".join(f"{a:.1f},{b:.1f}" for a, b in pts)
    m = ' marker-end="url(#a)"' if arrow else ""
    d = ' stroke-dasharray="5,4"' if dash else ""
    o.append(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="1.5"{d}{m}/>')


# --- layout pass ---
layout, y = [], 50
for spec in rows_spec:
    if spec.get("band"):
        y += 34   # room for the band title above the first row
    if spec["kind"] == "alt":
        ws_hs = [size(c) for c in spec["cells"]]
        h = max(h_ for _, h_ in ws_hs)
        tot = sum(w_ for w_, _ in ws_hs) + 30 * (len(ws_hs) - 1)
        xs, xx = [], XMAIN - tot / 2
        for w_, _ in ws_hs:
            xs.append(xx + w_ / 2)
            xx += w_ + 30
        w = tot
    else:
        w, h = size(spec["lines"])
        xs = [XMAIN]
    nh = note_size(spec["note"])[1] if spec.get("note") else 0
    rh = max(h, nh)
    layout.append((spec, y + (rh - h) / 2, h, rh, w, xs, ws_hs if spec["kind"] == "alt" else None))  # node centred in its row
    y += rh + GAP
TH = y - GAP + 30

o.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{TH:.0f}" viewBox="0 0 {W} {TH:.0f}" '
         f'font-family="DejaVu Sans, Arial, Helvetica, sans-serif" font-size="{FS}">')
o.append('<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto">'
         '<path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker></defs>')
o.append(f'<rect width="{W}" height="{TH:.0f}" fill="white"/>')

# --- bands (drawn first, behind everything) ---
band_starts = [(i, l[0]["band"]) for i, l in enumerate(layout) if l[0].get("band")]
for n, (i, title) in enumerate(band_starts):
    top = layout[i][1] - (layout[i][3] - layout[i][2]) / 2 - 40
    end_i = band_starts[n + 1][0] if n + 1 < len(band_starts) else len(layout) - 1
    last = layout[end_i - 1]
    bot = last[1] + last[2] + (last[3] - last[2]) / 2 + 14
    if n + 1 < len(band_starts):
        bot = layout[end_i][1] - (layout[end_i][3] - layout[end_i][2]) / 2 - 40 - 12
    o.append(f'<rect x="14" y="{top:.1f}" width="{W - 28}" height="{bot - top:.1f}" rx="8" fill="#fafafa" stroke="#bbb" stroke-width="1" stroke-dasharray="2,3"/>')
    text(W - 28, top + 18, title, bold=True, anchor="end", size_=13, fill="#555")

# --- connectors ---
for i in range(1, len(layout)):
    ps, ys, hs, rhs, ws_, xss, _ = layout[i - 1]
    cs, yc, hc, rhc, wc, xsc, _ = layout[i]
    ybot, ytop = ys + hs, yc
    mid = (ybot + ytop) / 2
    if len(xss) == 1 and len(xsc) == 1:
        line([(XMAIN, ybot), (XMAIN, ytop - 1)], True)
        continue
    allx = xss + xsc
    line([(min(allx), mid), (max(allx), mid)])
    for x in xss:
        line([(x, ybot), (x, mid)])
    for x in xsc:
        line([(x, mid), (x, ytop - 1)], True)

# --- nodes + notes ---
for spec, y0, h, rh, w, xs, ws_hs in layout:
    kind = spec["kind"]
    cells = spec["cells"] if kind == "alt" else [spec["lines"]]
    for idx, (ls, cx_) in enumerate(zip(cells, xs)):
        wi = ws_hs[idx][0] if ws_hs else w
        x = cx_ - wi / 2
        if kind == "io":
            s = 14
            o.append(f'<polygon points="{x+s:.1f},{y0:.1f} {x+wi:.1f},{y0:.1f} {x+wi-s:.1f},{y0+h:.1f} {x:.1f},{y0+h:.1f}" fill="#eef3ff" stroke="#333" stroke-width="1.5"/>')
        elif kind == "cond":
            s = 22
            o.append(f'<polygon points="{x+s:.1f},{y0:.1f} {x+wi-s:.1f},{y0:.1f} {x+wi:.1f},{y0+h/2:.1f} {x+wi-s:.1f},{y0+h:.1f} {x+s:.1f},{y0+h:.1f} {x:.1f},{y0+h/2:.1f}" fill="#fff4de" stroke="#333" stroke-width="1.5"/>')
        else:
            o.append(f'<rect x="{x:.1f}" y="{y0:.1f}" width="{wi:.1f}" height="{h:.1f}" rx="4" fill="#f4f4f4" stroke="#333" stroke-width="1.5"/>')
        for j, t in enumerate(ls):
            ty = y0 + PADY + LH * j + FS * 0.85
            text(cx_, ty, t, bold=(j == 0 and kind != "io") or (j == 0 and kind == "io"), fill="#111" if j == 0 else "#333")
    if spec.get("note"):
        nw, nh = note_size(spec["note"])
        nx = XNOTE0
        ny = y0 + h / 2 - nh / 2
        line([(xs[-1] + (ws_hs[-1][0] if ws_hs else w) / 2, y0 + h / 2), (nx, y0 + h / 2)], dash=True)
        o.append(f'<rect x="{nx:.1f}" y="{ny:.1f}" width="{nw:.1f}" height="{nh:.1f}" rx="4" fill="#fffdf4" stroke="#999" stroke-width="1" stroke-dasharray="5,4"/>')
        for j, t in enumerate(spec["note"]):
            text(nx + 12, ny + 20 + 18 * j, t, anchor="start", size_=12, fill="#444")

o.append("</svg>")
out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("segmentation_pipeline.svg")
out_path.write_text("\n".join(o))
print(f"wrote {out_path}")
