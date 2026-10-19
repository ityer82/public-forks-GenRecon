#!/usr/bin/env python3
"""Builds docs/pipeline.pdf, the documentation for run_full_pipeline.py.

Usage: uv run --with reportlab python docs/build_pipeline_pdf.py
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle,
)

OUT = Path(__file__).resolve().parent / "pipeline.pdf"
ACCENT = colors.HexColor("#1f4e79")
LIGHT = colors.HexColor("#eef3f8")

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["Normal"], fontSize=9.5, leading=13, alignment=TA_LEFT, spaceAfter=5)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=17, textColor=ACCENT, spaceBefore=4, spaceAfter=8)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, textColor=ACCENT, spaceBefore=10, spaceAfter=4, keepWithNext=1)
TITLE = ParagraphStyle("title", parent=ss["Title"], fontSize=24, textColor=ACCENT, spaceAfter=6)
SUB = ParagraphStyle("sub", parent=BODY, fontSize=11, alignment=1, textColor=colors.grey)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8, leading=10, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Helvetica-Bold", textColor=colors.white)
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=12, bulletIndent=2, spaceAfter=2)
CODE = ParagraphStyle("code", fontName="Courier", fontSize=7.0, leading=8.3, backColor=LIGHT,
                      borderPadding=5, spaceBefore=4, spaceAfter=8)


def P(t, s=BODY):
    return Paragraph(t, s)


def bullets(items):
    return [Paragraph(i, BUL, bulletText="•") for i in items]


def code(text):
    return Preformatted(text.strip("\n"), CODE)


def table(header, rows, widths):
    data = [[Paragraph(h, CELLB) for h in header]] + [[Paragraph(c, CELL) for c in r] for r in rows]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c5ced8")),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return [t, Spacer(1, 6)]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(20 * mm, 10 * mm, "GenRecon – run_full_pipeline.py")
    canvas.drawRightString(A4[0] - 20 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


s = []

# ── Title ───────────────────────────────────────────────────────────────────
s += [Spacer(1, 25 * mm), P("GenRecon End-to-End Pipeline", TITLE),
      P("<font name='Courier'>run_full_pipeline.py</font> – stages, output layout and CLI reference", SUB),
      Spacer(1, 10 * mm)]
s += bullets([
    "<b>1. Overview</b> – what the pipeline does, its modes and inputs",
    "<b>2. Stages</b> – every stage, in execution order",
    "<b>3. Output folder structure</b> – <font name='Courier'>runs/&lt;scene_name&gt;/</font> for both run types",
    "<b>4. CLI arguments</b> – all flags, defaults and validation rules",
])
s.append(PageBreak())

# ── 1. Overview ─────────────────────────────────────────────────────────────
s.append(P("1. Overview", H1))
s.append(P(
    "<font name='Courier'>run_full_pipeline.py</font> turns a set of photos (or a video, or a rectified stereo pair) of a "
    "real scene into a physics-ready Isaac Sim scene and a robot demo video. VGGT-Omega predicts camera poses and depth; "
    "the objects are segmented and reconstructed in 3D, converted to USD assets, composed into a scene, checked for "
    "inter-object penetration, and finally used in a robot demo. It is a Python port of "
    "<font name='Courier'>run_full_pipeline.sh</font>: every stage except Isaac Sim runs in-process as a typed function call "
    "(<font name='Courier'>genrecon/pipeline/stages.py</font>); Isaac Sim steps run as subprocesses in their own "
    "<font name='Courier'>isaacsim_venv</font> environment."))
s.append(P("Run modes", H2))
s += table(["Mode (<font name='Courier'>--mode</font>)", "Object source", "What happens", "Final deliverable"], [
    ["<b>pick-and-place</b> (default)",
     "Stage 0b: Qwen3-VL discovers object labels + boxes (conditioned on <font name='Courier'>--classes</font> if given).",
     "A pick/place pair is chosen (random, from <font name='Courier'>--pick_place_target</font>, by the scene agent, or from a "
     "<font name='Courier'>--mimic-robot</font> video caption). Fast path: only the objects are reconstructed (Stage P1–P4).",
     "<font name='Courier'>pick_place/pick_place.mp4</font>"],
    ["<b>full-scene</b>", "Grounding DINO with the required <font name='Courier'>--classes</font>.",
     "Full GenRecon reconstruction of the whole scene (background, floor, objects) and composed USD scene.",
     "<font name='Courier'>stage_3_genrecon/shapes/glb/scene.usda</font>, <font name='Courier'>final_objects/</font>"],
    ["<b>full-scene-with-robot</b>", "As full-scene.",
     "Full-scene, then a robot-collision demo against <font name='Courier'>--robot-target</font> (random class if omitted).",
     "As full-scene + <font name='Courier'>robot_collision/robot_collide.mp4</font>"],
], [33, 42, 62, 33])
s.append(P("Inputs", H2))
s += bullets([
    "<b>Image folder</b> (default): <font name='Courier'>run_full_pipeline.py &lt;image_folder&gt; &lt;scene_name&gt;</font>.",
    "<b>Video</b> (<font name='Courier'>--video</font>): frames are sampled with ffmpeg at 5 fps, resized to 960x540 into a temporary "
    "folder (deleted afterwards) and fed to Stage 0.",
    "<b>Stereo pair</b> (<font name='Courier'>--stage-0-use-stereo</font>): FoundationStereo replaces VGGT-Omega in Stage 0 "
    "(pick-and-place only; needs one left and one right image plus a ZED .conf calibration).",
    "<b>Robot episode video</b> (<font name='Courier'>--mimic-robot</font>): a video of a real pick-and-place; Qwen3-VL captions it and the caption selects the picked object and its target.",
])
s.append(P("Pipeline flow", H2))
s.append(code("""
Common      Stage 0 -> [0b discovery] -> [0c caption] -> Stage 1 segmentation -> Stage 2 MV-SAM3D
                                                         (Stage 1-2 run only if there are classes)

pick-and-place (fast path, returns after P4)
            [P0 scene agent] -> P1 align meshes -> P2 convert assets -> P3 compose scene
                             -> P3b penetration check/fix -> P4 Franka pick-and-place video

full-scene / full-scene-with-robot
            3 reconstruct -> 4 reprojection check -> 5 object meshes -> 6 floor + friction
            -> 7 mesh->GLB -> 8 convert assets -> 9 compose scene -> 9b penetration check
            -> [10 GLB bake, disabled] -> 11 final objects -> [12 robot collision demo]
"""))
s.append(P("Runtime behaviour", H2))
s += bullets([
    "<b>Resuming:</b> <font name='Courier'>--start-from-stage N</font> skips earlier stages and reuses their outputs on disk "
    "(Stage 0b/0c reuse <font name='Courier'>discovered_*.json</font> / <font name='Courier'>caption.json</font> so the class list cannot drift); "
    "<font name='Courier'>--stop-after-stage N</font> exits cleanly after stage N.",
    "<b>Logs:</b> <font name='Courier'>pipeline.log</font> (everything, including mirrored per-stage logs) and "
    "<font name='Courier'>debug_config.log</font> (Isaac Sim subprocess configuration) in the run folder.",
    "<b>Failure:</b> any exception is logged with a traceback and the process exits with status 1.",
])
s.append(PageBreak())

# ── 2. Stages ───────────────────────────────────────────────────────────────
s.append(P("2. Stages", H1))
s.append(P("Stage names match the <font name='Courier'>with stage(...)</font> blocks in <font name='Courier'>main()</font> and the logs. "
           "“Isaac” marks steps that run as a subprocess in <font name='Courier'>isaacsim_venv</font>."))
s.append(P("2.1 Common stages (all modes)", H2))
s += table(["Stage", "What it does", "Reads", "Writes"], [
    ["<b>0</b> VGGT-Omega export",
     "Loads <font name='Courier'>vggt_omega_1b_512.pt</font>, predicts poses + depth at 512 px, gravity-aligns the scene "
     "(<font name='Courier'>--rotate-horizontal-deg</font> adds a yaw), exports a COLMAP dataset and per-frame depth/confidence "
     "(point cloud capped at 1M points, filtered by <font name='Courier'>--vggt-conf-thres</font>). "
     "With <font name='Courier'>--stage-0-use-stereo</font>, FoundationStereo is used instead.",
     "image folder / video frames / stereo pair", "<font name='Courier'>stage_0_vggt/{rgb,depth,colmap}</font>"],
    ["<b>0b</b> Object discovery<br/>(pick-and-place)",
     "Qwen3-VL-8B-Instruct returns object labels and boxes (left/right view) for the table scene; with "
     "<font name='Courier'>--classes</font> it looks for exactly those objects. Result is cached.",
     "<font name='Courier'>stage_0_vggt/rgb</font>", "<font name='Courier'>discovered_classes.json</font>, <font name='Courier'>discovered_boxes.json</font>"],
    ["<b>0c</b> Episode captioning<br/>(<font name='Courier'>--mimic-robot</font>)",
     "Samples <font name='Courier'>--caption-num-frames</font> frames from the episode video; Qwen3-VL names the picked object, the hand used "
     "and the placement location. The caption chooses the pick/place pair; if the picked object is not among the detected classes, "
     "a random pair is used.",
     "episode video, detected classes", "<font name='Courier'>caption.json</font>"],
    ["<b>1</b> Segmentation",
     "Prompted SAM2 segmentation per class (boxes from Stage 0b in pick-and-place, Grounding DINO in full-scene), lifted to 3D with the "
     "Stage 0 depth (<font name='Courier'>--depth_conf_thres</font>, <font name='Courier'>--depth_edge_rtol</font>). Masks failing the hull-consistency "
     "check are rejected. Produces per-class masks and point clouds plus a background cloud. Runs only if classes exist.",
     "<font name='Courier'>stage_0_vggt</font>, classes",
     "<font name='Courier'>stage_1_segmentation/</font> (<font name='Courier'>labels.json</font>, per-class dirs, <font name='Courier'>background/</font>)"],
    ["<b>2</b> MV-SAM3D reconstruction",
     "Multi-view SAM3D generates one textured mesh per class: shape (stage 1 steps) then texture (stage 2 steps), using the best "
     "<font name='Courier'>--mvsam3d-top-k-views</font> views, optional shape-latent alignment across views, fixed seed. Meshes are "
     "collected and previews written (aligned <font name='Courier'>*_MESH.ply</font> and scene-scaled <font name='Courier'>*_PCL.ply</font>).",
     "<font name='Courier'>stage_0_vggt</font>, <font name='Courier'>stage_1_segmentation</font>",
     "<font name='Courier'>stage_2_mv_sam3d/&lt;label&gt;/{mesh.glb,*_MESH.ply,*_PCL.ply}</font>"],
], [30, 78, 28, 34])

s.append(P("2.2 Pick-and-place path (Stage P*)", H2))
s.append(P("After Stage 2 the pipeline takes the fast path and ends after P4; Stages 3–12 are not run."))
s += table(["Stage", "What it does", "Writes"], [
    ["<b>P0</b> Scene agent<br/>(<font name='Courier'>--ai-scene-agent</font>)",
     "Interactive Ollama agent (<font name='Courier'>--scene-agent-ollama-model</font>) picks pick/place targets, approach side, clearance, "
     "gripper width, lighting and camera; passed on as extra args to P4.",
     "<font name='Courier'>pick_place/scene_spec.json</font>"],
    ["<b>P1</b> Align meshes", "Scales/aligns every class mesh to the real scene using its Stage 1 point cloud as scale reference; sanitised label directories.",
     "<font name='Courier'>pick_place/glb/&lt;label&gt;/mesh.glb</font>"],
    ["<b>P2</b> Convert assets (Isaac)", "<font name='Courier'>convert_asset.py</font> builds <font name='Courier'>asset.usd</font> per object with convexDecomposition collision.",
     "<font name='Courier'>glb/&lt;label&gt;/asset.usd</font>, <font name='Courier'>convert_asset.log</font>"],
    ["<b>P3</b> Compose scene (Isaac)", "<font name='Courier'>compose_isaac_scene.py</font> assembles the objects on a table prop (optional kitchen room via <font name='Courier'>--room</font>).",
     "<font name='Courier'>glb/scene.usda</font>, <font name='Courier'>compose_isaac_scene.log</font>"],
    ["<b>P3b</b> Penetration check (Isaac)",
     "Detects inter-object penetration through PhysX contacts (<font name='Courier'>--penetration-check</font>). With <font name='Courier'>--penetration-fix</font> "
     "it writes +z lifts that clear pairs back into <font name='Courier'>scene.usda</font> (backup <font name='Courier'>scene.pre_penetration_fix.usda</font>); "
     "with <font name='Courier'>--penetration-settle</font> dynamic objects are then settled under gravity and the settled poses are written back.",
     "<font name='Courier'>penetration.json</font>, <font name='Courier'>penetration.log</font>"],
    ["<b>P4</b> Franka pick-and-place (Isaac)", "<font name='Courier'>demo_franka_pickplace.py</font>: Franka picks the target and places it on the place object (with clearance) or at <font name='Courier'>--place-offset</font>.",
     "<font name='Courier'>pick_place.mp4</font>, <font name='Courier'>pick_place_scene.usda</font>, <font name='Courier'>pick_place.log</font>"],
], [34, 100, 36])

s.append(P("2.3 Full-scene path (Stages 3–12)", H2))
s += table(["Stage", "What it does", "Writes"], [
    ["<b>3</b> Reconstruct scene", "<font name='Courier'>reconstruct_scene.py</font>: GenRecon chunked scene reconstruction from the Stage 0 views "
     "(<font name='Courier'>--num_imgs_per_scene</font>, <font name='Courier'>--fix_num_chunks</font>, <font name='Courier'>--max_chunks_per_group</font>, "
     "<font name='Courier'>--max_inflated_voxels</font>), using the Stage 1 masks when classes exist.",
     "<font name='Courier'>stage_3_genrecon/mesh.ply</font>, chunk files, <font name='Courier'>cameras.json</font>"],
    ["<b>4</b> Reprojection validation", "Renders synthetic views of the reconstruction at the input cameras for visual comparison.",
     "<font name='Courier'>stage_4_reproject_synth/{synth_views,compare_views}</font>"],
    ["<b>5</b> Object mesh extraction", "Cascading per-object extraction from the scene mesh using each object's point cloud and masks; remaining geometry becomes the "
     "background; floating fragments near each object are removed.",
     "<font name='Courier'>shapes/&lt;label&gt;_mesh.ply</font>, <font name='Courier'>_floaters.ply</font>, <font name='Courier'>background_mesh.ply</font>"],
    ["<b>6</b> Floor + friction", "Segments the floor out of the background; infers per-object friction from a material table "
     "(<font name='Courier'>--friction-table-path</font>) using an Ollama model (<font name='Courier'>--ollama-model</font>).",
     "<font name='Courier'>shapes/floor_mesh.ply</font>, <font name='Courier'>friction_assignments.json</font>"],
    ["<b>7</b> Mesh to GLB", "<font name='Courier'>mesh_to_glb.py</font> converts meshes to GLB and substitutes each object with its aligned MV-SAM3D mesh.",
     "<font name='Courier'>shapes/glb/&lt;label&gt;/mesh.glb</font>"],
    ["<b>8</b> Convert assets (Isaac)", "As P2, plus friction table with max combine mode.", "<font name='Courier'>shapes/glb/&lt;label&gt;/asset.usd</font>"],
    ["<b>9</b> Compose scene (Isaac)", "Composes background, floor and objects (no table prop; optional <font name='Courier'>--room</font>).", "<font name='Courier'>shapes/glb/scene.usda</font>"],
    ["<b>9b</b> Penetration check (Isaac)", "As P3b, on the full scene.", "<font name='Courier'>stage_3_genrecon/penetration.json</font>"],
    ["<b>10</b> Chunked GLB bake", "<font name='Courier'>chunked_to_glb.py</font> bakes a textured <font name='Courier'>scene.glb</font>. Currently disabled in code (<font name='Courier'>RUN_GLB = False</font>).", "<font name='Courier'>scene.glb</font> (when enabled)"],
    ["<b>11</b> Final objects", "Collects per-object deliverables: mesh, MV-SAM3D GLB and detection image.", "<font name='Courier'>final_objects/&lt;label&gt;/</font>, <font name='Courier'>background_mesh.ply</font>"],
    ["<b>12</b> Robot collision (Isaac)", "<font name='Courier'>demo_robot_collide.py</font>: the robot is driven into <font name='Courier'>--robot-target</font>. Needs Isaac (not with <font name='Courier'>--skip_isaac</font>).",
     "<font name='Courier'>robot_collision/robot_collide.mp4</font>, <font name='Courier'>.usda</font>, <font name='Courier'>.log</font>"],
], [34, 100, 36])
s.append(PageBreak())

# ── 3. Output structure ─────────────────────────────────────────────────────
s.append(P("3. Output folder structure", H1))
s.append(P("Everything is written to <font name='Courier'>runs/&lt;scene_name&gt;/</font>. Labels containing spaces are kept raw in "
           "<font name='Courier'>stage_2_mv_sam3d/</font> and <font name='Courier'>final_objects/</font> but sanitised (space/slash to underscore) in "
           "<font name='Courier'>stage_1_segmentation/</font>, <font name='Courier'>pick_place/glb/</font> and <font name='Courier'>shapes/</font>."))
s.append(P("3.1 Full-scene-with-robot (example: runs/kitchen_floor)", H2))
s.append(code("""
runs/kitchen_floor/
|-- pipeline.log                    full pipeline log
|-- debug_config.log                Isaac Sim subprocess config
|-- stage_0_vggt/
|   |-- rgb/                        input frames
|   |-- depth/                      frame_*_depth.npy, frame_*_depth_conf.npy
|   `-- colmap/                     cameras.txt, images.txt, points3D.{txt,ply}
|-- stage_1_segmentation/
|   |-- labels.json  hull_consistency_rejections.json  segmentation.log
|   |-- <label>/                    banana, bowl, plastic_cup, white_chair
|   |   |-- mask_bin/ mask_overlay/ mask_proj/    binary masks, overlays, projections
|   |   |-- mask_bin_rejected/ mask_overlay_rejected/   (only if some were rejected)
|   |   `-- point_cloud/<label>.ply
|   `-- background/point_cloud/
|-- stage_2_mv_sam3d/
|   |-- mv_sam3d.log
|   |-- input/                      images, per-class masks, da3_output.npz
|   `-- <raw label>/                mesh.glb, <label>_MESH.ply, <label>_PCL.ply
|-- stage_3_genrecon/
|   |-- mesh.ply  clean_points.ply  object_source_mesh.ply  cameras.json  args.json
|   |-- chunk_inputs.pt  to_glb_inputs.pt  chunk_transforms.json  coords_000..015.ply
|   |-- penetration.json  penetration.log  compose_isaac_scene.log  convert_asset.log
|   `-- shapes/
|       |-- <label>_mesh.ply  <label>_floaters.ply
|       |-- background_mesh.ply  floor_mesh.ply  friction_assignments.json
|       `-- glb/
|           |-- scene.usda  scene.pre_penetration_fix.usda
|           `-- <label>/ background/ floor/    asset.usd, manifest.json, mesh.glb, textures/
|-- stage_4_reproject_synth/
|   |-- synth_views/                re-rendered views
|   `-- compare_views/              input vs. synthetic comparisons
|-- final_objects/
|   |-- background_mesh.ply
|   `-- <raw label>/                mesh.ply, trellis_mesh.glb, detection.png
`-- robot_collision/
    `-- robot_collide.mp4  robot_collide_scene.usda  robot_collide.log
"""))
s.append(P("3.2 Pick-and-place (example: runs/kitchen_plates)", H2))
s.append(code("""
runs/kitchen_plates/
|-- pipeline.log  debug_config.log
|-- discovered_classes.json         Stage 0b labels (cached for --start-from-stage reruns)
|-- discovered_boxes.json           Stage 0b boxes per label (left/right view)
|-- caption.json                    only with --mimic-robot
|-- stage_0_vggt/                   rgb/  depth/  colmap/            (as above)
|-- stage_1_segmentation/           labels.json, <label>/ ...        (as above)
|   `-- white_bowl  white_plate_1  white_plate_2  silver_knife_1  silver_knife_2  glass_dish
|-- stage_2_mv_sam3d/               <raw label>/ ("white plate 1", "glass dish", ...)
`-- pick_place/
    |-- scene_spec.json             only with --ai-scene-agent
    |-- glb/
    |   |-- scene.usda  scene.pre_penetration_fix.usda
    |   `-- <label>/                asset.usd, manifest.json, mesh.glb, textures/
    |-- pick_place_scene.usda       scene as left by the demo
    |-- pick_place.mp4              final deliverable
    |-- penetration.json
    `-- pick_place.log  compose_isaac_scene.log  convert_asset.log  penetration.log
"""))
s.append(P("Key files", H2))
s += table(["File", "Content"], [
    ["<font name='Courier'>labels.json</font>", "Label to sanitised-name mapping for all segmented classes."],
    ["<font name='Courier'>hull_consistency_rejections.json</font>", "Per-class masks dropped by the 3D hull-consistency check."],
    ["<font name='Courier'>penetration.json</font>", "Detected penetrating pairs and the lifts/settling applied."],
    ["<font name='Courier'>friction_assignments.json</font>", "Per-object friction material (full-scene only)."],
    ["<font name='Courier'>scene.usda</font> / <font name='Courier'>scene.pre_penetration_fix.usda</font>", "Composed Isaac scene and its backup taken before the penetration fix."],
], [70, 100])
s.append(PageBreak())

# ── 4. CLI ──────────────────────────────────────────────────────────────────
s.append(P("4. CLI arguments", H1))
s.append(code("""
uv run python run_full_pipeline.py <image_folder> <scene_name> [options]
uv run python run_full_pipeline.py <video_file>   <scene_name> --video [options]
uv run python run_full_pipeline.py <scene_images> <scene_name> --mimic-robot episode.mp4

# pick-and-place with explicit targets
uv run python run_full_pipeline.py /path/to/images food2_vggt --classes "banana,bowl" \\
    --mode pick-and-place --pick_place_target banana --place-target bowl
# full scene with robot
uv run python run_full_pipeline.py /path/to/images kitchen_floor --mode full-scene-with-robot \\
    --classes "banana,bowl,plastic cup,white chair"
"""))
W = [56, 24, 90]


def c(t):
    return f"<font name='Courier'>{t}</font>"


def section(title, rows):
    data = [[Paragraph(h, CELLB) for h in ("Flag", "Default", "Description")]] + [[Paragraph(x, CELL) for x in r] for r in rows]
    t = Table(data, colWidths=[w * mm for w in W], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c5ced8")),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    s.append(P(title, H2))
    s.append(t)


section("4.1 Positional arguments and input", [
    [c("image_folder"), "required", "Folder of images, or a video file with <b>--video</b>."],
    [c("scene_name"), "required", "Name of the run; output goes to <b>runs/&lt;scene_name&gt;/</b>."],
    [c("--video"), "off", "Treat the first argument as a video: 5 fps, 960x540 frames in a temp folder. Not with <b>--stage-0-use-stereo</b>."],
    [c("--num_imgs_per_scene"), "32", "Number of views used by Stage 3 reconstruction."],
    [c("--skip-frames"), "-1", "Frame skipping passed to VGGT-Omega (-1 = none)."],
    [c("--max-frames"), "-1", "Use only the first N images (sorted by name); applied before <b>--skip-frames</b>."],
    [c("--rotate-horizontal-deg"), "0.0", "Extra yaw applied after gravity alignment in Stage 0."],
    [c("--vggt-conf-thres"), "20.0", "Confidence threshold for Stage 0's exported point cloud."],
    [c("--vggt-viewer"), "off", "Spawn the rerun point-cloud viewer after Stage 0."],
    [c("--stage-0-use-stereo"), "off", "FoundationStereo instead of VGGT-Omega; needs exactly one left and one right image. Pick-and-place only; not with <b>--skip-frames</b>, <b>--max-frames</b>, <b>--vggt-viewer</b>."],
    [c("--stereo-conf"), "*.conf in folder", "ZED calibration file; requires <b>--stage-0-use-stereo</b>."],
])
section("4.2 Mode and classes", [
    [c("--mode"), "pick-and-place", "<b>pick-and-place</b>, <b>full-scene</b> or <b>full-scene-with-robot</b>."],
    [c("--classes"), "none", "Comma-separated labels (\"banana,bowl\"). Required for both full-scene modes; in pick-and-place it conditions Stage 0b discovery."],
    [c("--detection-box-padding-frac"), "0.05", "Pick-and-place: outward padding of each Stage 0b box before prompting SAM2."],
    [c("--depth_conf_thres"), "50.0", "Stage 1: depth confidence percentile filter."],
    [c("--depth_edge_rtol"), "0.03", "Stage 1: relative tolerance for removing depth-edge points."],
])
section("4.3 MV-SAM3D (Stage 2)", [
    [c("--mvsam3d-stage1-steps"), "25", "Shape inference steps (upstream default 50)."],
    [c("--mvsam3d-stage2-steps"), "12", "Texture inference steps (upstream default 25)."],
    [c("--mvsam3d-top-k-views"), "5", "Keep the k views that best cover each object; 0 or ≥ view count disables pruning."],
    [c("--mvsam3d-seed"), "42", "Diffusion random seed."],
    [c("--[no-]mvsam3d-align-shape-latents"), "on", "Align each view's canonical frame before fusing to avoid crossed slabs."],
])
section("4.4 Reconstruction (Stages 3–12, full-scene)", [
    [c("--fix_num_chunks"), "16", "Number of chunks for scene reconstruction."],
    [c("--max_chunks_per_group"), "none", "Limit chunks per group."],
    [c("--max_inflated_voxels"), "none", "Limit inflated voxels."],
    [c("--friction-table-path"), "configs/materials/friction_table.example.yaml", "Material friction table used in Stage 6."],
    [c("--ollama-model"), "llama3.1:8b", "Ollama model for friction inference."],
    [c("--robot-target"), "random class", "Class hit by the robot in Stage 12. Only with <b>full-scene-with-robot</b>; not with <b>--pick_place_target</b>, <b>--ai-scene-agent</b>."],
])
section("4.5 Stage control and Isaac Sim", [
    [c("--start-from-stage"), "0", "Skip stages below N, reusing outputs on disk."],
    [c("--stop-after-stage"), "999", "Exit after stage N."],
    [c("--skip_isaac"), "off", "Skip Stages 7–9b (USD conversion/composition). Not with <b>full-scene-with-robot</b>."],
    [c("--[no-]penetration-check"), "on", "Detect inter-object penetration after composing (about 10 s)."],
    [c("--[no-]penetration-fix"), "on", "Write lifts that clear their pair back into scene.usda."],
    [c("--[no-]penetration-settle"), "on", "Settle dynamic objects under gravity after lifting; off = lift-only (capped at 20 mm)."],
    [c("--room"), "off", "Wrap the table in a static kitchen room backdrop."],
    [c("--room-asset-dir"), "assets/room", "Override the room asset directory."],
])
section("4.6 Pick-and-place targets", [
    [c("--pick_place_target"), "random", "Object to pick; must exactly match a <b>--classes</b> label. Pick-and-place only. Without it (and without <b>--ai-scene-agent</b>) a random pair is chosen, which needs at least 2 classes."],
    [c("--place-target"), "random", "Object to place onto; needs <b>--pick_place_target</b>, differs from it, and is in <b>--classes</b>."],
    [c("--place-offset"), "0.3,0.0,0.0", "DX,DY,DZ (m) placement when there is no place target."],
    [c("--place-target-clearance"), "0.05", "Clearance above the place target (m)."],
    [c("--gripper-open-width"), "0.06", "Gripper opening (m)."],
    [c("--approach-side"), "neg-y", "<b>neg-x</b>, <b>pos-x</b>, <b>neg-y</b> or <b>pos-y</b>."],
    [c("--ai-scene-agent"), "off", "Stage P0 agent chooses targets and scene settings. Not with <b>--pick_place_target</b>, <b>--robot-target</b>."],
    [c("--scene-agent-ollama-model"), "qwen2.5:7b", "Ollama model for the scene agent."],
    [c("--mimic-robot VIDEO"), "none", "Episode video; its Qwen3-VL caption selects pick/place. Not with <b>--pick_place_target</b> or <b>--ai-scene-agent</b>."],
    [c("--caption-num-frames"), "8", "Frames sampled from the video for captioning (≥ 1)."],
])
s.append(P("Validation rules", H2))
s += bullets([
    "Full-scene modes require <b>--classes</b> and reject <b>--pick_place_target</b>, <b>--place-target</b> and <b>--ai-scene-agent</b>.",
    "<b>--robot-target</b> is only valid with <b>--mode full-scene-with-robot</b>; <b>full-scene-with-robot</b> cannot be combined with <b>--skip_isaac</b>.",
    "<b>--mimic-robot</b> and <b>--stage-0-use-stereo</b> are pick-and-place only; <b>--stereo-conf</b> requires <b>--stage-0-use-stereo</b>.",
])

SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm,
                  bottomMargin=18 * mm, title="GenRecon End-to-End Pipeline",
                  author="GenRecon").build(s, onFirstPage=footer, onLaterPages=footer)
print(f"Wrote {OUT}")
