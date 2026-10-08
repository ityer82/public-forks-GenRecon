"""Run VGGT-Omega on frames from the Pick_And_Place 4-camera capture.

The capture folder holds one image folder per camera (left/right head, left/right hand),
each with frames named frame_XXXXXX.png. For every selected frame, the images of all
selected cameras are fed to VGGT-Omega together as one multi-view batch, and the result is
exported as a COLMAP-text dataset (+ depth maps, + points3D.ply).

Usage:
    uv run python scripts/run_vggt_pick_and_place.py                          # 1 frame, all 4 cameras
    uv run python scripts/run_vggt_pick_and_place.py --num_frames 5 --cameras left_head,right_head
    uv run python scripts/run_vggt_pick_and_place.py --start_frame 10 --frame_step 5 --num_frames 4 --viz
    uv run python scripts/run_vggt_pick_and_place.py --crop_hands                     # hand cams: middle half, rotated 90 deg cw
    uv run python scripts/run_vggt_pick_and_place.py --caption --caption_num_frames 8  # no VGGT: Qwen-VL describes the pick and place
    uv run python scripts/run_vggt_pick_and_place.py --caption --track                # ...then segment+track the picked object (default camera: left_head)
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from PIL import Image

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR / "vggt"))
sys.path.insert(0, str(REPO_DIR))

CAMERA_DIRS = {
    "left_head": "left_head_image",
    "right_head": "right_head_image",
    "left_hand": "left_hand_image",
    "right_hand": "right_hand_image",
}
HAND_CAMERAS = {"left_hand", "right_hand"}
DEFAULT_DATA_DIR = REPO_DIR.parent / "fisheye" / "data" / "Pick_And_Place_T1_1" / "frames"
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def parse_cameras(value: str) -> list[str]:
    cameras = [c.strip() for c in value.split(",") if c.strip()]
    unknown = [c for c in cameras if c not in CAMERA_DIRS]
    if unknown or not cameras:
        raise argparse.ArgumentTypeError(f"unknown cameras {unknown}; choose from {','.join(CAMERA_DIRS)}")
    return list(dict.fromkeys(cameras))


def list_frames(camera_dir: Path) -> dict[str, Path]:
    if not camera_dir.is_dir():
        raise FileNotFoundError(f"Camera folder not found: {camera_dir}")
    return {p.stem: p for p in sorted(camera_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS}


def crop_and_rotate(src: Path, dst: Path) -> None:
    """Keep the horizontal middle half (x in [W/4, 3W/4), full height) and rotate it 90 deg clockwise."""
    with Image.open(src) as img:
        w, h = img.size
        img.crop((w // 4, 0, 3 * w // 4, h)).transpose(Image.Transpose.ROTATE_270).save(dst)


def stage_inputs(data_dir: Path, cameras: list[str], frame_ids: list[str], staging_dir: Path, crop_hands: bool = False) -> None:
    """Copy the selected images into one flat folder as <frame>_<camera><ext> (frame-major order).

    With crop_hands, hand-camera images are cropped and rotated (see crop_and_rotate) instead of copied.
    """
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    per_camera = {cam: list_frames(data_dir / CAMERA_DIRS[cam]) for cam in cameras}
    for frame_id in frame_ids:
        for cam in cameras:
            src = per_camera[cam][frame_id]
            dst = staging_dir / f"{frame_id}_{cam}{src.suffix.lower()}"
            if crop_hands and cam in HAND_CAMERAS:
                crop_and_rotate(src, dst)
            else:
                shutil.copy2(src, dst)


def select_frames(data_dir: Path, cameras: list[str], start: int, step: int, count: int) -> list[str]:
    """Frame ids present in every selected camera, sliced as [start::step][:count]."""
    common = None
    for cam in cameras:
        ids = set(list_frames(data_dir / CAMERA_DIRS[cam]))
        common = ids if common is None else common & ids
    ordered = sorted(common)
    if not ordered:
        raise FileNotFoundError(f"No frames shared by cameras {cameras} in {data_dir}")
    selected = ordered[start::step][:count]
    if len(selected) < count:
        print(f"Warning: requested {count} frames but only {len(selected)} available from index {start} (step {step}).")
    if not selected:
        raise ValueError(f"No frames selected (start_frame={start}, {len(ordered)} common frames).")
    return selected


def run_caption(args, data_dir: Path, output_dir: Path) -> None:
    """Show evenly spaced frames of one camera to the Qwen VLM and save its description of the pick and place."""
    from genrecon.utils.object_discovery import discover_objects
    from genrecon.utils.pick_place_caption import caption_pick_and_place, sample_frames
    from run_full_pipeline import DISCOVERY_HF_MODEL

    frames = list(list_frames(data_dir / CAMERA_DIRS[args.caption_camera]).values())
    paths = sample_frames(frames, args.caption_num_frames)
    print(f"Captioning {len(paths)} frames of {args.caption_camera}: {[p.stem for p in paths]}")
    # Stage 0b detection first (on the two earliest frames, before the arms occlude the table);
    # discover_objects only reads .jpg files, so stage them as JPEGs in a temp dir.
    with tempfile.TemporaryDirectory(prefix="caption_detect_") as tmp:
        for p in frames[:2]:
            Image.open(p).convert("RGB").save(Path(tmp) / f"{p.stem}.jpg", quality=95)
        _, objects = discover_objects(Path(tmp), model_id=DISCOVERY_HF_MODEL)
    labels = [o["label"] for o in objects]
    print(f"Detected objects: {labels}")

    result = caption_pick_and_place(paths, DISCOVERY_HF_MODEL, objects=labels)
    result = {
        "camera": args.caption_camera, "frames": [p.stem for p in paths], "model": DISCOVERY_HF_MODEL,
        "detected_objects": objects, **result,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "caption.json").write_text(json.dumps(result, indent=2))
    print(f"Picked object:   {result['picked_object']}")
    print(f"Hand used:       {result['hand']}")
    print(f"Placed location: {result['placed_location']}")
    print(f"Summary:         {result['summary']}")
    if args.track:
        track = run_track(args, data_dir, output_dir, result)
        if track is not None:
            result["track"] = track
            (output_dir / "caption.json").write_text(json.dumps(result, indent=2))
    print(f"Done. Caption in {output_dir / 'caption.json'}")


def track_labels(args, caption: dict) -> list[str]:
    from genrecon.utils.pick_place_caption import SURFACE_WORDS

    """Objects to track: --track_labels if given, else the picked object plus the object it is
    placed on/in when that is one of the detected labels (not a bare surface such as the table)."""
    if args.track_labels:
        return args.track_labels
    labels = [caption["picked_object"]]
    placed = (caption["placed_location"] or "").strip().lower()
    detected = [o["label"] for o in caption["detected_objects"]]
    if placed in detected and placed != labels[0]:
        labels.append(placed)
    elif placed not in SURFACE_WORDS:
        print(f"Note: placed location {placed!r} is not a detected object label; tracking only the picked object.")
    else:
        print(f"Note: the object is placed on the bare surface ({placed!r}); tracking only the picked object.")
    return labels


def run_track(args, data_dir: Path, output_dir: Path, caption: dict) -> dict | None:
    """Segment + track the captioned picked object (and the object it is placed on/in, if any)
    through every frame of --track_camera (Qwen-VL locates each object in
    a few frames, GroundedSAM2 tracks all of them from their best boxes)."""
    import numpy as np
    from PIL import ImageDraw

    from genrecon.utils.pick_place_caption import sample_frames
    from genrecon.utils.pick_place_track import locate_objects, track_objects
    from run_full_pipeline import DISCOVERY_HF_MODEL

    hand = caption["hand"]
    if not caption["picked_object"] or (args.track_camera == "used_hand" and hand not in ("left", "right")):
        print(f"Track skipped: need a picked_object (and, for --track_camera used_hand, a left/right hand) from the caption (got {caption['picked_object']!r}, {hand!r}).")
        return None
    labels = track_labels(args, caption)
    camera = f"{hand}_hand" if args.track_camera == "used_hand" else args.track_camera
    frames = list(list_frames(data_dir / CAMERA_DIRS[camera]).values())
    candidates = sample_frames(frames, args.track_num_seed_frames)
    print(f"Tracking {labels} on {camera} ({len(frames)} frames), seed candidates: {[p.stem for p in candidates]}")

    seeds = locate_objects(candidates, DISCOVERY_HF_MODEL, labels)
    for label in labels:
        if label not in seeds:
            print(f"Warning: '{label}' was not found in any of the sampled {camera} frames; not tracked.")
    if not seeds:
        print("Track skipped: no object found.")
        return None
    track_dir = output_dir / "track"
    track_dir.mkdir(parents=True, exist_ok=True)
    for label, (seed_frame, seed_box) in seeds.items():
        seed_img = Image.open(seed_frame).convert("RGB")
        ImageDraw.Draw(seed_img).rectangle(seed_box, outline=(255, 0, 0), width=3)
        seed_img.save(track_dir / f"seed_{label.replace(' ', '_')}.png")
        print(f"Seed '{label}': {seed_frame.stem} box {[round(v) for v in seed_box]}")

    mask_dirs = track_objects(frames, seeds, work_dir=track_dir / "_work", out_dir=track_dir)
    objects = {}
    for label, mask_dir in mask_dirs.items():
        masks = sorted((mask_dir / "mask_bin").glob("*.png"))
        non_empty = sum(bool(np.asarray(Image.open(m)).any()) for m in masks)
        print(f"Tracked '{label}': {non_empty}/{len(frames)} frames with a non-empty mask in {mask_dir}")
        seed_frame, seed_box = seeds[label]
        objects[label] = {
            "seed_frame": seed_frame.stem, "seed_box": [float(v) for v in seed_box],
            "mask_dir": str(mask_dir), "frames_with_mask": non_empty,
        }
    return {"camera": camera, "frames_total": len(frames), "objects": objects}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR, help="Folder containing the per-camera image folders.")
    parser.add_argument("--cameras", type=parse_cameras, default=list(CAMERA_DIRS), help="Comma-separated subset of: " + ",".join(CAMERA_DIRS) + ". Default: all four.")
    parser.add_argument("--num_frames", type=int, default=1, help="Number of frames (timesteps) to use. Default: 1.")
    parser.add_argument("--start_frame", type=int, default=0, help="Index of the first frame in the sorted frame list. Default: 0.")
    parser.add_argument("--frame_step", type=int, default=1, help="Stride between selected frames. Default: 1.")
    parser.add_argument("--output_dir", type=Path, default=None, help="Output directory. Default: runs/vggt_pick_and_place/<data folder name>.")
    parser.add_argument("--checkpoint", type=Path, default=REPO_DIR / "checkpoints/vggt_omega/ckpts/vggt_omega_1b_512.pt")
    parser.add_argument("--image_resolution", type=int, default=512)
    parser.add_argument("--conf_thres", type=float, default=50.0, help="Confidence percentile threshold for exported points (0-100).")
    parser.add_argument("--max_points_k", type=int, default=1000, help="Max exported points, in thousands.")
    parser.add_argument("--align_to_gravity", action="store_true", help="Fit the ground plane and rotate the scene to be gravity-aligned.")
    parser.add_argument("--rotate_horizontal_deg", type=float, default=0.0, help="Extra yaw rotation (implies --align_to_gravity).")
    parser.add_argument("--crop_hands", action="store_true", help="For hand cameras, keep the horizontal middle half of the image (x in [W/4, 3W/4), full height) and rotate it 90 deg clockwise before running VGGT.")
    parser.add_argument("--viz",action="store_true", help="Also log the result to a rerun viewer.")
    parser.add_argument("--caption", action="store_true", help="Skip VGGT; instead have the Qwen VLM (same as run_full_pipeline.py) describe which object the robot picks up and where it places it, from frames of one camera. Writes caption.json to the output dir.")
    parser.add_argument("--caption_num_frames", type=int, default=8, help="With --caption: number of frames sampled evenly across the whole capture. Default: 8.")
    parser.add_argument("--caption_camera", choices=list(CAMERA_DIRS), default="left_head", help="With --caption: camera whose frames are described. Default: left_head.")
    parser.add_argument("--track", action="store_true", help="With --caption: after captioning, segment and track the picked object (and the object it is placed on/in, if any) through all frames of --track_camera (Qwen-VL finds the seed box, GroundedSAM2 tracks it). Masks go to <output_dir>/track/.")
    parser.add_argument("--track_num_seed_frames", type=int, default=8, help="With --track: number of evenly spaced frames in which Qwen-VL looks for the object to pick the seed box. Default: 8.")
    parser.add_argument("--track_camera", choices=list(CAMERA_DIRS) + ["used_hand"], default="left_head", help="With --track: camera whose frames are segmented and tracked; 'used_hand' = the hand camera on the side of the hand that picked the object (from the caption). Default: left_head.")
    parser.add_argument("--track_labels", type=lambda v: [x.strip() for x in v.split(",") if x.strip()], default=None, help="With --track: comma-separated objects to track, overriding the default (the picked object plus the object it is placed on/in, when that is a detected object).")
    args = parser.parse_args()
    if args.track and not args.caption:
        parser.error("--track requires --caption")
    if args.num_frames < 1 or args.frame_step < 1 or args.start_frame < 0 or args.caption_num_frames < 1 or args.track_num_seed_frames < 1:
        parser.error("--num_frames, --frame_step and --caption_num_frames must be >= 1 and --start_frame >= 0")
    return args


def main():
    args = parse_args()
    data_dir = args.data_dir.resolve()
    scene_name = data_dir.parent.name if data_dir.name == "frames" else data_dir.name
    output_dir = (args.output_dir or REPO_DIR / "runs" / "vggt_pick_and_place" / scene_name).resolve()

    if args.caption:
        if args.crop_hands or args.viz or args.align_to_gravity:
            print("Warning: --crop_hands/--viz/--align_to_gravity are ignored with --caption (VGGT is not run).")
        run_caption(args, data_dir, output_dir)
        return

    frame_ids = select_frames(data_dir, args.cameras, args.start_frame, args.frame_step, args.num_frames)
    print(f"Cameras: {args.cameras}; frames: {frame_ids}")
    staging_dir = output_dir / "input_images"
    stage_inputs(data_dir, args.cameras, frame_ids, staging_dir, crop_hands=args.crop_hands)

    # Heavy imports after arg parsing so --help stays fast.
    from demo_rerun import apply_gravity_alignment, export_colmap_dataset, load_model, log_to_rerun, run_model

    model = load_model(str(args.checkpoint))
    predictions = run_model(str(staging_dir), model, image_resolution=args.image_resolution)
    del model
    predictions = apply_gravity_alignment(predictions, args.align_to_gravity, args.rotate_horizontal_deg)

    export_dir = output_dir / "export"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_colmap_dataset(
        predictions,
        str(export_dir),
        conf_thres=args.conf_thres,
        max_points=args.max_points_k * 1000,
        save_depth=True,
    )

    if args.viz:
        import rerun as rr

        rr.init("vggt-omega-pick-and-place", spawn=True)
        log_to_rerun(
            predictions,
            conf_thres=args.conf_thres,
            mask_black_bg=False,
            mask_white_bg=False,
            show_cam=True,
            mask_sky=False,
            image_folder=str(staging_dir),
            max_points=args.max_points_k * 1000,
        )
    print(f"Done. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
