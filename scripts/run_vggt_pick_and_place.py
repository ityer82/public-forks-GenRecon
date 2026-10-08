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
"""
import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR / "vggt"))

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
    args = parser.parse_args()
    if args.num_frames < 1 or args.frame_step < 1 or args.start_frame < 0:
        parser.error("--num_frames and --frame_step must be >= 1 and --start_frame >= 0")
    return args


def main():
    args = parse_args()
    data_dir = args.data_dir.resolve()
    scene_name = data_dir.parent.name if data_dir.name == "frames" else data_dir.name
    output_dir = (args.output_dir or REPO_DIR / "runs" / "vggt_pick_and_place" / scene_name).resolve()

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
