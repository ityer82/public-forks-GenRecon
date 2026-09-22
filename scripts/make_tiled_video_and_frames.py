"""Tile a 4-camera pick-and-place capture into one grid video, and extract
every source video to a folder of PNG frames.

Given a data directory containing the 4 per-camera videos captured for a
manipulation task (see e.g. fisheye/data/Pick_And_Place_T1_1/, which has
left_hand_image.mp4, left_head_image.mp4, right_head_image.mp4,
right_hand_image.mp4), this:

1. Composites all 4 videos into a single 2x2 tiled video (head cameras on
   top, hand cameras on bottom), written to <output_dir>/tiled.mp4.
2. Extracts every frame of each source video to
   <output_dir>/frames/<video_stem>/frame_%06d.png.

Shells out to ffmpeg for both steps (no OpenCV dependency).

Usage:
    uv run python scripts/make_tiled_video_and_frames.py \
        /home/ss/Work/GitHub/fisheye/data/Pick_And_Place_T1_1 \
        --output-dir /tmp/pnp_test_out
"""
import argparse
import subprocess
from pathlib import Path

DEFAULT_VIDEO_NAMES = [
    "left_head_image.mp4",
    "right_head_image.mp4",
    "left_hand_image.mp4",
    "right_hand_image.mp4",
]


def build_tiled_video(videos: list[Path], out_path: Path) -> None:
    """Composite 4 videos into a 2x2 grid (order: top-left, top-right,
    bottom-left, bottom-right), scaling each to a common size first in case
    resolutions differ.
    """
    cmd = ["ffmpeg", "-y"]
    for video in videos:
        cmd += ["-i", str(video)]

    filter_complex = (
        "[0:v]scale=640:480[v0];"
        "[1:v]scale=640:480[v1];"
        "[2:v]scale=640:480[v2];"
        "[3:v]scale=640:480[v3];"
        "[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]"
    )
    cmd += ["-filter_complex", filter_complex, "-map", "[v]", str(out_path)]

    subprocess.run(cmd, check=True)


def extract_frames(video: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(video), str(out_dir / "frame_%06d.png")]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_dir", type=Path, help="Directory containing the 4 camera videos"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write tiled.mp4 and frames/ (default: data_dir itself)",
    )
    parser.add_argument(
        "--video-names",
        nargs=4,
        default=DEFAULT_VIDEO_NAMES,
        metavar=("TOP_LEFT", "TOP_RIGHT", "BOTTOM_LEFT", "BOTTOM_RIGHT"),
        help=f"Filenames of the 4 videos in data_dir (default: {DEFAULT_VIDEO_NAMES})",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = [args.data_dir / name for name in args.video_names]
    missing = [str(v) for v in videos if not v.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing expected video file(s): {missing}")

    print(f"Building tiled video from {[v.name for v in videos]}...")
    tiled_path = output_dir / "tiled.mp4"
    build_tiled_video(videos, tiled_path)
    print(f"Wrote {tiled_path}")

    frames_root = output_dir / "frames"
    for video in videos:
        frames_dir = frames_root / video.stem
        print(f"Extracting frames from {video.name} to {frames_dir}...")
        extract_frames(video, frames_dir)

    print("Done.")


if __name__ == "__main__":
    main()
