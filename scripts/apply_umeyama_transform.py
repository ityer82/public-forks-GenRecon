"""
Rescale/realign a VGGT-Omega-exported COLMAP-text dataset to match the metric
scale of a reference COLMAP reconstruction (e.g. ScanNet++'s iphone/colmap).

Fits a 7-DOF similarity transform (scale + rotation + translation) mapping
VGGT-Omega's camera centers onto the reference's camera centers (matched by
frame basename, Umeyama's method with a proper-rotation constraint), then
applies that same transform to every pose in images.txt and every point in
points3D.txt. cameras.txt (intrinsics) is copied unchanged since a world-space
similarity transform doesn't affect them.

Pass --apply_mode to test reduced-DOF variants instead of the full 7-DOF
transform, to isolate which component (scale / translation / rotation)
actually matters for fixing the reconstruction. In every mode, (s, R, t) are
all estimated once via the full 7-DOF Umeyama fit (rotation is required for a
meaningful scale/translation estimate when the two world frames differ by a
large rotation, as they do here) — --apply_mode only controls which of the
fitted components are kept when writing the output:
  - full              : apply fitted scale + rotation + translation (default)
  - scale_only        : apply fitted scale alone (R forced to I, t forced to 0)
  - scale_translation : apply fitted scale + fitted translation, rotation
                        forced to I after estimation (t is NOT re-fit)

Usage:
    uv run python scripts/apply_umeyama_transform.py \
        --src_sparse_dir runs/iphone_rgb_skip50/vggt_export/sparse/0 \
        --ref_colmap_dir /path/to/scannetpp/scene/iphone/colmap \
        --output_dir runs/iphone_rgb_skip50/colmap_scaled \
        [--apply_mode {full,scale_only,scale_translation}]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def qvec2rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def parse_images_txt(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray, int, str]]:
    """Returns {basename: (R_w2c, t_w2c, camera_id, raw_name)}."""
    poses = {}
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        parts = line.split(maxsplit=9)
        qw, qx, qy, qz = (float(x) for x in parts[1:5])
        tx, ty, tz = (float(x) for x in parts[5:8])
        cam_id = int(parts[8])
        raw_name = parts[9]
        basename = Path(raw_name).name
        R = qvec2rotmat(qw, qx, qy, qz)
        t = np.array([tx, ty, tz])
        poses[basename] = (R, t, cam_id, raw_name)
        i += 2
    return poses


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Fits dst ≈ s * R @ src + t (proper rotation, det(R) = +1). Returns (s, R, t, rmse)."""
    mu_src, mu_dst = src.mean(0), dst.mean(0)
    src_c, dst_c = src - mu_src, dst - mu_dst
    cov = dst_c.T @ src_c / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    var_src = (src_c**2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / var_src
    t = mu_dst - s * R @ mu_src
    resid = dst - (s * (R @ src.T).T + t)
    rmse = np.sqrt((resid**2).sum(1).mean())
    return s, R, t, rmse


def camera_center(R_w2c: np.ndarray, t_w2c: np.ndarray) -> np.ndarray:
    return -R_w2c.T @ t_w2c


def fit_transform(src_poses: dict, ref_poses: dict) -> tuple[float, np.ndarray, np.ndarray, float, int]:
    """Always fits the full 7-DOF similarity transform (rotation is required for a
    meaningful scale estimate when the two world frames differ by a large rotation,
    as they do here) — scale-only application, if requested, is handled by the
    caller discarding R/t after fitting, not by fitting scale in isolation.
    """
    common = sorted(set(src_poses) & set(ref_poses))
    if len(common) < 3:
        raise ValueError(f"Need >= 3 common frames to fit a similarity transform, found {len(common)}.")
    C_src = np.array([camera_center(*src_poses[n][:2]) for n in common])
    C_ref = np.array([camera_center(*ref_poses[n][:2]) for n in common])
    s, R, t, rmse = umeyama(C_src, C_ref)
    return s, R, t, rmse, len(common)


def transform_images_txt(src_images_txt: Path, dst_images_txt: Path, s: float, R_sim: np.ndarray, t_sim: np.ndarray) -> None:
    poses = parse_images_txt(src_images_txt)
    lines = []
    for image_id, name in enumerate(sorted(poses), start=1):
        R_w2c, t_w2c, cam_id, _ = poses[name]
        C = camera_center(R_w2c, t_w2c)
        C_new = s * R_sim @ C + t_sim
        R_c2w_new = R_sim @ R_w2c.T
        R_w2c_new = R_c2w_new.T
        t_w2c_new = -R_w2c_new @ C_new
        qw, qx, qy, qz = rotmat2qvec(R_w2c_new)
        tx, ty, tz = t_w2c_new
        lines.append(f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {cam_id} {name}\n\n")
    dst_images_txt.write_text("".join(lines), encoding="utf-8")


def transform_points3d_txt(src_path: Path, dst_path: Path, s: float, R_sim: np.ndarray, t_sim: np.ndarray) -> None:
    out_lines = []
    with src_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                out_lines.append(line)
                continue
            parts = stripped.split()
            point_id = parts[0]
            x, y, z = (float(v) for v in parts[1:4])
            rest = parts[4:]  # R G B ERROR [TRACK...]
            xyz_new = s * R_sim @ np.array([x, y, z]) + t_sim
            out_lines.append(f"{point_id} {xyz_new[0]} {xyz_new[1]} {xyz_new[2]} {' '.join(rest)}\n")
    dst_path.write_text("".join(out_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src_sparse_dir", required=True, help="VGGT-Omega export's sparse/0/ dir.")
    parser.add_argument("--ref_colmap_dir", required=True, help="Reference COLMAP dir (e.g. ScanNet++ iphone/colmap).")
    parser.add_argument("--output_dir", required=True, help="Where to write the transformed cameras/images/points3D.txt.")
    parser.add_argument(
        "--apply_mode",
        choices=["full", "scale_only", "scale_translation"],
        default="full",
        help="Which fitted components to apply (scale/rotation/translation are always "
        "estimated together via the full 7-DOF fit; this only controls what's kept "
        "when writing the output). See module docstring for details.",
    )
    args = parser.parse_args()

    src_dir = Path(args.src_sparse_dir)
    ref_dir = Path(args.ref_colmap_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_poses = parse_images_txt(src_dir / "images.txt")
    ref_poses = parse_images_txt(ref_dir / "images.txt")
    s, R_fit, t_fit, rmse, n_common = fit_transform(src_poses, ref_poses)

    print(f"[apply_umeyama_transform] matched {n_common} common frame(s)")
    print(f"[apply_umeyama_transform] fitted scale s      = {s:.6f}")
    print(f"[apply_umeyama_transform] fitted rotation R    =\n{R_fit}")
    print(f"[apply_umeyama_transform] det(R)               = {np.linalg.det(R_fit):.6f}")
    print(f"[apply_umeyama_transform] fitted translation t = {t_fit}")
    print(f"[apply_umeyama_transform] full-7-DOF camera-center RMSE = {rmse:.4f} m")

    if args.apply_mode == "scale_only":
        R_apply, t_apply = np.eye(3), np.zeros(3)
        print("[apply_umeyama_transform] apply_mode=scale_only: applying scale alone (R=I, t=0)")
    elif args.apply_mode == "scale_translation":
        R_apply, t_apply = np.eye(3), t_fit
        print("[apply_umeyama_transform] apply_mode=scale_translation: applying fitted scale + "
              "fitted translation, rotation forced to I (t not re-fit)")
    else:
        R_apply, t_apply = R_fit, t_fit

    (out_dir / "cameras.txt").write_text((src_dir / "cameras.txt").read_text(encoding="utf-8"), encoding="utf-8")
    transform_images_txt(src_dir / "images.txt", out_dir / "images.txt", s, R_apply, t_apply)
    transform_points3d_txt(src_dir / "points3D.txt", out_dir / "points3D.txt", s, R_apply, t_apply)

    print(f"[apply_umeyama_transform] wrote transformed COLMAP dataset to {out_dir}")


if __name__ == "__main__":
    main()
