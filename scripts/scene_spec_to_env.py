"""Prints `export KEY=VALUE` lines from a scene_spec.json (written by scripts/run_scene_agent.py)
for run_full_pipeline.sh's Stage P0 to `eval` -- bridges the interactive agent's JSON output back
into the same bash variables Stages P1-P4 already consume when the user drives them directly via
--pick_place_target/--place-target/etc.

Usage:
    eval "$(uv run python scripts/scene_spec_to_env.py --spec runs/<scene>/pick_place/scene_spec.json)"
"""
import argparse
import json
import shlex
from pathlib import Path


def _export(key: str, value) -> str:
    return f"export {key}={shlex.quote(str(value))}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text())
    lines = [_export("PICK_PLACE_TARGET", spec["pick_target"])]

    if spec.get("place_target"):
        lines.append(_export("PLACE_TARGET", spec["place_target"]))
        lines.append(_export("PLACE_TARGET_CLEARANCE", spec["place_target_clearance"]))
    else:
        dx, dy, dz = spec["place_offset"]
        lines.append(_export("PLACE_OFFSET", f"{dx},{dy},{dz}"))

    lines.append(_export("APPROACH_SIDE", spec["approach_side"]))
    lines.append(_export("GRIPPER_OPEN_WIDTH", spec["gripper_open_width"]))
    lines.append(_export("START_DISTANCE", spec["start_distance"]))

    lighting = spec["lighting"]
    lines.append(_export("DOME_LIGHT_INTENSITY", lighting["dome_intensity"]))
    lines.append(_export("DOME_LIGHT_COLOR", ",".join(str(c) for c in lighting["dome_color"])))
    lines.append(_export("DISTANT_LIGHT_INTENSITY", lighting["distant_intensity"]))
    lines.append(_export("DISTANT_LIGHT_ANGLE", lighting["distant_angle"]))
    lines.append(_export("DISTANT_LIGHT_ROTATION_DEG", ",".join(str(v) for v in lighting["distant_rotation_deg"])))

    camera = spec["camera"]
    lines.append(_export("CAMERA_MODE", camera["mode"]))
    lines.append(_export("CAMERA_DISTANCE_MULTIPLIER", camera["distance_multiplier"]))

    print("\n".join(lines))


if __name__ == "__main__":
    main()
