"""Interactively define an Isaac Sim pick-and-place scene using a LangGraph agent backed by a
local Ollama LLM (genrecon.utils.scene_agent), for run_full_pipeline.sh's --ai-scene-agent Stage
P0 to consume.

Must run after Stage 3 (mv-sam3d/TRELLIS.2 per-object mesh reconstruction) has populated
runs/<scene>/image_to_3d_meshes/<label>/mesh.glb for every --classes label -- the agent describes
each object's real-world size to the user from those meshes before asking anything.

Writes a single scene_spec.json capturing every decision (pick/place target, approach side,
lighting, camera) plus a `reasoning` field explaining how each was derived, so the spec is
self-documenting like manifest.json/friction_assignments.json already are.

Usage:
    uv run python scripts/run_scene_agent.py \
        --classes "banana,bowl" \
        --mesh_dir runs/<scene>/image_to_3d_meshes \
        --out_json runs/<scene>/pick_place/scene_spec.json
"""
import argparse
import json
from pathlib import Path

from genrecon.utils.logger import logger
from genrecon.utils.scene_agent import run_scene_agent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classes", required=True, help="Comma-separated class labels, same spelling as run_full_pipeline.sh's --classes.")
    parser.add_argument("--mesh_dir", type=Path, required=True, help="Directory containing <label>/mesh.glb per class (runs/<scene>/image_to_3d_meshes).")
    parser.add_argument("--out_json", type=Path, required=True)
    parser.add_argument("--ollama_model", default="llama3.1:8b")
    parser.add_argument("--ollama_host", default="http://localhost:11434")
    parser.add_argument("--start_distance", type=float, default=0.5)
    parser.add_argument("--gripper_open_width", type=float, default=0.06)
    args = parser.parse_args()

    class_labels = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not class_labels:
        raise SystemExit("--classes produced no labels after splitting on ','.")

    result = run_scene_agent(
        class_labels,
        args.mesh_dir,
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host,
        start_distance=args.start_distance,
        gripper_open_width=args.gripper_open_width,
    )

    spec = {
        "pick_target": result["pick_target"],
        "place_target": result["place_target"],
        "place_offset": result["place_offset"],
        "place_target_clearance": result["place_target_clearance"],
        "approach_side": result["approach_side"],
        "start_distance": result["start_distance"],
        "gripper_open_width": result["gripper_open_width"],
        "lighting": result["lighting"],
        "camera": result["camera"],
        "reasoning": result["reasoning"],
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(spec, indent=2))
    logger.info(f"Wrote scene spec to {args.out_json}")


if __name__ == "__main__":
    main()
