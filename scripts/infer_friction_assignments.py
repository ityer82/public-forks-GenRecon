"""Infer per-class-label friction coefficients against a fixed floor material using a
LangGraph agent backed by a local Ollama LLM, for IsaacSim's convert_asset.py to consume.

Reads labels.json ({raw_label: sanitized_dirname}, the file COB-GS's detector writes and
reconstruct_scene.py consumes) and, for every asset_id (sanitized dirname) except
--background_label, looks it up against --friction_table (a YAML material-pair
coefficient table, see genrecon/utils/friction_table.py) via the LangGraph agent in
genrecon/utils/friction_agent.py. Writes one entry per asset_id into --out_json, keyed
by the SAME sanitized dirname convert_asset.py uses as asset_id.

'floor' is never a COB-GS-detected label (it's carved out of background_mesh.ply by
extract_floor_mesh.py), so it never appears in labels.json -- this script always
synthesizes a forced --floor_label entry with static_friction=dynamic_friction=0,
regardless of labels.json's contents.

'background' is intentionally never given a table entry (see convert_asset.py's
--friction-table help): it's a single room-scan mesh, not a single material, so there's
no honest label to match against the table. It stays on convert_asset.py's own
--static-friction/--dynamic-friction global defaults.

Usage:
    uv run python scripts/infer_friction_assignments.py \
        --labels_json runs/<scene>/genrecon_output/segmentation_raw/masks/classes/labels.json \
        --friction_table configs/materials/friction_table.example.yaml \
        --out_json runs/<scene>/genrecon_output/shapes/friction_assignments.json
"""
import argparse
import json
from pathlib import Path

from genrecon.utils.friction_agent import run_friction_agent
from genrecon.utils.friction_table import DEFAULT_FLOOR_MATERIAL, load_friction_table
from genrecon.utils.logger import logger


def run_friction_assignments(
    labels_json: Path,
    friction_table_path: Path,
    out_json: Path,
    *,
    floor_label: str = "floor",
    background_label: str = "background",
    default_static_friction: float = 0.5,
    default_dynamic_friction: float = 0.5,
    floor_material: str = DEFAULT_FLOOR_MATERIAL,
    ollama_model: str = "llama3.1:8b",
    ollama_host: str = "http://localhost:11434",
) -> None:
    labels = json.loads(labels_json.read_text())
    table = load_friction_table(friction_table_path)

    assignments = {}
    for raw_label, dirname in labels.items():
        if dirname == background_label:
            continue
        result = run_friction_agent(
            raw_label,
            floor_material=floor_material,
            table=table,
            default_static_friction=default_static_friction,
            default_dynamic_friction=default_dynamic_friction,
            ollama_model=ollama_model,
            ollama_host=ollama_host,
        )
        assignments[dirname] = {
            "static_friction": result["static_friction"],
            "dynamic_friction": result["dynamic_friction"],
            "matched_material": result.get("matched_material"),
            "confidence": result.get("confidence"),
            "reasoning": result.get("reasoning"),
            "used_fallback": result["used_fallback"],
        }
        logger.info(
            f"friction[{dirname}] ({raw_label!r}): static={result['static_friction']}, "
            f"dynamic={result['dynamic_friction']}, matched={result.get('matched_material')!r}, "
            f"fallback={result['used_fallback']}"
        )

    assignments[floor_label] = {
        "static_friction": 0.0,
        "dynamic_friction": 0.0,
        "matched_material": floor_material,
        "confidence": "forced",
        "reasoning": "floor is always authored with zero friction; combined with frictionCombineMode=max "
                     "at convert_asset.py, this makes floor-object contacts use the object's own mu.",
        "used_fallback": False,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(assignments, indent=2))
    logger.info(f"Wrote {len(assignments)} friction assignments to {out_json}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels_json", type=Path, required=True)
    parser.add_argument("--friction_table", type=Path, required=True)
    parser.add_argument("--out_json", type=Path, required=True)
    parser.add_argument("--floor_label", default="floor", help="Must match convert_asset.py's --floor-label.")
    parser.add_argument("--background_label", default="background", help="Must match convert_asset.py's --background-label.")
    parser.add_argument("--default_static_friction", type=float, default=0.5, help="Fallback when the agent finds no confident match -- should mirror convert_asset.py's --static-friction.")
    parser.add_argument("--default_dynamic_friction", type=float, default=0.5, help="Fallback when the agent finds no confident match -- should mirror convert_asset.py's --dynamic-friction.")
    parser.add_argument("--floor_material", default=DEFAULT_FLOOR_MATERIAL, help="Fixed material name the agent looks up mu(object_material, floor_material) against.")
    parser.add_argument("--ollama_model", default="llama3.1:8b")
    parser.add_argument("--ollama_host", default="http://localhost:11434")
    args = parser.parse_args()

    run_friction_assignments(
        args.labels_json,
        args.friction_table,
        args.out_json,
        floor_label=args.floor_label,
        background_label=args.background_label,
        default_static_friction=args.default_static_friction,
        default_dynamic_friction=args.default_dynamic_friction,
        floor_material=args.floor_material,
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host,
    )


if __name__ == "__main__":
    main()
