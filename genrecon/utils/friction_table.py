"""Schema/loader for the material-pair friction reference table (YAML) consumed by
scripts/infer_friction_assignments.py's LangGraph agent to look up mu(material, floor)
for a scene's class labels.
"""
from dataclasses import dataclass
from pathlib import Path

import yaml

# Fixed default floor material the agent matches every class label against. Not a
# per-run flag yet (mu(material, floor) would need to also account for what the floor
# itself is made of) -- revisit if multiple floor types are needed.
DEFAULT_FLOOR_MATERIAL = "hardwood"


@dataclass(frozen=True)
class FrictionPairEntry:
    material_a: str
    material_b: str
    static_friction: float
    dynamic_friction: float
    source: str = ""
    notes: str = ""


def load_friction_table(path: Path) -> list[FrictionPairEntry]:
    with open(path) as f:
        raw = yaml.safe_load(f) or []
    return [
        FrictionPairEntry(
            material_a=entry["material_a"],
            material_b=entry["material_b"],
            static_friction=float(entry["static_friction"]),
            dynamic_friction=float(entry["dynamic_friction"]),
            source=entry.get("source", ""),
            notes=entry.get("notes", ""),
        )
        for entry in raw
    ]


def find_pairs_for_material(table: list[FrictionPairEntry], floor_material: str) -> list[FrictionPairEntry]:
    """Every entry where one side of the pair matches floor_material (case-insensitive) --
    the candidate set handed to the LLM's select_best_match node, filtered down from the
    full table so the prompt only has to reason over the *other* material name in each
    pair (the one actually describing the object, not the floor)."""
    target = floor_material.strip().lower()
    return [
        entry
        for entry in table
        if entry.material_a.strip().lower() == target or entry.material_b.strip().lower() == target
    ]


def other_material(entry: FrictionPairEntry, floor_material: str) -> str:
    """The non-floor side of a candidate pair -- the name shown to the LLM/returned as
    matched_material."""
    target = floor_material.strip().lower()
    return entry.material_b if entry.material_a.strip().lower() == target else entry.material_a
