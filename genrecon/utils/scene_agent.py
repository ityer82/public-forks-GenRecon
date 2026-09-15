"""LangGraph agent that interactively defines an Isaac Sim pick-and-place scene with the user,
once mv-sam3d/TRELLIS.2 (see run_full_pipeline.sh's Stage 3) has produced a real-world-scale mesh
for every --classes label at runs/<scene>/image_to_3d_meshes/<label>/mesh.glb.

Unlike genrecon.utils.friction_agent (a batch agent -- one graph.invoke() per class label, no
human turns), this is a structured wizard: a fixed sequence of nodes, each asking the user one
question about the scene (what to pick, where to place it, which side the robot approaches from,
lighting, camera), using a local Ollama LLM only to resolve free-text answers against a small set
of valid choices -- never to invent scene parameters outright. Every LLM call is wrapped so a bad
reply degrades to a safe default (mirroring today's hardcoded Isaac Sim behavior) instead of
raising or silently producing an unusable scene.

The assembled decisions are handed back as a SceneAgentState -- scripts/run_scene_agent.py persists
it to runs/<scene>/pick_place/scene_spec.json, which run_full_pipeline.sh's Stage P0 reads to drive
Stages P1-P4 in place of --pick_place_target/--place-target/etc.
"""
import json
from pathlib import Path
from typing import TypedDict

import trimesh
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from genrecon.utils.logger import logger

# Anchors every lighting/camera default to demo_franka_pickplace.py's current hardcoded values
# (add_lighting()/setup_camera()) -- a bad or unparseable LLM reply falls back to exactly today's
# render, not an arbitrary guess.
DEFAULT_LIGHTING = {
    "dome_intensity": 1000.0,
    "dome_color": [1.0, 1.0, 1.0],
    "distant_intensity": 3000.0,
    "distant_angle": 1.0,
    "distant_rotation_deg": [-45.0, 30.0, 0.0],
}
DEFAULT_CAMERA = {"mode": "angled", "distance_multiplier": 2.0}

# Keeps offsets "well under the Panda's ~0.85m reach", per demo_franka_pickplace.py's own
# --place-offset docstring -- the same constraint the agent's place-offset node clamps to.
MAX_PLACE_OFFSET_METERS = 0.6

APPROACH_SIDE_CHOICES = ["neg-x", "pos-x", "neg-y", "pos-y"]
CAMERA_MODE_CHOICES = ["angled", "overhead"]


class SceneAgentState(TypedDict):
    class_labels: list[str]
    mesh_dir: str
    object_summaries: dict[str, dict]
    pick_target: str | None
    place_target: str | None
    place_offset: list[float] | None
    place_target_clearance: float
    approach_side: str
    start_distance: float
    gripper_open_width: float
    lighting: dict
    camera: dict
    reasoning: dict[str, str]


def _parse_json_object(text: str) -> dict | None:
    """Tolerant JSON parse: the LLM sometimes wraps its reply in prose or code fences."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None


def _prompt_user(question: str) -> str:
    return input(f"\n{question}\n> ").strip()


def _describe_objects(state: SceneAgentState) -> dict:
    mesh_dir = Path(state["mesh_dir"])
    summaries: dict[str, dict] = {}
    print("\nObjects reconstructed from the scene:")
    for label in state["class_labels"]:
        mesh_path = mesh_dir / label / "mesh.glb"
        extents = None
        center = None
        if mesh_path.is_file():
            try:
                scene = trimesh.load(mesh_path, force="scene")
                bounds = scene.bounds  # (2, 3): [min, max]
                extents = (bounds[1] - bounds[0]).tolist()
                center = ((bounds[0] + bounds[1]) / 2.0).tolist()
            except Exception as e:
                logger.warning(f"scene_agent: could not read bbox for '{label}' from {mesh_path}: {e}")
        summaries[label] = {"extents_m": extents, "center_m": center}
        if extents is not None:
            print(f"  - {label}: ~{extents[0]:.2f} x {extents[1]:.2f} x {extents[2]:.2f} m")
        else:
            print(f"  - {label}: (mesh not found at {mesh_path}, size unknown)")
    return {"object_summaries": summaries}


def _make_ask_pick_target_node(llm: ChatOllama):
    def ask_pick_target(state: SceneAgentState) -> dict:
        labels = state["class_labels"]
        while True:
            answer = _prompt_user(f"Which object should the robot pick up? ({', '.join(labels)})")
            if answer in labels:
                return {"pick_target": answer, "reasoning": {**state["reasoning"], "pick_target": "exact match"}}
            resolved, reasoning = _resolve_label(llm, answer, labels, purpose="the object the user wants picked up")
            if resolved is not None:
                return {"pick_target": resolved, "reasoning": {**state["reasoning"], "pick_target": reasoning}}
            print(f"  Couldn't match '{answer}' to one of {labels} -- please try again.")

    return ask_pick_target


def _resolve_label(llm: ChatOllama, free_text: str, labels: list[str], *, purpose: str) -> tuple[str | None, str]:
    """Maps a free-text answer to one of `labels` via the LLM, or (None, reason) if no confident
    match -- callers re-prompt the user rather than guessing."""
    prompt = (
        f"A user was asked to identify {purpose}. Their answer was: '{free_text}'. "
        f"Which of these exact labels did they mean: {', '.join(labels)}? "
        'Reply with ONLY a JSON object {"label": "<one of the listed labels>", "confidence": "high"|"low"}, '
        'or {"label": "NO_MATCH", "confidence": "low"} if none of them plausibly match.'
    )
    try:
        reply = llm.invoke(prompt).content
    except Exception as e:
        logger.warning(f"scene_agent: label resolution failed for '{free_text}': {e}")
        return None, f"LLM call failed: {e}"

    parsed = _parse_json_object(reply)
    if parsed is None or "label" not in parsed:
        return None, f"unparseable LLM reply: {reply!r}"
    chosen = str(parsed["label"]).strip()
    confidence = str(parsed.get("confidence", "low")).strip().lower()
    if chosen == "NO_MATCH" or chosen not in labels or confidence != "high":
        return None, f"no confident match (LLM chose {chosen!r}, confidence={confidence!r})"
    return chosen, f"resolved '{free_text}' -> '{chosen}'"


def _make_ask_place_location_node(llm: ChatOllama):
    def ask_place_location(state: SceneAgentState) -> dict:
        other_labels = [l for l in state["class_labels"] if l != state["pick_target"]]
        options_text = ", ".join(other_labels) if other_labels else "(none)"
        answer = _prompt_user(
            f"Where should '{state['pick_target']}' be placed? Name another object to place it "
            f"above/into ({options_text}), or describe an offset/direction (e.g. 'move it 20cm to the "
            "right')."
        )
        if answer in other_labels:
            clearance_answer = _prompt_user(
                f"Clearance above '{answer}' in meters before releasing? (default {0.05}, press enter to keep)"
            )
            clearance = _safe_float(clearance_answer, default=0.05)
            return {
                "place_target": answer,
                "place_offset": None,
                "place_target_clearance": clearance,
                "reasoning": {**state["reasoning"], "place_location": f"place above '{answer}', clearance={clearance}m"},
            }

        if other_labels:
            resolved, reasoning = _resolve_label(
                llm, answer, other_labels, purpose="which other object to place the picked item above"
            )
            if resolved is not None:
                return {
                    "place_target": resolved,
                    "place_offset": None,
                    "place_target_clearance": 0.05,
                    "reasoning": {**state["reasoning"], "place_location": reasoning},
                }

        offset, reasoning = _resolve_place_offset(llm, answer)
        return {
            "place_target": None,
            "place_offset": offset,
            "place_target_clearance": 0.05,
            "reasoning": {**state["reasoning"], "place_location": reasoning},
        }

    return ask_place_location


def _resolve_place_offset(llm: ChatOllama, free_text: str) -> tuple[list[float], str]:
    default = [0.3, 0.0, 0.0]
    prompt = (
        f"A user described where to place an object, relative to its current position, as: "
        f"'{free_text}'. Convert this to a world-frame [dx, dy, dz] offset in meters (+x is one "
        "direction, +y is perpendicular to it, +z is up). Keep every component's magnitude under "
        f"{MAX_PLACE_OFFSET_METERS} (a small robot arm's reach). Reply with ONLY a JSON object "
        '{"dx": <float>, "dy": <float>, "dz": <float>}.'
    )
    try:
        reply = llm.invoke(prompt).content
    except Exception as e:
        logger.warning(f"scene_agent: place-offset inference failed for '{free_text}': {e}")
        return default, f"LLM call failed, using default {default}: {e}"

    parsed = _parse_json_object(reply)
    if parsed is None or not all(k in parsed for k in ("dx", "dy", "dz")):
        return default, f"unparseable LLM reply, using default {default}: {reply!r}"
    try:
        offset = [
            max(-MAX_PLACE_OFFSET_METERS, min(MAX_PLACE_OFFSET_METERS, float(parsed[k])))
            for k in ("dx", "dy", "dz")
        ]
    except (TypeError, ValueError):
        return default, f"non-numeric offset in LLM reply, using default {default}: {parsed!r}"
    return offset, f"parsed offset {offset} from '{free_text}' (clamped to +/-{MAX_PLACE_OFFSET_METERS}m)"


def _safe_float(text: str, *, default: float) -> float:
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _make_ask_approach_side_node(llm: ChatOllama):
    def ask_approach_side(state: SceneAgentState) -> dict:
        answer = _prompt_user(
            "Which side should the robot's base stand on, relative to "
            f"'{state['pick_target']}'? Options: {', '.join(APPROACH_SIDE_CHOICES)} (or describe it, "
            "e.g. 'from the front', 'no preference')."
        )
        if answer in APPROACH_SIDE_CHOICES:
            return {"approach_side": answer, "reasoning": {**state["reasoning"], "approach_side": "exact match"}}

        prompt = (
            f"A user described which side a robot should approach an object from as: '{answer}'. "
            f"Map this to one of: {', '.join(APPROACH_SIDE_CHOICES)} ('neg-y' is a reasonable default "
            "for 'no preference' or an ambiguous answer). Reply with ONLY a JSON object "
            '{"approach_side": "<one of the listed options>"}.'
        )
        try:
            reply = llm.invoke(prompt).content
            parsed = _parse_json_object(reply)
            chosen = str(parsed.get("approach_side", "")).strip() if parsed else ""
        except Exception as e:
            logger.warning(f"scene_agent: approach-side inference failed for '{answer}': {e}")
            chosen = ""

        if chosen not in APPROACH_SIDE_CHOICES:
            chosen = "neg-y"
            reasoning = f"could not resolve '{answer}', defaulting to '{chosen}'"
        else:
            reasoning = f"resolved '{answer}' -> '{chosen}'"
        return {"approach_side": chosen, "reasoning": {**state["reasoning"], "approach_side": reasoning}}

    return ask_approach_side


def _make_ask_lighting_node(llm: ChatOllama):
    def ask_lighting(state: SceneAgentState) -> dict:
        answer = _prompt_user(
            "Describe the lighting for the scene (e.g. 'bright studio light', 'dim warm indoor "
            "light', 'outdoor daylight'), or press enter for the default."
        )
        if not answer:
            return {"lighting": dict(DEFAULT_LIGHTING), "reasoning": {**state["reasoning"], "lighting": "default (no answer given)"}}

        prompt = (
            f"A user described a lighting mood for a robotics simulation render as: '{answer}'. "
            "Map it to concrete USD light parameters as a JSON object with keys: "
            '"dome_intensity" (float, reasonable range 200-3000, default 1000), '
            '"dome_color" ([r,g,b] floats 0-1, default [1,1,1], warm tints lean toward [1, 0.9, 0.75]), '
            '"distant_intensity" (float, reasonable range 500-6000, default 3000), '
            '"distant_angle" (float, softness of the shadow-casting light in degrees, default 1.0), '
            '"distant_rotation_deg" ([x,y,z] floats, default [-45, 30, 0]). '
            "Reply with ONLY that JSON object, no other keys, no explanation."
        )
        try:
            reply = llm.invoke(prompt).content
        except Exception as e:
            logger.warning(f"scene_agent: lighting inference failed for '{answer}': {e}")
            return {"lighting": dict(DEFAULT_LIGHTING), "reasoning": {**state["reasoning"], "lighting": f"LLM call failed, using default: {e}"}}

        parsed = _parse_json_object(reply)
        lighting = _clamp_lighting(parsed) if parsed else None
        if lighting is None:
            return {"lighting": dict(DEFAULT_LIGHTING), "reasoning": {**state["reasoning"], "lighting": f"unparseable reply, using default: {reply!r}"}}
        return {"lighting": lighting, "reasoning": {**state["reasoning"], "lighting": f"mapped '{answer}' -> {lighting}"}}

    return ask_lighting


def _clamp_lighting(parsed: dict) -> dict:
    try:
        lighting = dict(DEFAULT_LIGHTING)
        if "dome_intensity" in parsed:
            lighting["dome_intensity"] = max(50.0, min(10000.0, float(parsed["dome_intensity"])))
        if "dome_color" in parsed and len(parsed["dome_color"]) == 3:
            lighting["dome_color"] = [max(0.0, min(1.0, float(c))) for c in parsed["dome_color"]]
        if "distant_intensity" in parsed:
            lighting["distant_intensity"] = max(0.0, min(20000.0, float(parsed["distant_intensity"])))
        if "distant_angle" in parsed:
            lighting["distant_angle"] = max(0.0, min(90.0, float(parsed["distant_angle"])))
        if "distant_rotation_deg" in parsed and len(parsed["distant_rotation_deg"]) == 3:
            lighting["distant_rotation_deg"] = [float(v) for v in parsed["distant_rotation_deg"]]
        return lighting
    except (TypeError, ValueError):
        return None


def _make_ask_camera_node(llm: ChatOllama):
    def ask_camera(state: SceneAgentState) -> dict:
        answer = _prompt_user(
            f"Camera framing? Options: {', '.join(CAMERA_MODE_CHOICES)} (or describe it, e.g. "
            "'looking straight down', 'from the side', 'zoomed in'), or press enter for the default."
        )
        if not answer:
            return {"camera": dict(DEFAULT_CAMERA), "reasoning": {**state["reasoning"], "camera": "default (no answer given)"}}
        if answer in CAMERA_MODE_CHOICES:
            return {"camera": {"mode": answer, "distance_multiplier": DEFAULT_CAMERA["distance_multiplier"]}, "reasoning": {**state["reasoning"], "camera": "exact mode match"}}

        prompt = (
            f"A user described a camera framing preference as: '{answer}'. Map it to a JSON object "
            f'with keys: "mode" (one of {CAMERA_MODE_CHOICES!r}; "overhead" for a top-down/bird\'s-eye '
            'view, "angled" for a 3/4 view from the side -- the default), "distance_multiplier" '
            "(float, reasonable range 1.0-4.0, default 2.0; lower is a tighter/closer shot, higher is "
            "wider/further back). Reply with ONLY that JSON object."
        )
        try:
            reply = llm.invoke(prompt).content
        except Exception as e:
            logger.warning(f"scene_agent: camera inference failed for '{answer}': {e}")
            return {"camera": dict(DEFAULT_CAMERA), "reasoning": {**state["reasoning"], "camera": f"LLM call failed, using default: {e}"}}

        parsed = _parse_json_object(reply)
        camera = _clamp_camera(parsed) if parsed else None
        if camera is None:
            return {"camera": dict(DEFAULT_CAMERA), "reasoning": {**state["reasoning"], "camera": f"unparseable reply, using default: {reply!r}"}}
        return {"camera": camera, "reasoning": {**state["reasoning"], "camera": f"mapped '{answer}' -> {camera}"}}

    return ask_camera


def _clamp_camera(parsed: dict) -> dict:
    try:
        camera = dict(DEFAULT_CAMERA)
        mode = str(parsed.get("mode", camera["mode"])).strip()
        camera["mode"] = mode if mode in CAMERA_MODE_CHOICES else camera["mode"]
        if "distance_multiplier" in parsed:
            camera["distance_multiplier"] = max(1.0, min(4.0, float(parsed["distance_multiplier"])))
        return camera
    except (TypeError, ValueError):
        return None


def confirm_and_write_spec(state: SceneAgentState) -> dict:
    print("\n── Scene summary ──")
    print(f"  Pick target:    {state['pick_target']}")
    if state["place_target"]:
        print(f"  Place location: above '{state['place_target']}' (+{state['place_target_clearance']}m clearance)")
    else:
        print(f"  Place location: offset {state['place_offset']}")
    print(f"  Approach side:  {state['approach_side']}")
    print(f"  Lighting:       {state['lighting']}")
    print(f"  Camera:         {state['camera']}")
    _prompt_user("Press enter to confirm and generate the scene (answers are not re-editable in this pass).")
    return {}


def build_scene_agent_graph(llm: ChatOllama):
    graph = StateGraph(SceneAgentState)
    graph.add_node("describe_objects", _describe_objects)
    graph.add_node("ask_pick_target", _make_ask_pick_target_node(llm))
    graph.add_node("ask_place_location", _make_ask_place_location_node(llm))
    graph.add_node("ask_approach_side", _make_ask_approach_side_node(llm))
    graph.add_node("ask_lighting", _make_ask_lighting_node(llm))
    graph.add_node("ask_camera", _make_ask_camera_node(llm))
    graph.add_node("confirm_and_write_spec", confirm_and_write_spec)

    graph.set_entry_point("describe_objects")
    graph.add_edge("describe_objects", "ask_pick_target")
    graph.add_edge("ask_pick_target", "ask_place_location")
    graph.add_edge("ask_place_location", "ask_approach_side")
    graph.add_edge("ask_approach_side", "ask_lighting")
    graph.add_edge("ask_lighting", "ask_camera")
    graph.add_edge("ask_camera", "confirm_and_write_spec")
    graph.add_edge("confirm_and_write_spec", END)
    return graph.compile()


def run_scene_agent(
    class_labels: list[str],
    mesh_dir: Path,
    *,
    ollama_model: str,
    ollama_host: str,
    start_distance: float = 0.5,
    gripper_open_width: float = 0.06,
) -> SceneAgentState:
    """Runs the wizard once, interactively, and returns the final state -- pick_target,
    place_target/place_offset/place_target_clearance, approach_side, lighting, camera, reasoning."""
    llm = ChatOllama(model=ollama_model, base_url=ollama_host)
    app = build_scene_agent_graph(llm)
    initial_state: SceneAgentState = {
        "class_labels": class_labels,
        "mesh_dir": str(mesh_dir),
        "object_summaries": {},
        "pick_target": None,
        "place_target": None,
        "place_offset": None,
        "place_target_clearance": 0.05,
        "approach_side": "neg-y",
        "start_distance": start_distance,
        "gripper_open_width": gripper_open_width,
        "lighting": dict(DEFAULT_LIGHTING),
        "camera": dict(DEFAULT_CAMERA),
        "reasoning": {},
    }
    return app.invoke(initial_state)
