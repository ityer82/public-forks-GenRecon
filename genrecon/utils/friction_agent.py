"""LangGraph agent that maps a free-text object class label (e.g. "banana", "plastic
cup" -- produced upstream by COB-GS's open-vocabulary detector, not a fixed taxonomy) to
a friction coefficient against a fixed floor material, by asking a local Ollama LLM to
(1) infer the object's likely real-world material and (2) pick the closest matching
entry from a genrecon.utils.friction_table.FrictionPairEntry candidate list.

Every LLM call is wrapped so a bad/unparseable response, or Ollama being unreachable,
falls back to caller-supplied global friction defaults instead of raising -- one bad
label must not abort a whole scene's friction-assignment batch.
"""
import json
from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from genrecon.utils.friction_table import FrictionPairEntry, find_pairs_for_material, other_material
from genrecon.utils.logger import logger

# Common household-object materials, plus every material name that actually appears in
# configs/materials/friction_table.example.yaml (concrete, brick, leather, ice) so the
# inference step can propose an exact table match when one genuinely exists, instead of
# only ever landing on a near-neighbor word (e.g. "stone" for a concrete object) that then
# has to be reconciled by select_best_match's fuzzy step.
MATERIAL_VOCABULARY = [
    "plastic", "metal", "glass", "wood", "rubber", "fabric", "ceramic", "cardboard", "stone", "organic",
    "leather", "concrete", "brick", "ice",
]


class FrictionAgentState(TypedDict):
    class_label: str
    floor_material: str
    default_static_friction: float
    default_dynamic_friction: float
    candidates: list[FrictionPairEntry]
    inferred_material: str | None
    matched_entry: FrictionPairEntry | None
    matched_material: str | None
    confidence: str | None
    reasoning: str | None
    static_friction: float
    dynamic_friction: float
    used_fallback: bool


def _parse_json_object(text: str) -> dict | None:
    """Tolerant JSON parse: the LLM sometimes wraps its reply in prose or code fences."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None


def _make_infer_material_node(llm: ChatOllama):
    def infer_material_from_label(state: FrictionAgentState) -> dict:
        prompt = (
            "You are helping assign physical friction properties to objects in a robotics "
            "simulation. Given an object described as '{label}', what single material from "
            "this list is it most likely made of: {vocab}? Reply with only the one word, "
            "no punctuation or explanation."
        ).format(label=state["class_label"], vocab=", ".join(MATERIAL_VOCABULARY))
        try:
            reply = llm.invoke(prompt).content.strip().lower()
        except Exception as e:
            logger.warning(f"friction_agent: material inference failed for '{state['class_label']}': {e}")
            return {"inferred_material": None}

        word = reply.split()[0].strip(".,\"'") if reply else ""
        return {"inferred_material": word or None}

    return infer_material_from_label


def _make_retrieve_candidates_node(table: list[FrictionPairEntry]):
    def retrieve_candidates(state: FrictionAgentState) -> dict:
        return {"candidates": find_pairs_for_material(table, state["floor_material"])}

    return retrieve_candidates


# Few-shot examples for select_best_match's prompt -- worked cases (including deliberate
# NO_MATCH cases) that teach the LLM to gate on genuine physical similarity rather than
# always forcing a pick from a non-empty candidate list. Written generically (not tied to
# any one friction table's actual candidate names) so they still apply if the table's
# material coverage changes. Bare instructions alone ("don't force a match") were not
# enough to stop a small local model from picking the least-bad candidate anyway --
# concrete worked examples, including explicit NO_MATCH ones, are what actually shifts
# behavior for this kind of task.
SELECT_BEST_MATCH_EXAMPLES = """Example 1:
Inferred material: wood
Candidates: wood, metal, stone, leather
Correct answer: {"material": "wood", "confidence": "high"}
(Exact match -- always prefer an exact or clear-synonym match when the list contains one.)

Example 2:
Inferred material: plastic
Candidates: wood, metal, stone, leather, concrete, brick, ice
Correct answer: {"material": "NO_MATCH", "confidence": "low"}
(Plastic is a smooth, semi-rigid polymer -- none of these rigid/mineral/tanned-hide
materials are genuinely similar to it. A non-empty candidate list does not mean one of
them is a good match; forcing "metal" or "wood" here would be wrong.)

Example 3:
Inferred material: organic
Candidates: wood, metal, stone, leather, concrete, brick, ice
Correct answer: {"material": "NO_MATCH", "confidence": "low"}
(Soft biological materials like fruit skin or peel have no genuine analog among these
rigid, mineral, or tanned-hide materials.)

Example 4:
Inferred material: stone
Candidates: wood, metal, stone, leather
Correct answer: {"material": "stone", "confidence": "high"}

Example 5:
Inferred material: rubber
Candidates: wood, metal, stone, leather, concrete
Correct answer: {"material": "NO_MATCH", "confidence": "low"}
(Rubber's elasticity/compliance has no analog among these rigid materials -- resist the
temptation to pick "leather" just because both are sometimes described as "flexible".)
"""


def _make_select_best_match_node(llm: ChatOllama):
    def select_best_match(state: FrictionAgentState) -> dict:
        if not state["candidates"] or not state["inferred_material"]:
            return {"matched_entry": None, "confidence": "low", "reasoning": "no candidates or no inferred material"}

        candidate_names = sorted({other_material(e, state["floor_material"]) for e in state["candidates"]})
        prompt = (
            "You are matching an object's material to the closest entry in a friction-coefficient "
            "table. Pick a candidate ONLY if it is genuinely physically similar to the inferred "
            "material in a way relevant to sliding friction (comparable rigidity, surface texture, "
            "and compliance). Do NOT pick a candidate merely because it is the least-bad option in "
            "a non-empty list -- if no candidate is genuinely similar, you MUST reply NO_MATCH "
            "rather than force a weak match. Study these worked examples first:\n\n"
            "{examples}\n"
            "Now classify this real case:\n"
            "Inferred material: {material}\n"
            "Candidates: {candidates}\n"
            "Reply with ONLY a JSON object of the form "
            '{{"material": "<one of the listed materials>", "confidence": "high"|"low"}}, or '
            '{{"material": "NO_MATCH", "confidence": "low"}} if none of them are genuinely similar.'
        ).format(examples=SELECT_BEST_MATCH_EXAMPLES, material=state["inferred_material"], candidates=", ".join(candidate_names))

        try:
            reply = llm.invoke(prompt).content
        except Exception as e:
            logger.warning(f"friction_agent: best-match selection failed for '{state['class_label']}': {e}")
            return {"matched_entry": None, "confidence": "low", "reasoning": f"LLM call failed: {e}"}

        parsed = _parse_json_object(reply)
        if parsed is None or "material" not in parsed:
            return {"matched_entry": None, "confidence": "low", "reasoning": f"unparseable LLM reply: {reply!r}"}

        chosen = str(parsed["material"]).strip()
        confidence = str(parsed.get("confidence", "low")).strip().lower()
        if chosen == "NO_MATCH" or chosen not in candidate_names:
            return {"matched_entry": None, "confidence": confidence, "reasoning": f"no confident match (LLM chose {chosen!r})"}

        matched_entry = next(e for e in state["candidates"] if other_material(e, state["floor_material"]) == chosen)
        return {"matched_entry": matched_entry, "matched_material": chosen, "confidence": confidence, "reasoning": f"matched '{state['inferred_material']}' -> '{chosen}'"}

    return select_best_match


def validate_or_fallback(state: FrictionAgentState) -> dict:
    if state["matched_entry"] is not None and state.get("confidence") == "high":
        entry = state["matched_entry"]
        return {
            "static_friction": entry.static_friction,
            "dynamic_friction": entry.dynamic_friction,
            "used_fallback": False,
        }

    logger.warning(
        f"friction_agent: no confident material match for class label '{state['class_label']}' "
        f"(inferred_material={state.get('inferred_material')!r}, confidence={state.get('confidence')!r}) "
        f"-- falling back to default friction {state['default_static_friction']}/{state['default_dynamic_friction']}."
    )
    return {
        "static_friction": state["default_static_friction"],
        "dynamic_friction": state["default_dynamic_friction"],
        "used_fallback": True,
    }


def build_friction_graph(llm: ChatOllama, table: list[FrictionPairEntry]):
    graph = StateGraph(FrictionAgentState)
    graph.add_node("infer_material_from_label", _make_infer_material_node(llm))
    graph.add_node("retrieve_candidates", _make_retrieve_candidates_node(table))
    graph.add_node("select_best_match", _make_select_best_match_node(llm))
    graph.add_node("validate_or_fallback", validate_or_fallback)

    graph.set_entry_point("infer_material_from_label")
    graph.add_edge("infer_material_from_label", "retrieve_candidates")
    graph.add_edge("retrieve_candidates", "select_best_match")
    graph.add_edge("select_best_match", "validate_or_fallback")
    graph.add_edge("validate_or_fallback", END)
    return graph.compile()


def run_friction_agent(
    class_label: str,
    *,
    floor_material: str,
    table: list[FrictionPairEntry],
    default_static_friction: float,
    default_dynamic_friction: float,
    ollama_model: str,
    ollama_host: str,
) -> dict:
    """Runs the graph once for a single class label and returns the final state dict --
    static_friction, dynamic_friction, matched_material, confidence, reasoning, used_fallback."""
    llm = ChatOllama(model=ollama_model, base_url=ollama_host)
    app = build_friction_graph(llm, table)
    initial_state: FrictionAgentState = {
        "class_label": class_label,
        "floor_material": floor_material,
        "default_static_friction": default_static_friction,
        "default_dynamic_friction": default_dynamic_friction,
        "candidates": [],
        "inferred_material": None,
        "matched_entry": None,
        "matched_material": None,
        "confidence": None,
        "reasoning": None,
        "static_friction": default_static_friction,
        "dynamic_friction": default_dynamic_friction,
        "used_fallback": True,
    }
    return app.invoke(initial_state)
