"""Single-shot vision-LLM call that proposes open-vocabulary object-class labels from a
handful of Stage 0's exported representative images, for --discover-classes (an alternative
to manually typing --classes). Unlike friction_agent/scene_agent, this call is image-grounded
and has no LangGraph state machine -- it's one prompt, one reply, one parse.

Two backends share the same prompt/sampling/parsing code (see discover_object_classes):
- "ollama" (default): a local Ollama daemon via langchain_ollama.ChatOllama.
- "hf": a locally-downloaded HF transformers checkpoint (default: gemma-3-12b-it), run
  in-process, no Ollama daemon required. See
  https://ai.google.dev/gemma/docs/capabilities/vision/image for the image-text-to-text
  pipeline() usage this follows.
"""
import base64
import json
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from genrecon.utils.logger import logger

DISCOVERY_PROMPT = (
    "You are looking at {n} photos of the same tabletop scene from different viewpoints. "
    "Focus only on the center of each image, where the objects of interest sit on the table. "
    "Ignore the left and right edges of the frame, where robot arms may be visible -- a robot "
    "arm is not an object to list.\n"
    "List every distinct, nameable object on/in the scene as short noun phrases suitable as "
    "text prompts for an open-vocabulary object detector (e.g. \"banana\", \"ceramic bowl\"). "
    "Rules:\n"
    "- Do NOT include background surfaces or structure: table, floor, wall, shelf, counter, "
    "background.\n"
    "- Do NOT include robot arms, grippers, or any part of a robot, even if visible at the "
    "left or right edges of the image.\n"
    "- Merge duplicates: if the same kind of object appears in multiple photos or multiple "
    "times, list its category once.\n"
    "- If multiple labels could refer to the same physical object (e.g. one label is a more "
    "generic or more specific version of another, like \"bottle\" vs \"water bottle\", or "
    "\"cup\" vs \"coffee cup\"), include only the single most specific label, not both.\n"
    "- Use short (1-3 word) noun phrases, lowercase, singular.\n"
    "Respond with ONLY a JSON object of the form {{\"objects\": [\"label1\", \"label2\", ...]}}. "
    "No prose, no markdown fences."
)


def _parse_json_object(text: str) -> dict | None:
    """Tolerant JSON parse: the LLM sometimes wraps its reply in prose or code fences."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None


def _select_evenly_spaced(paths: list[Path], n: int) -> list[Path]:
    if n <= 0:
        raise ValueError(f"num_images must be positive, got {n}")
    if not paths or n >= len(paths):
        return paths
    step = (len(paths) - 1) / (n - 1) if n > 1 else 0
    idxs = sorted({round(i * step) for i in range(n)})
    return [paths[i] for i in idxs]


def _encode_image_data_uri(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _drop_specificity_duplicates(labels: list[str]) -> list[str]:
    """Backstop for the VLM ignoring DISCOVERY_PROMPT's specificity-merge rule: drops any label
    whose words are a proper subset of another label's words (e.g. "bottle" is dropped when
    "water bottle" is also present), keeping the more specific phrasing. Labels with no subset
    relation (e.g. "apple" vs "orange") are left alone -- they're different objects, not
    different specificities of the same one."""
    token_sets = {label: set(label.split()) for label in labels}
    to_drop = {
        a for a in labels
        for b in labels
        if a != b and token_sets[a] < token_sets[b]
    }
    if to_drop:
        logger.info(f"Discovery: dropping generic labels superseded by a more specific one: {sorted(to_drop)}")
    return [label for label in labels if label not in to_drop]


def _parse_and_postprocess(reply: str, model_label: str) -> list[str]:
    """Shared reply-handling for both backends: parse the {"objects": [...]} JSON reply,
    dedupe/normalize labels, and drop specificity duplicates. Raises RuntimeError on an
    unparseable reply or zero usable labels."""
    parsed = _parse_json_object(reply)
    if parsed is None or not isinstance(parsed.get("objects"), list):
        raise RuntimeError(
            f"Object-class discovery: could not parse a JSON {{'objects': [...]}} reply from "
            f"'{model_label}'. Raw reply:\n{reply}"
        )

    seen: set[str] = set()
    labels: list[str] = []
    for raw in parsed["objects"]:
        if not isinstance(raw, str):
            continue
        label = raw.strip().lower()
        if label and label not in seen:
            seen.add(label)
            labels.append(label)

    if not labels:
        raise RuntimeError(f"Object-class discovery: model '{model_label}' returned zero usable object labels.")

    return _drop_specificity_duplicates(labels)


def _discover_object_classes_ollama(
    export_images_dir: Path,
    vlm_model: str,
    num_images: int,
    ollama_host: str | None = None,
) -> list[str]:
    """Asks a local vision-LLM (Ollama) to propose open-vocabulary object-class labels from
    `num_images` evenly-spaced frames in `export_images_dir` (Stage 0's export_dir/images,
    always .jpg). Returns a deduped, lowercase-normalized, whitespace-stripped list of labels.

    Raises RuntimeError if no images are found, the model is unreachable, the reply can't be
    parsed, or zero labels are discovered -- discovery failing silently into an empty --classes
    run is worse than a loud crash, since it would silently skip Stages 1-3.
    """
    image_paths = sorted(export_images_dir.glob("*.jpg"))
    if not image_paths:
        raise RuntimeError(f"No .jpg images found in {export_images_dir} for class discovery.")

    selected = _select_evenly_spaced(image_paths, num_images)
    logger.info(f"Discovery: sampling {len(selected)}/{len(image_paths)} frames from {export_images_dir}")

    content = [{"type": "text", "text": DISCOVERY_PROMPT.format(n=len(selected))}]
    for p in selected:
        content.append({"type": "image_url", "image_url": {"url": _encode_image_data_uri(p)}})

    # temperature=0: discovery should be reproducible across repeated calls on the same
    # images (a non-deterministic default previously let two runs against the same run_dir
    # disagree on class names -- see run_full_pipeline.py's discovered_classes.json cache,
    # which is the other half of that fix).
    kwargs = {"model": vlm_model, "keep_alive": 0, "temperature": 0}
    if ollama_host:
        kwargs["base_url"] = ollama_host
    llm = ChatOllama(**kwargs)

    try:
        reply = llm.invoke([HumanMessage(content=content)]).content
    except Exception as e:
        raise RuntimeError(f"Object-class discovery failed calling Ollama model '{vlm_model}': {e}") from e

    return _parse_and_postprocess(reply, vlm_model)


def _discover_object_classes_hf(
    export_images_dir: Path,
    model_id: str,
    num_images: int,
) -> list[str]:
    """HF-transformers counterpart to _discover_object_classes_ollama: runs the same
    DISCOVERY_PROMPT against the same evenly-spaced frames, but through a locally-downloaded
    Gemma vision-language checkpoint (default: gemma-3-12b-it) instead of Ollama. Follows the
    image-text-to-text `pipeline()` usage documented at
    https://ai.google.dev/gemma/docs/capabilities/vision/image.

    Runs in-process (gemma3 is natively supported by the project's pinned transformers version,
    unlike gemma4, which needed an isolated transformers>=5 subprocess). Uses device=0 rather
    than device_map="auto" so this doesn't require the accelerate package -- gemma-3-12b-it fits
    on one GPU.
    """
    import torch
    from PIL import Image
    from transformers import GenerationConfig, pipeline

    image_paths = sorted(export_images_dir.glob("*.jpg"))
    if not image_paths:
        raise RuntimeError(f"No .jpg images found in {export_images_dir} for class discovery.")

    selected = _select_evenly_spaced(image_paths, num_images)
    logger.info(f"Discovery: sampling {len(selected)}/{len(image_paths)} frames from {export_images_dir}")

    content = [{"type": "image", "image": Image.open(p).convert("RGB")} for p in selected]
    content.append({"type": "text", "text": DISCOVERY_PROMPT.format(n=len(selected))})
    messages = [{"role": "user", "content": content}]

    pipe = None
    try:
        device = 0 if torch.cuda.is_available() else -1
        pipe = pipeline(task="image-text-to-text", model=model_id, device=device, dtype="auto")
        config = GenerationConfig(do_sample=False, max_new_tokens=512)  # greedy, mirrors Ollama's temperature=0
        output = pipe(messages, return_full_text=False, generate_kwargs={"generation_config": config})
        reply = output[0]["generated_text"]
    except Exception as e:
        raise RuntimeError(f"Object-class discovery failed calling HF model '{model_id}': {e}") from e
    finally:
        # Free VRAM before Stage 1 (its own subprocess) needs it -- mirrors the Ollama path's
        # keep_alive=0 unload; there's no daemon managing residency here.
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return _parse_and_postprocess(reply, model_id)


def discover_object_classes(
    export_images_dir: Path,
    vlm_model: str,
    num_images: int,
    ollama_host: str | None = None,
    backend: str = "ollama",
    hf_model: str | None = None,
) -> list[str]:
    """Asks a vision-LLM to propose open-vocabulary object-class labels from `num_images`
    evenly-spaced frames in `export_images_dir` (Stage 0's export_dir/images, always .jpg).
    Returns a deduped, lowercase-normalized, whitespace-stripped list of labels.

    backend="ollama" (default) calls a local Ollama daemon with model tag `vlm_model`
    (unchanged legacy path). backend="hf" runs a locally-downloaded HF `hf_model` checkpoint
    via transformers instead.

    Raises RuntimeError if no images are found, the model is unreachable, the reply can't be
    parsed, or zero labels are discovered -- discovery failing silently into an empty --classes
    run is worse than a loud crash, since it would silently skip Stages 1-3.
    """
    if backend == "hf":
        if not hf_model:
            raise ValueError("discover_object_classes(backend='hf') requires hf_model to be set.")
        return _discover_object_classes_hf(export_images_dir, hf_model, num_images)
    return _discover_object_classes_ollama(export_images_dir, vlm_model, num_images, ollama_host=ollama_host)
