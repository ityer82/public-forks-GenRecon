"""Caption a pick-and-place episode: a local Qwen3-VL checkpoint (the same one object_discovery
uses) is shown a chronological subset of frames from one camera in a single multi-image
message and asked which object the robot picks up and where it places it.
"""
import gc
import json
import re
from pathlib import Path

from genrecon.utils.logger import logger

SURFACE_WORDS = {"table", "table surface", "tabletop", "surface", "floor", "ground", "desk"}  # not distinct objects

MAX_SIDE = 768  # downscale frames so the long side is <= this, bounding the visual token count

CAPTION_PROMPT = (
    "These frames are in chronological order from a robot performing a pick-and-place task. "
    "Identify the object the robot picks up (the one its gripper grasps and lifts) and where it "
    "puts it down (the surface, object or container it ends up on or in). Describe the object by "
    "its color, material and kind. Also say which of the robot's two hands picks it up: the camera "
    "is the robot's head looking forward, so the robot's left hand appears on the left side of the "
    "image and its right hand on the right. Output only JSON: "
    "{\"picked_object\": str, \"hand\": \"left\" or \"right\", \"placed_location\": str, "
    "\"summary\": str} where summary is one or two sentences describing the whole action, "
    "including which hand was used."
)


def _build_caption_prompt(objects: list[str] | None) -> str:
    """CAPTION_PROMPT, prefixed with the detected object labels (if any) so the answer uses them."""
    if not objects:
        return CAPTION_PROMPT
    return (
        f"The objects on the table are: {', '.join(objects)}. "
        + CAPTION_PROMPT
        + " Write picked_object exactly as one of those object labels, and write placed_location "
        "as one of those labels too (the object or container it ends up on or in), or as the "
        "table surface if it is put down on the bare table."
    )


def _parse_json_object(text: str) -> dict | None:
    """Tolerant parse: the model wraps its reply in ```json fences or prose."""
    match = re.search(r"\{.*\}", text, re.S)
    if match is None:
        return None
    try:
        parsed = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def caption_pick_and_place(
    image_paths: list[Path], model_id: str, objects: list[str] | None = None, max_new_tokens: int = 400,
) -> dict:
    """Returns {"picked_object", "hand", "placed_location", "summary", "raw_reply"}. All but raw_reply are
    None if the reply isn't a parsable JSON object (raw_reply is always kept)."""
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    logger.info(f"Caption: querying {model_id} on {len(image_paths)} frames")
    model = None
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()

        content = []
        for i, path in enumerate(image_paths, 1):
            image = Image.open(path).convert("RGB")
            image.thumbnail((MAX_SIDE, MAX_SIDE))
            content.append({"type": "text", "text": f"Frame {i} ({path.stem}):"})
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": _build_caption_prompt(objects)})
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        reply = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    except Exception as e:
        raise RuntimeError(f"Captioning failed calling HF model '{model_id}': {e}") from e
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    parsed = _parse_json_object(reply) or {}
    if not parsed:
        logger.warning(f"Caption: could not parse a JSON object from the reply:\n{reply}")
    return {
        "picked_object": parsed.get("picked_object"),
        "hand": parsed.get("hand"),
        "placed_location": parsed.get("placed_location"),
        "summary": parsed.get("summary"),
        "raw_reply": reply,
    }


def sample_frames(paths: list[Path], n: int) -> list[Path]:
    """`n` evenly spaced paths (first and last included) from the chronologically sorted `paths`."""
    if not paths:
        return []
    last = len(paths) - 1
    count = min(n, len(paths))
    idx = sorted({round(i * last / (count - 1)) if count > 1 else 0 for i in range(count)})
    return [paths[i] for i in idx]


def select_pick_place(caption: dict, classes: list[str]) -> tuple[str | None, str | None]:
    """(pick, place) class labels for the scene from a caption: the picked object if it is one of
    `classes`, and the placed-on/in object if it is another one of `classes`. Each is returned in
    its `classes` spelling (matched case/whitespace-insensitively), or None when unusable --
    place is None for a bare surface such as the table."""
    by_key = {c.strip().lower(): c for c in classes}

    def match(value):
        return by_key.get((value or "").strip().lower())

    pick, place = match(caption.get("picked_object")), match(caption.get("placed_location"))
    return pick, (place if place != pick else None)
