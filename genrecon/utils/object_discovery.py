"""One-shot open-vocabulary object discovery + detection for --discover-classes (an alternative
to manually typing --classes). A local Qwen3-VL checkpoint is queried once per image of the
stereo pair (left, right) and returns, for every object on the table, a short visually
specific label and a bounding box. Boxes are paired across the two views by y-axis overlap
(the pair is roughly rectified, so the same object occupies the same rows in both views); an
object is kept only when its left and right boxes agree, and is named by its left label.

Asking for both views' boxes in a single call was tried and makes the model copy the left
boxes into the right view, so each image is queried on its own.
"""
import gc
import json
import re
from pathlib import Path

from genrecon.utils.logger import logger

DETECTION_PROMPT = (
    "This is a tabletop scene. Ignore robot arms, the table and background. List every distinct "
    "object on the table. Give each a short, visually specific label (start with its most "
    "distinctive color or material, so similar objects such as a plate and a bowl can be told "
    "apart) and a tight bounding box. Output only JSON: "
    "[{\"label\": str, \"bbox_2d\": [x1, y1, x2, y2]}] with coordinates relative to the image on "
    "a 0-1000 grid."
)

MIN_Y_IOU = 0.6  # minimum y-range overlap for a left/right box pair


def _parse_json_list(text: str) -> list | None:
    """Tolerant parse: the model wraps its reply in ```json fences or prose."""
    match = re.search(r"\[.*\]", text, re.S)
    if match is None:
        return None
    try:
        parsed = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, list) else None


def _detections_from_reply(reply: str, width: int, height: int) -> list[tuple[str, list[float]]]:
    """[(label, [x1, y1, x2, y2] in pixels)]; malformed entries and degenerate boxes dropped."""
    parsed = _parse_json_list(reply)
    if parsed is None:
        raise RuntimeError(f"Object discovery: could not parse a JSON list from the reply:\n{reply}")
    out = []
    for det in parsed:
        try:
            label = str(det["label"]).strip().lower()
            x1, y1, x2, y2 = (float(v) for v in det["bbox_2d"])
        except (KeyError, TypeError, ValueError):
            continue
        if not label or not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            continue
        out.append((label, [x1 / 1000 * width, y1 / 1000 * height, x2 / 1000 * width, y2 / 1000 * height]))
    return out


def _y_iou(a: list[float], b: list[float]) -> float:
    inter = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = max(a[3], b[3]) - min(a[1], b[1])
    return inter / union if union > 0 else 0.0


def pair_by_y_overlap(left, right, min_y_iou: float = MIN_Y_IOU):
    """Greedy best-first one-to-one pairing of left and right (label, box) detections by y-range
    IoU. Returns [(left_label, left_box, right_box)] in left-detection order; left detections
    with no partner are omitted (logged)."""
    # Objects in the same table row overlap in y with each other too, so an identical label
    # breaks the tie (it only orders candidates; y-overlap is still what qualifies a pair).
    scored = sorted(
        ((left[i][0] == right[j][0], _y_iou(left[i][1], right[j][1]), i, j)
         for i in range(len(left)) for j in range(len(right))),
        reverse=True,
    )
    partner, used_right = {}, set()
    for _, iou, i, j in scored:
        if iou < min_y_iou:
            continue
        if i not in partner and j not in used_right:
            partner[i] = j
            used_right.add(j)
    pairs = []
    for i, (label, box) in enumerate(left):
        if i in partner:
            pairs.append((label, box, right[partner[i]][1]))
        else:
            logger.warning(f"Discovery: dropping '{label}' -- no right-view box overlaps it in y")
    return pairs


def _find_stereo_pair(images_dir: Path) -> tuple[Path, Path]:
    images = sorted(images_dir.glob("*.jpg"))
    left = [p for p in images if "left" in p.name.lower()]
    right = [p for p in images if "right" in p.name.lower()]
    if len(images) != 2 or len(left) != 1 or len(right) != 1:
        raise RuntimeError(
            f"Object discovery expects exactly one left and one right .jpg in {images_dir}, "
            f"found {[p.name for p in images]}."
        )
    return left[0], right[0]


def discover_objects(export_images_dir: Path, model_id: str) -> list[dict]:
    """Returns [{"label", "left_box", "right_box"}] (pixel xyxy on the left/right image), labels
    unique and lowercase. Raises RuntimeError if the images are not a stereo pair, the model
    fails or its reply can't be parsed, or no object survives the left/right pairing -- an empty
    result would silently skip Stages 1-3."""
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    left_path, right_path = _find_stereo_pair(export_images_dir)
    logger.info(f"Discovery: querying {model_id} on {left_path.name} and {right_path.name}")

    model = None
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()

        per_view = []
        for path in (left_path, right_path):
            image = Image.open(path).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": DETECTION_PROMPT},
            ]}]
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=600, do_sample=False)
            reply = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            per_view.append(_detections_from_reply(reply, *image.size))
    except Exception as e:
        raise RuntimeError(f"Object discovery failed calling HF model '{model_id}': {e}") from e
    finally:
        # Free VRAM before Stage 1 and MV-SAM3D need it.
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    objects, seen = [], set()
    for label, left_box, right_box in pair_by_y_overlap(*per_view):
        if label in seen:
            logger.warning(f"Discovery: dropping duplicate label '{label}'")
            continue
        seen.add(label)
        objects.append({"label": label, "left_box": left_box, "right_box": right_box})

    if not objects:
        raise RuntimeError(
            f"Object discovery: no object had matching left and right boxes "
            f"(left={len(per_view[0])}, right={len(per_view[1])} detections)."
        )
    return objects
