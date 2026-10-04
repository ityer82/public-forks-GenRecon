"""One-shot object discovery + detection: pick-and-place mode's only box-retrieval mechanism. A
local Qwen3-VL checkpoint is queried once per image of the stereo pair (left, right) and
returns a short visually specific label and a bounding box for each object found. For a genuine
stereo pair, boxes are paired across the two views by y-axis overlap (the pair is roughly
rectified, so the same object occupies the same rows in both views); an object is kept only when
its left and right boxes agree, and is named by its left label. When no genuine stereo pair is
available and the first two frames of a sequential capture are used as a stand-in (see
_find_stereo_pair), the y-overlap premise doesn't hold -- the two frames are just two moments of
a moving camera, not a synchronized rig -- so that cross-view check is skipped and every
left-view detection is trusted directly.

With no explicit class list, the prompt free-invents both labels and boxes for everything on
the table ("discovery"). With an explicit --classes list, the prompt is conditioned to look for
exactly those objects instead, and each detection's label is reconciled back to its requested
spelling (see discover_objects's `classes` param).

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
    "object on the table; if several instances of the same kind exist, list each one separately. Give each a short, visually specific label (start with its most "
    "distinctive color or material, so similar objects such as a plate and a bowl can be told "
    "apart) and a tight bounding box. Output only JSON: "
    "[{\"label\": str, \"bbox_2d\": [x1, y1, x2, y2]}] with coordinates relative to the image on "
    "a 0-1000 grid."
)

MIN_Y_IOU = 0.6  # minimum y-range overlap for a left/right box pair


def _build_prompt(classes: list[str] | None) -> str:
    """The unconditioned free-discovery prompt (DETECTION_PROMPT) when `classes` is empty,
    otherwise a variant conditioned to look for exactly those objects."""
    if not classes:
        return DETECTION_PROMPT
    class_list = ", ".join(classes)
    return (
        "This is a tabletop scene. Ignore robot arms, the table and background. Find each of "
        f"the following objects if it is present on the table: {class_list}. For each one "
        "found, give its exact label copied verbatim from that list and a tight bounding box. "
        "If several instances of one listed object exist, report each one separately. "
        "Do not report any object that isn't in the list. Output only JSON: "
        "[{\"label\": str, \"bbox_2d\": [x1, y1, x2, y2]}] with coordinates relative to the "
        "image on a 0-1000 grid."
    )


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


def _find_stereo_pair(images_dir: Path) -> tuple[Path, Path, bool]:
    """A literal left/right-named pair (a genuine 2-image stereo capture) is used as-is. Otherwise
    this is a multi-frame export (N sequential frames from a single moving camera, named e.g.
    frame_000001.jpg) -- fall back to its first two frames by filename, treated as the "left" and
    "right" views for detection-box pairing purposes. Returns (left_path, right_path, is_stereo);
    is_stereo is False in the fallback case, telling discover_objects to skip the y-overlap
    cross-view check (see module docstring)."""
    images = sorted(images_dir.glob("*.jpg"))
    left = [p for p in images if "left" in p.name.lower()]
    right = [p for p in images if "right" in p.name.lower()]
    if len(images) == 2 and len(left) == 1 and len(right) == 1:
        return left[0], right[0], True
    if len(images) >= 2:
        logger.info(
            f"Discovery: no left/right-named stereo pair in {images_dir}; falling back to its "
            f"first two frames ({images[0].name}, {images[1].name}) as the left/right views."
        )
        return images[0], images[1], False
    raise RuntimeError(
        f"Object discovery needs at least 2 .jpg images in {images_dir}, found {[p.name for p in images]}."
    )


def discover_objects(
    export_images_dir: Path, model_id: str, classes: list[str] | None = None,
) -> tuple[str, list[dict]]:
    """Returns (left_image_name, [{"label", "left_box", "right_box"}]) (pixel xyxy on the
    left/right image). left_image_name is the actual filename used as the "left" view (see
    _find_stereo_pair) -- callers must use it rather than re-deriving it from a "left" substring
    match, which only holds for a genuine 2-image stereo capture. Raises RuntimeError if there
    aren't at least 2 images, the model fails or its reply can't be parsed, or no object
    survives the left/right pairing -- an empty result would silently skip Stages 1-3.

    With `classes` given, the prompt is conditioned to look for exactly those objects (see
    _build_prompt) and each detection's label is reconciled back to its requested spelling
    (case-insensitive match), so labels come back identical to the `classes` strings --
    downstream --pick_place_target/--place-target matching and mesh dirnames depend on that
    exact identity. Detections that don't match any requested class are dropped, and a warning
    is logged for any requested class that goes undetected. Labels are lowercased only in the
    unconditioned (classes=None) free-discovery case."""
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    left_path, right_path, is_stereo = _find_stereo_pair(export_images_dir)
    prompt = _build_prompt(classes)
    logger.info(
        f"Discovery: querying {model_id} on {left_path.name} and {right_path.name}"
        + (f" (conditioned on classes: {', '.join(classes)})" if classes else "")
    )

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
                {"type": "text", "text": prompt},
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

    if is_stereo:
        paired = pair_by_y_overlap(*per_view)
    else:
        logger.info(
            "Discovery: not a genuine stereo pair -- skipping the y-overlap cross-view check "
            "and trusting every left-view detection directly."
        )
        paired = [(label, box, box) for label, box in per_view[0]]

    named = []
    for label, left_box, right_box in paired:
        if classes:
            matched = next((c for c in classes if c.strip().lower() == label), None)
            if matched is None:
                logger.warning(f"Discovery: dropping detection '{label}' -- not in --classes")
                continue
            label = matched
        named.append((label, left_box, right_box))

    # Several instances of one label (e.g. two knives) become separate classes "<label> 1",
    # "<label> 2", ... in left-to-right order, because every later stage keys masks, meshes and
    # prims by a single label per object. A label seen once keeps its plain name.
    counts = {}
    for label, _, _ in named:
        counts[label] = counts.get(label, 0) + 1
    objects, next_index = [], {}
    for label, left_box, right_box in sorted(named, key=lambda d: d[1][0]):
        if counts[label] > 1:
            next_index[label] = next_index.get(label, 0) + 1
            label = f"{label} {next_index[label]}"
        objects.append({"label": label, "left_box": left_box, "right_box": right_box})
    seen_base = {l for l, _, _ in named}

    if classes:
        missing = [c for c in classes if c not in seen_base]
        if missing:
            logger.warning(f"Discovery: requested class(es) not detected: {', '.join(missing)}")

    if not objects:
        if is_stereo:
            raise RuntimeError(
                f"Object discovery: no object had matching left and right boxes "
                f"(left={len(per_view[0])}, right={len(per_view[1])} detections)."
            )
        raise RuntimeError(f"Object discovery: no object detected in {left_path.name}.")
    return left_path.name, objects
