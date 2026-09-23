"""Gemma-via-Ollama detection helpers: an alternative to detection_utils.py's Grounding
DINO functions, selected via detect_and_segment.py's --detector_backend gemma. Mirrors
detection_utils.py's (input_boxes, confidences, class_names) / {class_name: {"frame_idx",
"boxes", "confidences"}} return shapes so detect_and_segment.py's call sites can branch
with minimal changes.

Validated empirically (see project conversation history): gemma4:31b's box_2d output,
descaled from Google's documented 0-1000-normalized [y1, x1, y2, x2] grid
(https://ai.google.dev/gemma/docs/capabilities/vision/image), produced SAM2 box-prompt
masks scoring 0.92-0.99 across 18 test boxes spanning 4 scenes. The smaller gemma4:12b tag
undershot object extents often enough (~44% of boxes) to silently break SAM2 (small
fragment masks, not reliably flagged by SAM2's own score) -- gemma4:12b is not recommended
for this role.
"""
import base64
import json
import os
import urllib.request

import numpy as np

BOX2D_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "box_2d": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4},
            "label": {"type": "string"},
        },
        "required": ["box_2d", "label"],
    },
}


def _b64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _call_ollama(prompt, image_b64, model, ollama_host=None, timeout=180):
    # NOTE: never pass num_predict on a request that includes images -- image tokens count
    # against that budget, and even a generous-looking cap (e.g. 500-600) can make Ollama
    # return an EMPTY response with done_reason="length" before any text is generated.
    # Hit this directly during testing; omit num_predict entirely for image-bearing calls.
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "format": BOX2D_SCHEMA,
        "options": {"temperature": 0},
        # keep_alive=0: unload the model right after this call. Without this, Ollama's
        # default 5-minute keep-alive leaves gemma4:31b (~19GB) resident in GPU memory
        # well past Stage 1, and Stage 3 (MV-SAM3D) starting within that window has caused
        # real CUDA OOMs (see runs/t2/pipeline.log's first attempt).
        "keep_alive": 0,
    }
    url = f"{(ollama_host or 'http://localhost:11434').rstrip('/')}/api/generate"
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read())
    return result.get("response", "")


def gemma_box_2d_to_xyxy(box_2d, width, height):
    """Descale Gemma's normalized [y1, x1, y2, x2] (0-1000 grid) to absolute pixel xyxy."""
    y1, x1, y2, x2 = box_2d
    return [
        x1 / 1000 * width,
        y1 / 1000 * height,
        x2 / 1000 * width,
        y2 / 1000 * height,
    ]


def pad_box_xyxy(box, width, height, pad_frac):
    """Expand a box outward by pad_frac of its own width/height on each side, clamped to
    image bounds. Mitigates the undershoot failure mode found in testing (a box that
    doesn't fully enclose its object makes SAM2 truncate the mask at the wrong edge).
    Grounding DINO's boxes get no padding today and shouldn't -- its failure mode is
    different -- so this is scoped to the gemma backend only."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    dx, dy = w * pad_frac, h * pad_frac
    return [
        max(0.0, x1 - dx),
        max(0.0, y1 - dy),
        min(float(width), x2 + dx),
        min(float(height), y2 + dy),
    ]


def _match_label(gemma_label, candidate_classes):
    """Best-effort match of Gemma's (possibly rephrased) label back to one of
    candidate_classes: case-insensitive substring match in either direction, mirroring
    detection_utils.match_detections_to_classes's tolerance for free-text phrasing."""
    gl = gemma_label.strip().lower()
    for cls in candidate_classes:
        cl = cls.strip().lower()
        if cl == gl or cl in gl or gl in cl:
            return cls
    return None


def detect_frame_boxes_gemma(image_path, candidate_classes, width, height, ollama_model,
                              ollama_host=None, box_padding_frac=0.0):
    """Runs one Gemma/Ollama call on a single frame, detecting all candidate_classes at
    once (validated as reliable and far cheaper than one-object-per-call).

    Returns (input_boxes: np.ndarray[N, 4] absolute xyxy, confidences: list[float] (always
    1.0 -- Gemma has no native per-box confidence, unlike Grounding DINO), class_names:
    list[str] (Gemma's raw returned labels; match to candidate_classes via
    match_detections_to_classes_gemma)).
    """
    prompt = (
        f"Detect each of the following objects, if present: {', '.join(candidate_classes)}. "
        f"Output only ```json"
    )
    raw = _call_ollama(prompt, _b64_image(image_path), ollama_model, ollama_host)
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return np.zeros((0, 4), dtype=np.float32), [], []

    input_boxes, confidences, class_names = [], [], []
    for det in parsed:
        box_2d = det.get("box_2d")
        label = det.get("label")
        if not box_2d or not label or len(box_2d) != 4:
            continue
        box = gemma_box_2d_to_xyxy(box_2d, width, height)
        if box_padding_frac:
            box = pad_box_xyxy(box, width, height, box_padding_frac)
        input_boxes.append(box)
        confidences.append(1.0)
        class_names.append(label)

    if not input_boxes:
        return np.zeros((0, 4), dtype=np.float32), [], []
    return np.array(input_boxes, dtype=np.float32), confidences, class_names


def match_detections_to_classes_gemma(class_names, candidate_classes, warn=True):
    """Gemma-flavored counterpart to detection_utils.match_detections_to_classes. Gemma is
    prompted with the exact candidate_classes strings (not free-vocabulary captioning), so
    matching is closer to exact, but a tolerant substring match is kept as a safety net
    since Gemma sometimes rephrases (e.g. "grey bowl" vs "small bowl"). Every detection
    carries the same confidence (1.0), so unlike the Grounding DINO version there is no
    meaningful best-of-N tie-break within a single frame -- first match per class wins.

    Returns (kept_indices, matched_classes), same shape as
    detection_utils.match_detections_to_classes.
    """
    kept_indices, matched_classes, seen_classes = [], [], set()
    for i, label in enumerate(class_names):
        cls = _match_label(label, candidate_classes)
        if cls is None:
            if warn:
                print(f"[warn] dropping unmatched Gemma detection: '{label}'")
            continue
        if cls in seen_classes:
            continue
        seen_classes.add(cls)
        kept_indices.append(i)
        matched_classes.append(cls)
    return kept_indices, matched_classes


def _select_evenly_spaced_indices(n_total, n_sample):
    if n_sample <= 0 or n_sample >= n_total:
        return list(range(n_total))
    step = (n_total - 1) / (n_sample - 1) if n_sample > 1 else 0
    return sorted({round(i * step) for i in range(n_sample)})


def detect_classes_in_frames_gemma(video_dir, frame_names, candidate_classes, ollama_model,
                                    num_sample_frames=8, ollama_host=None, box_padding_frac=0.0):
    """Gemma-flavored counterpart to detection_utils.detect_classes_in_all_frames.

    Cost asymmetry vs. Grounding DINO: Grounding DINO is cheap enough to run on every
    frame in the folder; a per-frame VLM call is not. Instead of scanning every frame,
    evenly samples num_sample_frames (mirroring genrecon/utils/object_discovery.py's
    _select_evenly_spaced pattern) and runs one Gemma call per sampled frame requesting
    ALL candidate_classes at once. This trades "guaranteed single best frame" (Grounding
    DINO's true highest-confidence pick) for "good enough frame, few Gemma calls" -- a
    deliberate, documented tradeoff. Since Gemma has no real per-box confidence, the first
    sampled frame containing a class wins (no best-frame tie-break).

    Returns {class_name: {"frame_idx": int, "boxes": list[[x1,y1,x2,y2]] (always length
    1), "confidences": list[float] (always [1.0])}}, matching
    detect_classes_in_all_frames's shape.
    """
    from PIL import Image

    sample_idxs = _select_evenly_spaced_indices(len(frame_names), num_sample_frames)
    class_to_detections = {}

    for frame_idx in sample_idxs:
        img_path = os.path.join(video_dir, frame_names[frame_idx])
        with Image.open(img_path) as im:
            width, height = im.size

        input_boxes, confidences, class_names = detect_frame_boxes_gemma(
            img_path, candidate_classes, width, height, ollama_model,
            ollama_host=ollama_host, box_padding_frac=box_padding_frac,
        )
        if len(class_names) == 0:
            continue

        kept_indices, matched_classes = match_detections_to_classes_gemma(
            class_names, candidate_classes, warn=False)

        for i, cls in zip(kept_indices, matched_classes):
            if cls in class_to_detections:
                continue  # first sampled frame containing this class wins
            class_to_detections[cls] = {
                "frame_idx": frame_idx,
                "boxes": [input_boxes[i].tolist()],
                "confidences": [confidences[i]],
            }

    for cls in candidate_classes:
        if cls not in class_to_detections:
            print(f"[warn] class '{cls}' not detected in any sampled frame")

    return class_to_detections
