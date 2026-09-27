"""PaliGemma2-based detection helpers: an alternative to detection_utils.py's Grounding DINO
functions, selected via detect_and_segment.py's --detector_backend hf. Mirrors
detection_utils.py's (input_boxes, confidences, class_names) / {class_name: {"frame_idx",
"boxes", "confidences"}} return shapes so detect_and_segment.py's call sites can branch with
minimal changes.

Uses PaliGemma's *native* task-prefix detect prompt ("detect {label1} ; {label2} ; ..."), not
Gemma's chat/JSON box_2d convention -- gemma-3-12b-it/gemma-3-27b-it were both tested for box
detection and produced hallucinated/templated boxes (not grounded to real object positions);
PaliGemma2 (default: google/paligemma2-3b-pt-448) reliably grounds boxes at a fraction of the
size. Its reply is a run of 4 `<locNNNN>` tokens per detection on PaliGemma's own 0-1024
normalized grid (order ymin, xmin, ymax, xmax), see https://ai.google.dev/gemma/docs/paligemma.

Runs in-process: paligemma is natively supported by the project's pinned transformers version
and PaliGemma2-3B fits on one GPU with plain `.to(device)` (no accelerate/device_map="auto"
needed). detect_and_segment.py already runs as its own subprocess (see main_light.py), so
process exit is the GPU-memory cleanup -- no persistent worker/atexit machinery needed here.
"""
import re

import numpy as np

_LOC_RUN_RE = re.compile(r"((?:<loc\d{4}>){4})\s*([^;<]*)")
_LOC_TOKEN_RE = re.compile(r"<loc(\d{4})>")


def pad_box_xyxy(box, width, height, pad_frac):
    """Expand a box outward by pad_frac of its own width/height on each side, clamped to
    image bounds. Mitigates the undershoot failure mode found in testing (a box that
    doesn't fully enclose its object makes SAM2 truncate the mask at the wrong edge).
    Grounding DINO's boxes get no padding today and shouldn't -- its failure mode is
    different -- so this is scoped to the hf backend only."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    dx, dy = w * pad_frac, h * pad_frac
    return [
        max(0.0, x1 - dx),
        max(0.0, y1 - dy),
        min(float(width), x2 + dx),
        min(float(height), y2 + dy),
    ]


def _match_label(detected_label, candidate_classes):
    """Best-effort match of a detected (possibly rephrased) label back to one of
    candidate_classes: case-insensitive substring match in either direction, mirroring
    detection_utils.match_detections_to_classes's tolerance for free-text phrasing."""
    gl = detected_label.strip().lower()
    for cls in candidate_classes:
        cl = cls.strip().lower()
        if cl == gl or cl in gl or gl in cl:
            return cls
    return None


def match_detections_to_classes_hf(class_names, candidate_classes, warn=True):
    """PaliGemma-flavored counterpart to detection_utils.match_detections_to_classes. The model
    is prompted with the exact candidate_classes strings (not free-vocabulary captioning), so
    matching is closer to exact, but a tolerant substring match is kept as a safety net
    since it sometimes rephrases. Every detection carries the same confidence (1.0), so unlike
    the Grounding DINO version there is no meaningful best-of-N tie-break within a single
    frame -- first match per class wins.

    Returns (kept_indices, matched_classes), same shape as
    detection_utils.match_detections_to_classes.
    """
    kept_indices, matched_classes, seen_classes = [], [], set()
    for i, label in enumerate(class_names):
        cls = _match_label(label, candidate_classes)
        if cls is None:
            if warn:
                print(f"[warn] dropping unmatched detection: '{label}'")
            continue
        if cls in seen_classes:
            continue
        seen_classes.add(cls)
        kept_indices.append(i)
        matched_classes.append(cls)
    return kept_indices, matched_classes


def paligemma_loc_run_to_xyxy(loc_run, width, height):
    """Descale one <loc><loc><loc><loc> run (PaliGemma's own 0-1024 grid, order
    ymin, xmin, ymax, xmax) to absolute pixel xyxy."""
    ymin, xmin, ymax, xmax = (int(n) for n in _LOC_TOKEN_RE.findall(loc_run))
    return [xmin / 1024 * width, ymin / 1024 * height, xmax / 1024 * width, ymax / 1024 * height]


class PaliGemmaDetector:
    """Loads a PaliGemma(2) checkpoint once for the whole Stage 1 detection run (this script is
    already its own subprocess -- see detect_and_segment.py -- so there's no persistent-worker
    or atexit cleanup needed; process exit frees the GPU memory)."""

    def __init__(self, model_id):
        import torch
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.bfloat16
        ).to(self._device)
        self._model.eval()

    def detect(self, image_path, candidate_classes):
        """Runs one PaliGemma detect call on a single frame, detecting all candidate_classes at
        once. Returns the raw decoded reply (kept special tokens -- the <locNNNN> tokens ARE the
        output to parse)."""
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        prompt = "detect " + " ; ".join(candidate_classes)
        inputs = self._processor(text=prompt, images=image, return_tensors="pt").to(
            self._device, self._model.dtype
        )
        input_len = inputs["input_ids"].shape[-1]
        with self._torch.inference_mode():
            generation = self._model.generate(**inputs, max_new_tokens=200, do_sample=False)
            generation = generation[0][input_len:]
        return self._processor.decode(generation, skip_special_tokens=False)


def detect_frame_boxes_hf(image_path, candidate_classes, width, height, detector, box_padding_frac=0.0):
    """Runs PaliGemma on a single frame with ONE joint prompt ("detect a ; b ; c"). Returns
    (input_boxes: np.ndarray[N, 4] absolute xyxy, confidences: list[float] (always 1.0 -- no
    native per-box confidence, unlike Grounding DINO), class_names: list[str] (raw returned
    labels; match to candidate_classes via match_detections_to_classes_hf)). Note the joint
    prompt can silently omit a class that is visible; callers should check for missing classes."""
    input_boxes, confidences, class_names = [], [], []
    decoded = detector.detect(image_path, list(candidate_classes))
    for loc_run, label in _LOC_RUN_RE.findall(decoded):
        box = paligemma_loc_run_to_xyxy(loc_run, width, height)
        if box_padding_frac:
            box = pad_box_xyxy(box, width, height, box_padding_frac)
        input_boxes.append(box)
        confidences.append(1.0)
        class_names.append(label.strip())

    if not input_boxes:
        return np.zeros((0, 4), dtype=np.float32), [], []
    return np.array(input_boxes, dtype=np.float32), confidences, class_names


def detect_classes_in_frames_hf(video_dir, frame_names, candidate_classes, detector,
                                 box_padding_frac=0.0):
    """PaliGemma-flavored counterpart to detection_utils.detect_classes_in_all_frames. Assumes
    the input is exactly a stereo pair (a left and a right frame) and scans both with one joint
    multi-class prompt per frame; the first frame containing a class wins. Returns
    {class_name: {"frame_idx", "boxes", "confidences"}}, matching detect_classes_in_all_frames's
    shape. Used only for manual --classes runs; discovery runs take their boxes from Qwen3-VL
    (genrecon/utils/object_discovery.py)."""
    import os

    from PIL import Image

    if len(frame_names) != 2:
        raise ValueError(
            f"PaliGemma class scan expects exactly 2 frames (left and right), got "
            f"{len(frame_names)} in {video_dir}."
        )
    class_to_detections = {}

    for frame_idx in range(len(frame_names)):
        img_path = os.path.join(video_dir, frame_names[frame_idx])
        with Image.open(img_path) as im:
            width, height = im.size

        input_boxes, confidences, class_names = detect_frame_boxes_hf(
            img_path, candidate_classes, width, height, detector, box_padding_frac=box_padding_frac,
        )
        kept_indices, matched_classes = match_detections_to_classes_hf(
            class_names, candidate_classes, warn=False)

        for i, cls in zip(kept_indices, matched_classes):
            if cls in class_to_detections:
                continue  # first frame containing this class wins
            class_to_detections[cls] = {
                "frame_idx": frame_idx,
                "boxes": [input_boxes[i].tolist()],
                "confidences": [confidences[i]],
            }

    for cls in candidate_classes:
        if cls not in class_to_detections:
            print(f"[warn] class '{cls}' not detected in either frame")

    return class_to_detections
