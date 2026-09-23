"""PaliGemma2-based detection helpers: an alternative to gemma_detection_utils.py's Ollama/Gemma
box detection, selected via detect_and_segment.py's --detector_backend hf. Mirrors
gemma_detection_utils.py's (input_boxes, confidences, class_names) / {class_name: {"frame_idx",
"boxes", "confidences"}} return shapes so detect_and_segment.py's call sites can branch with
minimal changes.

Uses PaliGemma's *native* task-prefix detect prompt ("detect {label1} ; {label2} ; ..."), not
Gemma's chat/JSON box_2d convention -- gemma-3-12b-it/gemma-3-27b-it were both tested for box
detection and produced hallucinated/templated boxes (not grounded to real object positions);
PaliGemma2 (default: google/paligemma2-3b-pt-448) reliably grounds boxes at a fraction of the
size. Its reply is a run of 4 `<locNNNN>` tokens per detection on PaliGemma's own 0-1024
normalized grid (order ymin, xmin, ymax, xmax) -- distinct from Gemma's documented 0-1000
box_2d grid, see https://ai.google.dev/gemma/docs/paligemma.

Runs in-process: paligemma is natively supported by the project's pinned transformers version
(unlike gemma4, which needed an isolated transformers>=5 subprocess), and PaliGemma2-3B fits on
one GPU with plain `.to(device)` (no accelerate/device_map="auto" needed). detect_and_segment.py
already runs as its own subprocess (see main_light.py), so process exit is the GPU-memory
cleanup -- no persistent worker/atexit machinery needed here, unlike the old gemma4 design.
"""
import re

import numpy as np

from gemma_detection_utils import (
    pad_box_xyxy,
    match_detections_to_classes_gemma,
    _select_evenly_spaced_indices,
)

_LOC_RUN_RE = re.compile(r"((?:<loc\d{4}>){4})\s*([^;<]*)")
_LOC_TOKEN_RE = re.compile(r"<loc(\d{4})>")


def paligemma_loc_run_to_xyxy(loc_run, width, height):
    """Descale one <loc><loc><loc><loc> run (PaliGemma's own 0-1024 grid, order
    ymin, xmin, ymax, xmax -- NOT Gemma's 0-1000 box_2d grid) to absolute pixel xyxy."""
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
    """Runs one PaliGemma detect call on a single frame via `detector`, detecting all
    candidate_classes at once. Returns (input_boxes: np.ndarray[N, 4] absolute xyxy,
    confidences: list[float] (always 1.0 -- no native per-box confidence, same as the Ollama/
    Gemma backend), class_names: list[str] (raw returned labels; match to candidate_classes via
    match_detections_to_classes_gemma))."""
    decoded = detector.detect(image_path, candidate_classes)

    input_boxes, confidences, class_names = [], [], []
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
                                 num_sample_frames=8, box_padding_frac=0.0):
    """PaliGemma-flavored counterpart to gemma_detection_utils.detect_classes_in_frames_gemma.
    Same even-sampling-of-num_sample_frames tradeoff (a per-frame VLM call is too slow to run on
    every frame). Returns {class_name: {"frame_idx", "boxes", "confidences"}}, matching
    detect_classes_in_frames_gemma's shape."""
    import os

    from PIL import Image

    sample_idxs = _select_evenly_spaced_indices(len(frame_names), num_sample_frames)
    class_to_detections = {}

    for frame_idx in sample_idxs:
        img_path = os.path.join(video_dir, frame_names[frame_idx])
        with Image.open(img_path) as im:
            width, height = im.size

        input_boxes, confidences, class_names = detect_frame_boxes_hf(
            img_path, candidate_classes, width, height, detector, box_padding_frac=box_padding_frac,
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
