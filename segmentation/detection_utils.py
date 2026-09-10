"""GroundingDINO detection helpers shared across the anchor-frame detect step,
the empty-mask recovery loop, and the multi-frame per-class detection scan in
detect_and_segment.py.
"""
import os

import numpy as np
import torch
from torchvision.ops import box_convert

from groundingdino.util.inference import load_image, predict


def detect_frame_boxes(grounding_model, image_source, image, caption, box_threshold, text_threshold, remove_combined):
    """Runs GroundingDINO on a single already-loaded frame.

    Returns (input_boxes: np.ndarray[N, 4] absolute xyxy, confidences: list[float],
    class_names: list[str]).
    """
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=caption,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        remove_combined=remove_combined,
    )
    if boxes.numel() == 0:
        return np.zeros((0, 4), dtype=np.float32), [], []
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    return input_boxes, confidences.numpy().tolist(), labels


def match_detections_to_classes(class_names, confidences, candidate_classes, warn=True):
    """Matches free-form GroundingDINO phrases to a fixed candidate-class list.

    For each candidate class, keeps only its single highest-confidence
    detection -- one instance per class.

    Returns (kept_indices, matched_classes): parallel lists indexing into the
    original class_names/confidences, one entry per surviving detection.
    """
    confidences_arr = np.array(confidences)
    class_ids = []
    for phrase in class_names:
        for i, cls in enumerate(candidate_classes):
            if cls in phrase:
                class_ids.append(i)
                break
        else:
            class_ids.append(None)
            if warn:
                print(f"[warn] dropping unmatched detection: '{phrase}'")

    matched_indices = [i for i, cid in enumerate(class_ids) if cid is not None]

    kept_indices = []
    for cls_idx in range(len(candidate_classes)):
        cls_member_indices = [i for i in matched_indices if class_ids[i] == cls_idx]
        if not cls_member_indices:
            continue
        best_i = max(cls_member_indices, key=lambda i: confidences_arr[i])
        kept_indices.append(best_i)
    kept_indices = sorted(kept_indices)
    matched_classes = [candidate_classes[class_ids[i]] for i in kept_indices]
    return kept_indices, matched_classes


def detect_classes_in_all_frames(video_dir, frame_names, grounding_model, candidate_classes,
                                  detect_caption, box_threshold=0.3, text_threshold=0.45):
    """Scans every frame for the given candidate classes.

    Unlike single-anchor-frame detection, a class missing from frame 0 is
    still found if it appears anywhere else in the video. For each class,
    only its single best (highest max-confidence) frame is kept as the SAM2
    seed, and within that frame only the single highest-confidence detection
    of the class is kept -- so a class contributes exactly one box from
    exactly one frame, avoiding duplicate-detection/duplicate-tracking of
    the same object.

    Returns {class_name: {"frame_idx": int, "boxes": list[[x1, y1, x2, y2]]
    (always length 1), "confidences": list[float]}} for every class detected
    in at least one frame. Classes never detected anywhere are omitted
    (caller should warn).
    """
    class_to_detections = {}
    for frame_idx, frame_name in enumerate(frame_names):
        img_path = os.path.join(video_dir, frame_name)
        image_source, image = load_image(img_path)
        input_boxes, confidences, class_names = detect_frame_boxes(
            grounding_model, image_source, image, detect_caption,
            box_threshold, text_threshold, remove_combined=True,
        )
        if len(class_names) == 0:
            continue
        kept_indices, matched_classes = match_detections_to_classes(
            class_names, confidences, candidate_classes, warn=False)
        if not kept_indices:
            continue

        per_class_here = {}
        for i, cls in zip(kept_indices, matched_classes):
            per_class_here.setdefault(cls, []).append(i)

        for cls, idxs in per_class_here.items():
            frame_max_confidence = max(confidences[i] for i in idxs)
            best_so_far = class_to_detections.get(cls)
            if best_so_far is not None and best_so_far["confidence"] >= frame_max_confidence:
                continue
            class_to_detections[cls] = {
                "frame_idx": frame_idx,
                "boxes": [input_boxes[i].tolist() for i in idxs],
                "confidences": [confidences[i] for i in idxs],
                "confidence": frame_max_confidence,
            }

    for cls in candidate_classes:
        if cls not in class_to_detections:
            print(f"[warn] class '{cls}' not detected in any frame")

    for entry in class_to_detections.values():
        del entry["confidence"]

    return class_to_detections
