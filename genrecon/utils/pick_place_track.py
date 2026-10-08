"""Segment + track one named object through a single camera's frames: a local Qwen3-VL checkpoint
(the one object_discovery uses) finds the object's box in a few sampled frames, and the best box
seeds Stage 1's GroundedSAM2 video tracker (segmentation/detect_and_segment.py), which
propagates the mask through every frame.
"""
import gc
import json
import shutil
import sys
from pathlib import Path

from genrecon.utils.logger import logger
from genrecon.utils.object_discovery import _detections_from_reply

REPO_DIR = Path(__file__).resolve().parents[2]

EDGE_PX = 2  # a box edge within this many pixels of the image border counts as touching it

LOCATE_PROMPT = (
    "Find the {label} in this image. Output only JSON: "
    "[{{\"label\": str, \"bbox_2d\": [x1, y1, x2, y2]}}] with a tight box on a 0-1000 grid, "
    "or [] if it is not visible."
)


def locate_objects(
    image_paths: list[Path], model_id: str, labels: list[str], max_new_tokens: int = 200,
) -> dict[str, tuple[Path, list[float]]]:
    """For each label returns (image_path, [x1, y1, x2, y2] in pixels) of its detection with the
    largest box (the object nearest the camera) over all images; labels found in no image are
    omitted. Boxes touching the left, right or bottom image edge are skipped when any other exists:
    they tend to span the object plus its surroundings. The top edge doesn't count -- objects
    enter the hand camera's view from there, so a box touching it is normal."""
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    logger.info(f"Track: locating {labels} with {model_id} in {len(image_paths)} frames")
    model = None
    candidates = {label: [] for label in labels}  # label -> [(area, path, box, image_size)]
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            for label in labels:
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": LOCATE_PROMPT.format(label=label)},
                ]}]
                inputs = processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
                ).to(device)
                with torch.inference_mode():
                    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                reply = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
                try:
                    detections = _detections_from_reply(reply, *image.size)
                except RuntimeError:  # unparsable reply: treat as "not visible" in this frame
                    detections = []
                for _, box in detections:
                    logger.info(f"Track: {label} in {path.stem}: box {[round(v) for v in box]}")
                    candidates[label].append(((box[2] - box[0]) * (box[3] - box[1]), path, box, image.size))
    except Exception as e:
        raise RuntimeError(f"Locating {labels} failed calling HF model '{model_id}': {e}") from e
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    found = {}
    for label, cands in candidates.items():
        if not cands:
            continue
        inside = [c for c in cands if c[2][0] > EDGE_PX and c[2][2] < c[3][0] - EDGE_PX and c[2][3] < c[3][1] - EDGE_PX]
        _, path, box, _ = max(inside or cands, key=lambda c: c[0])
        found[label] = (path, box)
    return found


def track_objects(
    frames: list[Path], seeds: dict[str, tuple[Path, list[float]]],
    work_dir: Path, out_dir: Path, padding_frac: float = 0.05,
) -> dict[str, Path]:
    """Runs GroundedSAM2 mask tracking over `frames` (sorted, all one camera) for every label in
    `seeds` ({label: (seed_frame, seed_box in pixels)}), each seeded on its own frame and tracked
    jointly. Returns {label: mask directory} (out_dir/<label>/mask_bin and mask_overlay, one PNG
    per frame). `work_dir` holds the JPEG copies of the frames (SAM2 reads JPEG only) and is
    removed afterwards."""
    from PIL import Image

    sys.path.insert(0, str(REPO_DIR / "segmentation"))
    from main_light import run_segmentation

    # Labels travel through the comma-separated --classes flag.
    seeds = {label.replace(",", " ").strip(): seed for label, seed in seeds.items()}
    rgb_dir = work_dir / "rgb"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    rgb_dir.mkdir(parents=True)
    try:
        for p in frames:
            Image.open(p).convert("RGB").save(rgb_dir / f"{p.stem}.jpg", quality=95)
        boxes_json = work_dir / "boxes.json"
        boxes_json.write_text(json.dumps({
            "left_image": f"{next(iter(seeds.values()))[0].stem}.jpg",
            "objects": {
                label: {"left_box": box, "right_box": box, "image": f"{frame.stem}.jpg"}
                for label, (frame, box) in seeds.items()
            },
        }))
        out_dir.mkdir(parents=True, exist_ok=True)
        run_segmentation(
            "track", str(work_dir), str(out_dir), ",".join(seeds),
            text="classes", flat_output=True, skip_pc_segment=True,
            detection_box_padding_frac=padding_frac, boxes_json=str(boxes_json),
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    labels_json = out_dir / "labels.json"
    if not labels_json.exists():
        raise FileNotFoundError(f"Expected tracking labels.json not found at {labels_json}")
    label_dirs = json.loads(labels_json.read_text())
    return {label: out_dir / label_dirs[label] for label in seeds}
