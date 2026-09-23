import json
import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# `track_utils`/`video_utils`/`detection_utils` are local files copied
# alongside this script (see segmentation/ dir); `sam2` and `groundingdino`
# are editable-installed workspace packages (third_party/sam2,
# third_party/groundingdino).
from track_utils import sample_points_from_masks
from video_utils import create_video_from_images
from groundingdino.util.inference import load_model, load_image
from detection_utils import detect_frame_boxes, match_detections_to_classes, detect_classes_in_all_frames
from gemma_detection_utils import (
    detect_frame_boxes_gemma, match_detections_to_classes_gemma, detect_classes_in_frames_gemma,
)

from scene.colmap_loader import (
    read_extrinsics_binary, read_extrinsics_text,
    read_intrinsics_binary, read_intrinsics_text,
    read_points3D_binary, read_points3D_text,
)
from scene.frustum_utils import lift_box_to_frustum, reproject_frustum_area_fraction

import argparse

# FIXME: figure how does this influence the G-DINO model
torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

parser = argparse.ArgumentParser(description='extract mask')
parser.add_argument('--resolution', type=int, default=-1)
parser.add_argument('--dataset_root', type=str, default='',
                     help="Base directory containing this scene's images/ (and, for "
                          "3D-reprojection gating, sparse/0/) -- e.g. a VGGT-Omega "
                          "--export-for-3dgs output dir. Used directly, no <scene> "
                          "subdir is appended.")
parser.add_argument('--output', type=str, default='')
parser.add_argument('--scene', type=str, default='')
parser.add_argument('--text', type=str, default='')
parser.add_argument('--frame_idx', type=int, default=0)
parser.add_argument('--classes', type=str, default=None,
                     help="Comma-separated candidate sub-classes for open-world, "
                          "class-based segmentation (e.g. \"apple,banana,orange\"). "
                          "When set, each detected object is tracked and saved "
                          "under its matched class instead of one merged mask.")
parser.add_argument('--flat_output', action='store_true',
                     help="Skip the <scene> path segment: write directly under "
                          "--output/masks/<text> instead of --output/<scene>/masks/<text>. "
                          "Useful when --output is already scene-specific.")
parser.add_argument('--colmap_dir', type=str, default=None,
                     help="Path to a COLMAP sparse model (cameras.txt/images.txt/points3D.txt) "
                          "for this scene. Defaults to <dataset_base_dir>/sparse/0, which is "
                          "where upstream pipelines conventionally place it. Used to lift "
                          "detected boxes into 3D frustums that gate re-detection during "
                          "tracking (see --no_3d_reprojection).")
parser.add_argument('--no_3d_reprojection', action='store_true',
                     help="Disable 3D-frustum-gated re-detection even if a COLMAP model is "
                          "found at --colmap_dir/the default location: fall back to the old "
                          "behavior of re-running the detector on every frame with an empty "
                          "propagated mask.")
parser.add_argument('--reproj_area_threshold', type=float, default=0.02,
                     help="Minimum reprojected 3D-frustum area, as a fraction of frame area, "
                          "for a frame to be considered 'object should still be visible' and "
                          "thus worth a detector re-run when its propagated mask is empty.")
parser.add_argument('--frustum_box_margin', type=float, default=0.0,
                     help="Pixel margin added around a detection box when sampling COLMAP "
                          "points to estimate its 3D frustum's depth extent.")
parser.add_argument('--detector_backend', choices=['groundingdino', 'gemma'], default='groundingdino',
                     help="Box-detection backend. 'groundingdino' (default) is unchanged "
                          "existing behavior. 'gemma' uses a local Ollama-served Gemma "
                          "vision model instead (see gemma_detection_utils.py) -- validated "
                          "with gemma4:31b; the smaller gemma4:12b tag is not recommended, "
                          "see gemma_detection_utils.py's module docstring.")
parser.add_argument('--detection_vlm_model', type=str, default='gemma4:31b',
                     help="Ollama model tag used when --detector_backend gemma.")
parser.add_argument('--detection_ollama_host', type=str, default=None,
                     help="Ollama base URL override for --detector_backend gemma "
                          "(defaults to http://localhost:11434).")
parser.add_argument('--detection_num_sample_frames', type=int, default=8,
                     help="--detector_backend gemma, --classes mode only: number of evenly-"
                          "spaced frames sent to Gemma for the initial multi-class scan, "
                          "instead of Grounding DINO's cheap every-frame scan (a per-frame "
                          "VLM call is not cheap enough to run on every frame).")
parser.add_argument('--detection_box_padding_frac', type=float, default=0.05,
                     help="--detector_backend gemma only: outward padding applied to each "
                          "detected box (as a fraction of its own width/height) before it's "
                          "used to prompt SAM2. Mitigates the undershoot failure mode found "
                          "in testing, where a box that doesn't fully enclose its object "
                          "makes SAM2 truncate the mask at the wrong edge. Grounding DINO's "
                          "boxes get no padding (different failure mode, not needed).")
args = parser.parse_args()

multi_class = args.classes is not None
candidate_classes = [c.strip() for c in args.classes.split(',') if c.strip()] if args.classes is not None else []

# init sam image predictor and video predictor model
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sam2_checkpoint = os.path.join(REPO_ROOT, "checkpoints", "sam2", "ckpts", "sam2_hiera_large.pt")
model_cfg = "sam2_hiera_l.yaml"

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
image_predictor = SAM2ImagePredictor(sam2_image_model)

# build grounding dino model (skipped for --detector_backend gemma, which needs no local
# detection model -- it calls out to Ollama instead)
device = "cuda" if torch.cuda.is_available() else "cpu"
grounding_model = None
if args.detector_backend == 'groundingdino':
    grounding_model = load_model(
        model_config_path=os.path.join(REPO_ROOT, "checkpoints", "groundingdino", "GroundingDINO_SwinB_cfg.py"),
        model_checkpoint_path=os.path.join(REPO_ROOT, "checkpoints", "groundingdino", "ckpts", "groundingdino_swinb_cogcoor.pth"),
        device=device
    )
# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot

def downsample_images(images_folder, output_path, resolution):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for filename in os.listdir(images_folder):
        if filename.endswith(('.png', '.jpg', '.JPG', '.jpeg', '.bmp')):
            img_path = os.path.join(images_folder, filename)
            img = cv2.imread(img_path)

            if img is not None:
                new_width = img.shape[1] // resolution
                new_height = img.shape[0] // resolution

                downsampled_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

                output_img_path = os.path.join(output_path, filename)
                cv2.imwrite(output_img_path, downsampled_img)
            else:
                print(f"None: {img_path}")

output_path = args.output
scene = args.scene
text = args.text
resolution = args.resolution

# In multi-class mode, the detection caption is built from the candidate
# classes (GroundingDINO's period-separated multi-phrase syntax), while
# `text` continues to serve as the output-directory label under masks/<text>/.
detect_caption = " . ".join(candidate_classes) + " ." if multi_class else text

# `dataset_base_dir` is the scene root that both the frame images and (when
# available) the COLMAP sparse model live under -- passed straight through
# from --dataset_root, no dataset-type-keyed lookup.
dataset_base_dir = args.dataset_root

# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
video_dir = os.path.join(dataset_base_dir, 'images')
# scan all the JPEG frame names in this directory
if resolution != -1:
    print(os.path.exists(video_dir+f"_{resolution}"))
    if not os.path.exists(video_dir+f"_{resolution}"):
        print("create dataset....")
        downsample_images(video_dir, video_dir+f"_{resolution}", resolution)
    video_dir = video_dir+f"_{resolution}"

frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
]
frame_names.sort(key=lambda p: os.path.splitext(p)[0])

num_frames = len(frame_names)

# init video predictor state
inference_state = video_predictor.init_state(video_path=video_dir)

ann_frame_idx = args.frame_idx  # the frame index we interact with (unused when args.classes scans all frames)
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
"""
Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for specific frame(s)
"""

if args.classes is not None:
    # Multi-frame class scan: each class is seeded from its own single best
    # (highest-confidence) frame instead of one shared anchor frame, so a
    # class missing from frame 0 can still be found and seeded without
    # spreading one class's boxes across several frames (which would risk
    # duplicate detections/tracks for the same object).
    if args.detector_backend == 'gemma':
        class_to_detections = detect_classes_in_frames_gemma(
            video_dir, frame_names, candidate_classes, args.detection_vlm_model,
            num_sample_frames=args.detection_num_sample_frames,
            ollama_host=args.detection_ollama_host,
            box_padding_frac=args.detection_box_padding_frac,
        )
    else:
        class_to_detections = detect_classes_in_all_frames(
            video_dir, frame_names, grounding_model, candidate_classes, detect_caption,
        )
    OBJECT_CLASSES = []
    input_boxes = []
    object_frame_idxs = []
    confidences = []
    for cls in candidate_classes:
        det = class_to_detections.get(cls)
        if det is None:
            continue
        for box, conf in zip(det["boxes"], det["confidences"]):
            OBJECT_CLASSES.append(cls)
            input_boxes.append(np.array(box, dtype=np.float32))
            object_frame_idxs.append(det["frame_idx"])
            confidences.append(conf)

    max_confidence = max(confidences) if confidences else 0.0
    labels = [f"{cls} {conf:.2f}" for cls, conf in zip(OBJECT_CLASSES, confidences)]
    OBJECTS = labels
else:
    # prompt the detector to get the box coordinates on the single anchor frame
    img_path = os.path.join(video_dir, frame_names[ann_frame_idx])
    image_source, image = load_image(img_path)

    if args.detector_backend == 'gemma':
        anchor_height, anchor_width, _ = image_source.shape
        gemma_candidate_classes = candidate_classes if multi_class else [text]
        input_boxes, confidences, class_names = detect_frame_boxes_gemma(
            img_path, gemma_candidate_classes, anchor_width, anchor_height,
            args.detection_vlm_model, ollama_host=args.detection_ollama_host,
            box_padding_frac=args.detection_box_padding_frac,
        )
    else:
        input_boxes, confidences, class_names = detect_frame_boxes(
            grounding_model, image_source, image, detect_caption,
            box_threshold=0.3, text_threshold=0.45, remove_combined=multi_class,
        )

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(class_names, confidences)
    ]

    confidences_arr = np.array(confidences)

    if multi_class:
        if args.detector_backend == 'gemma':
            kept_indices, OBJECT_CLASSES = match_detections_to_classes_gemma(
                class_names, candidate_classes)
        else:
            kept_indices, OBJECT_CLASSES = match_detections_to_classes(
                class_names, confidences, candidate_classes)
        high_confidence_indices = kept_indices
        max_confidence = np.max(confidences_arr) if len(confidences_arr) else 0.0
    else:
        max_confidence = np.max(confidences_arr)
        threshold = max_confidence * 0.30
        high_confidence_indices = np.where(np.abs(confidences_arr - max_confidence) <= threshold)[0]
        OBJECT_CLASSES = [None] * len(high_confidence_indices)

    labels = [labels[i] for i in high_confidence_indices]
    input_boxes = [input_boxes[i] for i in high_confidence_indices]

    OBJECTS = labels
    object_frame_idxs = [ann_frame_idx] * len(OBJECTS)

"""
Step 2.6: Lift each detected box to a 3D frustum (for re-detection gating in Step 4)
"""
frame_pose_lookup = {}  # frame stem -> (extr, intr)
object_frustums = {}  # obj_id -> 8 world-space corners, or None
frustum_gating_active = False

if not args.no_3d_reprojection:
    colmap_dir = args.colmap_dir or os.path.join(dataset_base_dir, "sparse", "0")
    if not os.path.isdir(colmap_dir):
        print(f"[3d_reprojection] No COLMAP model found at {colmap_dir}; "
              f"falling back to unconditional re-detection on empty-mask frames.")
    else:
        try:
            try:
                cam_extrinsics = read_extrinsics_binary(os.path.join(colmap_dir, "images.bin"))
                cam_intrinsics = read_intrinsics_binary(os.path.join(colmap_dir, "cameras.bin"))
            except FileNotFoundError:
                cam_extrinsics = read_extrinsics_text(os.path.join(colmap_dir, "images.txt"))
                cam_intrinsics = read_intrinsics_text(os.path.join(colmap_dir, "cameras.txt"))
            try:
                points_xyz, _, _ = read_points3D_binary(os.path.join(colmap_dir, "points3D.bin"))
            except FileNotFoundError:
                points_xyz, _, _ = read_points3D_text(os.path.join(colmap_dir, "points3D.txt"))

            for extr in cam_extrinsics.values():
                stem = os.path.splitext(extr.name)[0]
                frame_pose_lookup[stem] = (extr, cam_intrinsics[extr.camera_id])

            for object_id, box in enumerate(input_boxes, start=1):
                seed_frame_idx = object_frame_idxs[object_id - 1]
                seed_stem = os.path.splitext(frame_names[seed_frame_idx])[0]
                pose = frame_pose_lookup.get(seed_stem)
                if pose is None:
                    object_frustums[object_id] = None
                    continue
                extr, intr = pose
                object_frustums[object_id] = lift_box_to_frustum(
                    points_xyz, extr, intr, box, box_margin=args.frustum_box_margin,
                )

            frustum_gating_active = any(v is not None for v in object_frustums.values())
            if not frustum_gating_active:
                print("[3d_reprojection] No object's box had enough nearby COLMAP points to "
                      "build a frustum; falling back to unconditional re-detection.")
        except Exception as e:
            print(f"[3d_reprojection] Failed to load COLMAP model at {colmap_dir} ({e}); "
                  f"falling back to unconditional re-detection.")
            frame_pose_lookup = {}
            object_frustums = {}
            frustum_gating_active = False

"""
Step 2.5: Save detections overlay(s) for sanity-checking Stage 1 output
"""
detections_dir = os.path.join(".", output_path) if args.flat_output else os.path.join(".", output_path, scene)
os.makedirs(detections_dir, exist_ok=True)

for anchor_frame_idx in sorted(set(object_frame_idxs)):
    frame_object_indices = [i for i, f in enumerate(object_frame_idxs) if f == anchor_frame_idx]
    frame_img_path = os.path.join(video_dir, frame_names[anchor_frame_idx])
    frame_image_source, _ = load_image(frame_img_path)
    frame_boxes = [input_boxes[i] for i in frame_object_indices]
    frame_labels = [OBJECTS[i] for i in frame_object_indices]

    image_predictor.set_image(frame_image_source)
    frame_masks, _, _ = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=frame_boxes,
        multimask_output=False,
    )
    # convert the mask shape to (n, H, W)
    # Note: image_predictor.predict() internally does masks.squeeze(0), which only
    # strips the box-batch dimension when there is exactly one box. So with a single
    # box, masks already comes back as (1, H, W) -- no reshaping needed. With multiple
    # boxes, it comes back as (n_boxes, 1, H, W) and needs the middle dim squeezed.
    if frame_masks.ndim == 4:
        frame_masks = frame_masks.squeeze(1)

    detections_img = cv2.cvtColor(frame_image_source, cv2.COLOR_RGB2BGR)
    detections = sv.Detections(
        xyxy=np.array(frame_boxes),
        mask=np.array(frame_masks).astype(bool),
        class_id=np.zeros(len(frame_boxes), dtype=int),
    )
    box_annotator = sv.BoxAnnotator()
    detections_img = box_annotator.annotate(scene=detections_img, detections=detections)
    label_annotator = sv.LabelAnnotator()
    detections_img = label_annotator.annotate(scene=detections_img, detections=detections, labels=frame_labels)
    mask_annotator = sv.MaskAnnotator()
    detections_img = mask_annotator.annotate(scene=detections_img, detections=detections)

    frame_classes = [OBJECT_CLASSES[i] for i in frame_object_indices]
    if len(set(object_frame_idxs)) == 1 or any(cls is None for cls in frame_classes):
        out_name = "detections.png"
    else:
        object_tag = "_".join(sorted(set(cls.strip().replace(" ", "_").replace("/", "_") for cls in frame_classes)))
        out_name = f"detection_{object_tag}.png"
        # Guard against two anchor frames resolving to the same class set.
        if os.path.exists(os.path.join(detections_dir, out_name)):
            suffix = 2
            while os.path.exists(os.path.join(detections_dir, f"detection_{object_tag}_{suffix}.png")):
                suffix += 1
            out_name = f"detection_{object_tag}_{suffix}.png"
    cv2.imwrite(os.path.join(detections_dir, out_name), detections_img)
    print(f"[save] Detections overlay: {os.path.join(detections_dir, out_name)}")

"""
Step 3: Register each object's positive points to video predictor with seperate add_new_points call
"""

PROMPT_TYPE_FOR_VIDEO = "box"  # or "point"

assert PROMPT_TYPE_FOR_VIDEO in ["point", "box", "mask"], "SAM 2 video predictor only support point/box/mask prompt"

# If you are using point prompts, we uniformly sample positive points based on the mask
if PROMPT_TYPE_FOR_VIDEO == "point":
    # sample the positive points from mask for each objects
    all_sample_points = sample_points_from_masks(masks=masks, num_points=10)

    for object_id, (label, points) in enumerate(zip(OBJECTS, all_sample_points), start=1):
        labels = np.ones((points.shape[0]), dtype=np.int32)
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=object_frame_idxs[object_id - 1],
            obj_id=object_id,
            points=points,
            labels=labels,
        )
# Using box prompt
elif PROMPT_TYPE_FOR_VIDEO == "box":
    for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=object_frame_idxs[object_id - 1],
            obj_id=object_id,
            box=box,
        )
# Using mask prompt is a more straightforward way
elif PROMPT_TYPE_FOR_VIDEO == "mask":
    for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
        labels = np.ones((1), dtype=np.int32)
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=object_frame_idxs[object_id - 1],
            obj_id=object_id,
            mask=mask
        )
else:
    raise NotImplementedError("SAM 2 video predictor only support point/box/mask prompts")

"""
Step 4: Propagate the video predictor to get the segmentation results for each frame
"""
video_segments = {}  # video_segments contains the per-frame segmentation results
distinct_anchor_frames = sorted(set(object_frame_idxs))
if len(distinct_anchor_frames) == 1:
    anchor_frame_idx = distinct_anchor_frames[0]
    if anchor_frame_idx == 0:
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state,
                                                                                              start_frame_idx=anchor_frame_idx,
                                                                                              reverse=False):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
    elif anchor_frame_idx == num_frames - 1:
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state,
                                                                                              start_frame_idx=anchor_frame_idx,
                                                                                              reverse=True):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
    else:
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state,
                                                                                              start_frame_idx=anchor_frame_idx,
                                                                                              reverse=False):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state,
                                                                                              start_frame_idx=anchor_frame_idx,
                                                                                              reverse=True):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
else:
    # Objects were seeded from several different frames (one per class, in
    # --classes mode) -- there's no single anchor frame to branch on, so
    # always cover the full video: forward from the start and backward from
    # the end, merging into video_segments like the single-anchor "middle
    # frame" case above.
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state,
                                                                                          start_frame_idx=0,
                                                                                          reverse=False):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state,
                                                                                          start_frame_idx=num_frames - 1,
                                                                                          reverse=True):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

def _mask_is_empty(mask):
    xyxy = sv.mask_to_xyxy(mask)[0]
    return xyxy[0] == 0 and xyxy[1] == 0 and xyxy[2] == 0 and xyxy[3] == 0


# valid_segments_dix[frame_idx] is the set of object_ids whose mask is empty
# in that frame -- tracked per-object (not just OBJECTS[0], the first object)
# so a class other than the first one going empty/losing its track is still
# detected and re-detected below, instead of silently drifting unnoticed.
valid_segments_dix = {}
for frame_idx, segments in video_segments.items():
    valid_segments_dix[frame_idx] = {
        obj_id for obj_id, mask in segments.items() if _mask_is_empty(mask)
    }

# class_name -> every object_id seeded for that class (usually one, but
# --classes mode can seed multiple instances of the same class from
# different frames).
class_to_object_ids = {}
for _obj_id, _cls in enumerate(OBJECT_CLASSES, start=1):
    class_to_object_ids.setdefault(_cls, []).append(_obj_id)

global_idx = 0
while global_idx < len(frame_names):
    missing_ids = valid_segments_dix.get(global_idx, set())
    if not missing_ids:
        global_idx += 1
        continue
    else:
        missing_classes = sorted({OBJECT_CLASSES[obj_id - 1] for obj_id in missing_ids})
        if frustum_gating_active:
            # Only worth a detector re-run if one of the *missing* objects'
            # 3D frustum still reprojects to a substantial footprint here --
            # otherwise the empty mask is plausibly correct (object out of
            # frame) rather than a lost track, so skip re-detecting.
            frame_stem = os.path.splitext(frame_names[global_idx])[0]
            pose = frame_pose_lookup.get(frame_stem)
            significant = pose is not None and any(
                reproject_frustum_area_fraction(object_frustums[obj_id], pose[0], pose[1]) >= args.reproj_area_threshold
                for obj_id in missing_ids if object_frustums.get(obj_id) is not None
            )
            if not significant:
                global_idx += 1
                continue

        img_path = os.path.join(video_dir, frame_names[global_idx])
        print(f"empty mask for classes {missing_classes}: " + img_path)
        image_source, image = load_image(img_path)
        if args.detector_backend == 'gemma':
            redetect_height, redetect_width, _ = image_source.shape
            input_boxes_det, confidences_det, class_names_det = detect_frame_boxes_gemma(
                img_path, missing_classes, redetect_width, redetect_height,
                args.detection_vlm_model, ollama_host=args.detection_ollama_host,
                box_padding_frac=args.detection_box_padding_frac,
            )
        else:
            input_boxes_det, confidences_det, class_names_det = detect_frame_boxes(
                grounding_model, image_source, image, detect_caption,
                box_threshold=0.5,
                text_threshold=0.5,
                remove_combined=multi_class,
            )
        if len(input_boxes_det) == 0:
            global_idx += 1
            continue

        # Re-run the same phrase-to-class matching used for the initial
        # seed detections (instead of positionally zipping OBJECTS against
        # whatever boxes come back in the detector's incidental order),
        # restricted to the classes actually missing here, so a re-detect
        # can never rebind one class's object_id to another class's box.
        if args.detector_backend == 'gemma':
            kept_indices, matched_classes = match_detections_to_classes_gemma(
                class_names_det, missing_classes)
        else:
            kept_indices, matched_classes = match_detections_to_classes(
                class_names_det, confidences_det, missing_classes)
        reseed_obj_ids = []
        reseed_boxes = []
        for kept_idx, matched_cls in zip(kept_indices, matched_classes):
            box = input_boxes_det[kept_idx]
            for obj_id in class_to_object_ids.get(matched_cls, []):
                if obj_id not in missing_ids:
                    # Already tracking fine at this frame -- don't clobber it.
                    continue
                reseed_obj_ids.append(obj_id)
                reseed_boxes.append(box)

        unmatched = sorted(set(missing_classes) - set(matched_classes))
        if unmatched:
            print(f"[re-detect] no match this frame for classes {unmatched}; leaving their track as-is")

        if len(reseed_obj_ids) == 0:
            global_idx += 1
            continue
        else:
            print(f"[re-detect] reseeding object_ids {reseed_obj_ids} "
                  f"(classes {[OBJECT_CLASSES[i - 1] for i in reseed_obj_ids]}) at frame {global_idx}")
            image_predictor.set_image(image_source)

            # prompt SAM 2 image predictor to get the mask for the object
            masks, scores, logits = image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=reseed_boxes,
                multimask_output=False,
            )
            # convert the mask shape to (n, H, W)
            if masks.ndim == 2:
                masks = masks[None]
                scores = scores[None]
                logits = logits[None]
            elif masks.ndim == 4:
                masks = masks.squeeze(1)

            assert PROMPT_TYPE_FOR_VIDEO in ["point", "box",
                                             "mask"], "SAM 2 video predictor only support point/box/mask prompt"

            # If you are using point prompts, we uniformly sample positive points based on the mask
            if PROMPT_TYPE_FOR_VIDEO == "point":
                # sample the positive points from mask for each objects
                all_sample_points = sample_points_from_masks(masks=masks, num_points=10)

                for object_id, points in zip(reseed_obj_ids, all_sample_points):
                    labels = np.ones((points.shape[0]), dtype=np.int32)
                    _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=global_idx,
                        obj_id=object_id,
                        points=points,
                        labels=labels,
                    )
            # Using box prompt
            elif PROMPT_TYPE_FOR_VIDEO == "box":
                for object_id, box in zip(reseed_obj_ids, reseed_boxes):
                    _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=global_idx,
                        obj_id=object_id,
                        box=box,
                    )
            # Using mask prompt is a more straightforward way
            elif PROMPT_TYPE_FOR_VIDEO == "mask":
                for object_id, mask in zip(reseed_obj_ids, masks):
                    labels = np.ones((1), dtype=np.int32)
                    _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=global_idx,
                        obj_id=object_id,
                        mask=mask
                    )
            else:
                raise NotImplementedError("SAM 2 video predictor only support point/box/mask prompts")

            reseeded_ids = set(reseed_obj_ids)
            max_zero_index = global_idx

            for idx in range(global_idx + 1, len(valid_segments_dix)):
                if valid_segments_dix.get(idx, set()) & reseeded_ids:
                    max_zero_index = idx
                else:
                    break
            print(f"Last frame with empty mask: {max_zero_index}")
            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=max_zero_index-global_idx, start_frame_idx=global_idx):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                valid_segments_dix[out_frame_idx] = {
                    obj_id for obj_id, mask in video_segments[out_frame_idx].items() if _mask_is_empty(mask)
                }

    global_idx += 1
"""
Step 5: Visualize the segment results across the video and save them
"""

save_dir = "."
save_dir = os.path.join(save_dir, output_path, "masks", text) if args.flat_output \
    else os.path.join(save_dir, output_path, scene, "masks", text)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}


def mask_overlay(img_bgr, mask, alpha=0.5, color=(0, 0, 255)):
    """Gray-converted image with a red mask overlay blended at `alpha`."""
    gray_bgr = cv2.cvtColor(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    overlay = gray_bgr.copy()
    overlay[mask.astype(bool)] = color
    return cv2.addWeighted(overlay, alpha, gray_bgr, 1 - alpha, 0)


def save_binary_mask(path, mask):
    """Write the true binary segmentation mask as a single-channel 0/255 PNG."""
    cv2.imwrite(path, mask.astype(np.uint8) * 255)


def save_mask_exports(class_dir, frame_stem, img_bgr, mask):
    """Save the true binary mask + grayscale/overlay pair for a frame under class_dir
    (multi-class mode: class_dir/mask_bin, class_dir/mask_overlay)."""
    mask_bin_dir = os.path.join(class_dir, "mask_bin")
    mask_overlay_dir = os.path.join(class_dir, "mask_overlay")
    os.makedirs(mask_bin_dir, exist_ok=True)
    os.makedirs(mask_overlay_dir, exist_ok=True)
    save_binary_mask(os.path.join(mask_bin_dir, f"{frame_stem}.png"), mask)
    cv2.imwrite(os.path.join(mask_overlay_dir, f"{frame_stem}.png"), mask_overlay(img_bgr, mask))


def save_debug_exports(class_dir, frame_stem, img_bgr, mask):
    """Save a true binary mask + grayscale/overlay debug pair for a frame
    (single-class mode only: layout is untouched by the multi-class restructuring)."""
    mask_bin_dir = os.path.join(class_dir, "debug", "mask_bin")
    mask_overlay_dir = os.path.join(class_dir, "debug", "mask_overlay")
    os.makedirs(mask_bin_dir, exist_ok=True)
    os.makedirs(mask_overlay_dir, exist_ok=True)
    save_binary_mask(os.path.join(mask_bin_dir, f"{frame_stem}.png"), mask)
    cv2.imwrite(os.path.join(mask_overlay_dir, f"{frame_stem}.png"), mask_overlay(img_bgr, mask))


if multi_class:
    def sanitize_label(label):
        return label.strip().replace(" ", "_").replace("/", "_")

    ID_TO_CLASS = {i: cls for i, cls in enumerate(OBJECT_CLASSES, start=1)}
    label_dirs = {cls: sanitize_label(cls) for cls in set(OBJECT_CLASSES)}
    for dirname in label_dirs.values():
        os.makedirs(os.path.join(save_dir, dirname), exist_ok=True)
    with open(os.path.join(save_dir, "labels.json"), "w") as f:
        json.dump(label_dirs, f, indent=2)

    for frame_idx, segments in video_segments.items():
        masks_by_label = {}
        for obj_id, mask in segments.items():
            cls = ID_TO_CLASS[obj_id]
            masks_by_label.setdefault(cls, []).append(mask)
        img = cv2.imread(os.path.join(video_dir, frame_names[frame_idx]))
        frame_stem = os.path.splitext(frame_names[frame_idx])[0]
        for cls, mask_list in masks_by_label.items():
            merged = np.max(np.concatenate(mask_list, axis=0), axis=0)
            class_dir = os.path.join(save_dir, label_dirs[cls])
            save_mask_exports(class_dir, frame_stem, img, merged)
else:
    for frame_idx, segments in video_segments.items():
        img = cv2.imread(os.path.join(video_dir, frame_names[frame_idx]))

        object_ids = list(segments.keys())
        masks = list(segments.values())
        masks = np.concatenate(masks, axis=0)
        masks = np.max(masks, axis=0)
        frame_stem = os.path.splitext(frame_names[frame_idx])[0]
        save_binary_mask(os.path.join(save_dir, f"{frame_stem}.png"), masks)
        save_debug_exports(save_dir, frame_stem, img, masks)
    # # Visualization
    # detections = sv.Detections(
    #     xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
    #     mask=masks, # (n, h, w)
    #     class_id=np.array(object_ids, dtype=np.int32),
    # )
    # box_annotator = sv.BoxAnnotator()
    # annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    # label_annotator = sv.LabelAnnotator()
    # annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=[ID_TO_OBJECTS[i] for i in object_ids])
    # mask_annotator = sv.MaskAnnotator()
    # annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    # cv2.imwrite(os.path.join(save_dir, frame_names[frame_idx]), annotated_frame)