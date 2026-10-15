"""Have a Franka Panda arm pick up a segmented object from a composed GenRecon scene and place it
at a nearby offset -- the manipulation counterpart to demo_robot_collide.py's mobile-robot
collision proof: confirms a scene.usda from compose_isaac_scene.py (dynamic, RigidBodyAPI-bearing
objects with real friction/mass) is actually graspable, not just physically collidable.

Unlike demo_robot_collide.py, the arm is stationary and does all the work itself: a
PickPlaceController (RMPFlow cartesian-space control + a scripted 10-phase state machine --
approach, descend, grasp, lift, translate, lower, release, retract) drives the Franka toward the
target's *live* bbox center every step, so the exact pick height/position doesn't need to be
known in advance.

The exported --stage-output USD file captures the pre-grasp setup (object + Franka positioned to
reach it), not the post-place outcome -- like demo_robot_collide.py, there is no embedded
OmniGraph here (the state machine is pure Python), so re-opening that stage in the GUI will not
replay the pick-and-place on its own; the video (--output) is the only record of the run.

Usage:
    uv run demo_franka_pickplace.py --scene /path/to/pick_place/glb/scene.usda --pick-target banana
    uv run demo_franka_pickplace.py --scene /path/to/pick_place/glb/scene.usda --pick-target banana \
        --place-target bowl --gripper-open-width 0.06 --approach-side neg-y
"""

import argparse
import os
import sys

# Anchors both presets to add_lighting()'s original hardcoded values ("ambient") plus a second,
# dimmer/warmer indoor look ("room") -- see --lighting-mode's help. Kept at module scope so
# parse_args() can reference it for --lighting-mode's choices.
LIGHTING_PRESETS = {
    "ambient": {
        "dome_intensity": 1000.0,
        "dome_color": (1.0, 1.0, 1.0),
        "distant_intensity": 3000.0,
        "distant_angle": 1.0,
        "distant_rotation_deg": (-45.0, 30.0, 0.0),
    },
    "room": {
        "dome_intensity": 400.0,
        "dome_color": (1.0, 0.95, 0.85),
        "distant_intensity": 1500.0,
        "distant_angle": 3.0,
        "distant_rotation_deg": (-80.0, 10.0, 0.0),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True, help="Path to a composed scene.usda (from compose_isaac_scene.py).")
    parser.add_argument(
        "--pick-target",
        required=True,
        help="Prim name under the referenced scene's /World to pick up. Must be a dynamic "
        "(RigidBodyAPI-bearing) object from convert_asset.py.",
    )
    parser.add_argument(
        "--place-offset",
        type=float,
        nargs=3,
        default=[0.3, 0.0, 0.0],
        metavar=("DX", "DY", "DZ"),
        help="Offset (meters, in world XYZ) from the target's initial bbox center to place it at. "
        "Keep this modest (well under the Panda's ~0.85m reach) -- the robot base is positioned "
        "once at setup and does not reposition itself between pick and place.",
    )
    parser.add_argument(
        "--place-target",
        default=None,
        help="Prim name under the referenced scene's /World to place the picked object above, instead of "
        "using --place-offset -- e.g. to place a banana inside a bowl. The release point is this prim's "
        "live bbox XY center with Z at its bbox top plus --place-target-clearance (not its bbox center, "
        "which for a bowl-shaped target would sit below the rim and risk the gripper colliding with its walls).",
    )
    parser.add_argument(
        "--place-target-clearance",
        type=float,
        default=0.05,
        help="Extra height in meters above --place-target's bbox top at which the gripper releases. "
        "Only used when --place-target is set.",
    )
    parser.add_argument(
        "--gripper-open-width",
        type=float,
        default=0.05,
        help="Per-finger open distance in meters for the Franka's parallel gripper (passed as "
        "gripper_open_position). Increase this if grasps fail due to insufficient clearance around the target.",
    )
    parser.add_argument("--gui", action="store_true", help="Open an Isaac Sim window instead of running headless.")
    parser.add_argument("--output", default=None, help="Path to write the pick-and-place mp4 to (default: output/<scene>_<label>_pickplace.mp4).")
    parser.add_argument(
        "--stage-output",
        default=None,
        help="Path to export the pre-grasp setup USD stage to -- object + Franka positioned to reach it, "
        "before any picking happens (default: output/<scene>_<label>_pickplace_scene.usda).",
    )
    parser.add_argument("--start-distance", type=float, default=0.5, help="Distance in meters to place the Franka's base from the target, along --approach-side.")
    parser.add_argument(
        "--approach-side",
        choices=["neg-x", "pos-x", "neg-y", "pos-y"],
        default="neg-x",
        help="Which side of the target (in world XY) to place the Franka's base on, --start-distance away: "
        "'neg-x'/'pos-x' approach head-on along the target's own X axis (the default, 'neg-x', matches the "
        "original behavior); 'neg-y'/'pos-y' approach from the side (perpendicular to X) -- useful when other "
        "objects are laid out in a row along X and a head-on approach would have the arm reach along/through "
        "that row. (Not spelled '-x'/'+x'/etc. because argparse misparses a value that looks like an option.)",
    )
    parser.add_argument(
        "--events-dt",
        type=str,
        default="0.008,0.005,1,0.05,0.015,0.01,0.0025,1,0.008,0.05",
        help="Comma-separated 10 floats controlling PickPlaceController's per-phase speed (phases: "
        "0=move above target, 1=lower to grasp, 2=settle, 3=close grip, 4=lift, 5=transport to place "
        "xy, 6=descend to place height, 7=open grip, 8=retract, 9=return). Each phase's real duration "
        "is roughly (1/dt) physics steps, so a SMALLER value means MORE steps, i.e. slower/smoother "
        "motion; a LARGER value means fewer steps, i.e. faster/jerkier. Defaults to a slower-than-stock "
        "preset for phases 3/4/5 (grip/lift/transport) -- the phases most likely to fling or drop a "
        "held object from rapid acceleration -- versus the Isaac Sim example's own stock default of "
        "0.008,0.005,1,0.1,0.05,0.05,0.0025,1,0.008,0.08.",
    )
    parser.add_argument("--max-steps", type=int, default=2000, help="Safety cap on physics/render steps while the state machine runs, in case it never reports done.")
    parser.add_argument("--settle-steps", type=int, default=60, help="Additional steps to let the scene settle after placing, before the final displacement check.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--resolution", type=int, nargs=2, default=[1280, 720], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument(
        "--displacement-tolerance",
        type=float,
        default=0.05,
        help="Maximum horizontal distance (meters) between the target's final bbox center and the "
        "intended place position to count as a successful placement.",
    )
    parser.add_argument(
        "--lighting-mode",
        choices=list(LIGHTING_PRESETS),
        default="ambient",
        help="Selects a lighting preset (see LIGHTING_PRESETS): 'ambient' (default) is today's "
        "original hardcoded look -- uniform dome light + a sun-angled shadow-casting distant light. "
        "'room' is a dimmer, warmer indoor look -- a subdued bounced-light dome plus a near-overhead "
        "key light. Any of the five --dome-light-*/--distant-light-* flags below, if explicitly "
        "passed, overrides that one value from the chosen preset.",
    )
    parser.add_argument("--dome-light-intensity", type=float, default=None, help="add_lighting()'s ambient dome light intensity. Overrides --lighting-mode's preset value.")
    parser.add_argument(
        "--dome-light-color",
        default=None,
        help="Comma-separated R,G,B (each 0-1) for the dome light's color, e.g. '1,0.9,0.75' for a warm tint. Overrides --lighting-mode's preset value.",
    )
    parser.add_argument("--distant-light-intensity", type=float, default=None, help="add_lighting()'s key/shadow-casting distant light intensity. Overrides --lighting-mode's preset value.")
    parser.add_argument("--distant-light-angle", type=float, default=None, help="add_lighting()'s distant light softness (degrees). Overrides --lighting-mode's preset value.")
    parser.add_argument(
        "--distant-light-rotation-deg",
        default=None,
        help="Comma-separated X,Y,Z rotation (degrees) for the distant light. Overrides --lighting-mode's preset value.",
    )
    parser.add_argument(
        "--camera-mode",
        choices=["angled", "overhead"],
        default="angled",
        help="'angled' is today's default fixed look-at framing; 'overhead' looks straight down at the "
        "midpoint between pick and place positions.",
    )
    parser.add_argument(
        "--camera-distance-multiplier",
        type=float,
        default=2.0,
        help="Scales setup_camera()'s distance-from-subject formula -- lower is a tighter/closer shot, "
        "higher is wider/further back.",
    )
    return parser.parse_args()


args = parse_args()

# SimulationApp must be constructed before any other isaacsim/omni import.
from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": not args.gui})

# See demo_robot_collide.py for why this is needed: without it, PhysX drives simulated transforms
# through Fabric only, so get_bbox()'s stage queries and stage.Export() below would keep seeing
# each prim's pre-simulation pose instead of the live one.
import carb  # noqa: E402
from omni.physx.bindings._physx import SETTING_UPDATE_TO_USD  # noqa: E402

carb.settings.get_settings().set_bool(SETTING_UPDATE_TO_USD, True)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import RigidPrim  # noqa: E402
from isaacsim.core.utils.nucleus import get_assets_root_path  # noqa: E402
from isaacsim.core.utils.rotations import euler_angles_to_quat  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import PickPlaceController  # noqa: E402
from isaacsim.robot.manipulators.examples.franka.franka import Franka  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

SCENE_PRIM_PATH = "/World/Scene"
FRANKA_PRIM_PATH = "/World/Franka"
CAMERA_PRIM_PATH = "/World/PickPlaceCamera"
DOME_LIGHT_PRIM_PATH = "/World/DomeLight"
DISTANT_LIGHT_PRIM_PATH = "/World/DistantLight"
FRANKA_USD_PATH = "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
# convert_asset.py's collision authoring is a lossy proxy of the visual mesh (same caveat as
# demo_robot_collide.py's identical constant) -- a small settle-induced sink below the composed,
# pre-physics floor height is expected and corrected once, not treated as an error.
GROUND_PENETRATION_TOLERANCE = 0.002


def get_bbox(prim_path):
    stage = omni.usd.get_context().get_stage()
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bound_range = bbox_cache.ComputeWorldBound(stage.GetPrimAtPath(prim_path)).ComputeAlignedRange()
    return np.array(bound_range.GetMin()), np.array(bound_range.GetMax())


def _rotation_matrix_to_quat(m):
    """Standard (Shepperd's method) rotation-matrix-to-quaternion conversion under the column-
    vector convention (v' = m @ v) -- see setup_camera()'s comment for why this is used instead
    of Gf.Matrix3d.ExtractRotation().GetQuat(). Returns scalar-first (w, x, y, z). `m` must be a
    proper rotation (orthogonal, det=+1)."""
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def add_lighting(
    dome_intensity=1000.0,
    dome_color=(1.0, 1.0, 1.0),
    distant_intensity=3000.0,
    distant_angle=1.0,
    distant_rotation_deg=(-45.0, 30.0, 0.0),
):
    """GenRecon's composed scene.usda carries geometry/collision only, no lights -- without
    this the render is black. Defaults match the original hardcoded values, so callers that don't
    pass anything (or --ai-scene-agent isn't used) get byte-identical lighting to before these
    parameters existed."""
    stage = omni.usd.get_context().get_stage()

    dome_light = UsdLux.DomeLight.Define(stage, DOME_LIGHT_PRIM_PATH)
    dome_light.CreateIntensityAttr(dome_intensity)
    dome_light.CreateColorAttr(Gf.Vec3f(*dome_color))

    distant_light = UsdLux.DistantLight.Define(stage, DISTANT_LIGHT_PRIM_PATH)
    distant_light.CreateIntensityAttr(distant_intensity)
    distant_light.CreateAngleAttr(distant_angle)
    xformable = UsdGeom.Xformable(distant_light.GetPrim())
    xformable.AddRotateXYZOp().Set(Gf.Vec3f(*distant_rotation_deg))


def get_scene_floor_z(scene_prim_path):
    """The surface the Franka should stand on and the objects should rest on. Prefers
    <scene_prim_path>/Table's own top-surface Z (its bbox max Z) when compose_isaac_scene.py added
    a table -- otherwise the robot would spawn at the real GroundPlane height, a table's-height
    below the objects it needs to reach (the table sits *under* the composed floor/objects, not on
    top of the GroundPlane safety net beneath it -- see compose_isaac_scene.py's add_table_asset()).
    Falls back to the world Z that --ground-plane authored on <scene_prim_path>/GroundPlane (== min
    bbox Z across every composed asset, minus a small margin), instead of deriving floor height
    from any single object's own bbox -- this matters once a scene has multiple objects, since the
    pick target isn't necessarily the one resting lowest. Returns None if the referenced scene has
    neither a Table nor a GroundPlane (e.g. composed with --no-table --no-ground-plane).

    GroundPlane's Z is matched by op *type* (TypeTranslate) rather than by exact op name, since
    physicsUtils.add_ground_plane authors a plain AddTranslateOp() (no suffix) -- different from
    compose_isaac_scene.py's own set_world_pose(), which uses opSuffix="world_pose"."""
    stage = omni.usd.get_context().get_stage()
    table_prim = stage.GetPrimAtPath(f"{scene_prim_path}/Table")
    if table_prim.IsValid():
        _, table_max = get_bbox(f"{scene_prim_path}/Table")
        return float(table_max[2])
    ground_plane_prim = stage.GetPrimAtPath(f"{scene_prim_path}/GroundPlane")
    if not ground_plane_prim.IsValid():
        return None
    for op in UsdGeom.Xformable(ground_plane_prim).GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            return float(op.Get()[2])
    return None


# Rotation about Z (degrees) that points the Panda's local +X reach direction back at the target,
# for a base placed start_distance away on each --approach-side. E.g. 'neg-x' (base on the target's
# -X side) needs no rotation, since the arm already reaches along +X at the identity orientation.
_APPROACH_SIDE_ROTATION_DEG = {"neg-x": 0.0, "pos-x": 180.0, "neg-y": 90.0, "pos-y": -90.0}


def setup_scene(scene_path, pick_target, start_distance, gripper_open_width, approach_side):
    """References the composed scene, then places a Franka start_distance away from the target on
    the given --approach-side, rotated to face it (no heading search is needed, unlike
    demo_robot_collide.py's mobile Jetbot -- this robot never drives). Returns (target_prim_path,
    expected_floor_z, franka)."""
    add_reference_to_stage(usd_path=os.path.abspath(scene_path), prim_path=SCENE_PRIM_PATH)

    target_prim_path = f"{SCENE_PRIM_PATH}/{pick_target}"
    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(target_prim_path).IsValid():
        raise RuntimeError(f"No prim at {target_prim_path} -- check --pick-target matches a label in {scene_path}.")

    target_min, target_max = get_bbox(target_prim_path)
    target_center = (target_min + target_max) / 2.0

    scene_floor_z = get_scene_floor_z(SCENE_PRIM_PATH)
    if scene_floor_z is None:
        print(
            f"  WARNING: no {SCENE_PRIM_PATH}/GroundPlane found; falling back to '{pick_target}' "
            "bbox min as floor height (correct only if the target itself rests on the scene floor)."
        )
        scene_floor_z = float(target_min[2])

    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        raise RuntimeError("Could not resolve Isaac Sim assets root path (no Nucleus connection).")

    axis_index = 0 if approach_side.endswith("x") else 1
    sign = -1.0 if approach_side.startswith("neg") else 1.0
    base_position = np.array([target_center[0], target_center[1], scene_floor_z])
    base_position[axis_index] += sign * start_distance
    base_orientation = euler_angles_to_quat(np.array([0.0, 0.0, np.radians(_APPROACH_SIDE_ROTATION_DEG[approach_side])]))
    franka = Franka(
        prim_path=FRANKA_PRIM_PATH,
        name="franka",
        usd_path=assets_root_path + FRANKA_USD_PATH,
        position=base_position,
        orientation=base_orientation,
        gripper_open_position=np.array([gripper_open_width, gripper_open_width]),
    )
    return target_prim_path, scene_floor_z, franka


def correct_ground_penetration(target_prim_path, expected_floor_z, tolerance=GROUND_PENETRATION_TOLERANCE):
    """Same fix as demo_robot_collide.py's function of the same name: corrects a settle-induced
    sink below the target's authored, pre-physics floor height by teleporting the rigid body via
    RigidPrim (not editing USD xformOps directly, which PhysX would immediately overwrite)."""
    current_min, _ = get_bbox(target_prim_path)
    penetration = expected_floor_z - float(current_min[2])
    if penetration <= tolerance:
        return 0.0

    rigid_prim = RigidPrim(prim_paths_expr=target_prim_path)
    positions, orientations = rigid_prim.get_world_poses()
    positions = np.array(positions)
    positions[0, 2] += penetration
    rigid_prim.set_world_poses(positions=positions, orientations=orientations)
    return penetration


def setup_camera(resolution, look_at_position, look_at_size, mode="angled", distance_multiplier=2.0):
    """A fixed look-at camera framing the midpoint between pick and place locations (there is no
    vehicle to mount an onboard camera on here, unlike demo_robot_collide.py's Jetbot).

    mode='angled' (the original, default behavior) offsets the camera to the side and above the
    subject; mode='overhead' places it directly above, looking straight down -- distance_multiplier
    scales how far back/up the camera sits in either mode (default 2.0 matches the original
    hardcoded `* 2.0`)."""
    camera = Camera(prim_path=CAMERA_PRIM_PATH, resolution=tuple(resolution))
    camera.initialize()
    distance = max(look_at_size[0], look_at_size[1], 1.0) * distance_multiplier + 1.5

    if mode == "overhead":
        camera_position = look_at_position + np.array([0.0, 0.0, distance + look_at_size[2]])
    else:
        camera_position = look_at_position + np.array([-distance * 0.3, -distance, look_at_size[2] + 0.6])
    forward = look_at_position - camera_position
    forward = forward / np.linalg.norm(forward)
    # cross(forward, world_up) then cross(right, forward), in that order, is what makes
    # (right, up, -forward) a proper right-handed rotation (det=+1); the reversed order
    # (cross(world_up, forward) then cross(forward, right)) is a mirror-image reflection
    # (det=-1), which is not a valid rotation and produced a camera pointed roughly away
    # from (or orthogonal to) the intended look-at target instead.
    #
    # world_up=[0,0,1] degenerates (zero-length cross product) when forward is itself
    # ~vertical -- exactly mode='overhead' looking straight down -- so fall back to [0,1,0] as the
    # "up" reference there (world +Y ends up as the image's "up" direction instead).
    world_up = [0.0, 1.0, 0.0] if abs(forward[2]) > 0.999 else [0.0, 0.0, 1.0]
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    # Columns (right, up, -forward) is the native USD/Hydra camera convention (-Z forward, +Y
    # "up" in the camera's own local frame) under the standard column-vector rotation convention
    # (v' = R @ v) -- which is what set_world_pose()/Hydra actually use to compose transforms.
    # Gf.Matrix3d.ExtractRotation().GetQuat() looks like the obvious way to turn R into a
    # quaternion, but pxr.Gf matrices use the *row*-vector convention (v' = v @ M) internally, so
    # extracting a quaternion that way silently yields the quaternion for R's transpose (R's
    # inverse, since R is orthogonal) instead of R itself -- confirmed by comparing the exported
    # stage's actual authored quaternion against this R: R.T @ [0,0,-1] matched the intended look
    # direction, while R @ [0,0,-1] pointed the camera roughly along -Y, into empty ground plane,
    # away from the robot and bowl entirely (which is exactly the "video shows nothing" bug this
    # replaces). Building the quaternion directly from R with a standard rotation-matrix-to-quat
    # conversion (Shepperd's method) sidesteps Gf's convention entirely.
    orientation = _rotation_matrix_to_quat(np.stack([right, up, -forward], axis=1))
    # camera_axes="usd" is the documented no-op axes mode (+Y up, -Z forward, i.e. exactly the R
    # built above) -- "world" would apply an extra (+X forward, +Z up) basis conversion on top,
    # which is wrong for a matrix already built in the native convention.
    camera.set_world_pose(camera_position, orientation, camera_axes="usd")
    return camera


def set_scene_kinematic(enabled):
    """Toggles physics:kinematicEnabled on every rigid body under SCENE_PRIM_PATH. Held on through
    the pre-grasp pre-roll and the stage export so thin, slightly interpenetrating reconstructed
    objects (e.g. knives lying on a concave plate) aren't flung by PhysX depenetration before the
    exported scene is written -- the export then matches compose_isaac_scene.py's poses."""
    stage = omni.usd.get_context().get_stage()
    for prim in Usd.PrimRange(stage.GetPrimAtPath(SCENE_PRIM_PATH)):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr().Set(enabled)


def run_settle(world, num_frames):
    for _ in range(num_frames):
        world.step(render=True)


def main():
    scene_id = os.path.splitext(os.path.basename(os.path.abspath(args.scene)))[0]
    output_path = args.output or f"output/{scene_id}_{args.pick_target}_pickplace.mp4"
    stage_output_path = args.stage_output or f"output/{scene_id}_{args.pick_target}_pickplace_scene.usda"

    world = World()
    world.reset()

    target_prim_path, expected_floor_z, franka = setup_scene(
        args.scene, args.pick_target, args.start_distance, args.gripper_open_width, args.approach_side
    )
    lighting = dict(LIGHTING_PRESETS[args.lighting_mode])
    if args.dome_light_intensity is not None:
        lighting["dome_intensity"] = args.dome_light_intensity
    if args.dome_light_color is not None:
        lighting["dome_color"] = tuple(float(c) for c in args.dome_light_color.split(","))
    if args.distant_light_intensity is not None:
        lighting["distant_intensity"] = args.distant_light_intensity
    if args.distant_light_angle is not None:
        lighting["distant_angle"] = args.distant_light_angle
    if args.distant_light_rotation_deg is not None:
        lighting["distant_rotation_deg"] = tuple(float(v) for v in args.distant_light_rotation_deg.split(","))
    add_lighting(**lighting)
    set_scene_kinematic(True)
    world.scene.add(franka)
    world.reset()
    franka.initialize()

    # PickPlaceController never issues an explicit "open" action before its first grasp attempt
    # (only "close" at phase 3 and "open" at phase 7, the final release) -- without this, the
    # fingers stay wherever the Franka USD asset's own default joint state left them, and
    # --gripper-open-width would only ever take effect at the final release, not the grasp.
    franka.gripper.open()

    for _ in range(10):
        world.step(render=True)

    penetration = correct_ground_penetration(target_prim_path, expected_floor_z)
    if penetration:
        print(f"Corrected {args.pick_target} sinking by {penetration:.4f}m")
        for _ in range(5):
            world.step(render=True)

    initial_min, initial_max = get_bbox(target_prim_path)
    initial_center = (initial_min + initial_max) / 2.0
    initial_size = initial_max - initial_min

    if args.place_target:
        place_target_prim_path = f"{SCENE_PRIM_PATH}/{args.place_target}"
        if not omni.usd.get_context().get_stage().GetPrimAtPath(place_target_prim_path).IsValid():
            raise RuntimeError(
                f"No prim at {place_target_prim_path} -- check --place-target matches a label in {args.scene}."
            )
        place_target_min, place_target_max = get_bbox(place_target_prim_path)
        # XY from the place target's bbox center, but Z from its bbox *top* (+ clearance), not its
        # bbox center -- for a bowl-shaped target the bbox center sits below the rim, which would
        # send the gripper descending into the bowl's interior (risking a collision with its walls)
        # instead of releasing from just above it.
        place_position = np.array(
            [
                (place_target_min[0] + place_target_max[0]) / 2.0,
                (place_target_min[1] + place_target_max[1]) / 2.0,
                place_target_max[2] + args.place_target_clearance,
            ]
        )
    else:
        place_position = initial_center + np.array(args.place_offset)

    camera = setup_camera(
        args.resolution,
        (initial_center + place_position) / 2.0,
        initial_size,
        mode=args.camera_mode,
        distance_multiplier=args.camera_distance_multiplier,
    )
    for _ in range(5):
        world.step(render=True)

    # Release the scene bodies only now (the controller needs them dynamic), and before the export
    # so the exported stage doesn't carry the temporary kinematic flag.
    set_scene_kinematic(False)

    # Export the pre-grasp setup, not the post-place result -- see module docstring.
    os.makedirs(os.path.dirname(stage_output_path) or ".", exist_ok=True)
    omni.usd.get_context().get_stage().Export(stage_output_path)
    print(f"Exported pre-grasp setup stage to {stage_output_path}")

    events_dt = [float(v) for v in args.events_dt.split(",")]
    controller = PickPlaceController(
        name="pick_place_controller", gripper=franka.gripper, robot_articulation=franka, events_dt=events_dt
    )
    articulation_controller = franka.get_articulation_controller()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    resolution = camera.get_resolution()
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, tuple(resolution))
    step_count = 0
    try:
        while not controller.is_done() and step_count < args.max_steps:
            # Re-read the target's live bbox center every step (rather than freezing it at the
            # initial value) so a small pre-grasp settle/slide doesn't throw off the approach.
            current_min, current_max = get_bbox(target_prim_path)
            picking_position = (current_min + current_max) / 2.0
            actions = controller.forward(
                picking_position=picking_position,
                placing_position=place_position,
                current_joint_positions=franka.get_joint_positions(),
            )
            articulation_controller.apply_action(actions)
            world.step(render=True)
            frame_rgba = camera.get_rgba()
            if frame_rgba is not None and frame_rgba.size > 0:
                frame_bgr = cv2.cvtColor(frame_rgba[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
            step_count += 1
    finally:
        writer.release()

    if controller.is_done():
        print(f"Wrote {step_count} frames to {output_path} (state machine completed).")
    else:
        print(f"Wrote {step_count} frames to {output_path} (hit --max-steps {args.max_steps} before the state machine finished).")

    run_settle(world, args.settle_steps)

    final_min, final_max = get_bbox(target_prim_path)
    final_center = (final_min + final_max) / 2.0
    displacement_from_place = float(np.linalg.norm(final_center[:2] - place_position[:2]))
    print(f"Target '{args.pick_target}' bbox center: initial={initial_center.tolist()} place_goal={place_position.tolist()} final={final_center.tolist()}")
    print(f"Horizontal distance from place goal: {displacement_from_place:.4f}m")

    placed = displacement_from_place <= args.displacement_tolerance
    if placed:
        print(f"PASS: target placed within {displacement_from_place:.4f}m (<= {args.displacement_tolerance}m tolerance) of the goal.")
    else:
        print(
            f"FAIL: target ended {displacement_from_place:.4f}m from the place goal (> {args.displacement_tolerance}m tolerance) -- "
            "the grasp may have failed or slipped. Check --start-distance/--place-offset, or try a coarser "
            "--collision-approximation (convexHull) when converting the asset."
        )

    sys.stdout.flush()
    simulation_app.close()

    if not placed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
