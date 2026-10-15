"""Drive a Jetbot straight into a dynamic object from a composed GenRecon scene and confirm a
real physics collision displaced it -- the demo proof that a scene.usda from
compose_isaac_scene.py (background static collider + convex-decomposition dynamic objects) is
actually usable for robot interaction, not just a static-looking asset.

The robot's motion is an open-loop straight-line drive command (it starts already aimed at the
target); PhysX collision response, not scripted object motion, is what moves the target.

The exported --stage-output USD file captures the pre-drive setup (background + target + Jetbot
aimed at it), not the post-collision outcome -- it's meant to be a reusable, replayable scene:
an OmniGraph action graph (OnPlaybackTick -> DifferentialController -> IsaacArticulationController,
mirroring the wiring Isaac Sim's own test_differential_controller.py uses) is embedded in the
Jetbot so pressing Play in Isaac Sim drives it on its own at --speed, no external script needed.
This same graph is what drives the Jetbot during this script's own capture below, so the exported
stage reproduces exactly what the video shows. The post-collision outcome itself is only visible
in the exported --output video and this script's printed displacement/PASS-FAIL summary -- the
exported stage is the pre-drive setup, not a snapshot of this run's result.

Usage:
    uv run demo_robot_collide.py --scene /path/to/genrecon_output/shapes/glb/scene.usda --robot-target bowl
"""

import argparse
import math
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True, help="Path to a composed scene.usda (from compose_isaac_scene.py).")
    parser.add_argument(
        "--robot-target",
        default="bowl",
        help="Prim name under the referenced scene's /World to drive into. Must be a dynamic "
        "(RigidBodyAPI-bearing) object from convert_asset.py -- the 'background' label is a "
        "static collider by design and won't move.",
    )
    parser.add_argument("--gui", action="store_true", help="Open an Isaac Sim window instead of running headless.")
    parser.add_argument("--output", default=None, help="Path to write the collision mp4 to (default: output/<scene>_<label>_collide.mp4).")
    parser.add_argument(
        "--stage-output",
        default=None,
        help="Path to export the pre-drive setup USD stage to -- background + target + Jetbot "
        "aimed at it, before any driving happens (default: output/<scene>_<label>_collide_scene.usda). "
        "The post-collision outcome is only visible in the exported --output video, not this stage.",
    )
    parser.add_argument("--robot-scale", type=float, default=0.5, help="Uniform scale factor applied to the Jetbot (e.g. 0.5 for half its default size).")
    parser.add_argument("--start-distance", type=float, default=1.0, help="Distance in meters behind the target to start the robot from.")
    parser.add_argument("--speed", type=float, default=0.6, help="Linear drive speed in m/s.")
    parser.add_argument("--num-steps", type=int, default=360, help="Number of physics/render steps to simulate while driving.")
    parser.add_argument("--settle-steps", type=int, default=60, help="Additional steps to let the scene settle after driving, before the final displacement check.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--resolution", type=int, nargs=2, default=[1280, 720], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument(
        "--displacement-tolerance",
        type=float,
        default=0.03,
        help="Minimum horizontal displacement (meters) of the target's bbox center to count as a real collision.",
    )
    return parser.parse_args()


args = parse_args()

# SimulationApp must be constructed before any other isaacsim/omni import.
from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": not args.gui})

# By default PhysX drives simulated transforms through Fabric (the fast in-memory scene graph
# used for rendering) without writing them back onto the real USD stage's attributes -- so the
# live viewport/rendered video shows the correct collision, but get_bbox()'s stage queries and
# stage.Export() below would otherwise still see each prim's original, pre-simulation pose.
# Enabling this setting makes PhysX write simulated transforms back to USD every step.
import carb  # noqa: E402
from omni.physx.bindings._physx import SETTING_UPDATE_TO_USD  # noqa: E402

carb.settings.get_settings().set_bool(SETTING_UPDATE_TO_USD, True)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import omni.graph.core as og  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import RigidPrim  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from isaacsim.core.utils.nucleus import get_assets_root_path  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.robot.wheeled_robots.robots import WheeledRobot  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402

# The node types used by add_self_driving_graph() below; enable_extension is idempotent, so
# this is safe even if they're already loaded as part of the base app.
enable_extension("omni.graph.action")
enable_extension("isaacsim.core.nodes")
enable_extension("isaacsim.robot.wheeled_robots")

SCENE_PRIM_PATH = "/World/Scene"
JETBOT_PRIM_PATH = "/World/Jetbot"
JETBOT_DRIVE_GRAPH_PATH = "/World/JetbotDriveGraph"
# The Jetbot USD asset ships its own onboard front-facing camera at this path -- reused here as
# the capture camera (rather than authoring a separate static one) so the video is shot from the
# robot's own point of view as it drives.
JETBOT_CAMERA_PRIM_PATH = f"{JETBOT_PRIM_PATH}/chassis/rgb_camera/jetbot_camera"
DOME_LIGHT_PRIM_PATH = "/World/DomeLight"
DISTANT_LIGHT_PRIM_PATH = "/World/DistantLight"
JETBOT_USD_PATH = "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd"
WHEEL_DOF_NAMES = ["left_wheel_joint", "right_wheel_joint"]
# Jetbot's real wheel geometry at its default (unscaled) size (Isaac Sim's own
# test_differential_controller.py exercises the Jetbot asset with these exact values) -- required
# so DifferentialController's unicycle-model wheel speeds actually match the asset being driven.
# All of these (and the derived constants below) get multiplied by --robot-scale at runtime, since
# scaling the Jetbot prim uniformly scales its real wheel radius/base/height by the same factor.
JETBOT_WHEEL_RADIUS_DEFAULT = 0.03
# Measured from the asset's own wheel-joint axle offsets (physics:localPos0 y = +-0.0467), not the
# 0.1125 figure test_differential_controller.py uses -- that value overestimates this asset's real
# track width by ~20%, which made DifferentialController command a larger real turn rate than the
# seek controller intended for a given angular velocity.
JETBOT_WHEEL_BASE_DEFAULT = 0.0934
# The Jetbot asset's root-prim origin is not its ground-contact point: its wheels are children of
# the root at local xformOp:translate z ~= 0.0191 (axle height above the root), so the true
# wheel-ground-contact point sits at 0.0191 - JETBOT_WHEEL_RADIUS ~= -0.0109 relative to the root
# -- i.e. spawning at position.z = <ground height> embeds the wheels ~1.1cm into the ground.
JETBOT_WHEEL_LOCAL_Z_DEFAULT = 0.0191
GROUND_PENETRATION_TOLERANCE = 0.002
# Directions (degrees, standard math convention around +X) to try the robot's start position at,
# in preference order -- -X (180 deg) first to match this script's previous fixed default, then
# the other axes, then the diagonals.
CANDIDATE_START_ANGLES_DEG = [180.0, 0.0, 90.0, -90.0, 135.0, -135.0, 45.0, -45.0]
JETBOT_APPROX_HEIGHT_DEFAULT = 0.15  # rough Jetbot chassis height, for filtering obstacles that are entirely above/below it.
BACKGROUND_LABEL = "background"


def get_bbox(prim_path):
    stage = omni.usd.get_context().get_stage()
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bound_range = bbox_cache.ComputeWorldBound(stage.GetPrimAtPath(prim_path)).ComputeAlignedRange()
    return np.array(bound_range.GetMin()), np.array(bound_range.GetMax())


def get_obstacle_bboxes(target_prim_path):
    """World-space bboxes of every sibling prim under SCENE_PRIM_PATH except the target itself
    and the pipeline's own static floor prims (background, GroundPlane) -- those are the floor/
    walls the robot is expected to drive over, not obstacles to route around."""
    stage = omni.usd.get_context().get_stage()
    target_name = target_prim_path.rsplit("/", 1)[-1]
    excluded = {target_name, "background", "GroundPlane"}
    bboxes = []
    for child in stage.GetPrimAtPath(SCENE_PRIM_PATH).GetChildren():
        if child.GetName() in excluded:
            continue
        bmin, bmax = get_bbox(child.GetPath().pathString)
        if np.any(np.isinf(bmin)) or np.any(np.isinf(bmax)):
            continue
        bboxes.append((bmin, bmax))
    return bboxes


def segment_clears_bbox_xy(p0_xy, p1_xy, half_width, bmin_xy, bmax_xy, num_samples=20):
    """Approximates a 2D capsule-vs-AABB clearance check by sampling points along the p0->p1
    segment and testing each against the AABB expanded by half_width -- simple rather than an
    exact closest-distance computation, which is plenty for a pre-flight path check."""
    expanded_min = bmin_xy - half_width
    expanded_max = bmax_xy + half_width
    for t in np.linspace(0.0, 1.0, num_samples):
        point = p0_xy + t * (p1_xy - p0_xy)
        if np.all(point >= expanded_min) and np.all(point <= expanded_max):
            return False
    return True


def point_to_aabb_signed_distance(point_xy, bmin_xy, bmax_xy):
    """Distance from point_xy to the [bmin_xy, bmax_xy] rectangle's boundary -- positive if
    outside, negative (the penetration depth to the nearest edge) if inside."""
    dx = max(bmin_xy[0] - point_xy[0], 0.0, point_xy[0] - bmax_xy[0])
    dy = max(bmin_xy[1] - point_xy[1], 0.0, point_xy[1] - bmax_xy[1])
    if dx > 0.0 or dy > 0.0:
        return math.hypot(dx, dy)
    inside_dx = min(point_xy[0] - bmin_xy[0], bmax_xy[0] - point_xy[0])
    inside_dy = min(point_xy[1] - bmin_xy[1], bmax_xy[1] - point_xy[1])
    return -min(inside_dx, inside_dy)


def path_clearance(p0_xy, p1_xy, bboxes_xy, half_width, num_samples=20):
    """The worst-case (minimum) gap between the robot's path (a half_width-wide corridor from p0
    to p1) and the nearest obstacle, across all obstacles and sample points -- positive means the
    whole path stays at least that far from every obstacle; negative means it would overlap one
    by that depth. Used to rank candidate start poses by how badly (or well) they clear
    obstacles, rather than just a binary clear/not-clear verdict, so a fallback pose can still be
    the least-bad option instead of an arbitrary default."""
    worst = math.inf
    for t in np.linspace(0.0, 1.0, num_samples):
        point = p0_xy + t * (p1_xy - p0_xy)
        for bmin_xy, bmax_xy in bboxes_xy:
            worst = min(worst, point_to_aabb_signed_distance(point, bmin_xy, bmax_xy) - half_width)
    return worst


def find_clear_start_pose(target_prim_path, target_center, target_min, start_distance, robot_scale):
    """Tries candidate spawn directions around the target in CANDIDATE_START_ANGLES_DEG order
    (starting with -X, this script's previous fixed default) and returns the first whose
    straight-line path to the target doesn't pass through another scene object -- so the robot
    doesn't spawn with an already-obstructed path just because -X happened to have something in
    the way. Returns (start_position, orientation). If none are fully clear (e.g. a small,
    cluttered room where every direction's margin overlaps something), falls back to whichever
    candidate has the largest path_clearance() -- the least-obstructed option, which may still
    leave enough real (unpadded) room for the robot to reach the target -- rather than always
    defaulting to -X, which can be the most-blocked direction rather than merely a flagged one."""
    wheel_radius = JETBOT_WHEEL_RADIUS_DEFAULT * robot_scale
    wheel_base = JETBOT_WHEEL_BASE_DEFAULT * robot_scale
    wheel_local_z = JETBOT_WHEEL_LOCAL_Z_DEFAULT * robot_scale
    ground_contact_offset = wheel_radius - wheel_local_z
    approx_height = JETBOT_APPROX_HEIGHT_DEFAULT * robot_scale
    path_half_width = wheel_base / 2.0 + 0.05  # robot half-width plus a clearance margin.

    obstacle_bboxes = get_obstacle_bboxes(target_prim_path)
    target_xy = target_center[:2]
    robot_z_min, robot_z_max = target_min[2], target_min[2] + approx_height
    relevant_bboxes_xy = [
        (bmin[:2], bmax[:2]) for bmin, bmax in obstacle_bboxes if not (bmax[2] < robot_z_min or bmin[2] > robot_z_max)
    ]

    best_fallback = None  # (clearance, start_position, orientation, angle_deg)
    for angle_deg in CANDIDATE_START_ANGLES_DEG:
        angle = np.radians(angle_deg)
        direction_xy = np.array([np.cos(angle), np.sin(angle)])
        start_xy = target_xy + direction_xy * start_distance
        start_position = np.array([start_xy[0], start_xy[1], target_min[2] + ground_contact_offset])

        # Face from start toward target, i.e. the direction opposite direction_xy (Jetbot's
        # forward axis is +X at the identity orientation).
        yaw = np.arctan2(-direction_xy[1], -direction_xy[0])
        orientation = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])

        clearance = path_clearance(start_xy, target_xy, relevant_bboxes_xy, path_half_width)
        if best_fallback is None or clearance > best_fallback[0]:
            best_fallback = (clearance, start_position, orientation, angle_deg)

        if clearance >= 0.0:
            if angle_deg != CANDIDATE_START_ANGLES_DEG[0]:
                print(f"Default -X start position was obstructed; using a {angle_deg:.0f} deg start instead.")
            return start_position, orientation

    clearance, start_position, orientation, angle_deg = best_fallback
    print(
        f"Warning: no fully clear start position found among candidates; using the least-obstructed "
        f"option ({angle_deg:.0f} deg, clearance {clearance:.3f}m) -- the drive may still clip an obstacle."
    )
    return start_position, orientation


def setup_scene(scene_path, target_label, start_distance, robot_scale):
    """References the composed scene, then places a Jetbot start_distance from the target along
    whichever of CANDIDATE_START_ANGLES_DEG has a clear straight-line path (see
    find_clear_start_pose()), facing it. Returns (target_prim_path, expected_floor_z, jetbot),
    where expected_floor_z is the target's authored (pre-physics) bbox-min Z -- the height
    compose_isaac_scene.py's pivot/world-pose math already placed it at, used later to detect and
    correct any sinking caused by lossy collision approximation."""
    add_reference_to_stage(usd_path=os.path.abspath(scene_path), prim_path=SCENE_PRIM_PATH)

    target_prim_path = f"{SCENE_PRIM_PATH}/{target_label}"
    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(target_prim_path).IsValid():
        raise RuntimeError(f"No prim at {target_prim_path} -- check --robot-target matches a label in {scene_path}.")

    target_min, target_max = get_bbox(target_prim_path)
    target_center = (target_min + target_max) / 2.0

    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        raise RuntimeError("Could not resolve Isaac Sim assets root path (no Nucleus connection).")

    # The ground-contact offset (computed inside find_clear_start_pose from the scaled wheel
    # geometry) corrects for the Jetbot root prim's origin sitting above its actual
    # wheel-ground-contact point.
    start_position, facing_orientation = find_clear_start_pose(
        target_prim_path, target_center, target_min, start_distance, robot_scale
    )

    jetbot = WheeledRobot(
        prim_path=JETBOT_PRIM_PATH,
        name="jetbot",
        wheel_dof_names=WHEEL_DOF_NAMES,
        create_robot=True,
        usd_path=assets_root_path + JETBOT_USD_PATH,
        position=start_position,
        orientation=facing_orientation,
    )
    if robot_scale != 1.0:
        jetbot.set_local_scale(np.array([robot_scale, robot_scale, robot_scale]))
    return target_prim_path, float(target_min[2]), jetbot


def correct_ground_penetration(target_prim_path, expected_floor_z, tolerance=GROUND_PENETRATION_TOLERANCE):
    """Compares the target's current (post-physics-settle) bbox-min Z against its authored,
    pre-physics expected_floor_z. convert_asset.py's collision approximation (convexDecomposition
    for dynamic objects, meshSimplification for the static background) is a lossy proxy of the
    visual mesh, so a target can visibly settle a bit below its authored floor height even though
    physics itself is stable. Once world.reset()/world.step() have run, PhysX's USD write-back
    (SETTING_UPDATE_TO_USD) has already replaced the target's original xformOpOrder (translate/
    orient/scale/pivot/world_pose) with a plain live translate/orient/scale representing its
    simulated pose -- so correcting it means teleporting the rigid body via RigidPrim (which
    updates PhysX's own internal transform, not just a USD attribute PhysX would immediately
    overwrite again), not editing USD xformOps directly. If the discrepancy exceeds `tolerance`,
    teleports the target up by the measured amount and returns it; returns 0.0 otherwise."""
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


def zero_out_background_friction():
    """convert_asset.py gives the background static collider (the meshSimplification room-scan
    shell) the same PhysicsMaterial friction as every other asset converted in that run -- there's
    no per-object override at conversion time, so background/floor/targets all share one
    --static-friction/--dynamic-friction value. Zero it here on the referenced stage instead, so
    the background still collides (its RigidBodyAPI-less static mesh is unaffected) but nothing
    resting on or sliding across it is slowed down by it. No-ops if the scene has no prim named
    BACKGROUND_LABEL or it has no bound physics material."""
    stage = omni.usd.get_context().get_stage()
    background_prim = stage.GetPrimAtPath(f"{SCENE_PRIM_PATH}/{BACKGROUND_LABEL}")
    if not background_prim.IsValid():
        return

    material_prim = UsdShade.MaterialBindingAPI(background_prim).ComputeBoundMaterial("physics")[0].GetPrim()
    if not material_prim.IsValid():
        print(f"Warning: '{BACKGROUND_LABEL}' has no bound physics material -- friction left unchanged.")
        return

    material_api = UsdPhysics.MaterialAPI(material_prim)
    material_api.CreateStaticFrictionAttr().Set(0.0)
    material_api.CreateDynamicFrictionAttr().Set(0.0)
    print(f"Zeroed static/dynamic friction on '{material_prim.GetPath()}'.")


# Closed-loop seek-controller tuning: PID gains on heading error (rad/s of angular velocity per
# rad, rad/s, rad/s^-1 of heading error/integral/derivative respectively), a bound on the integral
# accumulator (anti-windup, in rad*s) and a cap on angular velocity so a large heading error (e.g.
# the robot manually repositioned to face away from the target) doesn't command an unrealistically
# fast spin.
SEEK_HEADING_KP = 2.0
SEEK_HEADING_KI = 0.5
SEEK_HEADING_KD = 0.3
SEEK_INTEGRAL_LIMIT = 1.0
SEEK_MAX_ANGULAR_VELOCITY = 2.5

# Runs inside an embedded OmniGraph ScriptNode (see add_self_driving_graph()) -- baked into the
# exported stage as the node's inputs:script string, not executed by this script itself. Each
# tick it reads the Jetbot's and target's *current* world positions (so it keeps working even if
# either prim is manually moved in the GUI after this stage is loaded) and outputs a steering
# command instead of driving a fixed heading, unlike the previous fixed-angularVelocity=0 version.
# Forward speed is read from db.inputs.speed (a real OmniGraph attribute, set up as
# "SeekController.inputs:speed" in add_self_driving_graph()) rather than baked into the script
# text, so it shows up as a normal editable pin in the GUI's Property panel and takes effect
# immediately -- editing a plain variable inside the script text wouldn't take effect without
# also resetting the ScriptNode's state:omni_initialized to force a recompile.
#
# setup() forces PhysX to write simulated transforms back onto the USD stage
# (SETTING_UPDATE_TO_USD) the moment this node initializes. Without it, the compute() below reads
# robot/target poses through the classic pxr.Usd stage while PhysX is actually simulating through
# Fabric -- which by default does NOT write back to that stage each frame -- so the poses read here
# would be frozen at their pre-simulation values for the whole run, heading_error would never
# update, and the commanded turn would never converge (it would just trace a constant-curvature
# circle forever instead of aligning and driving straight). Setting this here makes the fix
# self-contained in the graph: it works whether this stage is driven by this script or opened
# directly in the Isaac Sim GUI and played, with no external script required.
#
# compute() itself is a standard discrete PID on heading error: integral and previous-error state
# persist across ticks as module-level globals (the script executes once into a persistent module
# namespace, so `global` inside compute() is enough -- no db.internal_state plumbing needed), and
# dt comes from db.inputs.deltaSeconds (wired from OnPlaybackTick.outputs:deltaSeconds in
# add_self_driving_graph()) rather than an assumed fixed rate.
SEEK_CONTROLLER_SCRIPT = f"""
import math
import omni.usd

ROBOT_PATH = "{{robot_path}}"
TARGET_PATH = "{{target_path}}"
HEADING_KP = {SEEK_HEADING_KP}
HEADING_KI = {SEEK_HEADING_KI}
HEADING_KD = {SEEK_HEADING_KD}
INTEGRAL_LIMIT = {SEEK_INTEGRAL_LIMIT}
MAX_ANGULAR_VELOCITY = {SEEK_MAX_ANGULAR_VELOCITY}

_integral = 0.0
_prev_error = 0.0


def setup(db):
    import carb
    from omni.physx.bindings._physx import SETTING_UPDATE_TO_USD
    carb.settings.get_settings().set_bool(SETTING_UPDATE_TO_USD, True)


def _world_xy(stage, prim_path):
    from pxr import UsdGeom
    prim = stage.GetPrimAtPath(prim_path)
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    translation = matrix.ExtractTranslation()
    return translation[0], translation[1]


def _world_yaw(stage, prim_path):
    from pxr import Gf, UsdGeom
    prim = stage.GetPrimAtPath(prim_path)
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    forward = Gf.Vec3d(1.0, 0.0, 0.0) * matrix.ExtractRotationMatrix()
    return math.atan2(forward[1], forward[0])


def compute(db):
    global _integral, _prev_error

    stage = omni.usd.get_context().get_stage()
    robot_x, robot_y = _world_xy(stage, ROBOT_PATH)
    target_x, target_y = _world_xy(stage, TARGET_PATH)
    robot_yaw = _world_yaw(stage, ROBOT_PATH)

    desired_yaw = math.atan2(target_y - robot_y, target_x - robot_x)
    heading_error = math.atan2(math.sin(desired_yaw - robot_yaw), math.cos(desired_yaw - robot_yaw))

    dt = db.inputs.deltaSeconds
    if dt > 0.0:
        _integral = max(-INTEGRAL_LIMIT, min(INTEGRAL_LIMIT, _integral + heading_error * dt))
        derivative = (heading_error - _prev_error) / dt
    else:
        derivative = 0.0
    _prev_error = heading_error

    angular_velocity = HEADING_KP * heading_error + HEADING_KI * _integral + HEADING_KD * derivative
    angular_velocity = max(-MAX_ANGULAR_VELOCITY, min(MAX_ANGULAR_VELOCITY, angular_velocity))

    db.outputs.linearVelocity = db.inputs.speed
    db.outputs.angularVelocity = angular_velocity
    return True
"""


def add_self_driving_graph(robot_path, target_path, speed, robot_scale):
    """Embeds an OmniGraph action graph in the stage so pressing Play in Isaac Sim drives the
    Jetbot toward the target on its own at `speed`, with no external script required. Unlike a
    fixed-heading open-loop drive, a ScriptNode ("SeekController") recomputes the heading from
    the Jetbot's *current* world position to the target's *current* world position every tick
    (see SEEK_CONTROLLER_SCRIPT) and feeds a proportional steering command into
    DifferentialController -> IsaacArticulationController (the same wiring Isaac Sim's own
    test_differential_controller.py uses to drive a live Jetbot from a saved graph). Because the
    heading is recomputed live rather than baked into a fixed angularVelocity, the Jetbot (or the
    target) can be manually repositioned/reoriented in the GUI after this stage is loaded and it
    will still drive toward the target. `speed` is exposed as a real "inputs:speed" attribute on
    SeekController (rather than baked into the script text), so it can be tweaked live from the
    Property panel after loading the exported stage, with no script editing or node reload
    needed. This graph is what actually drives the robot, including during this script's own
    capture below, so the exported pre-drive stage reproduces exactly what the video shows once a
    user presses Play on it."""
    script = SEEK_CONTROLLER_SCRIPT.format(robot_path=robot_path, target_path=target_path)
    og.Controller.edit(
        {"graph_path": JETBOT_DRIVE_GRAPH_PATH, "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("SeekController", "omni.graph.scriptnode.ScriptNode"),
                ("DifferentialController", "isaacsim.robot.wheeled_robots.DifferentialController"),
                ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
            ],
            og.Controller.Keys.CREATE_ATTRIBUTES: [
                ("SeekController.inputs:speed", "double"),
                ("SeekController.inputs:deltaSeconds", "double"),
                ("SeekController.outputs:linearVelocity", "double"),
                ("SeekController.outputs:angularVelocity", "double"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "SeekController.inputs:execIn"),
                ("OnPlaybackTick.outputs:deltaSeconds", "SeekController.inputs:deltaSeconds"),
                ("SeekController.outputs:execOut", "DifferentialController.inputs:execIn"),
                ("SeekController.outputs:execOut", "ArticulationController.inputs:execIn"),
                ("SeekController.outputs:linearVelocity", "DifferentialController.inputs:linearVelocity"),
                ("SeekController.outputs:angularVelocity", "DifferentialController.inputs:angularVelocity"),
                ("DifferentialController.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("SeekController.inputs:script", script),
                ("SeekController.inputs:speed", float(speed)),
                ("DifferentialController.inputs:wheelRadius", JETBOT_WHEEL_RADIUS_DEFAULT * robot_scale),
                ("DifferentialController.inputs:wheelDistance", JETBOT_WHEEL_BASE_DEFAULT * robot_scale),
                ("ArticulationController.inputs:robotPath", robot_path),
                ("ArticulationController.inputs:jointNames", WHEEL_DOF_NAMES),
            ],
        },
    )


def add_lighting():
    """GenRecon's composed scene.usda carries geometry/collision only, no lights -- without
    this the render is black. A dome light gives even ambient coverage of both the background
    and the robot regardless of where they end up; a distant light adds directionality/shading
    so shapes are still readable instead of flat."""
    stage = omni.usd.get_context().get_stage()

    dome_light = UsdLux.DomeLight.Define(stage, DOME_LIGHT_PRIM_PATH)
    dome_light.CreateIntensityAttr(1000.0)

    distant_light = UsdLux.DistantLight.Define(stage, DISTANT_LIGHT_PRIM_PATH)
    distant_light.CreateIntensityAttr(3000.0)
    distant_light.CreateAngleAttr(1.0)
    xformable = UsdGeom.Xformable(distant_light.GetPrim())
    xformable.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 30.0, 0.0))


def setup_jetbot_camera(resolution):
    """Wraps the Jetbot asset's own onboard front-facing camera (a child prim rigidly attached to
    its chassis) as the capture camera, so the video is shot from the robot's point of view as it
    drives into the target. The asset authors this camera with a clippingRange far plane of 1e6
    units -- harmless for rendering, but against this ~1m-scale scene (the kitchen background +
    bowl content only spans about 1m) the frustum guide balloons into an enormous shape riding
    along with the robot -- even the previous 50-unit bound was still 50x the room size and reads
    as a giant dolly/cage when the exported stage is opened fresh and the viewport draws camera
    gizmos by default. Bound it to something actually proportional to the scene instead."""
    stage = omni.usd.get_context().get_stage()
    UsdGeom.Camera(stage.GetPrimAtPath(JETBOT_CAMERA_PRIM_PATH)).GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 5.0))
    camera = Camera(prim_path=JETBOT_CAMERA_PRIM_PATH, resolution=tuple(resolution))
    camera.initialize()
    return camera


def run_drive(world, camera, num_frames, fps, output_path):
    """Steps the world while capturing frames. Driving itself comes from the OmniGraph action
    graph added by add_self_driving_graph() (ticking automatically each step, since world.reset()
    already started the Kit timeline) -- not from any Python-side wheel command here."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    resolution = camera.get_resolution()
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, tuple(resolution))
    try:
        for _ in range(num_frames):
            world.step(render=True)
            frame_rgba = camera.get_rgba()
            if frame_rgba is None or frame_rgba.size == 0:
                continue
            frame_bgr = cv2.cvtColor(frame_rgba[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
    finally:
        writer.release()


def run_settle(world, num_frames):
    for _ in range(num_frames):
        world.step(render=True)


def main():
    scene_id = os.path.splitext(os.path.basename(os.path.abspath(args.scene)))[0]
    output_path = args.output or f"output/{scene_id}_{args.robot_target}_collide.mp4"
    stage_output_path = args.stage_output or f"output/{scene_id}_{args.robot_target}_collide_scene.usda"

    world = World()
    world.reset()

    target_prim_path, expected_floor_z, jetbot = setup_scene(
        args.scene, args.robot_target, args.start_distance, args.robot_scale
    )
    zero_out_background_friction()
    add_lighting()
    world.scene.add(jetbot)
    world.reset()
    jetbot.initialize()

    for _ in range(5):
        world.step(render=True)

    penetration = correct_ground_penetration(target_prim_path, expected_floor_z)
    if penetration:
        print(f"Corrected {args.robot_target} sinking by {penetration:.4f}m")
        for _ in range(5):
            world.step(render=True)

    initial_min, initial_max = get_bbox(target_prim_path)
    initial_center = (initial_min + initial_max) / 2.0

    camera = setup_jetbot_camera(args.resolution)

    # Added right before Export() (rather than earlier) so the pre-drive snapshot below still
    # captures the Jetbot at its start pose, undisturbed -- the graph starts ticking as soon as
    # it exists, since world.reset() already started the Kit timeline.
    add_self_driving_graph(JETBOT_PRIM_PATH, target_prim_path, args.speed, args.robot_scale)

    # Export the pre-drive setup, not the post-collision result: this is the reusable "scene"
    # artifact (background + bowl + Jetbot aimed at it, with its self-driving graph attached) --
    # pressing Play on it in Isaac Sim reproduces the same drive-and-collide this script runs
    # below, with no external script needed.
    os.makedirs(os.path.dirname(stage_output_path) or ".", exist_ok=True)
    omni.usd.get_context().get_stage().Export(stage_output_path)
    print(f"Exported pre-drive setup stage to {stage_output_path}")

    for _ in range(5):
        world.step(render=True)

    run_drive(world, camera, args.num_steps, args.fps, output_path)
    print(f"Wrote {args.num_steps} frames to {output_path}")

    run_settle(world, args.settle_steps)

    final_min, final_max = get_bbox(target_prim_path)
    final_center = (final_min + final_max) / 2.0
    horizontal_displacement = float(np.linalg.norm(final_center[:2] - initial_center[:2]))
    print(f"Target '{args.robot_target}' bbox center: initial={initial_center.tolist()} final={final_center.tolist()}")
    print(f"Horizontal displacement: {horizontal_displacement:.4f}m")

    collided = horizontal_displacement >= args.displacement_tolerance
    if collided:
        print(f"PASS: target displaced {horizontal_displacement:.4f}m (>= {args.displacement_tolerance}m tolerance) -- real collision occurred.")
    else:
        print(
            f"FAIL: target displaced only {horizontal_displacement:.4f}m (< {args.displacement_tolerance}m tolerance) -- "
            "robot may have missed the target, or collision response was too weak. Check --start-distance/--speed/--num-steps."
        )

    sys.stdout.flush()
    simulation_app.close()

    if not collided:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
