"""Convert TRELLIS.2 GLB output into physics-ready USD for Isaac Sim.

Reads handoff/<asset_id>/mesh.glb (+ manifest.json written by ../trellis2/generate.py) and
writes handoff/<asset_id>/asset.usd with: Y-up->Z-up + metric conversion, a ground-contact
pivot, and RigidBody/Collision/Mass/PhysicsMaterial authored via PhysX's own helpers (which
let PhysX cook collision at load/sim time -- no trimesh/VHACD needed). Updates the per-asset
manifest.json with status="converted", the collision approximation used, and the resulting
bbox extents, so scale/collision issues are visible before an asset is ever referenced into
a scene.

Usage:
    uv run convert_asset.py --input handoff/                        # batch: all handoff/*/mesh.glb
    uv run convert_asset.py --input handoff/chair/mesh.glb           # single asset
"""

import argparse
import json
import os
import sys
import time
import traceback


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        required=True,
        help="A mesh.glb file, or a handoff/ directory to batch-convert every <asset_id>/mesh.glb under it.",
    )
    parser.add_argument("--output", default=None, help="Output .usd path (single-file mode only; defaults to a sibling asset.usd).")
    parser.add_argument(
        "--density",
        type=float,
        default=1000.0,
        help="Density in kg/m^3 used for mass (TRELLIS.2 has no notion of real-world mass; "
        "~500-1000 covers wood/plastic-like props -- defaulting to the top of that range since a "
        "too-low auto-computed mass makes objects fling/spin unrealistically hard when struck). "
        "PhysX derives mass from this x collision volume.",
    )
    parser.add_argument("--static-friction", type=float, default=0.5)
    parser.add_argument("--dynamic-friction", type=float, default=0.5)
    parser.add_argument("--restitution", type=float, default=0.1)
    parser.add_argument(
        "--friction-table",
        default=None,
        help="Path to friction_assignments.json produced by GenRecon's "
        "scripts/infer_friction_assignments.py, keyed by asset_id (the same directory name "
        "matched against --background-label/--floor-label below). Missing table, or an "
        "asset_id absent from it, falls back to --static-friction/--dynamic-friction for that "
        "asset (with a warning) -- exactly today's behavior when this flag isn't passed.",
    )
    parser.add_argument(
        "--friction-combine-mode",
        default="average",
        choices=["average", "min", "multiply", "max"],
        help="PhysX frictionCombineMode authored via PhysxSchema.PhysxMaterialAPI on every "
        "PhysicsMaterial this script creates. Defaults to 'average' -- PhysX's own implicit "
        "default when nothing is authored -- so omitting this flag reproduces prior behavior "
        "exactly. Pass 'max' together with --friction-table so a floor asset's forced "
        "static_friction=dynamic_friction=0 is a no-op at any floor-object contact: PhysX then "
        "computes max(0, mu_object) == mu_object, i.e. the object's own looked-up "
        "mu(material, floor) value, not an averaged-down approximation.",
    )
    parser.add_argument(
        "--collision-approximation",
        default="convexDecomposition",
        choices=["convexDecomposition", "convexHull", "boundingCube"],
        help="PhysX collision approximation applied to dynamic (non-background) mesh prims. "
        "TRELLIS.2's O-Voxel meshes may be open-surfaced/non-manifold, which can make "
        "convexDecomposition cooking unreliable at sim time -- there is no reliable way to detect "
        "a cooking failure ahead of time from this script; if demo_drop_asset.py shows an asset "
        "falling through the floor, rerun conversion for that asset with convexHull or "
        "boundingCube instead.",
    )
    parser.add_argument(
        "--background-label",
        default="background",
        help="Asset directory name (batch mode only) treated as the scene background -- must match "
        "compose_isaac_scene.py's --background-label. Unlike every other (dynamic) asset, the "
        "background never gets a RigidBodyAPI: PhysX only allows exact triangle-mesh collision "
        "('none'/PhysxTriangleMeshCollisionAPI) on prims with no RigidBodyAPI at all -- with one "
        "(dynamic or even kinematic), omni.physx.scripts.utils.setCollider() silently downgrades "
        "'none' to convexHull. GenRecon's per-object crop (extract_object_mesh.py) can leave "
        "concave holes in the background where objects were cut out; a convex approximation would "
        "paper over those holes (or, for convexHull, erase all concavity in the whole background), "
        "letting objects that should fall through a hole rest on empty air instead. The background "
        "is authored as a plain static collider (setStaticCollider) so it keeps its exact shape, "
        "holes included, and needs no kinematic flag since it was never a rigid body to begin with.",
    )
    parser.add_argument(
        "--floor-label",
        default="floor",
        help="Asset directory name (batch mode only) treated as the segmented scene floor -- must "
        "match compose_isaac_scene.py's --floor-label. GenRecon's extract_floor_mesh.py crops the "
        "flat floor region out of the background into its own asset specifically so it can get an "
        "exact analytic-plane collider here instead of the background's decimated "
        "'meshSimplification' triangle-mesh proxy (see --background-label's help): that proxy is "
        "what lets objects fall through the visible floor even though the synthetic ground-plane "
        "safety net (an analytic plane) never lets anything fall through. Like the background, the "
        "floor never gets a RigidBodyAPI (it's fixed scene geometry); unlike the background, its "
        "render mesh carries no collision at all -- collision comes entirely from a sibling "
        "UsdGeom.Plane authored by author_floor_collision().",
    )
    parser.add_argument("--gui", action="store_true", help="Open an Isaac Sim window instead of running headless.")
    return parser.parse_args()


args = parse_args()

# SimulationApp must be constructed before any other isaacsim/omni import.
from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": not args.gui})

import asyncio  # noqa: E402
import omni.kit.asset_converter as converter  # noqa: E402
import omni.usd  # noqa: E402
from omni.physx.scripts import physicsUtils  # noqa: E402
from omni.physx.scripts import utils as physx_utils  # noqa: E402
from pxr import Gf, PhysxSchema, Usd, UsdGeom  # noqa: E402

# Damping/sleep/solver tuning applied to every dynamic object's PhysxRigidBodyAPI in
# author_physics() below -- without these, PhysX's own defaults (no damping, sleepThreshold=0)
# let residual jitter from the (necessarily lossy) convexDecomposition collision proxy
# recirculate forever instead of settling, so a struck object spins/wobbles unrealistically.
# Heavier than the stock Jetbot's own tuning (angularDamping=0.05) since these are inert
# kitchenware/props, not a precision-controlled locomotion articulation.
RIGID_BODY_LINEAR_DAMPING = 0.1
RIGID_BODY_ANGULAR_DAMPING = 0.4
RIGID_BODY_SLEEP_THRESHOLD = 0.001
RIGID_BODY_STABILIZATION_THRESHOLD = 0.001
RIGID_BODY_SOLVER_POSITION_ITERATIONS = 16
RIGID_BODY_SOLVER_VELOCITY_ITERATIONS = 4


def find_glb_assets(input_path, output_override):
    """Returns a list of (asset_dir, glb_path, output_usd_path) tuples.

    Paths are resolved to absolute up front: omni.kit.asset_converter's conversion task
    changes the process's working directory as a side effect, so any relative path used
    afterward (e.g. context.open_stage) silently re-resolves against that new cwd instead
    of the one this script was launched from -- doubling up path segments.
    """
    if os.path.isfile(input_path):
        asset_dir = os.path.dirname(os.path.abspath(input_path))
        output_path = os.path.abspath(output_override) if output_override else os.path.join(asset_dir, "asset.usd")
        return [(asset_dir, os.path.abspath(input_path), output_path)]

    assets = []
    for name in sorted(os.listdir(input_path)):
        asset_dir = os.path.abspath(os.path.join(input_path, name))
        glb_path = os.path.join(asset_dir, "mesh.glb")
        if os.path.isdir(asset_dir) and os.path.isfile(glb_path):
            assets.append((asset_dir, glb_path, os.path.join(asset_dir, "asset.usd")))
    return assets


async def convert_glb_to_usd(glb_path, usd_path):
    """Runs the omni.kit.asset_converter task; must be awaited from a live Kit app."""
    context = converter.AssetConverterContext()
    context.embed_textures = True  # PBR textures are only embedded for FBX/glTF export/import
    context.use_meter_as_world_unit = True
    context.convert_stage_up_z = True  # glTF is always Y-up; this repo's Isaac Sim stages are Z-up
    context.export_preview_surface = False  # keep MDL so metallic/roughness round-trip correctly
    manager = converter.get_instance()
    task = manager.create_converter_task(glb_path, usd_path, None, context)
    return await task.wait_until_finished()


def get_bbox(stage, prim_path):
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bound_range = bbox_cache.ComputeWorldBound(stage.GetPrimAtPath(prim_path)).ComputeAlignedRange()
    return bound_range.GetMin(), bound_range.GetMax()


def normalize_pivot(stage, prim_path):
    """Re-centers the root xform (XY) and drops it to bbox-min-Z, so a plain translate onto
    a ground plane places the object correctly without per-asset guesswork.

    Returns the PRE-pivot bbox (not re-fetched after adding the op): any geometry authored
    later as a *child* of this same prim_path -- e.g. author_floor_collision()'s
    FloorCollisionPlane -- inherits this pivot op too, so a Z value computed from the
    post-pivot bbox and then placed via a local translate on such a child gets the pivot
    shift applied twice (once baked into the value, once again through inheritance), landing
    it off from the actual (once-shifted) mesh position by a full pivot_z. Bbox extents
    (max - min) are translation-invariant, so returning the pre-pivot bbox doesn't change
    anything for callers that only care about size (e.g. the manifest's bbox_extents_m)."""
    bbox_min, bbox_max = get_bbox(stage, prim_path)
    xformable = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
    offset = Gf.Vec3d(-(bbox_min[0] + bbox_max[0]) / 2.0, -(bbox_min[1] + bbox_max[1]) / 2.0, -bbox_min[2])
    # The glTF-imported prim already has its own translate op (xformOpOrder starts with one);
    # AddTranslateOp() with no suffix collides with it ("xformOp:translate already exists").
    # A uniquely-suffixed op is appended at the end of xformOpOrder instead, applying this
    # pivot offset in world space on top of the existing local transform.
    xformable.AddTranslateOp(opSuffix="pivot").Set(offset)
    return bbox_min, bbox_max


def _author_friction_combine_mode(stage, material_path, mode):
    """Authors PhysX's frictionCombineMode on the PhysicsMaterial at material_path so a
    contact's effective friction is max(mu_a, mu_b) instead of PhysX's implicit
    'average' -- see --friction-combine-mode's help for why this matters for a floor
    asset's forced-zero friction."""
    prim = stage.GetPrimAtPath(material_path)
    physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(prim)
    physx_material_api.CreateFrictionCombineModeAttr().Set(mode)


def author_physics(stage, prim_path, args, static_friction, dynamic_friction):
    prim = stage.GetPrimAtPath(prim_path)
    physx_utils.setRigidBody(prim, approximationShape=args.collision_approximation, kinematic=False)
    physicsUtils.add_density(stage, prim_path, args.density)

    physx_rigid_body = PhysxSchema.PhysxRigidBodyAPI(prim)
    physx_rigid_body.CreateLinearDampingAttr().Set(RIGID_BODY_LINEAR_DAMPING)
    physx_rigid_body.CreateAngularDampingAttr().Set(RIGID_BODY_ANGULAR_DAMPING)
    physx_rigid_body.CreateSleepThresholdAttr().Set(RIGID_BODY_SLEEP_THRESHOLD)
    physx_rigid_body.CreateStabilizationThresholdAttr().Set(RIGID_BODY_STABILIZATION_THRESHOLD)
    physx_rigid_body.CreateSolverPositionIterationCountAttr().Set(RIGID_BODY_SOLVER_POSITION_ITERATIONS)
    physx_rigid_body.CreateSolverVelocityIterationCountAttr().Set(RIGID_BODY_SOLVER_VELOCITY_ITERATIONS)

    material_path = f"{prim_path}/PhysicsMaterial"
    physx_utils.addRigidBodyMaterial(
        stage,
        material_path,
        density=args.density,
        staticFriction=static_friction,
        dynamicFriction=dynamic_friction,
        restitution=args.restitution,
    )
    physicsUtils.add_physics_material_to_prim(stage, prim, material_path)
    _author_friction_combine_mode(stage, material_path, args.friction_combine_mode)


FLOOR_PLANE_MARGIN = 0.1  # extra half-extent (m) beyond the floor slab's own footprint


def author_floor_collision(stage, prim_path, args, bbox_min, bbox_max, static_friction, dynamic_friction):
    """Floor variant: the render mesh gets no collision at all -- instead, a sibling analytic
    UsdGeom.Plane collider (via physicsUtils.add_ground_plane, the same mechanism
    compose_isaac_scene.py's synthetic ground-plane safety net uses, which is why *that* never
    lets anything fall through) sized to the floor slab's own footprint.

    Positioned at the *midpoint* of the asset's local Z bbox, not bbox_max/bbox_min: this mesh
    isn't a solid slab with a real top/bottom face, it's a thin noisy point-sample of a single
    fitted plane (extract_floor_mesh.py keeps every vertex within +/-distance_threshold of the
    fit), so bbox_max/bbox_min are just the RANSAC tolerance ceiling/floor, offset from the true
    fitted height by ~distance_threshold. Placing the plane at either extreme biases it off the
    real surface by that amount -- enough for resting objects (positioned against the *true*
    surface) to start embedded in the collider, which PhysX's depenetration solver then resolves
    by launching them explosively on the first simulation step. The midpoint is the closest
    unbiased estimate of the fitted plane's actual height available from just the bbox. Like
    author_static_collision, no RigidBodyAPI (it's fixed scene geometry, not a dynamic body)."""
    extent_x = bbox_max[0] - bbox_min[0]
    extent_y = bbox_max[1] - bbox_min[1]
    plane_size = max(extent_x, extent_y) / 2.0 + FLOOR_PLANE_MARGIN
    plane_z = (bbox_min[2] + bbox_max[2]) / 2.0
    physicsUtils.add_ground_plane(
        stage, f"{prim_path}/FloorCollisionPlane", "Z", plane_size, Gf.Vec3f(0.0, 0.0, plane_z), Gf.Vec3f(0.5, 0.5, 0.5)
    )

    prim = stage.GetPrimAtPath(prim_path)
    material_path = f"{prim_path}/PhysicsMaterial"
    physx_utils.addRigidBodyMaterial(
        stage,
        material_path,
        density=args.density,
        staticFriction=static_friction,
        dynamicFriction=dynamic_friction,
        restitution=args.restitution,
    )
    physicsUtils.add_physics_material_to_prim(stage, prim, material_path)
    _author_friction_combine_mode(stage, material_path, args.friction_combine_mode)


def author_static_collision(stage, prim_path, args, static_friction, dynamic_friction):
    """Background variant of author_physics(): no RigidBodyAPI/mass (it's not a dynamic body,
    just fixed scene geometry), and 'meshSimplification' collision approximation (a decimated,
    still-concave triangle mesh) instead of a convex one, so concave crop holes are preserved.
    See --background-label's help for why concavity matters.

    Not 'none' (exact triangle mesh): GenRecon backgrounds routinely land in the
    millions-of-triangles range (e.g. ~18.5M for a single room crop), and PhysX's cooker
    silently fails on meshes that large -- confirmed via
    '[omni.physx.cooking.plugin] UjitsoMeshCookingContext: cooking failure' /
    '[omni.physx.plugin] Unable to create triangle mesh for: ...' warnings at sim start, which
    leave the prim with *no* collision at all (not an exception, so nothing else signals the
    failure) and every dynamic object dropped onto it falls straight through to the ground-plane
    safety net. 'meshSimplification' has PhysX cook a decimated concave proxy instead of the
    exact mesh, which succeeds at this vertex count while still respecting crop-hole concavity."""
    prim = stage.GetPrimAtPath(prim_path)
    physx_utils.setStaticCollider(prim, approximationShape="meshSimplification")

    material_path = f"{prim_path}/PhysicsMaterial"
    physx_utils.addRigidBodyMaterial(
        stage,
        material_path,
        density=args.density,
        staticFriction=static_friction,
        dynamicFriction=dynamic_friction,
        restitution=args.restitution,
    )
    physicsUtils.add_physics_material_to_prim(stage, prim, material_path)
    _author_friction_combine_mode(stage, material_path, args.friction_combine_mode)


def resolve_friction(asset_id, friction_table, args):
    """Returns (static_friction, dynamic_friction) for asset_id: looks it up in
    friction_table (parsed from --friction-table's JSON), falling back to
    args.static_friction/args.dynamic_friction (today's global CLI defaults) with a
    warning if a table was supplied but has no entry for this asset_id -- the
    'background' label is expected to always fall through to this path, since
    infer_friction_assignments.py never writes a 'background' key."""
    entry = friction_table.get(asset_id)
    if entry is None:
        if friction_table:
            print(
                f"  WARNING: no friction entry for '{asset_id}' in --friction-table, falling back to "
                f"--static-friction={args.static_friction}/--dynamic-friction={args.dynamic_friction}."
            )
        return args.static_friction, args.dynamic_friction
    return entry["static_friction"], entry["dynamic_friction"]


def process_asset(asset_dir, glb_path, usd_path, args, is_background, is_floor, static_friction, dynamic_friction):
    manifest_path = os.path.join(asset_dir, "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    success = asyncio.get_event_loop().run_until_complete(convert_glb_to_usd(glb_path, usd_path))
    if not success:
        manifest["status"] = "failed"
        manifest["conversion_error"] = "omni.kit.asset_converter task failed"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        return False

    context = omni.usd.get_context()
    context.open_stage(usd_path)
    stage = context.get_stage()

    default_prim = stage.GetDefaultPrim()
    prim_path = default_prim.GetPath().pathString if default_prim else "/World"

    bbox_min, bbox_max = normalize_pivot(stage, prim_path)
    if is_floor:
        collision_approximation = "analytic_plane"
        author_floor_collision(stage, prim_path, args, bbox_min, bbox_max, static_friction, dynamic_friction)
    elif is_background:
        collision_approximation = "meshSimplification"
        author_static_collision(stage, prim_path, args, static_friction, dynamic_friction)
    else:
        collision_approximation = args.collision_approximation
        author_physics(stage, prim_path, args, static_friction, dynamic_friction)
    stage.Save()

    extents = [bbox_max[i] - bbox_min[i] for i in range(3)]
    manifest["status"] = "converted"
    manifest["converted_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["usd_path"] = os.path.abspath(usd_path)
    manifest["collision_approximation"] = collision_approximation
    manifest["density"] = None if (is_background or is_floor) else args.density
    manifest["bbox_extents_m"] = extents
    manifest["static_friction"] = static_friction
    manifest["dynamic_friction"] = dynamic_friction
    manifest["friction_combine_mode"] = args.friction_combine_mode
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    if max(extents) > 20.0 or max(extents) < 0.02:
        print(
            f"  WARNING: bbox extents {extents} look implausible for a real-world object -- "
            f"TRELLIS.2 doesn't guarantee metric output scale, check/scale this asset manually."
        )

    context.close_stage()
    return True


def load_friction_table(path):
    if path is None:
        return {}
    with open(path) as f:
        return json.load(f)


def main():
    friction_table = load_friction_table(args.friction_table)
    assets = find_glb_assets(args.input, args.output)
    if not assets:
        print(f"No mesh.glb found under {args.input}", file=sys.stderr)
        sys.exit(1)

    results = []
    for asset_dir, glb_path, usd_path in assets:
        asset_id = os.path.basename(asset_dir.rstrip("/")) or asset_dir
        is_background = asset_id == args.background_label
        is_floor = asset_id == args.floor_label
        static_friction, dynamic_friction = resolve_friction(asset_id, friction_table, args)
        tag = ""
        if is_floor:
            tag = " (floor, analytic-plane collision)"
        elif is_background:
            tag = " (background, static triangle-mesh collision)"
        print(f"[convert] {asset_id}{tag}: {glb_path} -> {usd_path} (friction={static_friction}/{dynamic_friction})")
        try:
            ok = process_asset(asset_dir, glb_path, usd_path, args, is_background, is_floor, static_friction, dynamic_friction)
            results.append((asset_id, "ok" if ok else "failed"))
        except Exception:
            traceback.print_exc()
            results.append((asset_id, "failed"))

    print("\nSummary:")
    for asset_id, status in results:
        print(f"  {status:8s} {asset_id}")

    simulation_app.close()
    if any(status == "failed" for _, status in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
