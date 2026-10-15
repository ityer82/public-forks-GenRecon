"""Compose every converted asset (background + objects) from convert_asset.py into a single
USD stage, restoring each one's original relative world pose.

convert_asset.py's normalize_pivot() recenters each asset's root Xform to local origin
(XY centroid, Z at bbox-min) so demo_drop_asset.py can drop a single object without
per-asset guesswork -- but that throws away where the object actually sat relative to the
rest of the scene. That recentering is a pure translation (no rotation) and is applied
identically across the whole batch, so it survives as a plain xformOp:translate:pivot
attribute on each asset.usd. Referencing each asset in and adding an outer translate equal
to the negated pivot exactly undoes it, restoring the original relative layout.

Any non-background, non-floor asset whose restored world position leaves its bottom below the
floor plane's world height gets an extra upward nudge (see --floor-contact-margin) -- without
it, reconstruction noise can leave an object's lowest point embedded in the floor's analytic
collision plane, which PhysX resolves with an explosive depenetration launch at sim start.

Usage:
    uv run compose_isaac_scene.py --input shapes/glb --output shapes/glb/scene.usda
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Directory containing <label>/asset.usd + <label>/manifest.json (convert_asset.py's --input).")
    parser.add_argument("--output", required=True, help="Path to write the composed stage to, e.g. scene.usda.")
    parser.add_argument(
        "--background-label",
        default="background",
        help="Label of the background asset (default 'background', converted from "
        "GenRecon's extract_object_mesh.py background_mesh.ply -- mesh_to_glb.py strips "
        "the '_mesh' suffix from the ply stem for the asset directory name, so "
        "background_mesh.ply becomes label 'background', not 'background_mesh'; the "
        "original, uncropped mesh.ply is never converted/composed). Must match "
        "convert_asset.py's --background-label: that script authors this asset as a plain "
        "static collider (no RigidBodyAPI, exact triangle-mesh collision) instead of a "
        "convex dynamic rigid body, so it both stays fixed in place and keeps any concave "
        "crop holes intact instead of a convex approximation papering over them.",
    )
    parser.add_argument(
        "--floor-label",
        default="floor",
        help="Label of the segmented floor asset (default 'floor', converted from GenRecon's "
        "extract_floor_mesh.py floor_mesh.ply -- same '_mesh' suffix-stripping convention as "
        "--background-label). Must match convert_asset.py's --floor-label: that script gives "
        "this asset an analytic-plane collider instead of a triangle-mesh approximation. Purely "
        "a cosmetic log tag here -- no physics is authored in this file, it's already baked in "
        "by convert_asset.py, and the floor composes through the same reference + pivot-undo "
        "path as every other asset.",
    )
    parser.add_argument(
        "--floor-contact-margin",
        type=float,
        default=0.002,
        help="Minimum clearance (m) to leave between a lifted object's bottom and the floor "
        "plane (default 0.002). Reconstruction/pivot noise means an object's own lowest point "
        "can sit below the floor's true fitted height even though the floor plane itself is "
        "correctly placed -- that residual overlap makes PhysX's depenetration solver launch "
        "the object on the first simulation step, the same mechanism (just smaller magnitude) "
        "as the systematic bug author_floor_collision() already fixes. Any non-background, "
        "non-floor asset whose world bbox bottom is above floor_z + this margin is left alone; "
        "one that dips below is translated straight up until it clears by this margin.",
    )
    parser.add_argument(
        "--trim-background-floor-overlap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop background faces that fall inside the floor's analytic collision plane "
        "footprint (see --floor-overlap-margin), on both the background's render and collision "
        "mesh (the same prim serves both). extract_floor_mesh.py already crops most of the "
        "floor's footprint out of background_mesh.ply, but only with a small hull padding -- a "
        "thin band can still remain at/near floor height, close enough that a resting object "
        "contacts both the floor's exact analytic plane and background's decimated "
        "meshSimplification triangle-mesh proxy at once, and the two colliders disagree (the "
        "background's material may also differ from the floor's), producing jitter/odd rest "
        "behavior right at the seam. On by default; pass --no-trim-background-floor-overlap to "
        "restore the previous behavior. No-op if either asset is missing from --input.",
    )
    parser.add_argument(
        "--floor-overlap-margin",
        type=float,
        default=0.03,
        help="Extra margin (m), in every direction (XY and Z), added around the floor's analytic "
        "collision plane's own world bbox when deciding which background faces to drop for "
        "--trim-background-floor-overlap. Kept small and symmetric on purpose: this only needs "
        "to catch a thin leftover seam, not re-do extract_floor_mesh.py's crop.",
    )
    parser.add_argument(
        "--ground-plane",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add a physics-enabled ground plane beneath the composed scene's lowest point, "
        "as a safety net in case the (possibly gap-ridden, cropped) background asset doesn't "
        "fully cover the floor -- without it, dynamic objects with no other support fall "
        "indefinitely. On by default; pass --no-ground-plane to disable.",
    )
    parser.add_argument(
        "--table",
        dest="add_table",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reference a table asset (see --table-asset-path) positioned directly beneath the "
        "composed floor/objects, so the scene reads as objects resting on a table instead of the "
        "bare ground plane. On by default; pass --no-table to restore the previous ground-only "
        "look. If the asset can't be resolved/loaded, this degrades to a warning and continues "
        "exactly as --no-table would, rather than failing the whole composition.",
    )
    parser.add_argument(
        "--table-asset-path",
        default=str(Path(__file__).resolve().parent / "assets/table/table_instanceable.usd"),
        help="Local/absolute USD path, or a Nucleus-relative path (leading '/', resolved against "
        "get_assets_root_path()) for the table asset used by --add-table. Defaults to a small "
        "(~0.72m x 0.76m) table vendored locally in assets/table/ (see assets/table/README.md) -- "
        "chosen for its size and identity local transform over Isaac Sim's much larger "
        "SeattleLabTable asset.",
    )
    parser.add_argument(
        "--table-height",
        type=float,
        default=0.79,
        help="Fallback expected table top height in meters, used only if the referenced table "
        "asset's own bbox can't be read. Ignored when the asset loads normally -- its own bbox "
        "top drives placement.",
    )
    parser.add_argument(
        "--room",
        dest="add_room",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Place the scene inside a static, collision-free kitchen room (NVIDIA's Simple_Room shell "
        "plus a fridge, stove and counters -- see assets/room/README.md), purely for looks. Off by "
        "default. Requires the table (--table); if the room assets are missing this degrades to a "
        "warning.",
    )
    parser.add_argument(
        "--room-asset-dir",
        default=str(Path(__file__).resolve().parent / "assets/room"),
        help="Directory holding the vendored room assets (populate with assets/room/fetch.sh).",
    )
    parser.add_argument("--gui", action="store_true", help="Open an Isaac Sim window instead of running headless.")
    return parser.parse_args()


args = parse_args()

# SimulationApp must be constructed before any other isaacsim/omni import.
from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": not args.gui})

from isaacsim.core.utils.nucleus import get_assets_root_path  # noqa: E402
from omni.physx.scripts import physicsUtils  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt  # noqa: E402


def find_converted_assets(input_dir):
    """Returns [(label, asset_dir, usd_path), ...] for every <input_dir>/<label>/asset.usd
    whose sibling manifest.json has status == "converted". Mirrors convert_asset.py's
    find_glb_assets() directory-walk convention, keyed off asset.usd + manifest status
    instead of mesh.glb."""
    assets = []
    for name in sorted(os.listdir(input_dir)):
        asset_dir = os.path.abspath(os.path.join(input_dir, name))
        usd_path = os.path.join(asset_dir, "asset.usd")
        manifest_path = os.path.join(asset_dir, "manifest.json")
        if not (os.path.isdir(asset_dir) and os.path.isfile(usd_path)):
            continue
        if not os.path.isfile(manifest_path):
            print(f"  WARNING: skipping {name}: no manifest.json next to asset.usd")
            continue
        with open(manifest_path) as f:
            manifest = json.load(f)
        if manifest.get("status") != "converted":
            print(f"  WARNING: skipping {name}: manifest status is {manifest.get('status')!r}, not 'converted'")
            continue
        assets.append((name, asset_dir, usd_path))
    return assets


def sanitize_prim_name(label):
    """USD prim names must match [A-Za-z_][A-Za-z0-9_]*."""
    name = "".join(c if c.isalnum() or c == "_" else "_" for c in label)
    if not name or name[0].isdigit():
        name = f"_{name}"
    return name


def set_world_pose(xform, value):
    """xform.AddTranslateOp(opSuffix="world_pose") throws once the op is already in
    xformOpOrder (unlike some other Add*Op calls, it does not silently return the existing
    op) -- so the first call per asset uses AddTranslateOp to author it, and any later
    correction (the floor-contact lift) must instead find and re-Set() that same op."""
    for op in xform.GetOrderedXformOps():
        if op.GetOpName() == "xformOp:translate:world_pose":
            op.Set(value)
            return
    xform.AddTranslateOp(opSuffix="world_pose").Set(value)


def get_bbox(stage, prim_path):
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bound_range = bbox_cache.ComputeWorldBound(stage.GetPrimAtPath(prim_path)).ComputeAlignedRange()
    return bound_range.GetMin(), bound_range.GetMax()


def get_collision_top_z(stage, prim_path):
    """Highest world Z of the collision geometry under `prim_path` (instance proxies included), or None if it
    has none. get_bbox() measures only default-purpose (visual) geometry, which can differ from what PhysX
    collides with: the vendored table's collision box tops out ~15.5 mm below its visual top, so objects
    placed by the visual top sank onto the hidden collider."""
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide]
    )
    tops = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(prim_path), Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.IsA(UsdGeom.Boundable):
            bound = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if not bound.IsEmpty():
                tops.append(bound.GetMax()[2])
    return max(tops) if tops else None


def get_pivot_offset(prim):
    """Finds the xformOp:translate:pivot op authored by convert_asset.py's normalize_pivot()
    (composed in via the reference) and returns its value, or (0,0,0) with a warning if the
    asset predates that pivot-normalization step."""
    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        if op.GetOpName() == "xformOp:translate:pivot":
            return op.Get()
    print(f"  WARNING: no xformOp:translate:pivot found on {prim.GetPath()}, assuming (0,0,0)")
    return Gf.Vec3d(0.0, 0.0, 0.0)


def find_mesh_prims(root_prim):
    return [p for p in Usd.PrimRange(root_prim) if p.IsA(UsdGeom.Mesh)]


def trim_background_collision_near_floor(stage, background_prim_path, floor_plane_prim_path, margin):
    """Drops background_mesh faces whose world-space centroid falls inside the floor's analytic
    collision plane footprint (+margin in every direction), directly overriding
    faceVertexCounts/faceVertexIndices on the composed (referenced) mesh prim.

    Only ever subtracts faces and never touches `points` -- so unlike splitting the overlap
    region out into a second collision-only mesh prim, this stays cheap even for GenRecon's
    millions-of-triangle background meshes, at the cost of also removing those faces from the
    render (acceptable here: the excluded faces sit right in the floor's own footprint, so the
    visible floor mesh already covers that area). See --trim-background-floor-overlap's help
    for the physics problem this avoids.
    """
    floor_plane_prim = stage.GetPrimAtPath(floor_plane_prim_path)
    if not floor_plane_prim.IsValid():
        print(f"  WARNING: floor collision plane not found at {floor_plane_prim_path}, skipping background/floor overlap trim.")
        return

    plane_min, plane_max = get_bbox(stage, floor_plane_prim_path)
    xy_min = (plane_min[0] - margin, plane_min[1] - margin)
    xy_max = (plane_max[0] + margin, plane_max[1] + margin)
    plane_z = (plane_min[2] + plane_max[2]) / 2.0
    z_min, z_max = plane_z - margin, plane_z + margin

    background_prim = stage.GetPrimAtPath(background_prim_path)
    xform_cache = UsdGeom.XformCache()
    total_excluded = 0
    total_faces = 0
    for mesh_prim in find_mesh_prims(background_prim):
        mesh = UsdGeom.Mesh(mesh_prim)
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not counts or not indices:
            continue
        counts_arr = np.array(counts, dtype=np.int64)
        if np.any(counts_arr != 3):
            print(f"  WARNING: {mesh_prim.GetPath()} is not fully triangulated, skipping floor-overlap trim for it.")
            continue

        pts = np.array(points, dtype=np.float64)
        tris = np.array(indices, dtype=np.int64).reshape(-1, 3)
        centroids_local = pts[tris].mean(axis=1)

        local_to_world = xform_cache.GetLocalToWorldTransform(mesh_prim)
        m = np.array([[local_to_world[i][j] for j in range(4)] for i in range(4)])
        homog = np.concatenate([centroids_local, np.ones((len(centroids_local), 1))], axis=1)
        centroids_world = (homog @ m)[:, :3]

        exclude_mask = (
            (centroids_world[:, 0] >= xy_min[0]) & (centroids_world[:, 0] <= xy_max[0])
            & (centroids_world[:, 1] >= xy_min[1]) & (centroids_world[:, 1] <= xy_max[1])
            & (centroids_world[:, 2] >= z_min) & (centroids_world[:, 2] <= z_max)
        )
        excluded = int(exclude_mask.sum())
        total_faces += len(tris)
        if excluded == 0:
            continue

        kept_tris = tris[~exclude_mask]
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(kept_tris)))
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(kept_tris.reshape(-1).tolist()))
        total_excluded += excluded
        print(f"  {mesh_prim.GetPath()}: dropped {excluded}/{len(tris)} faces overlapping the floor footprint.")

    if total_excluded == 0:
        print(
            f"[compose] {background_prim_path}: no faces (of {total_faces}) overlapped the floor "
            f"footprint (margin={margin}); nothing trimmed."
        )


def resolve_table_usd_path(table_asset_path):
    """A local file (e.g. the vendored default under assets/table/) is used as-is. Otherwise,
    Nucleus-relative paths (leading '/') are resolved against get_assets_root_path(), same
    convention as demo_franka_pickplace.py's FRANKA_USD_PATH; anything else (relative path or
    full URL) is used as-is. Returns None if a Nucleus-relative path was given but no Nucleus
    connection could be resolved.

    Checking Path.is_file() first (rather than branching on a leading '/') matters because a local
    absolute path and a Nucleus-relative token both start with '/' on Linux."""
    if Path(table_asset_path).is_file():
        return str(Path(table_asset_path).resolve())
    if not table_asset_path.startswith("/"):
        return table_asset_path
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        return None
    return assets_root_path + table_asset_path


def add_table_asset(stage, table_asset_path, table_height, floor_world_z, floor_contact_margin, xy_min, xy_max):
    """References a table asset at /World/Table, positioned so its collision top surface sits directly
    beneath floor_world_z (mirroring the floor-contact lift objects get above the floor: here the
    table sits just below it instead), centered in XY under the union bbox of everything already
    composed. Returns the table's own bbox bottom Z (for the ground-plane safety net to sit under
    its legs), or None if the asset couldn't be loaded -- in which case no /World/Table prim is
    left behind, exactly as if --no-table had been passed."""
    usd_path = resolve_table_usd_path(table_asset_path)
    if usd_path is None:
        print(f"[compose] WARNING: could not resolve Nucleus assets root path for table asset {table_asset_path!r}; continuing without a table.")
        return None

    prim_path = "/World/Table"
    try:
        xform = UsdGeom.Xform.Define(stage, prim_path)
        # Direct Sdf-level reference (matching every other asset in compose_scene()'s loop), not
        # isaacsim.core.utils.stage.add_reference_to_stage -- that helper operates on
        # omni.usd.get_context().get_stage(), not this script's standalone Usd.Stage.CreateNew()
        # stage, so it would silently add the reference to the wrong (unrelated) stage.
        xform.GetPrim().GetReferences().AddReference(usd_path)
        bbox_min, bbox_max = get_bbox(stage, prim_path)
        if bbox_min[2] >= bbox_max[2]:
            raise RuntimeError(f"table asset at {usd_path} has an empty/invalid bounding box")
    except Exception as e:
        print(f"[compose] WARNING: table asset {usd_path!r} could not be loaded ({e}); continuing without a table.")
        stage.RemovePrim(prim_path)
        return None

    # The collision top, not the visual top, is the surface objects actually rest on.
    table_top_z = get_collision_top_z(stage, prim_path)
    if table_top_z is None:
        print("[compose] WARNING: table has no collision geometry; placing it by its visual top instead.")
        table_top_z = bbox_max[2] if bbox_max[2] > bbox_min[2] else table_height
    target_top_z = (floor_world_z if floor_world_z is not None else 0.0) - floor_contact_margin
    lift_z = target_top_z - table_top_z

    center_x = (xy_min[0] + xy_max[0]) / 2.0 if xy_min is not None else 0.0
    center_y = (xy_min[1] + xy_max[1]) / 2.0 if xy_min is not None else 0.0
    table_center_x = (bbox_min[0] + bbox_max[0]) / 2.0
    table_center_y = (bbox_min[1] + bbox_max[1]) / 2.0

    world_pose = Gf.Vec3d(center_x - table_center_x, center_y - table_center_y, lift_z)
    set_world_pose(xform, world_pose)

    bbox_min, bbox_max = get_bbox(stage, prim_path)
    print(
        f"[compose] table ({usd_path}): world_pose={tuple(world_pose)}, visual top={bbox_max[2]:.4f}, "
        f"collision top={get_collision_top_z(stage, prim_path) or float('nan'):.4f} (target {target_top_z:.4f})"
    )
    return float(bbox_min[2])


_ROOM_SHELL_REL = "Assets/Isaac/5.1/Isaac/Environments/Simple_Room/simple_room.usd"
_ROOM_FRIDGE_REL = "Assets/ArchVis/Residential/Appliances/Refrigerators/fridge.usd"
_ROOM_STOVE_REL = "Assets/ArchVis/Residential/Appliances/Oven/stove.usd"
# Simple_Room's interior, in its own local frame: floor top Z, the kitchen wall's inner face Y, and
# the Y (behind that wall's counters) the table is anchored at so it stands ~2m from the kitchen wall.
_ROOM_FLOOR_Z = -0.77
_ROOM_KITCHEN_WALL_Y = -3.3
_ROOM_TABLE_ANCHOR_Y = -1.3
_ROOM_DEACTIVATE_NAMES = {"table_low_327", "GroundPlane", "DomeLight"}


def make_room_static(room_prim):
    """Strips every physics collision/rigid-body schema under room_prim, so the room is purely
    aesthetic. Returns the number of prims that still report a collision/rigid-body API afterwards
    (expected 0)."""
    for prim in Usd.PrimRange(room_prim):
        for schema in list(prim.GetAppliedSchemas()):
            if "Collision" in schema or "RigidBody" in schema:
                prim.RemoveAppliedSchema(schema)
        if prim.HasAttribute("physics:collisionEnabled"):
            prim.GetAttribute("physics:collisionEnabled").Set(False)
    return sum(
        1
        for prim in Usd.PrimRange(room_prim)
        if any("Collision" in s or "RigidBody" in s for s in prim.GetAppliedSchemas())
    )


def add_procedural_box(stage, path, center, size, color):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*center))
    cube.AddScaleOp().Set(Gf.Vec3d(*size))
    cube.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])


def add_kitchen_room(stage, room_dir, table_center_xy, floor_z):
    """Adds a static, collision-free kitchen backdrop at /World/Room: NVIDIA's Simple_Room shell
    (floor/walls, with its own table/lights/ceiling deactivated so demo_franka_pickplace.py's lights
    illuminate the scene) dressed with a fridge, stove and procedural counters/cabinets along one
    wall. The room is rotated 180deg about Z so that wall ends up on the +Y side of the table --
    i.e. behind it as seen from the default camera on the -Y side. Returns True on success."""
    room_dir = Path(room_dir)
    shell_usd = room_dir / _ROOM_SHELL_REL
    if not shell_usd.is_file():
        print(f"[compose] WARNING: room assets not found at {shell_usd} (run assets/room/fetch.sh); continuing without a room.")
        return False

    room_path = "/World/Room"
    try:
        room = UsdGeom.Xform.Define(stage, room_path)
        # Rotation about Z takes the local anchor (0, ANCHOR_Y) to (0, -ANCHOR_Y).
        translate = Gf.Vec3d(table_center_xy[0], table_center_xy[1] + _ROOM_TABLE_ANCHOR_Y, floor_z - _ROOM_FLOOR_Z)
        room.AddTranslateOp().Set(translate)
        room.AddRotateZOp().Set(180.0)

        shell = UsdGeom.Xform.Define(stage, f"{room_path}/Shell")
        shell.GetPrim().GetReferences().AddReference(str(shell_usd))

        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        for child in shell.GetPrim().GetChildren():
            ceiling = False
            if child.IsA(UsdGeom.Imageable):
                rng = bbox_cache.ComputeWorldBound(child).ComputeAlignedRange()
                ceiling = (not rng.IsEmpty()) and (rng.GetMin()[2] - translate[2]) > 3.0
            if child.GetName() in _ROOM_DEACTIVATE_NAMES or ceiling:
                child.SetActive(False)

        wall_y = _ROOM_KITCHEN_WALL_Y
        for name, rel, x, depth in (("Fridge", _ROOM_FRIDGE_REL, -3.2, 0.7), ("Stove", _ROOM_STOVE_REL, 1.5, 0.69)):
            usd = room_dir / rel
            if not usd.is_file():
                print(f"[compose] WARNING: {usd} missing; skipping {name}.")
                continue
            xf = UsdGeom.Xform.Define(stage, f"{room_path}/{name}")
            xf.AddTranslateOp().Set(Gf.Vec3d(x, wall_y + depth / 2.0 + 0.02, _ROOM_FLOOR_Z))
            xf.AddScaleOp().Set(Gf.Vec3d(0.01, 0.01, 0.01))  # ArchVis assets are in centimeters
            xf.GetPrim().GetReferences().AddReference(str(usd))

        counter_z = _ROOM_FLOOR_Z
        cabinet_color, top_color = (0.85, 0.83, 0.78), (0.25, 0.25, 0.27)
        for i, (x0, x1) in enumerate(((-2.2, 0.88), (2.12, 4.3))):
            cx, sx = (x0 + x1) / 2.0, x1 - x0
            cy = wall_y + 0.3 + 0.01
            add_procedural_box(stage, f"{room_path}/Counter{i}", (cx, cy, counter_z + 0.43), (sx, 0.6, 0.86), cabinet_color)
            add_procedural_box(stage, f"{room_path}/CounterTop{i}", (cx, cy, counter_z + 0.88), (sx, 0.64, 0.04), top_color)
            add_procedural_box(stage, f"{room_path}/Upper{i}", (cx, wall_y + 0.18, counter_z + 2.0), (sx, 0.35, 0.7), cabinet_color)

        remaining = make_room_static(stage.GetPrimAtPath(room_path))
    except Exception as e:
        print(f"[compose] WARNING: could not build the kitchen room ({e}); continuing without a room.")
        stage.RemovePrim(room_path)
        return False

    print(f"[compose] room at {tuple(translate)}: collision/rigid-body APIs remaining under {room_path}: {remaining}")
    return True


def compose_scene(
    assets,
    output_path,
    background_label,
    floor_label,
    floor_contact_margin,
    add_ground_plane,
    trim_background_floor_overlap,
    floor_overlap_margin,
    add_table,
    table_asset_path,
    table_height,
    add_room=False,
    room_asset_dir=None,
):
    stage = Usd.Stage.CreateNew(output_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    # Floor first: every other asset's below-the-floor check needs floor_world_z already known.
    assets = sorted(assets, key=lambda asset: asset[0] != floor_label)

    min_z = None
    xy_min = None
    xy_max = None
    floor_world_z = None
    floor_plane_prim_path = None
    for label, asset_dir, usd_path in assets:
        prim_path = f"/World/{sanitize_prim_name(label)}"
        xform = UsdGeom.Xform.Define(stage, prim_path)
        xform.GetPrim().GetReferences().AddReference(os.path.abspath(usd_path))
        # Reading ops here sees the composed value, including the referenced pivot op.
        pivot = get_pivot_offset(xform.GetPrim())
        world_pose = Gf.Vec3d(-pivot[0], -pivot[1], -pivot[2])
        set_world_pose(xform, world_pose)

        is_background = label == background_label
        is_floor = label == floor_label

        # get_bbox() is ComputeWorldBound(), so this already reflects the world_pose translate.
        bbox_min, bbox_max = get_bbox(stage, prim_path)

        if is_floor:
            floor_world_z = (bbox_min[2] + bbox_max[2]) / 2.0
            floor_plane_prim_path = f"{prim_path}/FloorCollisionPlane"
        elif is_background:
            if trim_background_floor_overlap and floor_plane_prim_path is not None:
                trim_background_collision_near_floor(stage, prim_path, floor_plane_prim_path, floor_overlap_margin)
        elif floor_world_z is not None and bbox_min[2] < floor_world_z + floor_contact_margin:
            lift = floor_world_z + floor_contact_margin - bbox_min[2]
            old_bottom = bbox_min[2]
            world_pose = Gf.Vec3d(world_pose[0], world_pose[1], world_pose[2] + lift)
            set_world_pose(xform, world_pose)
            bbox_min, bbox_max = get_bbox(stage, prim_path)
            print(f"[compose] {label}: lifted {lift:.4f}m to clear floor (bottom was {old_bottom:.4f}, floor {floor_world_z:.4f})")

        min_z = bbox_min[2] if min_z is None else min(min_z, bbox_min[2])
        if not is_floor and not is_background:
            xy_min = (bbox_min[0], bbox_min[1]) if xy_min is None else (min(xy_min[0], bbox_min[0]), min(xy_min[1], bbox_min[1]))
            xy_max = (bbox_max[0], bbox_max[1]) if xy_max is None else (max(xy_max[0], bbox_max[0]), max(xy_max[1], bbox_max[1]))

        if is_floor:
            tag = " (floor, analytic-plane collision)"
        elif is_background:
            tag = " (background, static)"
        else:
            tag = ""
        print(f"[compose] {label}{tag}: pivot={tuple(pivot)} -> world_pose={tuple(world_pose)}")

    if add_table:
        table_bottom_z = add_table_asset(stage, table_asset_path, table_height, floor_world_z, floor_contact_margin, xy_min, xy_max)
        if table_bottom_z is not None:
            min_z = table_bottom_z if min_z is None else min(min_z, table_bottom_z)
            if add_room:
                table_min, table_max = get_bbox(stage, "/World/Table")
                table_center_xy = ((table_min[0] + table_max[0]) / 2.0, (table_min[1] + table_max[1]) / 2.0)
                add_kitchen_room(stage, room_asset_dir, table_center_xy, table_bottom_z)
    if add_room and not add_table:
        print("[compose] WARNING: --room needs --table (the room is anchored to the table); skipping the room.")

    if add_ground_plane:
        floor_z = (min_z if min_z is not None else 0.0) - 0.01
        physicsUtils.add_ground_plane(
            stage, "/World/GroundPlane", "Z", 100.0, Gf.Vec3f(0.0, 0.0, floor_z), Gf.Vec3f(0.5, 0.5, 0.5)
        )
        print(f"[compose] ground plane at Z={floor_z:.4f} (scene min Z {min_z})")

    stage.GetRootLayer().Save()


def main():
    assets = find_converted_assets(args.input)
    if not assets:
        print(f"No converted assets found under {args.input}", file=sys.stderr)
        simulation_app.close()
        sys.exit(1)

    compose_scene(
        assets,
        args.output,
        args.background_label,
        args.floor_label,
        args.floor_contact_margin,
        args.ground_plane,
        args.trim_background_floor_overlap,
        args.floor_overlap_margin,
        args.add_table,
        args.table_asset_path,
        args.table_height,
        args.add_room,
        args.room_asset_dir,
    )
    print(f"\nWrote composed scene ({len(assets)} assets) to {args.output}")
    simulation_app.close()


if __name__ == "__main__":
    main()
