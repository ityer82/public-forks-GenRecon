"""Detect inter-object penetration in a composed Isaac Sim scene via PhysX contact reports.

Standalone side test: reads an existing scene.usda (e.g. runs/<run>/pick_place/glb/scene.usda,
written by compose_isaac_scene.py with convert_asset.py's rigid-body/collision assets), zeroes
gravity, steps the sim a few frames, and aggregates PhysX contact points per object pair. A
contact point with negative `separation` is a penetration of that depth. For each penetrating
pair the smaller object is lifted along +z (binary search, whole mm) to find the smallest lift that
clears the pair, then all lifts are applied together as a joint check.

By default nothing is modified: results go to stdout and an optional JSON. With --fix, the lifts
that worked are written back into the scene file (a backup `<scene>.pre_penetration_fix.<ext>` is
kept), except for lifts that would make another pair worse, which are dropped.

Run from the IsaacSim project (it owns the isaacsim environment):
    cd /home/ss/Work/GitHub/IsaacSim && uv run \
        /home/ss/Work/GitHub/public-forks-GenRecon/scripts/isaac_detect_penetration.py \
        --scene /home/ss/Work/GitHub/public-forks-GenRecon/runs/kitchen_plates/pick_place/glb/scene.usda \
        --out_json /home/ss/Work/GitHub/public-forks-GenRecon/runs/kitchen_plates/penetration.json [--fix]

Caveat: colliders are the convexDecomposition/convexHull approximations, not the exact meshes,
so shallow concavities (plate rims) can give small false positives/negatives, and lifts can be
larger than the true overlap.
"""

import argparse
import json
import shutil
import sys
import traceback
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=Path, required=True, help="Composed scene .usda/.usd to test.")
    parser.add_argument("--out_json", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=3, help="Physics frames to run; frame 0 is the least perturbed.")
    parser.add_argument(
        "--min_depth_mm", type=float, default=0.5, help="Penetration below this is treated as touching, not flagged."
    )
    parser.add_argument("--max_lift_mm", type=int, default=30, help="Upper bound of the +z lift binary search (whole mm).")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Write the +z lifts that cleared their pair back into --scene (backup kept). A lift that would "
        "make any other pair worse is dropped. Pairs that no lift clears are reported, never moved.",
    )
    parser.add_argument(
        "--max_fix_lift_mm",
        type=int,
        default=20,
        help="With --fix, never write a lift larger than this (convex colliders can inflate the needed lift).",
    )
    parser.add_argument("--gui", action="store_true")
    return parser.parse_args()


sys.stdout.reconfigure(line_buffering=True)  # simulation_app.close() can exit without flushing
args = parse_args()

# SimulationApp must be constructed before any other isaacsim/omni import.
from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": not args.gui})

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from omni.physx import get_physx_simulation_interface  # noqa: E402
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402


def object_of(path: str, object_paths: list[str]) -> str | None:
    for obj in object_paths:
        if path == obj or path.startswith(obj + "/"):
            return obj
    return None


def run_pass(object_paths, steps):
    """Plays the timeline `steps` frames and returns {pair: {frame: [contact dicts]}}."""
    frame = {"i": 0}
    contacts = defaultdict(lambda: defaultdict(list))

    def on_contact(headers, data):
        for h in headers:
            a = object_of(str(PhysicsSchemaTools.intToSdfPath(h.actor0)), object_paths)
            b = object_of(str(PhysicsSchemaTools.intToSdfPath(h.actor1)), object_paths)
            if a is None or b is None or a == b:
                continue
            for i in range(h.contact_data_offset, h.contact_data_offset + h.num_contact_data):
                cp = data[i]
                contacts[tuple(sorted((a, b)))][frame["i"]].append(
                    {
                        "sep": float(cp.separation),
                        "pos": np.array([cp.position[0], cp.position[1], cp.position[2]]),
                    }
                )

    sub = get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for i in range(steps):
        frame["i"] = i
        simulation_app.update()
    timeline.stop()  # restores authored poses
    del sub
    return contacts


def first_frame_points(per_frame):
    return per_frame[min(per_frame)]


def depth_mm(points):
    return max(0.0, -min(p["sep"] for p in points) * 1000.0)


def pair_key(pair):
    return " vs ".join(p.split("/")[-1] for p in pair)


def outer_translate_op(stage, obj_path):
    """The object's outermost translate op, i.e. a pure world-frame move. The
    `xformOp:translate:world_pose` op sits innermost in xformOpOrder, so it is scaled/rotated by the
    asset's orient+scale ops and is not a world-frame translation."""
    return next(
        o for o in UsdGeom.Xformable(stage.GetPrimAtPath(obj_path)).GetOrderedXformOps()
        if o.GetOpName() == "xformOp:translate"
    )


def set_lifts(stage, orig, lifts_mm):
    """Sets every object's outer translate to its original value plus a +z lift (mm, default 0).
    Absolute, so repeated calls never accumulate float error."""
    for obj, base in orig.items():
        outer_translate_op(stage, obj).Set(base + Gf.Vec3d(0.0, 0.0, lifts_mm.get(obj, 0.0) / 1000.0))


def write_back(scene_path: Path, applied_mm: dict[str, float]) -> Path:
    """Adds each lift to the object's outer translate in the scene file itself. Run on a fresh stage
    (the search stage's temporary edits live only in its session layer and are never saved)."""
    backup = scene_path.with_name(f"{scene_path.stem}.pre_penetration_fix{scene_path.suffix}")
    if not backup.exists():
        shutil.copy2(scene_path, backup)
    stage = Usd.Stage.Open(str(scene_path))
    for obj, dz in applied_mm.items():
        op = outer_translate_op(stage, obj)
        op.Set(op.Get() + Gf.Vec3d(0.0, 0.0, dz / 1000.0))
    stage.GetRootLayer().Save()
    return backup


def main():
    ctx = omni.usd.get_context()
    ctx.open_stage(str(args.scene.resolve()))
    stage = ctx.get_stage()
    # Every temporary edit below (gravity, report APIs, trial lifts) goes to the session layer so the
    # scene file is only ever changed by write_back().
    stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))

    # Objects = direct children of the default prim (e.g. /World/silver_knife_1).
    root = stage.GetDefaultPrim() or stage.GetPrimAtPath("/World")
    object_paths = [str(c.GetPath()) for c in root.GetChildren() if c.IsA(UsdGeom.Xformable)]
    orig = {o: outer_translate_op(stage, o).Get() for o in object_paths if any(
        op.GetOpName() == "xformOp:translate" for op in UsdGeom.Xformable(stage.GetPrimAtPath(o)).GetOrderedXformOps()
    )}

    # Gravity off so nothing falls/settles during the test; keep the existing PhysicsScene if any.
    scene_prim = next((p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)), None)
    scene = UsdPhysics.Scene(scene_prim) if scene_prim else UsdPhysics.Scene.Define(stage, "/physicsScene")
    scene.CreateGravityMagnitudeAttr(0.0)

    # Contact reports on every rigid body, threshold 0 so every contact point is delivered.
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
            # Keep bodies where they were authored: scene gravity alone did not stop them falling/being
            # kicked apart (a knife moved ~15 mm in the first frame), which corrupted the lift tests.
            body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            body.CreateDisableGravityAttr(True)
            body.CreateMaxDepenetrationVelocityAttr(1e-4)

    # Which object of a pair is movable: only dynamic bodies (static background/floor/table are never moved);
    # if both are dynamic, the one with the smaller bbox volume.
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    volume = {}
    dynamic = set()
    for o in object_paths:
        size = cache.ComputeWorldBound(stage.GetPrimAtPath(o)).ComputeAlignedRange().GetSize()
        volume[o] = size[0] * size[1] * size[2]
        if any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in Usd.PrimRange(stage.GetPrimAtPath(o))):
            dynamic.add(o)

    def pick_movable(pair):
        candidates = [o for o in pair if o in dynamic]
        return min(candidates or pair, key=lambda o: volume[o])

    # The first play after loading reports contacts that were already partly resolved (a 13 mm floor
    # overlap showed up as 0), and stopping does not restore authored poses. So discard one warm-up pass and
    # reset the poses before every measurement.
    run_pass(object_paths, 1)
    set_lifts(stage, orig, {})
    contacts = run_pass(object_paths, args.steps)
    set_lifts(stage, orig, {})

    results = []
    for pair, per_frame in contacts.items():
        pts = first_frame_points(per_frame)
        d = depth_mm(pts)
        if d < args.min_depth_mm:
            continue
        movable = pick_movable(pair)
        obstacle = pair[0] if movable == pair[1] else pair[1]
        results.append(
            {
                "pair": list(pair),
                "movable": movable,
                "obstacle": obstacle,
                "max_penetration_mm": d,
                "num_penetrating_points": sum(1 for p in pts if -p["sep"] * 1000.0 >= args.min_depth_mm),
            }
        )
    results.sort(key=lambda r: -r["max_penetration_mm"])

    def penetrating_depths(lifts_mm):
        """{pair: depth_mm} of pairs penetrating with the given {object: +z lift mm} applied."""
        set_lifts(stage, orig, lifts_mm)
        after = run_pass(object_paths, 1)
        set_lifts(stage, orig, {})
        depths = {pair: depth_mm(first_frame_points(f)) for pair, f in after.items()}
        return {pair: d for pair, d in depths.items() if d >= args.min_depth_mm}

    # Up-lift search: smallest +z lift (whole mm) at which this pair no longer penetrates, by binary search
    # (assumes clearance is monotonic in lift; if even --max_lift_mm does not clear, lift_mm stays None).
    def clears(r, dz):
        return tuple(r["pair"]) not in penetrating_depths({r["movable"]: dz})

    for r in results:
        r["lift_mm"] = None
        if not clears(r, args.max_lift_mm):
            continue
        lo, hi = 0, args.max_lift_mm  # invariant: hi clears; answer in (lo, hi] unless 0 clears
        if clears(r, 0):
            r["lift_mm"] = 0
            continue
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if clears(r, mid):
                hi = mid
            else:
                lo = mid
        r["lift_mm"] = hi

    # Joint check: lift every movable object by the max of its per-pair lifts, all at once, and re-run.
    lifts = defaultdict(float)
    for r in results:
        if r["lift_mm"] is not None:
            lifts[r["movable"]] = max(lifts[r["movable"]], r["lift_mm"])
    joint_remaining = {pair_key(pair): round(d, 2) for pair, d in penetrating_depths(dict(lifts)).items()}

    print(f"\n{'pair (movable / fixed)':<44}{'depth':>7}  required +z lift")
    for r in results:
        name = f"{r['movable'].split('/')[-1]} / {r['obstacle'].split('/')[-1]}"
        lift = f"{r['lift_mm']} mm" if r["lift_mm"] is not None else f"not cleared within {args.max_lift_mm} mm"
        print(f"{name:<44}{r['max_penetration_mm']:>7.2f}  {lift}")
    print(f"joint lift {({k.split('/')[-1]: v for k, v in lifts.items()})} mm -> remaining overlaps: {joint_remaining or 'none'}")
    if not results:
        print("no penetrating pairs found")

    payload = {"scene": str(args.scene), "pairs": results, "joint_lift_mm": dict(lifts), "joint_remaining_mm": joint_remaining}

    if args.fix:
        payload.update(fix_lifts(results, penetrating_depths))

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.out_json}")

    if args.fix and payload["applied_lifts_mm"]:
        ctx.close_stage()  # release the search stage before editing the file
        backup = write_back(args.scene, payload["applied_lifts_mm"])
        print(f"fix: wrote {args.scene} (backup: {backup})")


def fix_lifts(results, penetrating_depths):
    """Chooses which lifts to write: only lifts that cleared their pair and are <= --max_fix_lift_mm, minus any
    object whose lift makes another pair worse than before (e.g. a case lifted off a printer pushed deeper into
    headphones). Mutates `results` with `fixed` / `skipped_reason`; returns the JSON fields to add."""
    baseline = {tuple(r["pair"]): r["max_penetration_mm"] for r in results}

    candidates: dict[str, float] = {}
    for r in results:
        r["fixed"] = False
        if r["lift_mm"] is None:
            r["skipped_reason"] = f"no +z lift up to {args.max_lift_mm} mm clears it"
        elif r["lift_mm"] > args.max_fix_lift_mm:
            r["skipped_reason"] = f"needed lift {r['lift_mm']} mm exceeds --max_fix_lift_mm {args.max_fix_lift_mm}"
        else:
            candidates[r["movable"]] = max(candidates.get(r["movable"], 0), r["lift_mm"])

    dropped: dict[str, str] = {}
    after = dict(baseline)
    while candidates:
        after = penetrating_depths(candidates)
        worse = [p for p, d in after.items() if d > baseline.get(p, 0.0) + args.min_depth_mm]
        drop = {o for p in worse for o in p if o in candidates}
        if not drop:
            break
        for o in drop:
            reason = ", ".join(pair_key(p) for p in worse if o in p)
            dropped[o] = f"lift {candidates.pop(o):g} mm would worsen {reason}"
        after = dict(baseline)  # re-evaluated on the next iteration (or stays baseline if nothing is left)

    for r in results:
        if r["movable"] in dropped:
            r["skipped_reason"] = dropped[r["movable"]]
        elif r["movable"] in candidates:
            r["fixed"] = tuple(r["pair"]) not in after
            if not r["fixed"]:
                r["skipped_reason"] = "still penetrating after the lift"

    remaining = {pair_key(p): round(d, 2) for p, d in after.items()}
    print("\nfix: applied lifts (mm):", {k.split("/")[-1]: v for k, v in candidates.items()} or "none")
    for r in results:
        if not r["fixed"]:
            print(f"WARNING: not fixed: {pair_key(r['pair'])} ({r['max_penetration_mm']:.2f} mm): {r.get('skipped_reason')}")
    print("fix: remaining overlaps after fix:", remaining or "none")
    return {"applied_lifts_mm": dict(candidates), "remaining_after_fix_mm": remaining}


try:
    main()
except Exception:
    traceback.print_exc()
    sys.stderr.flush()
finally:
    simulation_app.close()
