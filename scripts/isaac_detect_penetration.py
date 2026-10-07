"""Detect inter-object penetration in a composed Isaac Sim scene via PhysX contact reports.

Standalone side test: reads an existing scene.usda (e.g. runs/<run>/pick_place/glb/scene.usda,
written by compose_isaac_scene.py with convert_asset.py's rigid-body/collision assets), zeroes
gravity, steps the sim a few frames, and aggregates PhysX contact points per object pair. A
contact point with negative `separation` is a penetration of that depth. For each penetrating
pair the smaller object is lifted along +z (binary search, whole mm) to find the smallest lift that
clears it. Movers are processed bottom-up: an object resting on another mover is searched with that mover
already at its final lift. All lifts are then applied together as a joint check.

By default nothing is modified: results go to stdout and an optional JSON. With --fix, the lifts
that worked are written back into the scene file (a backup `<scene>.pre_penetration_fix.<ext>` is
kept), except for lifts that would make another pair worse, which are dropped.

With --settle (default) the +z lift is only a transient "clear the overlap" step, and EVERY dynamic object (not just
the lifted ones) is then dropped under gravity, everything static frozen, until it comes to rest. The settled poses
are written back as `xformOp:translate:settle` / `xformOp:orient:settle` ops ahead of each object's existing ops.
This also lands objects that start floating above their support (which no overlap test can see), so nothing falls
when the simulation starts. There are no limits on how far or how much a settle moves/rotates an object.

Run from the isaacsim/ project (it owns the isaacsim environment):
    cd /home/ss/Work/GitHub/public-forks-GenRecon/isaacsim && uv run \
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
    parser.add_argument(
        "--settle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="With --fix: after lifting, let every dynamic object settle under gravity and write back the settled poses "
        "(lifts up to --max_lift_mm are then allowed, since they are transient). --no-settle writes lift-only results.",
    )
    parser.add_argument("--settle_steps", type=int, default=300, help="Max physics frames of the settle pass.")
    parser.add_argument(
        "--settle_max_speed", type=float, default=0.1,
        help="Linear speed cap (m/s) of the settling bodies, to keep each physics substep well under a thin shell's thickness.",
    )
    parser.add_argument("--trace_settle", action="store_true", help="Print per-frame pose / contact depth of the settle pass.")
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


def run_pass(object_paths, steps, on_frame=None):
    """Plays the timeline `steps` frames and returns {pair: {frame: [contact dicts]}}. `on_frame(i)` is called
    after each frame (while the simulated poses are still in place) and may return True to stop early."""
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
        if on_frame is not None and on_frame(i):
            break
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


class TempEdits:
    """Session-layer attribute edits that can be undone (the search stage is reused after the settle pass)."""

    def __init__(self):
        self.saved = []

    def set(self, attr, value):
        self.saved.append((attr, attr.Get()))
        attr.Set(value)

    def restore(self):
        for attr, old in reversed(self.saved):
            attr.Set(old) if old is not None else attr.Clear()
        self.saved.clear()


def world_matrix(stage, obj_path):
    """Current composed local-to-world matrix (fresh cache: poses change every frame while playing)."""
    return UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(obj_path))


def run_settle(stage, scene, orig, object_paths, lifts_mm, startup_world):
    """Applies `lifts_mm` ({object: +z lift mm}, 0 for the rest), then lets exactly the objects in it fall and come to rest under
    gravity while every other rigid body is frozen (kinematic). Returns ({object: info}, {pair: final depth mm}).
    info: lift_mm, displacement_mm (object center), rotation_deg, steps, converged, delta (4x4 rows, world-frame rigid delta authored pose -> settled pose,
    row-vector convention: p_settled = p_authored * delta)."""
    settle_objs = list(lifts_mm)
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    authored = {o: startup_world[o] for o in settle_objs}  # the file's poses, before any play touched the stage
    center = {o: bbox_cache.ComputeWorldBound(stage.GetPrimAtPath(o)).ComputeCentroid() for o in settle_objs}

    edits = TempEdits()
    for o in object_paths:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(o)):
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            if o in settle_objs:
                body = PhysxSchema.PhysxRigidBodyAPI(prim)
                edits.set(body.CreateDisableGravityAttr(), False)
                edits.set(body.CreateMaxDepenetrationVelocityAttr(), 0.5)
                edits.set(body.CreateLinearDampingAttr(), 1.0)
                edits.set(body.CreateAngularDampingAttr(), 1.0)
                # No tunneling through thin shells (plate floors are a few mm thick; a free fall reaches ~10 mm
                # per 1/60 s substep): CCD plus a velocity cap well under the shell thickness per substep.
                edits.set(body.CreateEnableCCDAttr(), True)
                edits.set(body.CreateMaxLinearVelocityAttr(), args.settle_max_speed)
                edits.set(body.CreateMaxAngularVelocityAttr(), 90.0)  # deg/s
            else:
                edits.set(UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(), True)
    edits.set(scene.CreateGravityMagnitudeAttr(), 9.81)
    edits.set(PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim()).CreateEnableCCDAttr(), True)

    state = {"prev": None, "stable": 0, "steps": 0, "mats": {}, "trace": []}

    def on_frame(i):
        mats = {o: world_matrix(stage, o) for o in settle_objs}
        state["steps"] = i + 1
        state["mats"] = mats
        if args.trace_settle:
            state["trace"].append({o: (m.ExtractTranslation(), (authored[o].GetInverse() * m).ExtractRotation().GetAngle()) for o, m in mats.items()})
        prev = state["prev"]
        state["prev"] = mats
        if prev is None:
            return False
        moved = False
        for o in settle_objs:
            d_pos = (mats[o].ExtractTranslation() - prev[o].ExtractTranslation()).GetLength()
            d_rot = (prev[o].GetInverse() * mats[o]).ExtractRotation().GetAngle()  # degrees
            moved = moved or d_pos > 1e-5 or d_rot > 0.01
        state["stable"] = 0 if moved else state["stable"] + 1
        return state["stable"] >= 10

    try:
        set_lifts(stage, orig, lifts_mm)
        if args.trace_settle:
            for o in settle_objs:
                pre = world_matrix(stage, o)
                print(f"trace pre-play {o.split('/')[-1]}: rot vs startup {(startup_world[o].GetInverse() * pre).ExtractRotation().GetAngle():.2f} deg, "
                      f"rot authored(at settle start) vs startup {(startup_world[o].GetInverse() * authored[o]).ExtractRotation().GetAngle():.2f} deg, "
                      f"dz vs startup {(pre.ExtractTranslation() - startup_world[o].ExtractTranslation())[2]*1000:.2f} mm")
        contacts = run_pass(object_paths, args.settle_steps, on_frame)
    finally:
        edits.restore()
        set_lifts(stage, orig, {})

    if args.trace_settle:
        for i, poses in enumerate(state["trace"]):
            if i % 3 and i != len(state["trace"]) - 1:
                continue
            deps = {pair_key(p): round(depth_mm(f[i]), 2) for p, f in contacts.items() if i in f}
            print(f"trace {i:3d}: " + "; ".join(f"{o.split('/')[-1]} z={t[2]*1000:.2f}mm rot={a:.1f}deg" for o, (t, a) in poses.items()) + f" | depth {deps}")
    converged = state["stable"] >= 10
    info = {}
    for o in settle_objs:
        delta = authored[o].GetInverse() * state["mats"][o]
        moved_center = delta.Transform(center[o]) - center[o]
        displacement_mm = moved_center.GetLength() * 1000.0
        rotation_deg = delta.ExtractRotation().GetAngle()
        info[o] = {
            "lift_mm": lifts_mm[o],
            "displacement_mm": round(displacement_mm, 2),
            "rotation_deg": round(rotation_deg, 2),
            "steps": state["steps"],
            "converged": converged,
            "delta": [list(row) for row in delta],
        }
    final = {pair: depth_mm(per_frame[max(per_frame)]) for pair, per_frame in contacts.items()}
    return info, {pair: d for pair, d in final.items() if d >= args.min_depth_mm}


def apply_settle_ops(prim, delta):
    """Composes the world-frame rigid `delta` outside everything the object already has, via two new outermost
    ops `xformOp:translate:settle` + `xformOp:orient:settle` (p' = T * O * existing chain). An existing pair of
    settle ops (earlier fix) is updated in place. The asset's own ops and `outer_translate_op` are untouched."""
    xf = UsdGeom.Xformable(prim)
    ops = {o.GetOpName(): o for o in xf.GetOrderedXformOps()}
    t_name, o_name = "xformOp:translate:settle", "xformOp:orient:settle"
    if t_name in ops and o_name in ops:
        old = Gf.Matrix4d(1.0)
        old.SetRotate(Gf.Quatd(ops[o_name].Get()))
        old.SetTranslateOnly(Gf.Vec3d(ops[t_name].Get()))
        delta = old * delta
        t_op, o_op = ops[t_name], ops[o_name]
    else:
        t_op = xf.AddTranslateOp(opSuffix="settle")
        o_op = xf.AddOrientOp(opSuffix="settle", precision=UsdGeom.XformOp.PrecisionDouble)
        rest = [o for o in xf.GetOrderedXformOps() if o.GetOpName() not in (t_name, o_name)]
        xf.SetXformOpOrder([t_op, o_op] + rest)
    t_op.Set(Gf.Vec3d(delta.ExtractTranslation()))
    o_op.Set(Gf.Quatd(delta.ExtractRotationQuat()))


def write_back(scene_path: Path, applied_mm: dict[str, float], settled: dict | None = None) -> Path:
    """Adds each lift-only object's lift to its outer translate in the scene file itself, and applies each settled
    object's pose delta (which already contains its lift) as settle ops. Run on a fresh stage (the search stage's
    temporary edits live only in its session layer and are never saved)."""
    backup = scene_path.with_name(f"{scene_path.stem}.pre_penetration_fix{scene_path.suffix}")
    if not backup.exists():
        shutil.copy2(scene_path, backup)
    stage = Usd.Stage.Open(str(scene_path))
    for obj, dz in applied_mm.items():
        op = outer_translate_op(stage, obj)
        op.Set(op.Get() + Gf.Vec3d(0.0, 0.0, dz / 1000.0))
    for obj, info in (settled or {}).items():
        apply_settle_ops(stage.GetPrimAtPath(obj), Gf.Matrix4d(*[v for row in info["delta"] for v in row]))
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

    startup_world = {o: world_matrix(stage, o) for o in object_paths}

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

    # Bottom-up up-lift search: each mover's smallest +z lift (whole mm) at which it no longer penetrates anything
    # that is static or already decided, by binary search (assumes clearance is monotonic in lift; if even
    # --max_lift_mm does not clear, the mover gets no lift). Obstacles that are themselves movers (a phone on a
    # book on the table) are decided first, so the lift of the object resting on them is searched against their
    # final lifted pose instead of their authored one.
    movers = {r["movable"] for r in results}
    zmin = {o: cache.ComputeWorldBound(stage.GetPrimAtPath(o)).ComputeAlignedRange().GetMin()[2] for o in movers}
    below = {m: {r["obstacle"] for r in results if r["movable"] == m and r["obstacle"] in movers} for m in movers}
    order, pending = [], set(movers)
    while pending:
        ready = [m for m in pending if not (below[m] & pending)] or list(pending)  # cycle: lowest first
        m = min(ready, key=lambda o: (zmin[o], volume[o]))
        order.append(m)
        pending.discard(m)

    decided: dict[str, float] = {}
    for m in order:
        waiting = {o for o in movers if o not in decided and o != m}
        pairs = [tuple(r["pair"]) for r in results if r["movable"] == m and r["obstacle"] not in waiting]

        def clears(pair, dz):
            return pair not in penetrating_depths({**decided, m: dz})

        # Per pair, so a pair no lift clears (case under headphones) does not veto the lift that clears the others.
        needed = [0]
        for pair in pairs:
            if clears(pair, 0):
                continue
            if not clears(pair, args.max_lift_mm):
                continue
            lo, hi = 0, args.max_lift_mm  # invariant: hi clears, lo does not
            while hi - lo > 1:
                mid = (lo + hi) // 2
                lo, hi = (lo, mid) if clears(pair, mid) else (mid, hi)
            needed.append(hi)
        decided[m] = float(max(needed))

    final = penetrating_depths(decided)
    for r in results:
        r["lift_mm"] = int(decided[r["movable"]]) if tuple(r["pair"]) not in final else None

    lifts = dict(decided)
    joint_remaining = {pair_key(pair): round(d, 2) for pair, d in final.items()}

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
        settle_fn = (lambda lifts: run_settle(stage, scene, orig, object_paths, {**{o: 0.0 for o in sorted(dynamic)}, **lifts}, startup_world)) if args.settle else None
        payload.update(fix_lifts(results, penetrating_depths, settle_fn))

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.out_json}")

    if args.fix and (payload["applied_lifts_mm"] or payload["settled"]):
        ctx.close_stage()  # release the search stage before editing the file
        backup = write_back(args.scene, payload["applied_lifts_mm"], payload["settled"])
        print(f"fix: wrote {args.scene} (backup: {backup})")


def fix_lifts(results, penetrating_depths, settle_fn=None):
    """Chooses which lifts to write: only lifts that cleared their pair and are <= the cap (--max_lift_mm when
    settling, since the lift is transient; else --max_fix_lift_mm), minus any object whose lift makes another pair
    worse than before (e.g. a case lifted off a printer pushed deeper into headphones). With `settle_fn`
    (offsets -> (info, final depths), see run_settle) every dynamic object is then settled under gravity (kept
    lifts are only the starting poses) and the settled poses replace the lifts, unconditionally. Mutates `results` with `fixed` / `skipped_reason`; returns the JSON
    fields to add."""
    baseline = {tuple(r["pair"]): r["max_penetration_mm"] for r in results}
    cap = args.max_lift_mm if settle_fn else args.max_fix_lift_mm

    candidates: dict[str, float] = {}
    for r in results:
        r["fixed"] = False
        if r["lift_mm"] is None:
            r["skipped_reason"] = f"no +z lift up to {args.max_lift_mm} mm clears it"
        elif r["lift_mm"] > cap:
            r["skipped_reason"] = f"needed lift {r['lift_mm']} mm exceeds {cap} mm"
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

    settled: dict[str, dict] = {}
    settle_depths: dict = {}
    if settle_fn:
        info, settle_depths = settle_fn(dict(candidates))
        for o, i in info.items():
            moved = i["displacement_mm"] >= 0.01 or i["rotation_deg"] >= 0.01
            print(f"settle: {o.split('/')[-1]}: lift {i['lift_mm']:g} mm -> moved {i['displacement_mm']} mm, rotated {i['rotation_deg']} deg, "
                  f"{i['steps']} frames, converged={i['converged']}")
            if moved:
                settled[o] = i
        candidates = {}  # the settled poses replace every lift
        after = {}

    for r in results:
        o = r["movable"]
        if settle_fn:
            r["fixed"] = settle_depths.get(tuple(r["pair"]), 0.0) < args.min_depth_mm
            if not r["fixed"]:
                r["skipped_reason"] = dropped.get(o) or f"still penetrating {settle_depths[tuple(r['pair'])]:.2f} mm after the settle"
        elif o in dropped:
            r["skipped_reason"] = dropped[o]
        elif o in candidates:
            r["fixed"] = tuple(r["pair"]) not in after
            if not r["fixed"]:
                r["skipped_reason"] = "still penetrating after the lift"

    lift_only = dict(candidates)
    remaining = {pair_key(p): round(d, 2) for p, d in (settle_depths if settle_fn else after).items()}
    print("\nfix: applied lifts (mm):", {k.split("/")[-1]: v for k, v in lift_only.items()} or "none")
    print("fix: settled objects:", [k.split("/")[-1] for k in settled] or "none")
    for r in results:
        if not r["fixed"]:
            print(f"WARNING: not fixed: {pair_key(r['pair'])} ({r['max_penetration_mm']:.2f} mm): {r.get('skipped_reason')}")
    print("fix: remaining overlaps after fix:", remaining or "none")
    return {"applied_lifts_mm": lift_only, "settled": settled, "remaining_after_fix_mm": remaining}


try:
    main()
except Exception:
    traceback.print_exc()
    sys.stderr.flush()
finally:
    simulation_app.close()
