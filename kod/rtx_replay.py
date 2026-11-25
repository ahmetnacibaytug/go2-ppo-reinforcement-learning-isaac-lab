"""Go2 poz kaydından RTX video."""
import argparse
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("trajectories", nargs="+", type=Path)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--fps", type=int, default=60, help="50 = sadece kayıttaki adımlar, 60 = iki kayıt arası ara değer")
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
parser.add_argument("--rt_subframes", type=int, default=8)
parser.add_argument("--cam_smooth", type=float, default=0.12, help="Kamera takibi yumuşatma katsayısı (1 = yumuşatma yok)")
parser.add_argument("--repeat_every", type=int, default=10, help="Her N. karede render tekrarlanabilirliğini ölç")
parser.add_argument("--max_frames", type=int, default=0, help="Bölüm başına en fazla kare (0 = hepsi)")
parser.add_argument("--ffmpeg", default="ffmpeg")
parser.add_argument("--title", default="")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": args.width, "height": args.height})
try:
    import numpy as np
    import omni.replicator.core as rep
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import GroundPlane
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.storage.native import get_assets_root_path
    from PIL import Image, ImageDraw, ImageFont
    from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics

    world = World(stage_units_in_meters=1.0, physics_dt=1 / 200, rendering_dt=1 / 50)
    GroundPlane("/World/Ground", size=200.0, color=np.array([0.66, 0.68, 0.71]))
    stage = omni.usd.get_context().get_stage()
    for i in range(-30, 31):
        for name, pts in ((f"x{i+30}", [(i, -30, 0.001), (i, 30, 0.001)]), (f"y{i+30}", [(-30, i, 0.001), (30, i, 0.001)])):
            c = UsdGeom.BasisCurves.Define(stage, f"/World/Grid/{name}")
            c.CreatePointsAttr(pts); c.CreateCurveVertexCountsAttr([2]); c.CreateTypeAttr("linear")
            c.CreateWidthsAttr([0.008, 0.008]); c.CreateDisplayColorAttr([(0.45, 0.47, 0.5)])
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun"); sun.CreateIntensityAttr(2800.0)
    sun.CreateAngleAttr(1.0)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-50, 0, 35))
    UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(550.0)
    add_reference_to_stage(get_assets_root_path() + "/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd", "/World/Go2")

    gravity_off = 0
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith("/World/Go2") and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)
            gravity_off += 1

    robot = world.scene.add(SingleArticulation("/World/Go2", name="go2"))
    world.reset()
    dof_names = list(robot.dof_names)
    controller = robot.get_articulation_controller()

    cam_prim = UsdGeom.Camera.Define(stage, "/World/FollowCam")
    cam_prim.CreateFocalLengthAttr(20.0)
    cam_xform = UsdGeom.Xformable(cam_prim).AddTransformOp()

    def place_camera(tx, ty):
        eye = Gf.Vec3d(tx + 1.6, ty - 1.6, 0.8)
        view = Gf.Matrix4d().SetLookAt(eye, Gf.Vec3d(tx, ty, 0.27), Gf.Vec3d(0, 0, 1))
        cam_xform.Set(view.GetInverse())

    place_camera(0.0, 0.0)
    rp = rep.create.render_product("/World/FollowCam", (args.width, args.height))
    ann = rep.AnnotatorRegistry.get_annotator("rgb"); ann.attach(rp)
    fsize = max(18, args.height // 36)
    try:
        font = ImageFont.load_default(size=fsize)
    except TypeError:
        font = ImageFont.load_default()

    def capture():
        world.render()
        rep.orchestrator.step(rt_subframes=args.rt_subframes)  # Fizik ilerler.
        return np.ascontiguousarray(np.asarray(ann.get_data())[:, :, :3])

    def slerp(qa, qb, a):
        dot = float(np.dot(qa, qb))
        if dot < 0:
            qb, dot = -qb, -dot
        if dot > 0.9995:
            r = qa + a * (qb - qa)
        else:
            th = np.arccos(dot)
            r = (np.sin((1 - a) * th) * qa + np.sin(a * th) * qb) / np.sin(th)
        return r / np.linalg.norm(r)

    def set_state(pos, quat, q):
        robot.set_world_pose(position=pos, orientation=quat)
        robot.set_joint_positions(q)
        robot.set_joint_velocities(np.zeros(len(dof_names), dtype=np.float32))
        robot.set_linear_velocity(np.zeros(3))
        robot.set_angular_velocity(np.zeros(3))
        controller.apply_action(ArticulationAction(joint_positions=q))

    frames_dir = args.out.parent / (args.out.stem + "_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    n_frame, meta_segments = 0, []
    pos_err, joint_err, repeat_diff, still_diff = [], [], [], []
    for traj in args.trajectories:
        d = np.load(traj)
        names = [str(n) for n in d["joint_names"]]
        order = [names.index(n) for n in dof_names]
        step_dt = float(d["step_dt"][0])
        stride = max(1, round(1.0 / (args.fps * step_dt)))
        origin = d["root_pos"][0].copy(); origin[2] = 0.0
        n_rec = len(d["joint_pos"])
        rec_hz = 1.0 / step_dt
        if args.fps > rec_hz + 1e-6:
            # Poz ve eklem: doğrusal. Yönelim: SLERP.
            times = np.arange(0.0, (n_rec - 1) * step_dt + 1e-9, 1.0 / args.fps)
            idx = [t / step_dt for t in times]
        else:
            idx = list(range(0, n_rec, stride))
        if args.max_frames:
            idx = idx[: args.max_frames]
        seg = traj.stem.replace("joint_traj_", "")
        meta_segments.append({"segment": seg, "trajectory": str(traj), "checkpoint": str(d["checkpoint"][0]),
                              "steps": int(len(d["joint_pos"])), "frames": len(idx), "stride": stride,
                              "command": [round(float(v), 3) for v in d["command"][0]]})
        cam = np.array(d["root_pos"][0][:2] - origin[:2], dtype=float)
        prev = None
        for j, kf in enumerate(idx):
            k0 = int(np.floor(kf)); k1 = min(k0 + 1, n_rec - 1); a = float(kf - k0)
            k = k0 if a < 0.5 else k1
            pos = ((1 - a) * d["root_pos"][k0] + a * d["root_pos"][k1] - origin).astype(np.float64)
            q = ((1 - a) * d["joint_pos"][k0] + a * d["joint_pos"][k1])[order].astype(np.float32)
            quat = slerp(d["root_quat_wxyz"][k0].astype(np.float64), d["root_quat_wxyz"][k1].astype(np.float64), a)
            set_state(pos, quat, q)
            cam = pos[:2] if j == 0 else (1 - args.cam_smooth) * cam + args.cam_smooth * pos[:2]
            place_camera(float(cam[0]), float(cam[1]))
            rgb = capture()
            rpos, _ = robot.get_world_pose()
            pos_err.append(float(np.linalg.norm(np.asarray(rpos) - pos)))
            joint_err.append(float(np.abs(np.asarray(robot.get_joint_positions()) - q).max()))
            if args.repeat_every and j % args.repeat_every == 0:
                set_state(pos, quat, q)
                rgb2 = capture()
                repeat_diff.append(float(np.abs(rgb.astype(np.int16) - rgb2.astype(np.int16)).mean()))
            if seg == "stand" and prev is not None:
                still_diff.append(float(np.abs(rgb.astype(np.int16) - prev.astype(np.int16)).mean()))
            prev = rgb
            img = Image.fromarray(rgb)
            draw = ImageDraw.Draw(img)
            c, m = d["command"][k], d["measured"][k]
            text = (f"{args.title}  |  {seg}  |  t={kf * step_dt:5.2f}s\n"
                    f"cmd  vx={c[0]:+.2f} vy={c[1]:+.2f} m/s  wz={c[2]:+.2f} rad/s\n"
                    f"meas vx={m[0]:+.2f} vy={m[1]:+.2f} m/s  wz={m[2]:+.2f} rad/s")
            draw.multiline_text((int(fsize * 0.9), int(fsize * 0.7)), text, fill=(20, 20, 20), font=font, spacing=6)
            img.save(frames_dir / f"f{n_frame:05d}.png")
            n_frame += 1

    cmd = [args.ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-framerate", str(args.fps),
           "-i", str(frames_dir / "f%05d.png"), "-c:v", "libx264", "-preset", "slow", "-crf", "16",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.out)]
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL, timeout=3000)
    Image.open(frames_dir / f"f{n_frame // 2:05d}.png").convert("RGB").save(args.out.with_suffix(".jpg"), quality=92)

    def stats(v, scale=1.0):
        return None if not v else {"mean": round(float(np.mean(v)) * scale, 4), "max": round(float(np.max(v)) * scale, 4), "n": len(v)}

    meta = {"renderer": "Isaac Sim 6.0.1 RTX (replicator rgb annotator)", 
            "resolution": [args.width, args.height], "fps": args.fps, "rt_subframes": args.rt_subframes,
            "frames": n_frame, "duration_s": round(n_frame / args.fps, 2),
            "robot_links_gravity_disabled": gravity_off, "camera_target_ema": args.cam_smooth,
            "robot_state_filtering": "none",
            "frame_states": ("recorded 50 Hz physics states, one per frame" if args.fps <= 50 else
                             f"{args.fps} fps: each frame interpolated between the two neighbouring recorded 50 Hz "
                             "states (linear joints/position, slerp base orientation); no smoothing across more than 2 samples"),
            "fidelity": {"root_pos_err_mm": stats(pos_err, 1000.0), "joint_err_mrad": stats(joint_err, 1000.0),
                         "repeat_capture_pixel_diff": stats(repeat_diff), "stand_consecutive_pixel_diff": stats(still_diff)},
            "source": "recorded physics states (play.py --record_joints; policy ran in Isaac Sim 5.1 / Isaac Lab 2.3.2)",
            "segments": meta_segments}
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print("RTX_REPLAY_OK", json.dumps(meta["fidelity"]), flush=True)
except BaseException:
    import sys
    import traceback
    traceback.print_exc()
    sys.stdout.flush(); sys.stderr.flush()
    print("RTX_REPLAY_FAILED", flush=True)
finally:
    app.close()
