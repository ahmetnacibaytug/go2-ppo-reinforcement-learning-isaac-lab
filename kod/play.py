"""Değerlendirme."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Go2 policy testi ve video")
parser.add_argument("--checkpoint", type=Path, default=None)
parser.add_argument("--log_root", type=Path, default=Path(__file__).resolve().parent / "logs",
                    help="--checkpoint verilmezse son checkpoint burada aranır")
parser.add_argument("--num_envs", type=int, default=None,
                    help="Varsayılan: --suite için 16, diğerlerinde 1")
parser.add_argument("--cmd", type=float, nargs=3, default=None, metavar=("VX", "VY", "WZ"))
parser.add_argument("--suite", action="store_true", help="Sabit komutlu test senaryoları (video yok)")
parser.add_argument("--turn_sweep", action="store_true",
                    help="--suite'e iki yönde 0.5/0.6/0.7 rad/s dönüşleri ekler")
parser.add_argument("--demo", action="store_true", help="Tek video: ileri, yana, dönüş, dur")
parser.add_argument("--record_joints", action="store_true",
                    help="RTX render için gövde pozu ve eklem açılarını kaydet (joint_traj_<senaryo>.npz)")
parser.add_argument("--robustness", action="store_true", help="Testte randomizasyon, gürültü ve itmeleri aç")
parser.add_argument("--steps", type=int, default=400, help="Senaryo başına adım sayısı")
parser.add_argument("--warmup", type=int, default=50, help="Metriklere girmeyen ilk adım sayısı")
parser.add_argument("--no_video", action="store_true")
parser.add_argument("--video_backend", choices=("rtx", "kinematic"), default="kinematic",
                    help="rtx = Isaac kamerası, kinematic = kaydedilen pozlardan GIF")
parser.add_argument("--kinematic_fps", type=int, default=12, help="kinematic GIF fps (1..30)")
parser.add_argument("--no_gif", action="store_true")
parser.add_argument("--no_export", action="store_true")
parser.add_argument("--no_plot", action="store_true")
parser.add_argument("--arrows", action="store_true", help="Komut ve gerçek hız oklarını göster")
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.suite and (args.demo or args.cmd):
    parser.error("--suite, --demo, --cmd aynı anda kullanılamaz")
if args.demo and args.cmd:
    parser.error("--demo ve --cmd aynı anda kullanılamaz")
if args.turn_sweep and not args.suite:
    parser.error("--turn_sweep yalnızca --suite ile kullanılabilir")
if args.num_envs is None:
    args.num_envs = 16 if args.suite else 1
if args.steps <= 0 or args.num_envs <= 0 or args.warmup < 0 or args.warmup >= args.steps:
    parser.error("--steps/num_envs pozitif, --warmup ise 0 <= warmup < steps olmalı")
record_video = not args.no_video and not args.suite and args.video_backend == "rtx"
record_kinematic = not args.no_video and not args.suite and args.video_backend == "kinematic"
if not (1 <= args.kinematic_fps <= 30):
    parser.error("--kinematic_fps 1..30 olmalı")
if record_video:
    args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import shutil
import subprocess
from datetime import datetime, timezone

from kinematic_video import save_trajectory, render_trajectory
from reward_math import current_contact_slip

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

from evaluation_utils import (
    SCENARIOS, SCENARIOS_TURN_SWEEP, command_for_demo, save_csv, save_json,
    summarize_across_environments, summarize_scenario, summarize_environment_errors,
)
from play_checks import validate_scenario_commands
from go2_env_cfg import Go2FlatEnvCfg_PLAY
from rsl_rl_ppo_cfg import Go2FlatPPORunnerCfg, validate_rsl_rl_version


def latest_checkpoint(log_root: Path, experiment: str) -> Path:
    root = log_root.expanduser().resolve() / "rsl_rl" / experiment
    candidates = []
    for folder in root.glob("*"):
        if folder.is_dir():
            models = list(folder.glob("model_*.pt"))
            if models:
                last = max(models, key=lambda p: int(p.stem.split("_")[-1]))
                candidates.append((last.stat().st_mtime, last))
    if not candidates:
        raise FileNotFoundError(f"Checkpoint bulunamadı: {root}; --checkpoint ile dosya göster.")
    return max(candidates, key=lambda item: item[0])[1]


def export_policy(runner: OnPolicyRunner, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    neural_net = getattr(runner.alg, "policy", None)
    if neural_net is None:
        neural_net = runner.alg.actor_critic
    normalizer = getattr(neural_net, "actor_obs_normalizer", None)
    export_policy_as_jit(neural_net, normalizer=normalizer, path=str(output), filename="policy.pt")
    export_policy_as_onnx(neural_net, normalizer=normalizer, path=str(output), filename="policy.onnx")
    for name in ("policy.pt", "policy.onnx"):
        path = output / name
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError(f"Export işlemi dosya oluşturmadı: {path}")
    print(f"[OK] TorchScript + ONNX: {output}")


def mp4_to_gif(video_dir: Path):
    if shutil.which("ffmpeg") is None:
        print("[WARN] ffmpeg bulunamadı; MP4 kaydedilmiş olabilir, GIF atlandı.")
        return None
    mp4s = list(video_dir.glob("*.mp4"))
    if not mp4s:
        print("[WARN] MP4 bulunamadı; GIF oluşturulamıyor.")
        return None
    mp4 = max(mp4s, key=lambda p: p.stat().st_mtime)
    gif = mp4.with_suffix(".gif")
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(mp4),
        "-filter_complex",
        "fps=15,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse",
        "-loop", "0", str(gif),
    ]
    subprocess.run(command, check=True)
    if not gif.exists() or gif.stat().st_size == 0:
        raise RuntimeError("GIF dönüşümünde dosya oluşturulmadı")
    print(f"[OK] Video: {mp4}\n[OK] GIF: {gif}")
    return gif


def maybe_plot(csv_file: Path, destination: Path):
    try:
        from plot_results import plot_gait, plot_timeseries
        plot_timeseries(csv_file, destination)
        print(f"[OK] Grafik: {destination}")
        gait = plot_gait(csv_file, destination.with_name("gait_diagram.png"))
        if gait is not None:
            print(f"[OK] Gait diyagramı: {gait}")
    except ImportError:
        print("[WARN] Matplotlib kurulu değil; grafik atlandı. CSV/JSON yazıldı.")


def get_policy_observations(env):
    obs = env.get_observations()
    return obs[0] if isinstance(obs, tuple) else obs


def set_command(env, command):
    term = env.unwrapped.command_manager.get_term("base_velocity")
    with torch.no_grad():
        term.command[:] = torch.tensor(command, device=env.unwrapped.device, dtype=term.command.dtype)
        term.is_standing_env[:] = False


JOINT_RECORDS: dict = {}


def run_test(env, runner, policy, *, scenario: str, command, steps: int, warmup: int, demo: bool,
             capture_kinematic: bool = False):
    base = env.unwrapped
    robot = base.scene["robot"]
    contact = base.scene.sensors["contact_forces"]
    foot_ids, foot_names = contact.find_bodies(".*_foot")
    robot_foot_ids, robot_foot_names = robot.find_bodies(".*_foot")
    if len(foot_ids) != 4 or len(robot_foot_ids) != 4:
        raise RuntimeError(f"Beklenen 4 ayak bulunamadı: sensor={foot_names}")
    robot_foot_ids = [robot_foot_ids[robot_foot_names.index(name)] for name in foot_names]
    network = getattr(runner.alg, "policy", getattr(runner.alg, "actor_critic", None))
    t = base.step_dt
    rows = []
    kin = None
    if capture_kinematic:
        kin = {"positions": [], "root_positions": [], "root_quaternions": [],
               "commands": [], "velocities": [], "sample_steps": [],
               "body_names": list(robot.body_names), "step_dt": t}
        stride = max(1, round(1.0 / (args.kinematic_fps * t)))
    air_times = {name: [] for name in foot_names}
    total_falls = 0
    env_mass_kg = robot.root_physx_view.get_masses().sum(dim=1)
    robot_mass_kg = float(env_mass_kg.mean().item())
    with torch.inference_mode():
        # Reset, inference_mode içinde.
        base.reset()
        if network is not None and hasattr(network, "reset"):
            network.reset(torch.ones(base.num_envs, device=base.device, dtype=torch.bool))
        foot_contact_sum = torch.zeros(len(foot_names), device=base.device)
        env_xy_err_sq = torch.zeros(base.num_envs, device=base.device)
        env_yaw_err_sq = torch.zeros(base.num_envs, device=base.device)
        env_falls = torch.zeros(base.num_envs, device=base.device, dtype=torch.int64)
        env_falls_total = torch.zeros(base.num_envs, device=base.device, dtype=torch.int64)
        env_valid_samples = torch.zeros(base.num_envs, device=base.device, dtype=torch.int64)
        # Warmup her reset'te yeniden başlar.
        steps_since_reset = torch.zeros(base.num_envs, device=base.device, dtype=torch.int64)
        for step in range(steps):
            if not simulation_app.is_running():
                print("[WARN] Simülasyon kapandı, test erken bitti")
                break
            current_command = command_for_demo(step, steps) if demo else command
            set_command(env, current_command)
            obs = get_policy_observations(env)
            actions = policy(obs)
            _, _, dones, _ = env.step(actions)
            if args.record_joints:
                rec = JOINT_RECORDS.setdefault(scenario, {
                    "root_pos": [], "root_quat_wxyz": [], "joint_pos": [], "command": [], "measured": [],
                    "joint_names": list(robot.joint_names), "step_dt": t})
                rec["root_pos"].append(robot.data.root_pos_w[0].detach().cpu().numpy().copy())
                rec["root_quat_wxyz"].append(robot.data.root_quat_w[0].detach().cpu().numpy().copy())
                rec["joint_pos"].append(robot.data.joint_pos[0].detach().cpu().numpy().copy())
                rec["command"].append(list(current_command))
                rec["measured"].append(torch.cat((robot.data.root_lin_vel_b[0, :2],
                                       robot.data.root_ang_vel_b[0, 2:3])).detach().cpu().numpy().copy())
            if kin is not None and step % stride == 0:
                kin["positions"].append(robot.data.body_pos_w[0].detach().cpu().numpy().copy())
                kin["root_positions"].append(robot.data.root_pos_w[0].detach().cpu().numpy().copy())
                kin["root_quaternions"].append(robot.data.root_quat_w[0].detach().cpu().numpy().copy())
                kin["commands"].append(list(current_command))
                kin["velocities"].append(torch.cat((robot.data.root_lin_vel_b[0,:2],
                                   robot.data.root_ang_vel_b[0,2:3])).detach().cpu().numpy().copy())
                kin["sample_steps"].append(step)
            if network is not None and hasattr(network, "reset"):
                network.reset(dones)
            steps_since_reset += 1
            steps_since_reset[dones.bool().reshape(-1)] = 0
            valid = steps_since_reset > warmup
            num_valid = int(valid.sum().item())

            def valid_mean(values):
                return (values[valid] if num_valid else values).mean().item()

            lin = robot.data.root_lin_vel_b[:, :2]
            cmd = torch.tensor(current_command, device=base.device, dtype=lin.dtype).expand(base.num_envs, 3)
            yaw = robot.data.root_ang_vel_b[:, 2]
            lin_err = torch.linalg.norm(cmd[:, :2] - lin, dim=1)
            yaw_err = torch.abs(cmd[:, 2] - yaw)
            first_contact = contact.compute_first_contact(base.step_dt)[:, foot_ids]
            last_air = contact.data.last_air_time[:, foot_ids]
            forces_w = contact.data.net_forces_w[:, foot_ids, :]
            foot_on_ground = forces_w[..., 2].abs() > 1.0
            vel_feet = robot.data.body_lin_vel_w[:, robot_foot_ids, :2]
            slip_proxy = current_contact_slip(forces_w, vel_feet)
            mech_power_proxy = (robot.data.applied_torque * robot.data.joint_vel).abs().sum(dim=1)
            fall_flags = base.termination_manager.terminated
            timeout_flags = base.termination_manager.time_outs
            total_falls += int(fall_flags.sum().item())
            env_falls_total += fall_flags.long()
            if step >= warmup:
                env_xy_err_sq += lin_err.square() * valid
                env_yaw_err_sq += yaw_err.square() * valid
                env_valid_samples += valid.long()
                env_falls += fall_flags.long()
                foot_contact_sum += (foot_on_ground & valid.unsqueeze(1)).float().sum(dim=0)
                for i, name in enumerate(foot_names):
                    samples = last_air[:, i][first_contact[:, i] & valid]
                    air_times[name].extend(samples.detach().cpu().tolist())
                rows.append({
                    "scenario": scenario, "step": step, "time_s": round(step * t, 4),
                    "cmd_vx": round(cmd[:, 0].mean().item(), 5),
                    "cmd_vy": round(cmd[:, 1].mean().item(), 5),
                    "cmd_wz": round(cmd[:, 2].mean().item(), 5),
                    "vx": round(valid_mean(lin[:, 0]), 5),
                    "vy": round(valid_mean(lin[:, 1]), 5),
                    "wz": round(valid_mean(yaw), 5),
                    "linear_error_mps": round(valid_mean(lin_err), 5),
                    "yaw_error_radps": round(valid_mean(yaw_err), 5),
                    "linear_error_sq_mean": round(lin_err.square().mean().item(), 7),
                    "yaw_error_sq_mean": round(yaw_err.square().mean().item(), 7),
                    "valid_env_count": num_valid,
                    "linear_error_sq_valid_sum": round((lin_err.square() * valid).sum().item(), 7),
                    "yaw_error_sq_valid_sum": round((yaw_err.square() * valid).sum().item(), 7),
                    "planar_speed_mps": round(valid_mean(lin.norm(dim=1)), 5),
                    "base_height_m": round(valid_mean(robot.data.root_pos_w[:, 2]), 5),
                    "slip_proxy_mps": round(valid_mean(slip_proxy), 5),
                    "abs_mech_power_proxy_w": round(valid_mean(mech_power_proxy), 5),
                    "terminated_count": int(fall_flags.sum().item()),
                    "timeout_count": int(timeout_flags.sum().item()),
                    **{f"contact_{name}": int(foot_on_ground[0, i].item()) for i, name in enumerate(foot_names)},
                })
            if step % 100 == 0:
                root_xyz = robot.data.root_pos_w[0].detach().cpu().tolist()
                feet_z = robot.data.body_pos_w[0, robot_foot_ids, 2].detach().cpu().tolist()
                print(f"[{scenario}][{step:04d}] base_xyz={tuple(round(x, 3) for x in root_xyz)} "
                      f"feet_z={dict(zip(foot_names, (round(z, 3) for z in feet_z)))} "
                      f"cmd={tuple(round(x, 2) for x in current_command)} "
                      f"vx={lin[0,0].item():.2f}, vy={lin[0,1].item():.2f}, wz={yaw[0].item():.2f}")
    if not rows:
        raise RuntimeError(f"{scenario}: warmup sonrası değerlendirme örneği kaydedilemedi")
    total_valid = int(env_valid_samples.sum().item())
    duty_factor = {
        name: foot_contact_sum[i].item() / total_valid for i, name in enumerate(foot_names)
    } if total_valid else None
    summary = summarize_scenario(
        rows, num_envs=base.num_envs, air_times=air_times,
        duty_factor=duty_factor, robot_mass_kg=robot_mass_kg,
    )
    summary["per_environment"] = summarize_environment_errors(
        env_xy_err_sq.detach().cpu().tolist(),
        env_yaw_err_sq.detach().cpu().tolist(),
        env_falls_total.detach().cpu().tolist(),
        env_valid_samples.detach().cpu().tolist(),
        fall_counts_after_warmup=env_falls.detach().cpu().tolist(),
    )
    summary["across_environments"] = summarize_across_environments(summary["per_environment"])
    summary["robot_mass_kg_min_max"] = [round(float(env_mass_kg.min().item()), 3),
                                        round(float(env_mass_kg.max().item()), 3)]
    summary["total_fall_terminations_including_warmup"] = total_falls
    return rows, summary, kin


def main():
    validate_rsl_rl_version()
    suite_scenarios = SCENARIOS_TURN_SWEEP if args.turn_sweep else SCENARIOS
    validate_scenario_commands(suite_scenarios)
    if args.cmd is not None:
        validate_scenario_commands({"custom": args.cmd})
    env_cfg = Go2FlatEnvCfg_PLAY()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    if args.device is not None:
        env_cfg.sim.device = args.device
    if args.robustness:
        env_cfg.enable_robustness_test()
    env_cfg.commands.base_velocity.debug_vis = args.arrows
    if args.cmd is not None:
        env_cfg.set_fixed_command(*args.cmd)
    elif args.suite or args.demo:
        env_cfg.set_fixed_command(0.0, 0.0, 0.0)
    agent_cfg = Go2FlatPPORunnerCfg()
    agent_cfg.device = env_cfg.sim.device
    agent_cfg.seed = args.seed
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else latest_checkpoint(args.log_root, agent_cfg.experiment_name)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    run_dir = checkpoint.parent
    eval_dir = run_dir / "evaluation" / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f") + ("_robust" if args.robustness else ""))
    eval_dir.mkdir(parents=True)
    env_cfg.log_dir = str(eval_dir)
    env = None
    trajectory_path = None
    try:
        env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array" if record_video else None)
        video_dir = eval_dir / "videos"
        if record_video:
            env = gym.wrappers.RecordVideo(
                env, video_folder=str(video_dir), step_trigger=lambda step: step == 0,
                video_length=args.steps, disable_logger=True,
            )
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        if not args.no_export:
            try:
                export_policy(runner, run_dir / "exported")
            except Exception as exc:
                print(f"[WARN] Export başarısız: {exc}; değerlendirmeye devam ediliyor.")
        scenarios = suite_scenarios if args.suite else {
            ("demo" if args.demo else "custom"): (0.8, 0.0, 0.0) if args.demo else (
                tuple(args.cmd) if args.cmd is not None else (0.8, 0.0, 0.3)
            )
        }
        all_rows = []
        summaries = []
        for name, target in scenarios.items():
            rows, summary, kin = run_test(
                env, runner, policy, scenario=name, command=target,
                steps=args.steps, warmup=args.warmup, demo=args.demo,
                capture_kinematic=record_kinematic,
            )
            if kin is not None:
                trajectory_path = save_trajectory(eval_dir / "videos" / "trajectory.npz", **kin)
                print(f"[OK] Poz kaydı: {trajectory_path}")
            all_rows.extend(rows)
            summaries.append(summary)
            across = summary["across_environments"]
            print(f"[SONUÇ] {name}: xy RMSE={summary['linear_xy_rmse_valid_mps']} m/s, "
                  f"yaw RMSE={summary['yaw_rmse_valid_radps']} rad/s, "
                  f"düşme={summary['total_fall_terminations_including_warmup']}, "
                  f"düşmeyen ortam={across['envs_without_fall']}/{across['num_envs']}")
        save_csv(eval_dir / "timeseries.csv", all_rows)
        if args.record_joints:
            import numpy as np
            for name, rec in JOINT_RECORDS.items():
                out = eval_dir / f"joint_traj_{name}.npz"
                np.savez_compressed(
                    out, root_pos=np.stack(rec["root_pos"]).astype("float32"),
                    root_quat_wxyz=np.stack(rec["root_quat_wxyz"]).astype("float32"),
                    joint_pos=np.stack(rec["joint_pos"]).astype("float32"),
                    command=np.asarray(rec["command"], dtype="float32"),
                    measured=np.stack(rec["measured"]).astype("float32"),
                    joint_names=np.asarray(rec["joint_names"], dtype=str),
                    step_dt=np.asarray([rec["step_dt"]], dtype="float64"),
                    checkpoint=np.asarray([str(checkpoint)], dtype=str))
                print(f"[OK] Eklem kaydı: {out} ({len(rec['joint_pos'])} adım)")
        report = {
            "checkpoint": str(checkpoint),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "steps_per_scenario": args.steps,
            "warmup_steps": args.warmup,
            "num_envs": args.num_envs,
            "seed": args.seed,
            "robustness_randomization": args.robustness,
            "evaluation_mode": "suite" if args.suite else "demo" if args.demo else "single",
            "video_backend": (args.video_backend if not args.no_video and not args.suite else "none"),
            "scenarios": summaries,
        }
        save_json(eval_dir / "report.json", report)
        if not args.no_plot:
            maybe_plot(eval_dir / "timeseries.csv", eval_dir / "velocity_tracking.png")
        print(f"[DONE] CSV + JSON -> {eval_dir}")
    finally:
        if env is not None:
            env.close()
    if record_kinematic:
        if trajectory_path is None:
            raise RuntimeError("GIF için poz kaydı oluşmadı")
        gif, mp4 = render_trajectory(trajectory_path, fps=args.kinematic_fps,
                                     make_mp4=not args.no_gif)
        print(f"[OK] GIF: {gif}")
        if mp4:
            print(f"[OK] MP4: {mp4}")
    if record_video and not args.no_gif:
        try:
            mp4_to_gif(video_dir)
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            print(f"[WARN] GIF oluşturma başarısız: {exc}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
