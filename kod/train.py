"""Go2 PPO eğitimi (Isaac Lab 2.3.2)."""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Go2 hız takibi PPO eğitimi")
parser.add_argument("--preset", choices=["smoke", "budget", "balanced", "full"], default="balanced")
parser.add_argument("--num_envs", type=int, default=None, help="Ortam sayısı (preset'i ezer)")
parser.add_argument("--max_iterations", type=int, default=None,
                    help="Toplam iterasyon (preset'i ezer); --resume ile de toplam hedeftir")
parser.add_argument("--save_interval", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume", type=Path, default=None, help="Devam edilecek model_*.pt")
parser.add_argument("--fresh", action="store_true", help="Sıfırdan başla (--resume ile kullanılamaz)")
parser.add_argument("--symmetry", action="store_true", help="Sol-sağ simetri veri artırımını aç")
parser.add_argument("--run_name", type=str, default="")
parser.add_argument("--log_root", type=Path, default=Path(__file__).resolve().parent / "logs",
                    help="Log klasörü (varsayılan: kod/logs)")
parser.add_argument("--no_randomization", action="store_true", help="Domain randomization'ı kapat")
parser.add_argument("--no_gain_randomization", action="store_true", help="Sadece PD kazanç randomizasyonunu kapat")
parser.add_argument("--no_finite_check", action="store_true", help="Her adımdaki NaN/Inf kontrolünü kapat")
parser.add_argument("--video", action="store_true", help="Eğitim sırasında video kaydet")
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--video_length", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.fresh and args.resume is not None:
    parser.error("--fresh ve --resume birlikte kullanılamaz")
if args.video:
    args.enable_cameras = True
if args.video_interval <= 0 or args.video_length <= 0:
    parser.error("--video_interval ve --video_length pozitif olmalı")
if not re.fullmatch(r"[A-Za-z0-9_.-]*", args.run_name):
    parser.error("--run_name yalnızca harf/rakam/_/.- içerebilir")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from go2_env_cfg import Go2FlatEnvCfg
from play_checks import resume_plan
from go2_symmetry import validate_go2_symmetry
from rsl_rl_ppo_cfg import Go2FlatPPORunnerCfg, enable_lr_symmetry, validate_rsl_rl_version
from sim_checks import check_actuator_gains, install_finite_check

PRESETS = {
    # (ortam sayısı, iterasyon)
    "smoke": (128, 10),
    "budget": (1024, 300),
    "balanced": (4096, 300),
    "full": (4096, 1500),
}


def configure_training(env_cfg, agent_cfg):
    num_envs, iterations = PRESETS[args.preset]
    env_cfg.scene.num_envs = args.num_envs if args.num_envs is not None else num_envs
    agent_cfg.max_iterations = args.max_iterations if args.max_iterations is not None else iterations
    if env_cfg.scene.num_envs <= 0 or agent_cfg.max_iterations <= 0:
        raise ValueError("num_envs ve max_iterations pozitif olmalı")
    agent_cfg.save_interval = min(agent_cfg.save_interval, max(1, agent_cfg.max_iterations // 2))
    if args.save_interval is not None:
        if args.save_interval <= 0:
            raise ValueError("save_interval pozitif olmalı")
        agent_cfg.save_interval = args.save_interval
    if args.device is not None:
        env_cfg.sim.device = args.device
    env_cfg.seed = agent_cfg.seed = args.seed
    agent_cfg.device = env_cfg.sim.device
    if args.no_randomization:
        env_cfg.events.physics_material = None
        env_cfg.events.add_base_mass = None
        env_cfg.events.actuator_gains = None
        env_cfg.events.push_robot = None
        env_cfg.observations.policy.enable_corruption = False
    if args.no_gain_randomization:
        env_cfg.events.actuator_gains = None
    if args.symmetry:
        enable_lr_symmetry(agent_cfg)
        print("[INFO] Sol-sağ simetri veri artırımı açık")
    return env_cfg, agent_cfg


def get_training_summary(env_cfg, agent_cfg):
    control_dt = env_cfg.sim.dt * env_cfg.decimation
    samples = env_cfg.scene.num_envs * agent_cfg.num_steps_per_env
    return {
        "num_envs": env_cfg.scene.num_envs,
        "physics_hz": round(1 / env_cfg.sim.dt, 3),
        "policy_hz": round(1 / control_dt, 3),
        "episode_policy_steps": int(env_cfg.episode_length_s / control_dt),
        "rollout_steps_per_env": agent_cfg.num_steps_per_env,
        "batch_size": samples,
        "mini_batch_size": samples // agent_cfg.algorithm.num_mini_batches,
        "optimizer_updates_per_iteration": (
            agent_cfg.algorithm.num_mini_batches * agent_cfg.algorithm.num_learning_epochs
        ),
        "iterations_requested": agent_cfg.max_iterations,
        "total_environment_transitions_requested": samples * agent_cfg.max_iterations,
    }


def write_metadata(log_dir, env_cfg, agent_cfg, rsl_version):
    log_dir.mkdir(parents=True, exist_ok=False)
    params_dir = log_dir / "params"
    params_dir.mkdir(parents=True)
    dump_yaml(str(params_dir / "env.yaml"), env_cfg)
    dump_yaml(str(params_dir / "agent.yaml"), agent_cfg)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_isaac_lab": "v2.3.2",
        "rsl_rl_installed": rsl_version,
        "torch_version": torch.__version__,
        "python": sys.version,
        "cuda_gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "preset": args.preset,
        "seed": args.seed,
        "randomization": not args.no_randomization,
        "actuator_gain_randomization": env_cfg.events.actuator_gains is not None,
        "resume_checkpoint": str(args.resume.resolve()) if args.resume else None,
        "symmetry": args.symmetry,
        "rel_standing_envs": env_cfg.commands.base_velocity.rel_standing_envs,
        "swing_clearance_weight": env_cfg.rewards.swing_clearance.weight,
        "initialization": "resume" if args.resume else "random_fresh",
        "training": get_training_summary(env_cfg, agent_cfg),
    }
    (params_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("[INFO] Eğitim ayarları:\n" + json.dumps(metadata["training"], indent=2))


def check_symmetry_integration(runner, env) -> dict:
    from go2_symmetry import compute_symmetric_states, mirror_policy_obs_lr

    cfg = getattr(runner.alg, "symmetry", None)
    if not cfg or not cfg.get("use_data_augmentation"):
        raise RuntimeError(f"RSL-RL PPO simetriyi etkinleştirmedi (alg.symmetry={cfg!r})")
    func = cfg.get("data_augmentation_func")
    func_name = func if isinstance(func, str) else f"{getattr(func, '__module__', '?')}:{getattr(func, '__name__', '?')}"
    if not func_name.endswith("compute_symmetric_states") or "go2_symmetry" not in func_name:
        raise RuntimeError(f"beklenmeyen data_augmentation_func: {func_name}")
    if isinstance(func, str):
        func = compute_symmetric_states
    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]
    if not hasattr(obs, "batch_size") or "policy" not in obs.keys():
        raise RuntimeError(f"TensorDict içinde 'policy' bekleniyor: {type(obs)}")
    n = int(obs.batch_size[0])
    actions = torch.zeros(n, env.unwrapped.action_manager.total_action_dim, device=obs["policy"].device)
    obs_aug, act_aug = func(obs=obs, actions=actions, env=env)
    if int(obs_aug.batch_size[0]) != 2 * n or act_aug.shape[0] != 2 * n:
        raise RuntimeError(f"Simetri sonrası {2 * n} örnek bekleniyor.")
    if not torch.equal(obs_aug["policy"][:n], obs["policy"]):
        raise RuntimeError("İlk gruptaki gözlemler değişmiş.")
    if not torch.allclose(mirror_policy_obs_lr(obs_aug["policy"][n:]), obs["policy"], atol=1e-6):
        raise RuntimeError("Çift aynalama orijinali vermedi.")
    cmd = obs["policy"][:, 9:12]
    cmd_m = obs_aug["policy"][n:, 9:12]
    if not (torch.allclose(cmd_m[:, 0], cmd[:, 0]) and torch.allclose(cmd_m[:, 1:], -cmd[:, 1:])):
        raise RuntimeError("Aynalanmış komutta vx korunmalı, vy ve wz işaret değiştirmeliydi")
    return {"alg_symmetry_enabled": True, "function": func_name, "obs_keys": sorted(obs.keys()),
            "batch": n, "augmented_batch": int(obs_aug.batch_size[0]),
            "real_obs_mirror_roundtrip_ok": True, "command_sign_ok": True}


def write_training_result(log_dir, env_cfg, agent_cfg, runner, start_iteration, iterations_run, elapsed_s):
    transitions = env_cfg.scene.num_envs * agent_cfg.num_steps_per_env * iterations_run
    checkpoints = sorted(log_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    result = {
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_clock_s": round(elapsed_s, 1),
        "wall_clock_min": round(elapsed_s / 60.0, 2),
        "start_iteration": int(start_iteration),
        "iterations_run_this_session": int(iterations_run),
        "target_total_iterations": int(agent_cfg.max_iterations),
        "last_iteration": int(runner.current_learning_iteration),
        "environment_transitions": transitions,
        "transitions_per_second": round(transitions / elapsed_s, 1) if elapsed_s > 0 else None,
        "cuda_gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "final_checkpoint": checkpoints[-1].name if checkpoints else None,
    }
    (log_dir / "params" / "training_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("[INFO] Eğitim özeti:\n" + json.dumps(result, indent=2))


def main():
    rsl_version = validate_rsl_rl_version()
    env_cfg, agent_cfg = configure_training(Go2FlatEnvCfg(), Go2FlatPPORunnerCfg())
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    experiment_dir = args.log_root.expanduser().resolve() / "rsl_rl" / agent_cfg.experiment_name
    log_dir = experiment_dir / (timestamp + ("_" + args.run_name if args.run_name else ""))
    agent_cfg.run_name = args.run_name
    env_cfg.log_dir = str(log_dir)
    print(f"[INFO] Logs -> {log_dir}")
    if args.resume and not args.resume.expanduser().is_file():
        raise FileNotFoundError(f"Checkpoint bulunamadı: {args.resume}")
    write_metadata(log_dir, env_cfg, agent_cfg, rsl_version)

    env = None
    try:
        env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array" if args.video else None)
        if args.video:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(log_dir / "videos" / "train"),
                step_trigger=lambda step: step % args.video_interval == 0,
                video_length=args.video_length,
                disable_logger=True,
            )
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        gains = check_actuator_gains(env)
        (log_dir / "params" / "actuator_gains.json").write_text(json.dumps(gains, indent=2), encoding="utf-8")
        sym = None
        if args.symmetry:
            robot = env.unwrapped.scene["robot"]
            sym = validate_go2_symmetry(
                robot.joint_names,
                robot.data.default_joint_pos[0].detach().cpu(),
                env.unwrapped.observation_manager.group_obs_dim["policy"][0],
                env.unwrapped.action_manager.total_action_dim,
            )
            print(f"[OK] Simetri ön kontrolü (robot): {sym}")
        if not args.no_finite_check:
            install_finite_check(env)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
        if args.symmetry:
            sym["rsl_rl_integration"] = check_symmetry_integration(runner, env)
            (log_dir / "params" / "symmetry_check.json").write_text(json.dumps(sym, indent=2), encoding="utf-8")
            print(f"[OK] Simetri RSL-RL entegrasyonu: {sym['rsl_rl_integration']}")
        checkpoint_iteration = None
        if args.resume:
            runner.load(str(args.resume.expanduser().resolve()))
            checkpoint_iteration = int(runner.current_learning_iteration)
        start_iteration, num_iterations = resume_plan(agent_cfg.max_iterations, checkpoint_iteration)
        runner.current_learning_iteration = start_iteration
        print(f"[INFO] Başlangıç iterasyonu {start_iteration}, bu oturumda {num_iterations} iterasyon, "
              f"son checkpoint model_{start_iteration + num_iterations - 1}.pt olacak")
        started = time.perf_counter()
        runner.learn(num_learning_iterations=num_iterations, init_at_random_ep_len=True)
        write_training_result(
            log_dir, env_cfg, agent_cfg, runner, start_iteration, num_iterations, time.perf_counter() - started
        )
        print(f"[DONE] Eğitim tamamlandı. Checkpoint: {log_dir}")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
