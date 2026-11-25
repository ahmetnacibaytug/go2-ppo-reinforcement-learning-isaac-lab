from importlib import metadata

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class Go2FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 300
    save_interval = 25
    experiment_name = "go2_flat_custom"
    logger = "tensorboard"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = None

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


def enable_lr_symmetry(agent_cfg: Go2FlatPPORunnerCfg) -> Go2FlatPPORunnerCfg:
    from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

    from go2_symmetry import compute_symmetric_states

    agent_cfg.algorithm.symmetry_cfg = RslRlSymmetryCfg(
        use_data_augmentation=True,
        use_mirror_loss=False,
        data_augmentation_func=compute_symmetric_states,
    )
    return agent_cfg


def validate_rsl_rl_version():
    from packaging.version import Version

    try:
        current = metadata.version("rsl-rl-lib")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("rsl-rl-lib kurulu değil. Isaac Lab v2.3.2 RL kurulumunu tamamla.") from exc
    parsed = Version(current)
    if not Version("3.0.1") <= parsed < Version("4.0.0"):
        raise RuntimeError(f"RSL-RL {current} bulundu; 3.x sürümü gerekiyor.")
    print(f"[INFO] RSL-RL {current}")
    return current
