"""Go2 hız takibi ortamı."""

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from reward_math import current_contact_slip, swing_clearance_reward, touchdown_swing_reward


# Ödül fonksiyonları
def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg,
    min_s: float = 0.12, max_s: float = 0.30,
) -> torch.Tensor:
    """Temasta verilen 0,12–0,30 s hava süresi ödülü."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = touchdown_swing_reward(first_contact, last_air_time, min_s=min_s, max_s=max_s)
    cmd = env.command_manager.get_command(command_name)
    moving = (torch.norm(cmd[:, :2], dim=1) > 0.1) | (torch.abs(cmd[:, 2]) > 0.1)
    return reward * moving


def stand_still(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, cmd_threshold: float = 0.1
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    dev = torch.sum(
        torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]),
        dim=1,
    )
    cmd = env.command_manager.get_command(command_name)
    is_still = (torch.norm(cmd[:, :2], dim=1) < cmd_threshold) & (torch.abs(cmd[:, 2]) < cmd_threshold)
    return dev * is_still


def swing_clearance(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_z: float, std: float, tanh_mult: float
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    feet_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return swing_clearance_reward(foot_z, feet_vel_xy, target_z=target_z, std=std, tanh_mult=tanh_mult)


def foot_slip(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_w = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    asset: Articulation = env.scene[asset_cfg.name]
    feet_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return current_contact_slip(forces_w, feet_vel_xy)


# Sahne
@configclass
class Go2SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )


# Komutlar
@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.10,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
        ),
    )


# Aksiyonlar
@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


# Gözlemler
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# Ödüller
FEET = ".*_foot"
# Sensör ve robot için ortak ayak sırası.
FEET_ORDERED = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
SWING_TARGET_Z_M = 0.08      # Yerdeki referans ~0.023 m.
SWING_CLEARANCE_STD = 0.01
SWING_TANH_MULT = 2.0

FEET_AIR_TIME_MIN_S = 0.12
FEET_AIR_TIME_MAX_S = 0.30

@configclass
class RewardsCfg:
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    feet_air_time = RewTerm(
        func=feet_air_time, weight=0.2,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET),
            "min_s": FEET_AIR_TIME_MIN_S,
            "max_s": FEET_AIR_TIME_MAX_S,
        },
    )

    foot_slip = RewTerm(
        func=foot_slip, weight=-0.08,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_ORDERED, preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_ORDERED, preserve_order=True),
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts, weight=-0.6,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"), "threshold": 1.0},
    )

    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)

    stand_still = RewTerm(
        func=stand_still, weight=-0.5,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    )
    swing_clearance = RewTerm(
        func=swing_clearance, weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET),
            "target_z": SWING_TARGET_Z_M, "std": SWING_CLEARANCE_STD, "tanh_mult": SWING_TANH_MULT,
        },
    )


# Sonlandırma
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


# Reset ve randomizasyon
@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material, mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass, mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains, mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform, mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")},
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale, mode="reset",
        params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity, mode="interval", interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


# Ortam
@configclass
class Go2FlatEnvCfg(ManagerBasedRLEnvCfg):
    scene: Go2SceneCfg = Go2SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        # Fizik 200 Hz, politika 50 Hz.
        self.sim.dt = 0.005
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # Temas sensörü: 200 Hz.
        self.scene.contact_forces.update_period = self.sim.dt

        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.eye = (2.0, 2.0, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)


@configclass
class Go2FlatEnvCfg_PLAY(Go2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.actuator_gains = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
        r = self.commands.base_velocity.ranges
        r.lin_vel_x, r.lin_vel_y, r.ang_vel_z = (0.8, 0.8), (0.0, 0.0), (0.3, 0.3)

    def enable_robustness_test(self):
        training_events = EventCfg()
        self.events.physics_material = training_events.physics_material
        self.events.add_base_mass = training_events.add_base_mass
        self.events.actuator_gains = training_events.actuator_gains
        self.events.push_robot = training_events.push_robot
        self.observations.policy.enable_corruption = True

    def set_fixed_command(self, vx: float, vy: float, wz: float):
        r = self.commands.base_velocity.ranges
        r.lin_vel_x, r.lin_vel_y, r.ang_vel_z = (vx, vx), (vy, vy), (wz, wz)
