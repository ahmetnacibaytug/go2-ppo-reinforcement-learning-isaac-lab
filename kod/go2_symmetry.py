"""sol-sağ veri artırımı - Isaac Lab ANYmal örneği"""

from __future__ import annotations

import torch

GO2_JOINT_ORDER = (
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
)
POLICY_OBS_DIM = 48

# FL ↔ FR, RL ↔ RR
_LR_PERM = (1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10)
# Kalça işaretleri terslenir.
_LR_SIGN = (-1.0,) * 4 + (1.0,) * 8

_LIN_SIGN = (1.0, -1.0, 1.0)
_ANG_SIGN = (-1.0, 1.0, -1.0)
_CMD_SIGN = (1.0, -1.0, -1.0)


def _const(values, like: torch.Tensor) -> torch.Tensor:
    return torch.tensor(values, dtype=like.dtype, device=like.device)


def mirror_joints_lr(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] != 12:
        raise ValueError(f"12 eklem bekleniyordu, gelen boyut {x.shape[-1]}")
    perm = torch.tensor(_LR_PERM, device=x.device)
    return x.index_select(-1, perm) * _const(_LR_SIGN, x)


def mirror_policy_obs_lr(obs: torch.Tensor) -> torch.Tensor:
    if obs.shape[-1] != POLICY_OBS_DIM:
        raise ValueError(f"Policy gözlemi {POLICY_OBS_DIM} boyut olmalı, gelen {obs.shape[-1]} "
                         "(gözlem terimleri değiştiyse simetri eşlemesini güncelle)")
    out = obs.clone()
    out[..., 0:3] = obs[..., 0:3] * _const(_LIN_SIGN, obs)
    out[..., 3:6] = obs[..., 3:6] * _const(_ANG_SIGN, obs)
    out[..., 6:9] = obs[..., 6:9] * _const(_LIN_SIGN, obs)
    out[..., 9:12] = obs[..., 9:12] * _const(_CMD_SIGN, obs)
    out[..., 12:24] = mirror_joints_lr(obs[..., 12:24])
    out[..., 24:36] = mirror_joints_lr(obs[..., 24:36])
    out[..., 36:48] = mirror_joints_lr(obs[..., 36:48])
    return out


def mirror_actions_lr(actions: torch.Tensor) -> torch.Tensor:
    return mirror_joints_lr(actions)


@torch.no_grad()
def compute_symmetric_states(env=None, obs=None, actions=None):
    obs_aug = None
    if obs is not None:
        n = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][:n] = obs["policy"][:]
        obs_aug["policy"][n:] = mirror_policy_obs_lr(obs["policy"])
    actions_aug = None
    if actions is not None:
        actions_aug = torch.cat((actions, mirror_actions_lr(actions)), dim=0)
    return obs_aug, actions_aug


def validate_go2_symmetry(joint_names, default_joint_pos, policy_obs_dim, action_dim, atol: float = 1e-4) -> dict:
    names = tuple(joint_names)
    if names != GO2_JOINT_ORDER:
        raise RuntimeError(f"Eklem sırası simetri varsayımıyla uyuşmuyor:\n  beklenen {GO2_JOINT_ORDER}\n  gelen    {names}")
    if policy_obs_dim != POLICY_OBS_DIM:
        raise RuntimeError(f"Policy gözlem boyutu {policy_obs_dim}, simetri {POLICY_OBS_DIM} bekliyor")
    if action_dim != 12:
        raise RuntimeError(f"Aksiyon boyutu {action_dim}, simetri 12 bekliyor")
    default = torch.as_tensor(default_joint_pos, dtype=torch.float32).flatten()
    mirrored = mirror_joints_lr(default)
    if not torch.allclose(mirrored, default, atol=atol):
        raise RuntimeError("Varsayılan duruş sol-sağ simetrik değil (kalça L = -R olmalı); "
                           f"default={default.tolist()} aynalanmış={mirrored.tolist()}")
    return {"joint_order_ok": True, "policy_obs_dim": policy_obs_dim, "action_dim": action_dim,
            "default_pose_mirror_max_abs_err": float((mirrored - default).abs().max())}
