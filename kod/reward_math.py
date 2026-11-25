"""Ödül hesapları."""

import torch


def touchdown_swing_reward(first_contact: torch.Tensor, last_air_time: torch.Tensor,
                           min_s: float = 0.12, max_s: float = 0.30) -> torch.Tensor:
    if not 0 < min_s < max_s:
        raise ValueError("0 < min_s < max_s olmalı")
    if first_contact.shape != last_air_time.shape or last_air_time.ndim != 2:
        raise ValueError("(num_envs, num_feet) boyutlu iki tensör bekleniyor")
    bounded = torch.clamp(last_air_time - min_s, min=0.0, max=max_s - min_s)
    return (bounded * first_contact).sum(dim=1)


def current_contact_slip(forces_w: torch.Tensor, feet_vel_xy: torch.Tensor,
                         threshold_n: float = 1.0) -> torch.Tensor:
    if (forces_w.ndim != 3 or feet_vel_xy.ndim != 3 or
            forces_w.shape[:2] != feet_vel_xy.shape[:2] or
            forces_w.shape[-1] != 3 or feet_vel_xy.shape[-1] != 2):
        raise ValueError("(N, F, 3) kuvvet ve (N, F, 2) ayak hızı bekleniyor")
    current_contact = forces_w[..., 2].abs() > threshold_n
    return (torch.linalg.vector_norm(feet_vel_xy, dim=-1) * current_contact).sum(dim=1)


def swing_clearance_reward(foot_z: torch.Tensor, feet_vel_xy: torch.Tensor, target_z: float = 0.08,
                           std: float = 0.01, tanh_mult: float = 2.0) -> torch.Tensor:
    if std <= 0 or tanh_mult <= 0:
        raise ValueError("std ve tanh_mult pozitif olmalı")
    if foot_z.ndim != 2 or feet_vel_xy.ndim != 3 or feet_vel_xy.shape[:2] != foot_z.shape or feet_vel_xy.shape[-1] != 2:
        raise ValueError("foot_z (N, F) ve feet_vel_xy (N, F, 2) bekleniyor")
    speed_gate = torch.tanh(tanh_mult * torch.linalg.vector_norm(feet_vel_xy, dim=-1))
    error = (foot_z - target_z).square() * speed_gate
    return torch.exp(-error.sum(dim=1) / std)
