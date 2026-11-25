from __future__ import annotations

import torch

GAIN_TOLERANCE = 1.0e-3


def _range(values: torch.Tensor) -> tuple[float, float]:
    return float(values.min().item()), float(values.max().item())


def check_actuator_gains(env) -> dict:
    base = env.unwrapped
    robot = base.scene["robot"]
    event = getattr(base.cfg.events, "actuator_gains", None)
    k_scale = d_scale = (1.0, 1.0)
    if event is not None:
        if event.params.get("operation") != "scale":
            raise RuntimeError("check_actuator_gains yalnızca 'scale' randomization için yazıldı")
        k_scale = tuple(event.params.get("stiffness_distribution_params") or (1.0, 1.0))
        d_scale = tuple(event.params.get("damping_distribution_params") or (1.0, 1.0))
    report = {}
    for name, actuator in robot.actuators.items():
        joints = actuator.joint_indices
        stiffness, damping = actuator.stiffness, actuator.damping
        default_k = robot.data.default_joint_stiffness[:, joints]
        default_d = robot.data.default_joint_damping[:, joints]
        for label, values in (("stiffness", stiffness), ("damping", damping),
                              ("default stiffness", default_k), ("default damping", default_d)):
            if not torch.isfinite(values).all():
                raise RuntimeError(f"Aktüatör '{name}' {label}: NaN/Inf")
            if (values <= 0).any():
                raise RuntimeError(
                    f"Aktüatör '{name}' {label} sıfır veya negatif (aralık {_range(values)}). "
                    "Kazanç randomization'ı sıfır taban değerinden ölçekliyor olabilir (eski Isaac Lab "
                    "sürümlerindeki explicit-actuator hatası). Eğitim başlatılmadı; kurulu sürümü kontrol et "
                    "veya geçici olarak --no_gain_randomization kullan."
                )
        for label, values, default, (low, high) in (
            ("stiffness", stiffness, default_k, k_scale), ("damping", damping, default_d, d_scale),
        ):
            ratio = values / default
            r_min, r_max = _range(ratio)
            if r_min < low - GAIN_TOLERANCE or r_max > high + GAIN_TOLERANCE:
                raise RuntimeError(
                    f"Aktüatör '{name}' {label} oranı [{r_min:.3f}, {r_max:.3f}], beklenen [{low}, {high}]"
                )
        report[name] = {
            "num_joints": int(stiffness.shape[1]),
            "stiffness_min_max": [round(v, 4) for v in _range(stiffness)],
            "damping_min_max": [round(v, 4) for v in _range(damping)],
            "configured_stiffness_min_max": [round(v, 4) for v in _range(default_k)],
            "configured_damping_min_max": [round(v, 4) for v in _range(default_d)],
            "randomized": event is not None,
        }
        print(f"[CHECK] Aktüatör '{name}': stiffness {report[name]['stiffness_min_max']} "
              f"(yapılandırılan {report[name]['configured_stiffness_min_max']}), "
              f"damping {report[name]['damping_min_max']} "
              f"(yapılandırılan {report[name]['configured_damping_min_max']})")
    if not report:
        raise RuntimeError("Robotta aktüatör bulunamadı")
    return report


def assert_finite(name: str, value) -> None:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and not torch.isfinite(value).all():
            bad = int((~torch.isfinite(value)).sum().item())
            raise FloatingPointError(f"{name}: NaN/Inf ({bad} / {value.numel()} eleman)")
        return
    if hasattr(value, "items"):
        for key, item in value.items():
            assert_finite(f"{name}[{key}]", item)


def install_finite_check(env) -> None:
    original_step = env.step
    counter = {"steps": 0}

    def checked_step(actions):
        assert_finite(f"policy step {counter['steps']}: actions", actions)
        result = original_step(actions)
        assert_finite(f"policy step {counter['steps']}: observations", result[0])
        assert_finite(f"policy step {counter['steps']}: rewards", result[1])
        counter["steps"] += 1
        return result

    env.step = checked_step
