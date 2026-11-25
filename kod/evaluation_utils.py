from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import fmean, pstdev

# Komut = (ileri m/s, yanal m/s, yaw rad/s)
SCENARIOS = {
    "forward": (0.8, 0.0, 0.0),
    "backward": (-0.4, 0.0, 0.0),
    "strafe": (0.0, 0.5, 0.0),
    "turn": (0.0, 0.0, 0.7),
    "stand": (0.0, 0.0, 0.0),
}
SCENARIOS_TURN_SWEEP = {
    **SCENARIOS,
    "turn_negative": (0.0, 0.0, -0.7),
    "turn_pos_0.6": (0.0, 0.0, 0.6),
    "turn_neg_0.6": (0.0, 0.0, -0.6),
    "turn_pos_0.5": (0.0, 0.0, 0.5),
    "turn_neg_0.5": (0.0, 0.0, -0.5),
}
CSV_FIELDS = (
    "scenario", "step", "time_s", "cmd_vx", "cmd_vy", "cmd_wz",
    "vx", "vy", "wz", "linear_error_mps", "yaw_error_radps",
    "linear_error_sq_mean", "yaw_error_sq_mean",
    "valid_env_count", "linear_error_sq_valid_sum", "yaw_error_sq_valid_sum",
    "planar_speed_mps", "base_height_m", "slip_proxy_mps", "abs_mech_power_proxy_w",
    "terminated_count", "timeout_count",
)
CONTACT_PREFIX = "contact_"
GRAVITY_MPS2 = 9.81
MIN_SPEED_FOR_COT_MPS = 0.1
IDENTICAL_ENV_RMSE_STD = 1.0e-3


def summarize_scenario(
    rows: list[dict], *, num_envs: int, air_times: dict[str, list[float]],
    duty_factor: dict[str, float] | None = None, robot_mass_kg: float | None = None,
) -> dict:
    if not rows:
        raise ValueError("boş test verisinden metrik üretilemez")
    if num_envs <= 0:
        raise ValueError("num_envs pozitif olmalı")
    mean_air = {
        foot: (round(fmean(times), 4) if times else None)
        for foot, times in air_times.items()
    }
    # RMSE: önce her ortamın hatasının karesi.
    linear_sq = [
        r.get("linear_error_sq_mean", (r["vx"] - r["cmd_vx"]) ** 2 + (r["vy"] - r["cmd_vy"]) ** 2)
        for r in rows
    ]
    yaw_sq = [r.get("yaw_error_sq_mean", (r["wz"] - r["cmd_wz"]) ** 2) for r in rows]
    valid_samples = sum(int(r.get("valid_env_count", 0)) for r in rows)
    if all("valid_env_count" in r for r in rows):
        weights = [int(r["valid_env_count"]) for r in rows]
    else:
        weights = [1] * len(rows)
    total_weight = sum(weights)

    def state_mean(getter):
        if total_weight == 0:
            return None
        return sum(getter(r) * w for r, w in zip(rows, weights)) / total_weight

    def rounded(value, digits=4):
        return None if value is None else round(value, digits)

    mean_power = state_mean(lambda r: r["abs_mech_power_proxy_w"])
    mean_speed = state_mean(lambda r: r.get("planar_speed_mps", math.hypot(r["vx"], r["vy"])))
    xy_valid = yaw_valid = None
    if valid_samples > 0:
        xy_valid = round(math.sqrt(sum(r.get("linear_error_sq_valid_sum", 0.0) for r in rows) / valid_samples), 4)
        yaw_valid = round(math.sqrt(sum(r.get("yaw_error_sq_valid_sum", 0.0) for r in rows) / valid_samples), 4)
    cost_of_transport = None
    if robot_mass_kg is not None:
        if robot_mass_kg <= 0:
            raise ValueError("robot_mass_kg pozitif olmalı")
        if mean_speed is not None and mean_speed >= MIN_SPEED_FOR_COT_MPS:
            cost_of_transport = round(mean_power / (robot_mass_kg * GRAVITY_MPS2 * mean_speed), 4)
    return {
        "scenario": rows[0]["scenario"],
        "observed_policy_steps": len(rows),
        "num_envs": num_envs,
        "duration_after_warmup_s": round(rows[-1]["time_s"] - rows[0]["time_s"], 3),
        "linear_xy_rmse_valid_mps": xy_valid,
        "yaw_rmse_valid_radps": yaw_valid,
        "valid_samples": valid_samples,
        "valid_sample_fraction": round(valid_samples / (num_envs * len(rows)), 4),
        "linear_xy_rmse_mps": round(math.sqrt(fmean(linear_sq)), 4),
        "linear_xy_mean_abs_error_mps": rounded(state_mean(lambda r: r["linear_error_mps"])),
        "yaw_rmse_radps": round(math.sqrt(fmean(yaw_sq)), 4),
        "yaw_mean_abs_error_radps": rounded(state_mean(lambda r: r["yaw_error_radps"])),
        "mean_height_m": rounded(state_mean(lambda r: r["base_height_m"])),
        "mean_slip_proxy_mps": rounded(state_mean(lambda r: r["slip_proxy_mps"])),
        "mean_abs_mech_power_proxy_w": rounded(mean_power),
        "mean_planar_speed_mps": rounded(mean_speed),
        "robot_mean_mass_kg": round(robot_mass_kg, 3) if robot_mass_kg is not None else None,
        "mech_cost_of_transport_proxy": cost_of_transport,
        "duty_factor_by_foot": (
            {foot: round(value, 4) for foot, value in duty_factor.items()} if duty_factor is not None else None
        ),
        "fall_terminations_after_warmup": int(sum(r["terminated_count"] for r in rows)),
        "timeouts_after_warmup": int(sum(r["timeout_count"] for r in rows)),
        "first_contact_air_time_s_by_foot": mean_air,
        "first_contacts_by_foot": {foot: len(times) for foot, times in air_times.items()},
    }


def summarize_environment_errors(
    sum_xy_sq: list[float], sum_yaw_sq: list[float], fall_counts: list[int], samples: int | list[int],
    fall_counts_after_warmup: list[int] | None = None,
) -> list[dict]:
    counts = [samples] * len(sum_xy_sq) if isinstance(samples, int) else list(samples)
    late = [None] * len(sum_xy_sq) if fall_counts_after_warmup is None else list(fall_counts_after_warmup)
    if not (len(sum_xy_sq) == len(sum_yaw_sq) == len(fall_counts) == len(counts) == len(late)):
        raise ValueError("Geçersiz per-env sayaçları")
    if any(l is not None and l > f for l, f in zip(late, fall_counts)):
        raise ValueError("warmup sonrası düşme sayısı toplamdan büyük olamaz")
    if any(c < 0 for c in counts) or (isinstance(samples, int) and samples <= 0):
        raise ValueError("Geçersiz per-env sayaçları")
    return [
        {"env_index": i,
         "linear_xy_rmse_mps": round(math.sqrt(max(x, 0.0) / n), 4) if n else None,
         "yaw_rmse_radps": round(math.sqrt(max(y, 0.0) / n), 4) if n else None,
         "valid_samples": int(n),
         "fall_terminations": int(falls),
         "fall_terminations_after_warmup": None if l is None else int(l)}
        for i, (x, y, falls, n, l) in enumerate(zip(sum_xy_sq, sum_yaw_sq, fall_counts, counts, late))
    ]


def _spread(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {"mean": round(fmean(values), 4), "std": round(pstdev(values), 4),
            "min": round(min(values), 4), "max": round(max(values), 4)}


def summarize_across_environments(per_environment: list[dict]) -> dict:
    if not per_environment:
        raise ValueError("per_environment boş")
    xy = [e["linear_xy_rmse_mps"] for e in per_environment if e["linear_xy_rmse_mps"] is not None]
    yaw = [e["yaw_rmse_radps"] for e in per_environment if e["yaw_rmse_radps"] is not None]
    survived = sum(1 for e in per_environment if e["fall_terminations"] == 0)
    late = [e.get("fall_terminations_after_warmup") for e in per_environment]
    survived_late = None if None in late else sum(1 for value in late if value == 0)
    xy_spread, yaw_spread = _spread(xy), _spread(yaw)
    return {
        "num_envs": len(per_environment),
        "envs_without_fall": survived,
        "survival_rate": round(survived / len(per_environment), 4),
        "envs_without_fall_after_warmup": survived_late,
        "survival_rate_after_warmup": (
            None if survived_late is None else round(survived_late / len(per_environment), 4)
        ),
        "envs_without_valid_samples": len(per_environment) - len(xy),
        "linear_xy_rmse_mps": xy_spread,
        "yaw_rmse_radps": yaw_spread,
        "environments_nearly_identical": bool(
            len(xy) > 1 and xy_spread["std"] < IDENTICAL_ENV_RMSE_STD and yaw_spread["std"] < IDENTICAL_ENV_RMSE_STD
        ),
    }


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = [key for key in (rows[0] if rows else {}) if key.startswith(CONTACT_PREFIX)]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*CSV_FIELDS, *extra])
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def command_for_demo(step: int, total: int) -> tuple[float, float, float]:
    if total <= 0 or not 0 <= step < total:
        raise ValueError("step aralığın dışında")
    fraction = step / total
    if fraction < 0.25:
        return SCENARIOS["forward"]
    if fraction < 0.50:
        return SCENARIOS["strafe"]
    if fraction < 0.75:
        return SCENARIOS["turn"]
    return SCENARIOS["stand"]
