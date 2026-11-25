import argparse
import csv
from collections import defaultdict
from pathlib import Path


def load_rows(csv_file: Path):
    scenarios = defaultdict(list)
    with csv_file.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            record = {k: (v if k == "scenario" else float(v)) for k, v in row.items() if v not in ("", None)}
            scenarios[record["scenario"]].append(record)
    if not scenarios:
        raise ValueError(f"CSV boş: {csv_file}")
    return dict(scenarios)


def plot_timeseries(csv_file: Path, output_file: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load_rows(csv_file)
    fig, axes = plt.subplots(3, len(data), figsize=(max(7, 4.0 * len(data)), 9), squeeze=False)
    components = (("vx", "cmd_vx", "Vx (m/s)"), ("vy", "cmd_vy", "Vy (m/s)"),
                  ("wz", "cmd_wz", "Yaw rate (rad/s)"))
    for col, (name, rows) in enumerate(data.items()):
        time = [row["time_s"] - rows[0]["time_s"] for row in rows]
        for line, (actual, desired, label) in enumerate(components):
            ax = axes[line][col]
            ax.plot(time, [r[actual] for r in rows], label="Actual", lw=1.4)
            ax.plot(time, [r[desired] for r in rows], label="Command", ls="--", lw=1.1)
            ax.set_ylabel(label)
            ax.grid(alpha=0.25)
            if line == 0:
                ax.set_title(name.title())
            if line == 2:
                ax.set_xlabel("Time after warm-up (s)")
            if col == 0 and line == 0:
                ax.legend(loc="best")
    fig.suptitle("Unitree Go2 — Velocity Tracking Evaluation", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_file


def stance_intervals(times, flags, step):
    bars, start, previous = [], None, None
    for time, flag in zip(times, flags):
        if start is not None and (not flag or time - previous > 1.5 * step):
            bars.append((start, previous - start + step))
            start = None
        if flag and start is None:
            start = time
        previous = time
    if start is not None:
        bars.append((start, previous - start + step))
    return bars


def plot_gait(csv_file: Path, output_file: Path):
    data = load_rows(csv_file)
    first = next(iter(data.values()))[0]
    feet = [key[len("contact_"):] for key in first if key.startswith("contact_")]
    if not feet:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(data), figsize=(max(7, 4.0 * len(data)), 3.2), squeeze=False, sharey=True)
    for col, (name, rows) in enumerate(data.items()):
        ax = axes[0][col]
        time = [row["time_s"] - rows[0]["time_s"] for row in rows]
        step = (time[1] - time[0]) if len(time) > 1 else 0.02
        for line, foot in enumerate(feet):
            flags = [row.get(f"contact_{foot}", 0.0) > 0.5 for row in rows]
            ax.broken_barh(stance_intervals(time, flags, step), (line - 0.35, 0.7))
        ax.set_yticks(range(len(feet)))
        ax.set_yticklabels(feet)
        if col == 0:
            ax.invert_yaxis()
        ax.set_xlim(0, time[-1] + step)
        ax.set_title(name.title())
        ax.set_xlabel("Time after warm-up (s)")
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Unitree Go2 — Gait Diagram (bar = stance, environment 0)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    target = args.out or args.csv.with_name("velocity_tracking.png")
    print(plot_timeseries(args.csv, target))
    gait_file = plot_gait(args.csv, target.with_name("gait_diagram.png"))
    if gait_file is not None:
        print(gait_file)
