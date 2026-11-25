from __future__ import annotations

from pathlib import Path


def save_trajectory(path: Path, *, positions, root_positions, root_quaternions,
                    commands, velocities, sample_steps, body_names, step_dt: float) -> Path:
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not positions:
        raise ValueError("GIF için kaydedilmiş kare yok")
    np.savez_compressed(
        path, body_positions=np.stack(positions).astype("float32"),
        root_positions=np.stack(root_positions).astype("float32"),
        root_quaternions=np.stack(root_quaternions).astype("float32"),
        commands=np.stack(commands).astype("float32"),
        velocities=np.stack(velocities).astype("float32"),
        sample_steps=np.array(sample_steps, dtype="int32"),
        body_names=np.asarray(body_names, dtype=str),
        step_dt=np.asarray([step_dt], dtype="float64"),
    )
    return path


def _rotation_wxyz(q):
    import numpy as np
    w, x, y, z = np.asarray(q, dtype=float)
    norm = max(float(np.linalg.norm([w, x, y, z])), 1.e-12)
    w, x, y, z = np.array([w, x, y, z]) / norm
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def render_trajectory(path: Path, *, fps: int = 12, make_mp4: bool = False) -> tuple[Path, Path | None]:
    import shutil
    import numpy as np
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

    if not (1 <= fps <= 30):
        raise ValueError("fps 1..30 olmalı")
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        positions = data["body_positions"]
        roots = data["root_positions"]
        quats = data["root_quaternions"]
        commands = data["commands"]
        velocities = data["velocities"]
        steps = data["sample_steps"]
        names = [str(s) for s in data["body_names"].tolist()]
        step_dt = float(data["step_dt"][0])
    if positions.ndim != 3 or positions.shape[2] != 3 or len(positions) == 0:
        raise ValueError("Geçersiz kayıt boyutu")
    if len(names) != positions.shape[1]:
        raise ValueError("Gövde adı ve konum sayısı uyuşmuyor")
    name_to_index = {name.lower(): index for index, name in enumerate(names)}
    legs = ("FL", "FR", "RL", "RR")
    color_for_leg = {"FL": "#0796a8", "FR": "#2b62d6", "RL": "#e39d35", "RR": "#a650be"}

    fig = plt.figure(figsize=(8, 5.5), dpi=100, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(left=0.02, right=0.98, top=0.87, bottom=0.04)
    fig.suptitle("Unitree Go2 | Simulated motion (kinematic reconstruction)", fontsize=12, weight="bold")
    body_box = np.array([
        [-.23, -.13, -.07], [.23, -.13, -.07], [.23, .13, -.07], [-.23, .13, -.07],
        [-.23, -.13, .07], [.23, -.13, .07], [.23, .13, .07], [-.23, .13, .07],
    ])
    edges = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))

    def update(frame):
        ax.cla()
        root = roots[frame]
        R = _rotation_wxyz(quats[frame])
        corners = body_box @ R.T + root
        for a,b in edges:
            p = corners[[a,b]]
            ax.plot(p[:,0],p[:,1],p[:,2], color="#202c38", linewidth=2.0)
        ax.scatter(root[0],root[1],root[2], s=15, color="#202c38")
        for leg in legs:
            points = [root]
            for suffix in ("hip", "thigh", "calf", "foot"):
                index = name_to_index.get(f"{leg}_{suffix}".lower())
                if index is not None:
                    points.append(positions[frame, index])
            if len(points) >= 2:
                pts = np.asarray(points)
                ax.plot(pts[:,0],pts[:,1],pts[:,2], '-o', linewidth=2.3,
                        markersize=3.5, color=color_for_leg[leg])
        x, y, z = root
        for delta in np.arange(-0.8, 1.0, .4):
            ax.plot([x-.85,x+.85],[y+delta,y+delta],[0,0], color="#dfebef", linewidth=.7)
            ax.plot([x+delta,x+delta],[y-.85,y+.85],[0,0], color="#dfebef", linewidth=.7)
        past = roots[max(0,frame-24):frame+1]
        if len(past) > 1:
            ax.plot(past[:,0],past[:,1],np.maximum(past[:,2]*0,0.015),
                    color="#4b788c", linewidth=1.5, alpha=.65)
        ax.set_xlim(x-.85,x+.85); ax.set_ylim(y-.85,y+.85)
        ax.set_zlim(-.07,max(.85,z+.4))
        ax.set_box_aspect((1.7,1.7,.92))
        ax.view_init(elev=23,azim=-58)
        ax.set_xlabel("x (m)",fontsize=8); ax.set_ylabel("y (m)",fontsize=8); ax.set_zlabel("z (m)",fontsize=8)
        ax.tick_params(labelsize=7)
        cmd = commands[frame]; vel = velocities[frame]
        ax.set_title(f"t = {steps[frame]*step_dt:5.2f}s   |   cmd (vx, vy, yaw) = ({cmd[0]:.2f}, {cmd[1]:.2f}, {cmd[2]:.2f})\n"
                     f"measured (vx, vy, yaw) = ({vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f})",fontsize=9,pad=8)
        return ()

    animation = FuncAnimation(fig, update, frames=len(positions), interval=1000/fps,
                              blit=False, repeat=False)
    gif_path = path.parent / "go2_kinematic.gif"
    mp4_path = None
    try:
        animation.save(str(gif_path), writer=PillowWriter(fps=fps))
        if make_mp4 and shutil.which("ffmpeg"):
            mp4_path = path.parent / "go2_kinematic.mp4"
            animation.save(str(mp4_path), writer=FFMpegWriter(fps=fps, bitrate=1600))
    finally:
        plt.close(fig)
    if not gif_path.is_file() or gif_path.stat().st_size == 0:
        raise RuntimeError("GIF oluşturulamadı")
    return gif_path, mp4_path
