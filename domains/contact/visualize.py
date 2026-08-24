# domains/contact/visualize.py
"""Rendering for the contact domain, deliberately implementation-agnostic:
every function here takes a Snapshot (or a sequence of them), never a raw
state vector, a pymunk object, or a PlanarFingertipParams. If the underlying
sim is ever swapped (see docs/stage1_env_spec.md's Units section on why that
might happen), only the adapter that builds Snapshots needs to change --
domains/contact/physics.py's to_snapshot() is that adapter today. Nothing
here changes.

Local disk only. Per project convention (limited wandb storage), nothing in
this module uploads anything anywhere -- no wandb import at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]


@dataclass(frozen=True)
class Snapshot:
    """One rendered instant. Everything a renderer needs to draw a frame,
    and nothing it doesn't -- no sim internals, no engine handles.

    `walls` is a generic list of line segments (any 2D sim can produce
    these, not just PyMunk), not a list of Portals -- this module has no
    idea what a Portal is, and shouldn't need to.
    """

    board_w_cm: float
    board_h_cm: float
    object_xy: Point
    object_angle_rad: float
    object_w_cm: float
    object_h_cm: float
    fingers: Dict[str, Point]          # e.g. {"L": (x, y), "R": (x, y)}
    finger_radius_cm: float
    touching: Dict[str, bool]          # same keys as fingers
    walls: Sequence[Tuple[Point, Point]] = ()


_FINGER_COLOR = {True: "#2e7d32", False: "#9e9e9e"}   # touching / not
_OBJECT_COLOR = "#d9a066"


def _rect_corners(cx: float, cy: float, w: float, h: float, theta: float) -> np.ndarray:
    """Four corners of a w x h rectangle centered at (cx, cy), rotated by
    theta radians. Computed by hand rather than via a patch's rotation-point
    API, whose anchor semantics differ across matplotlib versions."""
    hw, hh = w / 2.0, h / 2.0
    local = np.array([(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)])
    c, s = np.cos(theta), np.sin(theta)
    rot = local @ np.array([[c, -s], [s, c]]).T
    return rot + np.array([cx, cy])


def _draw_walls(ax, walls: Sequence[Tuple[Point, Point]], board_w: float, board_h: float) -> None:
    """Every wall segment, generic line data -- no Portal, no engine object.
    Falls back to a plain board outline if a Snapshot carries no wall data
    (e.g. one built before this field existed), so old callers don't break."""
    from matplotlib.patches import Polygon

    if not walls:
        ax.add_patch(Polygon([(0, 0), (board_w, 0), (board_w, board_h), (0, board_h)],
                             closed=True, fill=False, edgecolor="black", linewidth=1.5))
        return
    for (ax0, ay0), (ax1, ay1) in walls:
        ax.plot([ax0, ax1], [ay0, ay1], color="black", linewidth=3.0, solid_capstyle="butt")


def plot_snapshot(snap: Snapshot, ax=None):
    """One frame: walls, the object, both fingertips (colored by contact
    state). Returns the axes so callers can compose or save it."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon

    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 6.0 * snap.board_h_cm / snap.board_w_cm))

    _draw_walls(ax, snap.walls, snap.board_w_cm, snap.board_h_cm)

    corners = _rect_corners(snap.object_xy[0], snap.object_xy[1],
                            snap.object_w_cm, snap.object_h_cm, snap.object_angle_rad)
    ax.add_patch(Polygon(corners, closed=True, facecolor=_OBJECT_COLOR,
                         edgecolor="black", linewidth=1.0, zorder=3))

    for name, (fx, fy) in snap.fingers.items():
        touching = snap.touching.get(name, False)
        ax.add_patch(Circle((fx, fy), snap.finger_radius_cm,
                            facecolor=_FINGER_COLOR[touching],
                            edgecolor="black", linewidth=0.8, zorder=4))
        ax.annotate(name, (fx, fy), ha="center", va="center",
                   fontsize=8, color="white", zorder=5)

    ax.set_xlim(-2, snap.board_w_cm + 2)
    ax.set_ylim(-2, snap.board_h_cm + 2)
    ax.set_aspect("equal")
    return ax


def plot_trajectory(snapshots: Sequence[Snapshot], ax=None):
    """One static figure summarizing a whole rollout: board, start/end object
    poses, full path traces for the object and each fingertip, and markers
    where a fingertip's contact state changed. This is the plot to eyeball
    first -- it answers "did anything reach or drift oddly" in one image."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    if not snapshots:
        raise ValueError("plot_trajectory needs at least one snapshot")

    first = snapshots[0]
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 6.5 * first.board_h_cm / first.board_w_cm))

    _draw_walls(ax, first.walls, first.board_w_cm, first.board_h_cm)

    for snap, alpha in ((snapshots[0], 0.25), (snapshots[-1], 0.9)):
        corners = _rect_corners(snap.object_xy[0], snap.object_xy[1],
                                snap.object_w_cm, snap.object_h_cm, snap.object_angle_rad)
        ax.add_patch(Polygon(corners, closed=True, facecolor=_OBJECT_COLOR,
                             edgecolor="black", alpha=alpha, zorder=3))

    obj_path = np.array([s.object_xy for s in snapshots])
    ax.plot(obj_path[:, 0], obj_path[:, 1], color="black", linewidth=1.2,
           label="object", zorder=2)

    finger_names = list(first.fingers)
    colors = plt.cm.tab10.colors
    for i, name in enumerate(finger_names):
        path = np.array([s.fingers[name] for s in snapshots])
        ax.plot(path[:, 0], path[:, 1], color=colors[i % len(colors)],
               linewidth=1.0, linestyle="--", label=f"finger {name}", zorder=2)

        touching = [s.touching.get(name, False) for s in snapshots]
        for t in range(1, len(touching)):
            if touching[t] != touching[t - 1]:
                marker = "^" if touching[t] else "v"
                ax.plot(*path[t], marker=marker, color=colors[i % len(colors)],
                       markersize=8, zorder=6)

    ax.set_xlim(-2, first.board_w_cm + 2)
    ax.set_ylim(-2, first.board_h_cm + 2)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    return ax


def save_video(snapshots: Sequence[Snapshot], path: str, *,
              fps: float = 25.0, dpi: int = 100) -> str:
    """Renders every snapshot as a frame and writes an .mp4 to `path`, local
    disk only. Never call this on a path under anything that syncs to wandb
    or another remote store -- see the module docstring."""
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt

    if not snapshots:
        raise ValueError("save_video needs at least one snapshot")

    frames = []
    first = snapshots[0]
    figsize = (6.0, 6.0 * first.board_h_cm / first.board_w_cm)
    for tick, snap in enumerate(snapshots):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        plot_snapshot(snap, ax=ax)
        ax.set_title(f"tick {tick}", fontsize=9)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(frame)
        plt.close(fig)

    # macro_block_size=None: our frame size need not be a multiple of 16.
    with imageio.get_writer(path, fps=fps, macro_block_size=None) as writer:
        for frame in frames:
            writer.append_data(frame)
    return path
