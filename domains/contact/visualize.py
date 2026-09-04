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
from typing import Dict, Optional, Sequence, Tuple

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

    # Task overlay. Optional with defaults so every pre-existing caller keeps
    # working; a Snapshot without them renders exactly as it used to.
    # goal_xy is the DESIRED OBJECT POSITION, not a finger target.
    goal_xy: Optional[Point] = None
    arrival_eps_cm: Optional[float] = None
    # Which fingertip the policy actually drives. The other one is not absent:
    # under mask_inactive_finger it is servo-HELD wherever it spawned, i.e. an
    # obstacle in the object's path (measured: forbidden_contact 8% -> 17-19%
    # once pushes got long enough to travel). Contact colour cannot show this,
    # so agency gets its own encoding.
    active_finger: Optional[str] = None
    inactive_masked: Optional[bool] = None
    # RECONTACT's goal is a FINGERTIP target, not an object position, so it
    # cannot ride in goal_xy. Both are WORLD-frame here -- recontact stores its
    # targets in the object's frame, and physics.to_snapshot does that transform
    # so this module stays frame-agnostic (see the module docstring).
    # Eq 13's interface names BOTH fingertips, hence a dict: a single marker
    # cannot show a two-finger goal, and the tolerances are deliberately
    # asymmetric (anchor 0.3cm, retracted 2.0cm), so each carries its own.
    finger_goals: Optional[Dict[str, Point]] = None
    finger_goal_tol_cm: Optional[Dict[str, float]] = None


_FINGER_COLOR = {True: "#2e7d32", False: "#9e9e9e"}   # touching / not
_OBJECT_COLOR = "#d9a066"
# Deep purple, NOT tab10[0]: the finger paths take tab10 in order, so a blue
# goal star was the same colour as finger L's contact markers and vanished into
# them. Object tan / walls black / fingers blue+orange / nearest red / goal
# purple are mutually distinguishable, which is the whole point of the overlay.
_GOAL_COLOR = "#6a1b9a"
_TRAIL_COLOR = "#333333"
_NEAREST_COLOR = "#c62828"


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


def goal_dist(snap: Snapshot) -> float:
    """Distance to the goal in whatever the goal IS: object-centre to target
    for push, worst fingertip-to-target for recontact. nan without either.

    Worst rather than mean, and rather than the active finger's: Eq 13's
    interface is a conjunction, so the binding constraint is the finger that is
    furthest from where it must end up.
    """
    if snap.goal_xy is not None:
        return float(np.hypot(snap.object_xy[0] - snap.goal_xy[0],
                              snap.object_xy[1] - snap.goal_xy[1]))
    if snap.finger_goals:
        return max(
            float(np.hypot(snap.fingers[k][0] - g[0], snap.fingers[k][1] - g[1]))
            for k, g in snap.finger_goals.items() if k in snap.fingers)
    return float("nan")


def nearest_index(snapshots: Sequence[Snapshot]) -> Optional[int]:
    """Tick of closest approach -- E3's statistic, and the one thing a final
    frame cannot show. None when there is no goal to measure against."""
    ds = [goal_dist(s) for s in snapshots]
    live = [(d, i) for i, d in enumerate(ds) if d == d]
    return min(live)[1] if live else None


def _draw_goal(ax, snap: Snapshot) -> None:
    """Goal marker plus the arrival tolerance as a ring. arrival_eps is 0.4cm
    on a 50cm board -- under 1% of frame width -- so it is drawn as a ring with
    a fixed minimum radius, never as a dot that would vanish."""
    from matplotlib.patches import Circle

    def _ring_and_star(gx, gy, tol, marker, size):
        if tol is not None:
            r = max(float(tol), 0.02 * snap.board_w_cm)
            ax.add_patch(Circle((gx, gy), r, fill=False, edgecolor=_GOAL_COLOR,
                                linewidth=1.2, linestyle="--", zorder=2))
        ax.plot([gx], [gy], marker=marker, color=_GOAL_COLOR, markersize=size,
                markeredgecolor="white", markeredgewidth=1.3, zorder=7)

    if snap.goal_xy is not None:
        _ring_and_star(*snap.goal_xy, snap.arrival_eps_cm, "*", 17)
    # One marker PER FINGERTIP target, labelled, because Eq 13's interface is a
    # conjunction over both and a single star cannot say which finger is meant.
    for name, (gx, gy) in (snap.finger_goals or {}).items():
        tol = (snap.finger_goal_tol_cm or {}).get(name, snap.arrival_eps_cm)
        _ring_and_star(gx, gy, tol, "X", 11)
        ax.annotate(f"{name}*", (gx, gy), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=7,
                    color=_GOAL_COLOR, fontweight="bold", zorder=7)


def _draw_fingers(ax, snap: Snapshot) -> None:
    """Fill = contact (green touching / grey not). Edge = AGENCY: the driven
    fingertip gets a heavy solid ring, a masked one a dashed ring, because a
    servo-held fingertip looks identical to a working one otherwise."""
    from matplotlib.patches import Circle

    for name, (fx, fy) in snap.fingers.items():
        touching = snap.touching.get(name, False)
        is_active = (snap.active_finger is None or name == snap.active_finger)
        passive_masked = (not is_active) and bool(snap.inactive_masked)
        ax.add_patch(Circle(
            (fx, fy), snap.finger_radius_cm,
            facecolor=_FINGER_COLOR[touching],
            edgecolor="black" if is_active else "#444444",
            linewidth=2.0 if is_active else 0.9,
            linestyle="-" if not passive_masked else (0, (2, 1)),
            zorder=4))
        ax.annotate(name, (fx, fy), ha="center", va="center",
                    fontsize=8, color="white", zorder=5,
                    fontweight="bold" if is_active else "normal")


def plot_snapshot(snap: Snapshot, ax=None):
    """One frame: walls, goal + tolerance ring, the object, both fingertips
    (fill = contact, edge = which one the policy drives). Returns the axes so
    callers can compose or save it."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 6.0 * snap.board_h_cm / snap.board_w_cm))

    _draw_walls(ax, snap.walls, snap.board_w_cm, snap.board_h_cm)
    _draw_goal(ax, snap)

    corners = _rect_corners(snap.object_xy[0], snap.object_xy[1],
                            snap.object_w_cm, snap.object_h_cm, snap.object_angle_rad)
    ax.add_patch(Polygon(corners, closed=True, facecolor=_OBJECT_COLOR,
                         edgecolor="black", linewidth=1.0, zorder=3))

    _draw_fingers(ax, snap)

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
    _draw_goal(ax, first)

    for snap, alpha, tag in ((snapshots[0], 0.25, "start"),
                             (snapshots[-1], 0.9, "end")):
        corners = _rect_corners(snap.object_xy[0], snap.object_xy[1],
                                snap.object_w_cm, snap.object_h_cm, snap.object_angle_rad)
        ax.add_patch(Polygon(corners, closed=True, facecolor=_OBJECT_COLOR,
                             edgecolor="black", alpha=alpha, zorder=3,
                             label=f"object {tag}"))

    obj_path = np.array([s.object_xy for s in snapshots])
    ax.plot(obj_path[:, 0], obj_path[:, 1], color=_TRAIL_COLOR, linewidth=1.2,
           label="object path", zorder=2)

    # Closest approach: "ran out of time" and "never got close" produce the same
    # success rate and need opposite fixes, so the trajectory has to show which.
    # Marked on whatever goal_dist MEASURES -- the object for push, the driven
    # fingertip for recontact -- or the circle sits on the object while the
    # number beside it describes a finger.
    k = nearest_index(snapshots)
    if k is not None:
        _at = (snapshots[k].fingers[first.active_finger]
               if first.goal_xy is None and first.finger_goals
               and first.active_finger in snapshots[k].fingers
               else obj_path[k])
        ax.plot(*_at, marker="o", markerfacecolor="none",
                markeredgecolor=_NEAREST_COLOR, markeredgewidth=1.8,
                markersize=11, zorder=6,
                label=f"nearest {goal_dist(snapshots[k]):.2f}cm @ t{k}")

    finger_names = list(first.fingers)
    colors = plt.cm.tab10.colors
    for i, name in enumerate(finger_names):
        path = np.array([s.fingers[name] for s in snapshots])
        is_active = (first.active_finger is None or name == first.active_finger)
        ax.plot(path[:, 0], path[:, 1], color=colors[i % len(colors)],
               linewidth=1.4 if is_active else 0.8, linestyle="--",
               label=f"finger {name}" + ("" if is_active else " (held)"), zorder=2)

        touching = [s.touching.get(name, False) for s in snapshots]
        for t in range(1, len(touching)):
            if touching[t] != touching[t - 1]:
                marker = "^" if touching[t] else "v"
                ax.plot(*path[t], marker=marker, color=colors[i % len(colors)],
                       markersize=5, alpha=0.8, zorder=5)

    ax.set_xlim(-2, first.board_w_cm + 2)
    ax.set_ylim(-2, first.board_h_cm + 2)
    ax.set_aspect("equal")
    # Below the axes, not inset: at 50x30 the upper-right corner is INSIDE the
    # board, so an inset legend hides the region the object travels through.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3,
              fontsize=7, frameon=False)
    return ax


def _episode_caption(snapshots: Sequence[Snapshot], info: Optional[Dict] = None) -> str:
    """Static line: what episode this is and how it ended. Derived from the
    snapshots where possible so a caller that passes nothing still gets it."""
    d0, dN = goal_dist(snapshots[0]), goal_dist(snapshots[-1])
    bits = []
    if d0 == d0:
        bits.append(f"d0 {d0:.2f}cm -> final {dN:.2f}cm")
    k = nearest_index(snapshots)
    if k is not None:
        bits.append(f"nearest {goal_dist(snapshots[k]):.2f}cm @ t{k}")
    if info:
        if info.get("why"):
            bits.append(f"ended: {info['why']}")
        if "success" in info:
            bits.append("ARRIVED" if float(info["success"]) > 0.5 else "failed")
        if info.get("bin"):
            bits.append(f"bin {info['bin']}")
    act = snapshots[0].active_finger
    if act is not None:
        held = " (other held)" if snapshots[0].inactive_masked else ""
        bits.append(f"driving {act}{held}")
    return "   ".join(bits)


def _even_frame(frame: np.ndarray) -> np.ndarray:
    """Trim to even height and width.

    libx264 with yuv420p rejects an odd axis, and save_video passes
    macro_block_size=None so nothing pads frames to a multiple of 16 for us.
    The 80x60 recontact board renders 509px tall -- 6.0 * 60/80 + 0.6 = 5.1in,
    and 5.1 * 100dpi truncates to 509 -- so EVERY recontact clip came out
    0 bytes, with ffmpeg reporting nothing on stderr. Push's 50x30 board happens
    to give 420, which is why this only ever broke recontact.

    Trimming the ARRAY rather than choosing an even figsize, because
    matplotlib's own float rounding defeats the figsize approach: asking for
    exactly 5.10in still renders 509.
    """
    h, w = frame.shape[:2]
    return frame[:h - h % 2, :w - w % 2]


def save_video(snapshots: Sequence[Snapshot], path: str, *,
              fps: float = 25.0, dpi: int = 100,
              info: Optional[Dict] = None) -> str:
    """Renders every snapshot as a frame and writes an .mp4 to `path`, local
    disk only. Never call this on a path under anything that syncs to wandb
    or another remote store -- see the module docstring.

    `info` adds episode-level context to the caption (`why`, `success`, `bin`);
    everything else in the caption is derived from the snapshots themselves.
    """
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt

    if not snapshots:
        raise ValueError("save_video needs at least one snapshot")

    caption = _episode_caption(snapshots, info)
    # Trail whatever the task actually MOVES. Recontact's premise is that the
    # object stays put, so trailing the object draws a still dot and hides the
    # only motion in the clip; there the driven fingertip is the trajectory.
    _trail_finger = (snapshots[0].active_finger
                     if snapshots[0].goal_xy is None
                     and snapshots[0].finger_goals else None)
    trail = np.array([s.fingers[_trail_finger] if _trail_finger else s.object_xy
                      for s in snapshots])
    nearest = nearest_index(snapshots)

    frames = []
    first = snapshots[0]
    figsize = (6.0, 6.0 * first.board_h_cm / first.board_w_cm + 0.6)
    for tick, snap in enumerate(snapshots):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        plot_snapshot(snap, ax=ax)

        # Trail only up to the current tick, so the video never shows the
        # object's future -- and mark closest approach once it has happened.
        if tick:
            ax.plot(trail[:tick + 1, 0], trail[:tick + 1, 1],
                    color=_TRAIL_COLOR, linewidth=1.0, alpha=0.75, zorder=2)
        if nearest is not None and tick >= nearest:
            ax.plot(*trail[nearest], marker="o", markerfacecolor="none",
                    markeredgecolor=_NEAREST_COLOR, markeredgewidth=1.6,
                    markersize=10, zorder=6)

        d = goal_dist(snap)
        live = f"tick {tick}/{len(snapshots) - 1}"
        if d == d:
            live += f"    dist {d:.2f}cm"
        ax.set_title(f"{caption}\n{live}", fontsize=7.5, loc="left")
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(_even_frame(
            np.asarray(fig.canvas.buffer_rgba())[:, :, :3]).copy())
        plt.close(fig)

    # macro_block_size=None: our frame size need not be a multiple of 16.
    with imageio.get_writer(path, fps=fps, macro_block_size=None) as writer:
        for frame in frames:
            writer.append_data(frame)
    return path
