"""Drawing logic for a unit circle with points at given angles."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button


def default_thetas(n: int) -> np.ndarray:
    """Evenly spaced angles theta_0 .. theta_{n-1} on [0, 2*pi)."""
    return np.linspace(0, 2 * np.pi, n, endpoint=False)


def points_on_unit_circle(thetas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.cos(thetas), np.sin(thetas)


def _setup_axes(ax: plt.Axes, n: int, visible: int) -> None:
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(rf"Unit circle: showing $a_0$ through $a_{{{visible - 1}}}$ (${visible}/{n}$)")
    ax.grid(True, alpha=0.3)


def _is_numeric(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating))


def _active_delta_rays(
    thetas: np.ndarray,
    phi_sides: list[np.ndarray],
    delta_sides: list[np.ndarray],
    visible_count: int,
) -> list[tuple[float, float]]:
    """
    Collect (mid-angle, inner-radius) for every ± side with delta < 1.
    Both plus and minus partners are kept when both are active.
    """
    j = visible_count - 1
    rays: list[tuple[float, float]] = []
    for phi, delta in zip(phi_sides, delta_sides):
        for i in range(visible_count):
            d = delta[i, j]
            p = phi[i, j]
            if not _is_numeric(d) or not _is_numeric(p) or d >= 1:
                continue
            angle = (thetas[i] + float(p)) / 2
            rays.append((float(angle), float(d)))
    return rays


def _draw_perpendicular_bisectors(
    ax: plt.Axes,
    thetas: np.ndarray,
    phi_sides: list[np.ndarray],
    delta_sides: list[np.ndarray],
    visible_count: int,
) -> list:
    """
    For each visible a_i, draw the perpendicular bisector of the segment from
    the origin to a_i.  The bisector passes through the midpoint (cos θ_i/2,
    sin θ_i/2) and is clipped to the unit disk, stopping early wherever it
    hits an active delta line.
    """
    artists: list = []
    active_delta = _active_delta_rays(thetas, phi_sides, delta_sides, visible_count)

    for i in range(visible_count):
        theta_i = float(thetas[i])
        # Midpoint M of (a_i, origin): radius 1/2 at angle theta_i
        mx = np.cos(theta_i) / 2.0
        my = np.sin(theta_i) / 2.0
        # Perpendicular direction (tangent to the half-radius circle at M)
        dx = -np.sin(theta_i)
        dy = np.cos(theta_i)

        # The chord |M + t*(dx,dy)|² = 1 gives t = ±√3/2
        t_pos = np.sqrt(3.0) / 2.0
        t_neg = -t_pos

        for alpha, d_inner in active_delta:
            # Bisector ∩ ray at angle alpha:
            #   t  = -(1/2) tan(θ_i − α)
            #   s  = (1/2) / cos(α − θ_i)   (radial distance along ray)
            cos_diff = np.cos(alpha - theta_i)
            if abs(cos_diff) < 1e-10:
                continue
            t_sol = -0.5 * np.tan(theta_i - alpha)
            s_sol = 0.5 / cos_diff

            # Only clip if the crossing lands on the actual delta segment [d_inner, 1]
            if not (d_inner - 1e-9 <= s_sol <= 1.0 + 1e-9):
                continue

            if t_sol >= 0.0:
                t_pos = min(t_pos, t_sol)
            else:
                t_neg = max(t_neg, t_sol)

        x1 = mx + t_neg * dx
        y1 = my + t_neg * dy
        x2 = mx + t_pos * dx
        y2 = my + t_pos * dy

        artists.append(
            ax.plot(
                [x1, x2],
                [y1, y2],
                color="tab:purple",
                lw=1.5,
                alpha=0.75,
                zorder=3,
            )[0]
        )

    return artists


def _draw_delta_lines(
    ax: plt.Axes,
    thetas: np.ndarray,
    phi_sides: list[np.ndarray],
    delta_sides: list[np.ndarray],
    visible_count: int,
) -> list:
    """
    With a_0 .. a_j visible, draw every active ± delta ray: for each side,
    a segment at angle (theta[i] + phi[i][j]) / 2 from radius delta[i][j] to 1
    whenever delta[i][j] < 1.
    """
    artists: list = []
    for angle, d in _active_delta_rays(thetas, phi_sides, delta_sides, visible_count):
        x_inner = d * np.cos(angle)
        y_inner = d * np.sin(angle)
        x_outer = np.cos(angle)
        y_outer = np.sin(angle)
        artists.append(
            ax.plot(
                [x_inner, x_outer],
                [y_inner, y_outer],
                color="tab:orange",
                lw=1.5,
                alpha=0.75,
                zorder=3,
            )[0]
        )

    return artists


def _draw_points(ax: plt.Axes, thetas: np.ndarray) -> list:
    artists: list = []
    if thetas.size == 0:
        return artists

    x, y = points_on_unit_circle(thetas)
    artists.append(
        ax.scatter(x, y, s=90, color="tab:blue", zorder=5)
    )

    for i, (xi, yi, theta) in enumerate(zip(x, y, thetas)):
        artists.append(
            ax.annotate(
                rf"$a_{i}$",
                (xi, yi),
                textcoords="offset points",
                xytext=(10, 10),
                fontsize=12,
            )
        )
        artists.append(
            ax.annotate(
                rf"$\theta_{i}={theta:.2f}$",
                (xi, yi),
                textcoords="offset points",
                xytext=(10, -14),
                fontsize=9,
                color="dimgray",
            )
        )

    return artists


def draw_unit_circle(
    thetas: np.ndarray | list[float],
    *,
    show: bool = True,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Draw the unit circle and mark points a_0 .. a_{n-1} at theta_0 .. theta_{n-1}.
    """
    thetas = np.asarray(thetas, dtype=float)
    n = thetas.size
    if n < 1:
        raise ValueError("thetas must contain at least one angle")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    circle_t = np.linspace(0, 2 * np.pi, 256)
    ax.plot(np.cos(circle_t), np.sin(circle_t), color="black", lw=1.5, alpha=0.4)
    _draw_points(ax, thetas)
    _setup_axes(ax, n, n)

    if show:
        plt.tight_layout()
        plt.show()

    return fig, ax


def draw_unit_circle_interactive(
    thetas: np.ndarray | list[float],
    *,
    phi_p: np.ndarray | None = None,
    delta_p: np.ndarray | None = None,
    phi_m: np.ndarray | None = None,
    delta_m: np.ndarray | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Draw the unit circle and reveal points one at a time with Next / Back buttons.

    Initially only a_0 is shown.  Each Next reveals a_j; every visible a_i
    gets a delta ray for each ± side with delta[i][j] < 1.
    """
    thetas = np.asarray(thetas, dtype=float)
    n = thetas.size
    if n < 1:
        raise ValueError("thetas must contain at least one angle")

    sides = [
        (phi_p, delta_p),
        (phi_m, delta_m),
    ]
    phi_sides = [phi for phi, delta in sides if phi is not None and delta is not None]
    delta_sides = [delta for phi, delta in sides if phi is not None and delta is not None]
    draw_deltas = len(phi_sides) > 0
    for phi, delta in zip(phi_sides, delta_sides):
        if phi.shape != (n, n) or delta.shape != (n, n):
            raise ValueError("phi and delta matrices must have shape (n, n)")

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.12)

    circle_t = np.linspace(0, 2 * np.pi, 256)
    ax.plot(np.cos(circle_t), np.sin(circle_t), color="black", lw=1.5, alpha=0.4)

    visible_count = 1
    overlay_artists: list = []

    def refresh() -> None:
        nonlocal overlay_artists
        for artist in overlay_artists:
            artist.remove()
        overlay_artists = []
        if draw_deltas:
            overlay_artists.extend(
                _draw_delta_lines(ax, thetas, phi_sides, delta_sides, visible_count)
            )
            overlay_artists.extend(
                _draw_perpendicular_bisectors(
                    ax, thetas, phi_sides, delta_sides, visible_count
                )
            )
        overlay_artists.extend(_draw_points(ax, thetas[:visible_count]))
        _setup_axes(ax, n, visible_count)
        back_button.ax.set_visible(visible_count > 1)
        next_button.ax.set_visible(visible_count < n)
        fig.canvas.draw_idle()

    def on_next(_event) -> None:
        nonlocal visible_count
        if visible_count < n:
            visible_count += 1
            refresh()

    def on_back(_event) -> None:
        nonlocal visible_count
        if visible_count > 1:
            visible_count -= 1
            refresh()

    back_ax = fig.add_axes((0.32, 0.03, 0.14, 0.06))
    next_ax = fig.add_axes((0.54, 0.03, 0.14, 0.06))
    back_button = Button(back_ax, "Back")
    next_button = Button(next_ax, "Next")

    back_button.on_clicked(on_back)
    next_button.on_clicked(on_next)

    refresh()

    if show:
        plt.show()

    return fig, ax
