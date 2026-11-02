"""Time integration of an area-constrained phase field. Model-free: no foraminiferal concepts.

Solves u_t = diffusion * Laplacian(u) + reaction(u) by explicit Euler steps,
each followed by a projection onto a prescribed area. u is held at 0 outside
`allowed` and at 1 on `aperture`. The reaction term and the target area are
supplied by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .parameters import (AREA_TOLERANCE_EPS, DT_SAFETY, DTYPE, EDGE_THRESHOLD, NEWTON_STEPS,
                         WINDOW_SPILL, WINDOW_TAIL)


@dataclass(frozen=True)
class Window:
    """A rectangle [r0, r1) x [c0, c1) of a grid of size `shape`."""
    r0: int
    r1: int
    c0: int
    c1: int
    shape: tuple[int, int]

    @classmethod
    def around(cls, mask, margin):
        """Bounding box of mask (nonempty), widened by margin pixels and clipped to the grid."""
        rows = torch.nonzero(mask.any(1)).squeeze(1)
        cols = torch.nonzero(mask.any(0)).squeeze(1)
        n, m = mask.shape
        return cls(max(int(rows[0]) - margin, 0), min(int(rows[-1]) + margin + 1, n),
                   max(int(cols[0]) - margin, 0), min(int(cols[-1]) + margin + 1, m), (n, m))

    def widened(self, margin, sides):
        """Widen by margin on the given sides: four booleans (top, bottom, left, right)."""
        top, bottom, left, right = (margin if side else 0 for side in sides)
        n, m = self.shape
        return Window(max(self.r0 - top, 0), min(self.r1 + bottom, n),
                      max(self.c0 - left, 0), min(self.c1 + right, m), self.shape)

    @property
    def at_border(self):
        """Which sides (top, bottom, left, right) lie on the grid's own border."""
        return (self.r0 == 0, self.r1 == self.shape[0], self.c0 == 0, self.c1 == self.shape[1])

    def crop(self, a):
        return a[self.r0:self.r1, self.c0:self.c1]

    def paste(self, a):
        """a placed into a zero array of the full grid size."""
        full = a.new_zeros(self.shape)
        full[self.r0:self.r1, self.c0:self.c1] = a
        return full


def laplacian(u, h):
    """Five-point Laplacian; periodic, which is harmless as u is 0 on the outer ring."""
    return (torch.roll(u, 1, 0) + torch.roll(u, -1, 0) + torch.roll(u, 1, 1) + torch.roll(u, -1, 1) - 4*u) / h**2


def stable_dt(h, diffusion, reaction_bound, max_dt):
    """Explicit-Euler cap from the diffusion term and a bound on the reaction rate."""
    return min(max_dt, DT_SAFETY / (4*diffusion/h**2 + reaction_bound))


def corrected_phase_area(phase, desired, allowed, aperture, h):
    """Project phase onto area `desired`: u = clamp(u0 + lam*w, 0, 1) for a scalar lam.

    u0 is phase restricted to `allowed` with the aperture set to 1, and
    w = u0(1 - u0). Since w vanishes wherever u0 is 0 or 1, the correction only
    moves the diffuse interface; the excluded region and the aperture are left
    untouched without re-masking. The area is a nondecreasing, piecewise-linear
    function of lam. Newton's method, using the slope in the direction of the
    step, never overshoots on it; one step is normally exact to rounding.
    """
    u0 = torch.where(aperture, 1.0, phase.clamp(0.0, 1.0) * allowed)
    w = u0 * (1.0 - u0)
    target = desired / h**2
    u = u0
    lam = torch.zeros((), dtype=u0.dtype, device=u0.device)
    for _ in range(NEWTON_STEPS):
        error = target - u.sum()
        slope = torch.where(error > 0, (w * (u < 1)).sum(), (w * (u > 0)).sum())
        # A vanishing slope gives a huge step; the residual check then reports it.
        lam = lam + error / slope.clamp_min(1e-12)
        u = (u0 + lam * w).clamp(0.0, 1.0)
    return u, (target - u.sum()).abs() * h**2


def area_tolerance(area):
    """Largest acceptable projection residual: the sum is only accurate to about eps."""
    return AREA_TOLERANCE_EPS * torch.finfo(DTYPE).eps * area


def edge_values(u):
    """Largest u next to each side of the array (top, bottom, left, right).

    The outermost ring itself is held at 0, so this looks one pixel inside it.
    """
    return torch.stack((u[1, :].max(), u[-2, :].max(), u[:, 1].max(), u[:, -2].max()))


def check_phase(u, edges, residual, area):
    """Raise if the phase field went bad; return the four edge values as floats.

    edges and residual are running maxima since the last check, kept on the
    device; reading them together costs a single sync.
    """
    *edges, residual, finite = torch.cat((edges, torch.stack((residual, torch.isfinite(u).all().to(u.dtype))))).tolist()
    if not finite:
        raise RuntimeError("Nonfinite phase field.")
    if residual > area_tolerance(area):
        raise RuntimeError("Diffuse area correction failed; decrease the timestep.")
    return edges


def integrate(u, allowed, aperture, diffusion, reaction, target, save_times, dt_max, h, margin, on_snapshot):
    """Evolve u through save_times; returns the final u, the step count and the largest area error.

    reaction(u, window) gives the reaction term on the current window; target(t)
    the prescribed area. on_snapshot(u, t) is called at t = 0 and at every save time.

    The time stepping runs on a window around the region where u > 0, not the
    whole grid. The window's outer ring is held at u = 0, like the grid's, so the
    step works on it unchanged. At every save time the window is refitted. If u
    reached a side of the window during an interval, the window was too small:
    that side is widened by `margin` pixels and the interval is repeated. A side
    that lies on the grid border cannot be widened; there the usual edge check
    applies instead.
    """
    t, steps = 0.0, 0
    max_error = torch.zeros((), device=u.device, dtype=DTYPE)
    on_snapshot(u, t)
    window = Window.around(u > WINDOW_TAIL, margin)
    u_full = u
    for save_time in save_times:
        t_start, steps_start = t, steps
        while True:
            allowed_w = window.crop(allowed).clone()
            allowed_w[[0, -1], :] = False
            allowed_w[:, [0, -1]] = False
            aperture_w = window.crop(aperture)
            u = window.crop(u_full) * allowed_w
            # Nothing in this loop reads back from the device; problems are only
            # checked at the save time, from running maxima kept on the device.
            interval_error = torch.zeros((), device=u.device, dtype=DTYPE)
            edges = torch.zeros(4, device=u.device, dtype=DTYPE)
            while t < save_time - 1e-12:
                dt = min(dt_max, save_time - t)
                rhs = diffusion*laplacian(u, h) + reaction(u, window)
                u, residual = corrected_phase_area(u + dt*rhs, target(t + dt), allowed_w, aperture_w, h)
                t += dt
                steps += 1
                interval_error = torch.maximum(interval_error, residual)
                edges = torch.maximum(edges, edge_values(u))
            edges = check_phase(u, edges, interval_error, target(t))
            if any(e > EDGE_THRESHOLD and border for e, border in zip(edges, window.at_border)):
                raise RuntimeError("Growth reached the computational edge; increase half_width.")
            spilled = [e > WINDOW_SPILL and not border for e, border in zip(edges, window.at_border)]
            if not any(spilled):
                break
            window = window.widened(margin, spilled)
            t, steps = t_start, steps_start
        u_full = window.paste(u)
        max_error = torch.maximum(max_error, interval_error)
        on_snapshot(u_full, t)
        window = Window.around(u_full > WINDOW_TAIL, margin)
    return u_full, steps, float(max_error)
