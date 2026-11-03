"""Time integration of an area-constrained phase field. Model-free: no foraminiferal concepts.

Solves u_t = diffusion * Laplacian(u) + reaction(u) by explicit Euler steps,
each followed by a projection onto a prescribed area. u is held at 0 outside
`allowed` and at 1 on `aperture`. The reaction term and the target area are
supplied by the caller.
"""

import math

import torch

from .parameters import ImplementationParameters


def window_around(mask, margin):
    """Bounding box of mask (nonempty) as a pair of slices, widened by margin pixels.

    Slices clip at the grid end themselves; only the start needs clipping at 0.
    """
    rows = torch.nonzero(mask.any(1)).squeeze(1)
    cols = torch.nonzero(mask.any(0)).squeeze(1)
    return (
        slice(max(int(rows[0]) - margin, 0), int(rows[-1]) + margin + 1),
        slice(max(int(cols[0]) - margin, 0), int(cols[-1]) + margin + 1),
    )


def laplacian(u, h):
    """Nine-point Laplacian on the interior; zero on the outer ring, where u is held at 0.

    The stencil is [[1, 4, 1], [4, -20, 4], [1, 4, 1]] / (6 h^2): 2/3 of the
    five-point stencil plus 1/3 of its 45-degree rotation. That mix makes the
    leading error term (h^2/12) Laplacian^2 u, which is rotation invariant.
    """
    out = torch.zeros_like(u)
    out[1:-1, 1:-1] = (
        4*(
            u[:-2, 1:-1]
            + u[2:, 1:-1]
            + u[1:-1, :-2]
            + u[1:-1, 2:]
        )
        + u[:-2, :-2]
        + u[:-2, 2:]
        + u[2:, :-2]
        + u[2:, 2:]
        - 20*u[1:-1, 1:-1]
    )
    return out / (6 * h**2)


def stable_dt(h, diffusion, reaction_bound):
    """Explicit-Euler cap from the diffusion term and a bound on the reaction rate.

    The Laplacian's eigenvalues lie in [-16/(3 h^2), 0], so the diffusion term
    alone is stable for dt <= 2 / (16 diffusion / (3 h^2)).
    """
    return 1 / (8 * diffusion / (3 * h**2) + reaction_bound)


def corrected_phase_area(phase, desired, allowed, aperture, h, newton_steps):
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
    for _ in range(newton_steps):
        error = target - u.sum()
        # Only pixels that can still move in the direction of the step count.
        slope = (w * torch.where(error > 0, u < 1, u > 0)).sum()
        # A vanishing slope gives a huge step; the residual check then reports it.
        lam = lam + error / slope.clamp_min(1e-12)
        u = (u0 + lam * w).clamp(0.0, 1.0)
    return u, (target - u.sum()).abs() * h**2


def check_phase(u, edge, residual, area, q: ImplementationParameters):
    """Raise if the phase field went bad."""
    if not torch.isfinite(u).all():
        raise RuntimeError("Nonfinite phase field.")
    if residual > q.area_tolerance_eps * torch.finfo(u.dtype).eps * area:
        raise RuntimeError("Diffuse area correction failed; decrease the timestep.")
    if edge > q.window_spill:
        raise RuntimeError(
            "The chamber reached the edge of the time-stepping window: shrink the "
            "colony if it is near the grid border, otherwise increase window_margin."
        )


def integrate(
    u,
    allowed,
    aperture,
    diffusion,
    reaction,
    target,
    save_times,
    dt_max,
    q: ImplementationParameters,
    on_snapshot,
):
    """Evolve u through save_times; returns the final u, the step count and the largest area error.

    reaction(u, window) gives the reaction term on the current window, a pair of
    slices; target(t) the prescribed area. on_snapshot(u, t) is called at t = 0
    and at every save time.

    The time stepping runs on a window around the region where u > q.window_tail,
    widened by q.window_margin interface widths and refitted at every save time.
    The window's outer ring is held at u = 0, like the grid's, so the step works
    on it unchanged. If u comes close to the window border anyway, check_phase
    raises.
    """
    h = q.spacing
    margin = math.ceil(q.window_margin * q.interface_width_factor)
    t, steps = 0.0, 0
    max_error = torch.zeros((), device=u.device, dtype=u.dtype)
    on_snapshot(u, t)

    for save_time in save_times:
        window = window_around(u > q.window_tail, margin)
        allowed_w = allowed[window].clone()
        allowed_w[[0, -1], :] = False
        allowed_w[:, [0, -1]] = False
        aperture_w = aperture[window]
        u_w = u[window] * allowed_w
        # The pixels just inside the window's outer ring.
        edge_ring = torch.zeros_like(u_w)
        edge_ring[[1, -2], 1:-1] = 1.0
        edge_ring[1:-1, [1, -2]] = 1.0

        # Nothing in this loop reads back from the device; problems are only
        # checked at the save time, from running maxima kept on the device.
        interval_error = torch.zeros((), device=u.device, dtype=u.dtype)
        edge = torch.zeros((), device=u.device, dtype=u.dtype)
        while t < save_time - 1e-12:
            dt = min(dt_max, save_time - t)
            rhs = diffusion*laplacian(u_w, h) + reaction(u_w, window)
            u_w, residual = corrected_phase_area(
                u_w + dt*rhs,
                target(t + dt),
                allowed_w,
                aperture_w,
                h,
                q.newton_steps,
            )
            t += dt
            steps += 1
            interval_error = torch.maximum(interval_error, residual)
            edge = torch.maximum(edge, (u_w * edge_ring).amax())

        check_phase(u_w, edge, interval_error, target(t), q)
        u = torch.zeros_like(u)
        u[window] = u_w
        max_error = torch.maximum(max_error, interval_error)
        on_snapshot(u, t)

    return u, steps, float(max_error)
