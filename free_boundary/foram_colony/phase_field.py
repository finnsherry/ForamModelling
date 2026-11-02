"""Phase-field numerics: no foraminiferal concepts, only u, masks and the grid spacing."""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def double_well_coefficient(p):
    return 2 * p.curvature_coefficient / p.interface_width**2


def stable_dt(p):
    """Explicit-Euler cap from the diffusion, double-well and adhesion terms."""
    k = double_well_coefficient(p)
    return min(p.max_dt, 0.35 / (4*p.curvature_coefficient/p.spacing**2 + 3*k + p.adhesion_strength))


def phase_rhs(u, adhesion, p):
    """Allen–Cahn plus adhesion: du/dt before the area projection."""
    h = p.spacing
    lap = (np.roll(u, 1, 0) + np.roll(u, -1, 0) + np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4*u) / h**2
    return (p.curvature_coefficient*lap + double_well_coefficient(p)*u*(1-u)*(u-0.5)
            + adhesion*u*(1-u))


def corrected_phase_area(phase, desired, allowed, aperture, h):
    """Scalar multiplier correction of diffuse area, with fixed wall constraints."""
    u = np.clip(phase, 0.0, 1.0) * allowed
    u[aperture] = 1.0
    weights = u * (1.0 - u) * allowed
    weights[aperture] = 0.0
    for _ in range(15):
        error = desired / h**2 - u.sum()
        if abs(error) * h**2 < 5e-12:
            return u
        active = (u < 1.0) if error > 0 else (u > 0.0)
        denominator = weights[active].sum()
        if denominator < 1e-12:
            raise RuntimeError("Insufficient interface to enforce the requested area.")
        u = np.clip(u + (error / denominator) * weights * active, 0.0, 1.0)
        u *= allowed
        u[aperture] = 1.0
    raise RuntimeError("Diffuse area correction failed; decrease the timestep.")


def crisp_chamber(phase, allowed, aperture, area, h):
    """Threshold to the nearest pixel-area target while retaining the aperture."""
    count = int(round(area / h**2))
    candidates = np.flatnonzero(allowed & ~aperture)
    needed = count - int(aperture.sum())
    if needed <= 0 or needed > len(candidates):
        raise ValueError("Target area incompatible with aperture or remaining space.")
    chosen = candidates[np.argpartition(phase.flat[candidates], -needed)[-needed:]]
    mask = aperture.copy()
    mask.flat[chosen] = True
    _, components = ndimage.label(mask)
    if components != 1:
        raise RuntimeError("The new chamber disconnected; reduce growth/adhesion or revise the aperture.")
    return mask


def check_phase(u):
    if not np.isfinite(u).all():
        raise RuntimeError("Nonfinite phase field.")
    if max(u[1, :].max(), u[-2, :].max(), u[:, 1].max(), u[:, -2].max()) > .5:
        raise RuntimeError("Growth reached the computational edge; increase half_width.")
