"""Phase-field numerics: no foraminiferal concepts, only u, masks and the grid spacing."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .parameters import AREA_TOLERANCE_EPS, DT_SAFETY, DTYPE, NEWTON_STEPS


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


def double_well_coefficient(p):
    return 2 * p.curvature_coefficient / p.interface_width**2


def stable_dt(p):
    """Explicit-Euler cap from the diffusion, double-well and adhesion terms."""
    k = double_well_coefficient(p)
    return min(p.max_dt, DT_SAFETY / (4*p.curvature_coefficient/p.spacing**2 + 3*k + p.adhesion_strength))


def phase_rhs(u, adhesion, p):
    """Allen–Cahn plus adhesion: du/dt before the area projection."""
    h = p.spacing
    lap = (torch.roll(u, 1, 0) + torch.roll(u, -1, 0) + torch.roll(u, 1, 1) + torch.roll(u, -1, 1) - 4*u) / h**2
    return (p.curvature_coefficient*lap + double_well_coefficient(p)*u*(1-u)*(u-0.5)
            + adhesion*u*(1-u))


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


def crisp_chamber(phase, allowed, aperture, area, h):
    """Threshold to the nearest pixel-area target while retaining the aperture."""
    count = round(area / h**2)
    candidates = torch.nonzero((allowed & ~aperture).flatten()).squeeze(1)
    needed = count - int(aperture.sum())
    if needed <= 0 or needed > len(candidates):
        raise ValueError("Target area incompatible with aperture or remaining space.")
    chosen = candidates[torch.topk(phase.flatten()[candidates], needed).indices]
    mask = aperture.clone()
    mask.view(-1)[chosen] = True
    return mask
