"""How one chamber grows: the equation that is solved, and from which initial state.

The phase field u of the growing chamber evolves as

    u_t = gamma Laplacian(u) + reaction(u),
    reaction(u) = k u(1-u)(u-1/2) + adhesion u(1-u),     k = 2 gamma / ell^2,
    adhesion = (a / ell) exp(-d/R),

subject to h^2 sum(u) = target_area(t). gamma is p.curvature_coefficient, a
p.adhesion_speed, R p.adhesion_range, d the distance to existing chambers and
ell q.interface_width. A forcing g u(1-u) moves the interface at speed g ell,
so as ell -> 0 the boundary moves at gamma * curvature + a exp(-d/R) plus the
area correction, independent of ell. The Laplacian, the area constraint and
the time stepping are in solver.py; this module supplies the model-specific
pieces.
"""

import math

import torch
import torch.nn.functional as F

from .morphology import distance_to
from .parameters import ColonyParameters, ImplementationParameters


def target_area(initial_area, final_area, p: ColonyParameters):
    """Prescribed chamber area over time.

    A sigmoid from initial_area at t = 0 to final_area at t = 1, the formation time.
    """
    def sigmoid(t):
        return 1.0 / (1.0 + math.exp(-p.growth_rate * (t - p.growth_midpoint_fraction)))

    f0, f1 = sigmoid(0.0), sigmoid(1.0)
    scale = (final_area - initial_area) / (f1 - f0)

    def area(t):
        return initial_area + scale * (sigmoid(t) - f0)

    return area


def nucleate(labels, chamber, x, y, p: ColonyParameters, q: ImplementationParameters):
    """Initial u: a soft disc just outside the aperture, pointing away from the parent."""
    parent = labels == chamber.parent
    center = torch.stack((x[parent].mean(), y[parent].mean()))
    anchor = torch.stack((x[chamber.anchor], y[chamber.anchor]))
    outward = F.normalize(anchor - center, dim=0)
    blob_center = anchor + p.nucleus_offset*outward

    distance = torch.hypot(x - blob_center[0], y - blob_center[1])
    u = (1 / (1 + torch.exp((distance - p.nucleus_radius) / q.interface_width))) * chamber.allowed
    # Compact support prevents phase nucleation far from the seeded chamber.
    u[distance > p.nucleus_radius + q.nucleus_cutoff_widths*q.interface_width] = 0.0
    u[chamber.aperture] = 1.0
    return u


def adhesion_field(occupied, p: ColonyParameters, q: ImplementationParameters):
    """Attraction that decays with distance from any existing chamber.

    Zero beyond q.adhesion_cutoff_ranges adhesion ranges.
    """
    distance = distance_to(
        occupied,
        q.spacing,
        q.adhesion_cutoff_ranges * p.adhesion_range,
        q.torch_dtype,
    )
    return adhesion_coefficient(p, q) * torch.exp(-distance / p.adhesion_range)


def adhesion_coefficient(p: ColonyParameters, q: ImplementationParameters):
    """The adhesion forcing at the wall that moves the interface at p.adhesion_speed."""
    return p.adhesion_speed / q.interface_width


def double_well_coefficient(p: ColonyParameters, q: ImplementationParameters):
    return 2 * p.curvature_coefficient / q.interface_width**2


def reaction(u, adhesion, p: ColonyParameters, q: ImplementationParameters):
    """The non-diffusive part of du/dt: the double well plus adhesion."""
    return u * (1 - u) * (double_well_coefficient(p, q) * (u - 0.5) + adhesion)


def reaction_rate_bound(p: ColonyParameters, q: ImplementationParameters):
    """Upper bound on |d reaction / du| over u in [0, 1], for the explicit time-step limit."""
    return 3 * double_well_coefficient(p, q) + adhesion_coefficient(p, q)
