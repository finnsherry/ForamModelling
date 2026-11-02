"""How one chamber grows: the equation that is solved, and from which initial state.

The phase field u of the growing chamber evolves as

    u_t = gamma Laplacian(u) + reaction(u),
    reaction(u) = k u(1-u)(u-1/2) + adhesion u(1-u),     k = 2 gamma / ell^2,

subject to h^2 sum(u) = target_area(t). gamma is p.curvature_coefficient and
ell is p.interface_width. The Laplacian, the area constraint and the time
stepping are in solver.py; this module supplies the model-specific pieces.
"""

import math

import torch
import torch.nn.functional as F

from .morphology import distance_to
from .parameters import ADHESION_CUTOFF_RANGES


def target_area(initial_area, final_area, p):
    """Prescribed chamber area over time.

    A sigmoid from initial_area at t = 0 to final_area at p.formation_time.
    """
    end_time = p.formation_time
    rate, midpoint = p.growth_rate, p.growth_midpoint_fraction * end_time

    def sigmoid(t):
        return 1.0 / (1.0 + math.exp(-rate * (t - midpoint)))

    f0, f1 = sigmoid(0.0), sigmoid(end_time)
    scale = (final_area - initial_area) / (f1 - f0)

    def area(t):
        return initial_area + scale * (sigmoid(t) - f0)

    return area


def nucleate(labels, chamber, x, y, p):
    """Initial u: a soft disc just outside the aperture, pointing away from the parent."""
    parent = labels == chamber.parent
    center = torch.stack((x[parent].mean(), y[parent].mean()))
    anchor = torch.stack((x[chamber.anchor], y[chamber.anchor]))
    outward = F.normalize(anchor - center, dim=0)
    blob_center = anchor + p.nucleus_offset*outward

    distance = torch.hypot(x - blob_center[0], y - blob_center[1])
    u = (1 / (1 + torch.exp((distance - p.nucleus_radius) / p.interface_width))) * chamber.allowed
    # Compact support prevents phase nucleation far from the seeded chamber.
    u[distance > p.nucleus_support_radius] = 0.0
    u[chamber.aperture] = 1.0
    return u


def adhesion_field(occupied, p):
    """Attraction that decays with distance from any existing chamber.

    Zero beyond ADHESION_CUTOFF_RANGES adhesion ranges.
    """
    distance = distance_to(occupied, p.spacing, ADHESION_CUTOFF_RANGES * p.adhesion_range)
    return p.adhesion_strength * torch.exp(-distance / p.adhesion_range)


def double_well_coefficient(p):
    return 2 * p.curvature_coefficient / p.interface_width**2


def reaction(u, adhesion, p):
    """The non-diffusive part of du/dt: the double well plus adhesion."""
    return u * (1 - u) * (double_well_coefficient(p) * (u - 0.5) + adhesion)


def reaction_rate_bound(p):
    """Upper bound on |d reaction / du| over u in [0, 1], for the explicit time-step limit."""
    return 3 * double_well_coefficient(p) + p.adhesion_strength
