"""Model rules: seed geometry, where each chamber attaches, how it starts, what attracts it."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .morphology import component_containing, dilate, distance_to, is_connected
from .parameters import ADHESION_CUTOFF_RANGES, MIN_SEED_PIXELS, SEED_STYLES


@dataclass
class Chamber:
    """Fixed constraints for the chamber currently growing."""
    cell: int
    parent: int
    allowed: torch.Tensor     # where u may be nonzero
    aperture: torch.Tensor    # where u is held at 1
    anchor: tuple[int, int]   # grid index of the aperture centre


def make_seed(p, x, y):
    """Initial labels: one disc (unilocular) or two adjacent compartments (bilocular).

    For a bilocular seed, two overlapping generating discs are partitioned:
    compartment 1 retains its disc; compartment 2 occupies the second disc
    outside compartment 1. The resulting compartments have disjoint interiors
    and a common interface. This is geometry, not a universal embryonic shape.
    """
    labels = torch.zeros(x.shape, dtype=torch.int16, device=x.device)
    if p.seed_mode == "unilocular":
        labels[torch.hypot(x, y) <= p.central_radius] = 1
    else:
        r1 = p.first_piece_radius_fraction * p.central_radius
        r2 = r1 * p.second_piece_radius_ratio
        distance = p.seed_separation_fraction * (r1 + r2)
        angle = math.radians(p.seed_orientation_degrees)
        axis = (math.cos(angle), math.sin(angle))
        c1 = (-0.5 * distance * axis[0], -0.5 * distance * axis[1])
        c2 = (0.5 * distance * axis[0], 0.5 * distance * axis[1])
        first = torch.hypot(x - c1[0], y - c1[1]) <= r1
        second = torch.hypot(x - c2[0], y - c2[1]) <= r2
        labels[first] = 1
        labels[second & ~first] = 2
    reports = []
    for cell, name in enumerate(SEED_STYLES[p.seed_mode]["names"], start=1):
        mask = labels == cell
        if int(mask.sum()) < MIN_SEED_PIXELS or not is_connected(mask):
            raise ValueError("An initial compartment is missing, disconnected or under-resolved. Refine the grid or adjust seed geometry.")
        reports.append({"chamber":cell,"name":name,"pixel_area":float(mask.sum())*p.spacing**2,
                        "centroid":[float(x[mask].mean()),float(y[mask].mean())]})
    if not is_connected(labels > 0):
        raise ValueError("The central seed pieces do not connect; adjust their separation or grid spacing.")
    return labels, reports


def select_aperture(labels, parent, x, y, angle, guide_radius, radius):
    """Choose an exposed parent wall nearest the guide point at (guide_radius, angle)."""
    parent_mask = labels == parent
    candidates = dilate(parent_mask) & (labels == 0)
    candidates[[0, -1], :] = False
    candidates[:, [0, -1]] = False
    rows, columns = torch.nonzero(candidates, as_tuple=True)
    if len(rows) == 0:
        raise RuntimeError("The previous chamber has no exposed boundary.")
    theta = math.radians(angle)
    guide = (guide_radius * math.cos(theta), guide_radius * math.sin(theta))
    scores = (x[rows, columns] - guide[0])**2 + (y[rows, columns] - guide[1])**2
    selected = int(torch.argmin(scores))
    cy, cx = int(rows[selected]), int(columns[selected])
    distance = torch.hypot(x - x[cy, cx], y - y[cy, cx])
    aperture = candidates & (distance <= radius)
    # The outside pixel layer of a curved wall can be diagonally connected.
    aperture = component_containing(aperture, (cy, cx), kernel="square")
    return aperture, (cy, cx)


def place_chamber(labels, index, seed_chambers, x, y, p):
    parent, cell = index + seed_chambers, index + seed_chambers + 1
    allowed = labels == 0
    allowed[[0, -1], :] = False   # keep the outer ring empty
    allowed[:, [0, -1]] = False
    aperture, anchor = select_aperture(labels, parent, x, y,
        index*p.turning_angle_degrees + p.seed_orientation_degrees,
        p.guide_start_radius + index*p.guide_radius_increment, p.aperture_radius)
    if bool((aperture & ~allowed).any()):
        raise RuntimeError("Selected aperture intersects the computational boundary.")
    return Chamber(cell, parent, allowed, aperture, anchor)


def nucleate(labels, chamber, x, y, p):
    """Initial u: a soft disc just outside the aperture, pointing away from the parent."""
    parent = labels == chamber.parent
    center = torch.stack((x[parent].mean(), y[parent].mean()))
    anchor = torch.stack((x[chamber.anchor], y[chamber.anchor]))
    outward = anchor - center
    outward /= max(float(torch.linalg.norm(outward)), 1e-12)
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
    return (p.adhesion_strength * torch.exp(-distance / p.adhesion_range))
