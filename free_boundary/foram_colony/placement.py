"""Model rules: seed geometry, where each chamber attaches, how it starts, what attracts it."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.special import expit

from .parameters import SEED_STYLES


@dataclass
class Chamber:
    """Fixed constraints for the chamber currently growing."""
    cell: int
    parent: int
    allowed: np.ndarray       # where u may be nonzero
    aperture: np.ndarray      # where u is held at 1
    anchor: tuple[int, int]   # grid index of the aperture centre


def make_seed(p, x, y):
    """Two distinct adjacent compartments, or the original one-disc seed.

    For a bilocular seed, two overlapping generating discs are partitioned:
    compartment 1 retains its disc; compartment 2 occupies the second disc
    outside compartment 1. The resulting compartments have disjoint interiors
    and a common interface. This is geometry, not a universal embryonic shape.
    """
    labels = np.zeros_like(x, dtype=np.int16)
    if p.seed_mode == "unilocular":
        labels[np.hypot(x, y) <= p.central_radius] = 1
    else:
        r1 = 0.70 * p.central_radius
        r2 = r1 * p.second_piece_radius_ratio
        distance = p.seed_separation_fraction * (r1 + r2)
        axis = np.array([np.cos(np.deg2rad(p.seed_orientation_degrees)),
                         np.sin(np.deg2rad(p.seed_orientation_degrees))])
        c1, c2 = -0.5 * distance * axis, 0.5 * distance * axis
        first = np.hypot(x - c1[0], y - c1[1]) <= r1
        second = np.hypot(x - c2[0], y - c2[1]) <= r2
        labels[first] = 1
        labels[second & ~first] = 2
    reports = []
    for cell, name in enumerate(SEED_STYLES[p.seed_mode]["names"], start=1):
        mask = labels == cell
        _, components = ndimage.label(mask)
        if mask.sum() < 12 or components != 1:
            raise ValueError("An initial compartment is missing, disconnected or under-resolved. Refine the grid or adjust seed geometry.")
        reports.append({"chamber":cell,"name":name,"pixel_area":float(mask.sum()*p.spacing**2),
                        "centroid":[float(x[mask].mean()),float(y[mask].mean())]})
    _, components = ndimage.label(labels > 0)
    if components != 1:
        raise ValueError("The central seed pieces do not connect; adjust their separation or grid spacing.")
    return labels, reports


def select_aperture(labels, parent, x, y, angle, radius, guide_index=None):
    """Choose an exposed parent wall nearest a prescribed coiling guide point."""
    parent_mask = labels == parent
    candidates = ndimage.binary_dilation(parent_mask) & (labels == 0)
    candidates[[0, -1], :] = False
    candidates[:, [0, -1]] = False
    rows, columns = np.nonzero(candidates)
    if len(rows) == 0:
        raise RuntimeError("The previous chamber has no exposed boundary.")
    theta = np.deg2rad(angle)
    # This aperture-placement rule is a geometric input, not derived biology.
    guide_radius = 0.55 + 0.075 * ((parent - 1) if guide_index is None else guide_index)
    guide = guide_radius * np.array([np.cos(theta), np.sin(theta)])
    scores = (x[rows, columns] - guide[0])**2 + (y[rows, columns] - guide[1])**2
    selected = np.argmin(scores)
    cy, cx = int(rows[selected]), int(columns[selected])
    distance = np.hypot(x - x[cy, cx], y - y[cy, cx])
    aperture = candidates & (distance <= radius)
    # The outside pixel layer of a curved wall can be diagonally connected.
    parts, _ = ndimage.label(aperture, structure=np.ones((3, 3), dtype=int))
    aperture = parts == parts[cy, cx]
    return aperture, (cy, cx)


def place_chamber(labels, index, seed_chambers, x, y, p):
    parent, cell = index + seed_chambers, index + seed_chambers + 1
    allowed = labels == 0
    allowed[[0, -1], :] = False   # keep the outer ring empty
    allowed[:, [0, -1]] = False
    aperture, anchor = select_aperture(labels, parent, x, y,
        index*p.turning_angle_degrees + p.seed_orientation_degrees,
        p.aperture_radius, guide_index=index)
    if (aperture & ~allowed).any():
        raise RuntimeError("Selected aperture intersects the computational boundary.")
    return Chamber(cell, parent, allowed, aperture, anchor)


def nucleate(labels, chamber, x, y, p):
    """Initial u: a soft disc just outside the aperture, pointing away from the parent."""
    parent = labels == chamber.parent
    center = np.array([x[parent].mean(), y[parent].mean()])
    anchor = np.array([x[chamber.anchor], y[chamber.anchor]])
    outward = anchor - center
    outward /= max(np.linalg.norm(outward), 1e-12)
    blob_center = anchor + 0.07*outward
    distance = np.hypot(x - blob_center[0], y - blob_center[1])
    u = expit(-(distance - 0.13) / p.interface_width) * chamber.allowed
    # Compact support prevents phase nucleation far from the seeded chamber.
    u[distance > 0.32] = 0.0
    u[chamber.aperture] = 1.0
    return u


def adhesion_field(occupied, p):
    """Attraction that decays with distance from any existing chamber."""
    distance = ndimage.distance_transform_edt(~occupied) * p.spacing
    return p.adhesion_strength * np.exp(-distance / p.adhesion_range)
