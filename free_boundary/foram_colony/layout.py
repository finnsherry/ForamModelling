"""Colony layout: the initial seed, and where each new chamber attaches."""

import math
from dataclasses import dataclass

import torch

from .morphology import component_containing, dilate, is_connected
from .parameters import SEED_STYLES, ColonyParameters, ImplementationParameters


@dataclass
class Chamber:
    """Fixed constraints for the chamber currently growing."""
    cell: int
    parent: int
    allowed: torch.Tensor     # where u may be nonzero
    aperture: torch.Tensor    # where u is held at 1
    anchor: tuple[int, int]   # grid index of the aperture centre


def make_seed(p: ColonyParameters, q: ImplementationParameters, x, y):
    """Initial labels: one disc (unilocular) or two adjacent compartments (bilocular).

    For a bilocular seed, two overlapping generating discs are partitioned:
    compartment 1 retains its disc; compartment 2 occupies the second disc
    outside compartment 1. The resulting compartments have disjoint interiors
    and a common interface.
    """
    labels = torch.zeros(x.shape, dtype=torch.int16, device=x.device)
    if p.seed_mode == "unilocular":
        labels[torch.hypot(x, y) <= p.central_radius] = 1
    else:
        r1 = p.central_radius
        r2 = r1 * p.second_piece_radius_ratio
        distance = p.seed_separation_fraction * (r1 + r2)
        # The disc centres are at (-distance/2, 0) and (distance/2, 0).
        first = torch.hypot(x + 0.5*distance, y) <= r1
        second = torch.hypot(x - 0.5*distance, y) <= r2
        labels[first] = 1
        labels[second & ~first] = 2

    reports = []
    for cell, name in enumerate(SEED_STYLES[p.seed_mode]["names"], start=1):
        mask = labels == cell
        if int(mask.sum()) < q.min_seed_pixels or not is_connected(mask):
            raise ValueError(
                "An initial compartment is missing, disconnected or under-resolved. "
                "Refine the grid or adjust seed geometry."
            )
        reports.append({
            "chamber": cell,
            "name": name,
            "pixel_area": float(mask.sum()) * q.spacing**2,
            "centroid": [float(x[mask].mean()), float(y[mask].mean())],
        })
    if not is_connected(labels > 0):
        raise ValueError(
            "The central seed pieces do not connect; adjust their separation or grid_points."
        )
    return labels, reports


def select_aperture(labels, parent, allowed, x, y, angle, guide_radius, radius):
    """Choose an exposed, allowed parent wall nearest the guide point at (guide_radius, angle)."""
    candidates = dilate(labels == parent) & allowed
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


def place_chamber(labels, index, seed_chambers, x, y, p: ColonyParameters):
    parent, cell = index + seed_chambers, index + seed_chambers + 1
    allowed = labels == 0
    allowed[[0, -1], :] = False   # keep the outer ring empty
    allowed[:, [0, -1]] = False
    aperture, anchor = select_aperture(
        labels,
        parent,
        allowed,
        x,
        y,
        angle=index*p.turning_angle_degrees,
        guide_radius=p.guide_start_radius + index*p.guide_radius_increment,
        radius=p.aperture_radius,
    )
    return Chamber(cell, parent, allowed, aperture, anchor)
