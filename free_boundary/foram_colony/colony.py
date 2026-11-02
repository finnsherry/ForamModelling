"""The simulation driver: grows the colony chamber by chamber, place → grow → freeze.

Everything runs as torch tensors on `device`, in parameters.DTYPE. Results are
converted to numpy only at the end, for output.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .layout import make_seed, place_chamber
from .growth import adhesion_field, nucleate, reaction, reaction_rate_bound, target_area
from .morphology import dilate, is_connected
from .parameters import DTYPE, WINDOW_MARGIN, ColonyParameters
from .solver import integrate, stable_dt


@dataclass
class Record:
    """What the simulation accumulates for output."""
    frames: list = field(default_factory=list)
    history: list = field(default_factory=list)
    reports: list = field(default_factory=list)
    max_diffuse_error: float = 0.0


def chamber_targets(p, chamber_areas):
    """Final area of each new chamber: the supplied ones, or the geometric series from p."""
    if chamber_areas is None:
        chamber_areas = [p.first_chamber_area * p.chamber_area_ratio**i for i in range(p.new_chambers)]
    chamber_areas = [float(a) for a in chamber_areas]
    if len(chamber_areas) != p.new_chambers or not all(math.isfinite(a) and a > 0 for a in chamber_areas):
        raise ValueError("Supply one finite positive target area per new chamber.")
    return chamber_areas


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


def snapshot(record, labels, chamber, u, t, target, start_time, h):
    """Store a crisp view of the growing chamber for the animation and history."""
    binary = crisp_chamber(u, chamber.allowed, chamber.aperture, target(t), h)
    image = labels.clone()
    image[binary] = chamber.cell
    record.frames.append({"labels": image, "chamber": chamber.cell, "time": start_time + t})
    record.history.append([chamber.cell, t, start_time + t, target(t),
                           h**2*float(u.sum()), h**2*int(binary.sum())])


def grow(labels, chamber, final_area, start_time, x, y, p, record):
    """Evolve the new chamber's phase field from nucleation to final_area.

    Returns the final phase field and the number of time steps.
    """
    h = p.spacing
    u = nucleate(labels, chamber, x, y, p)
    seed_area = h**2 * float(u.sum())
    if final_area <= seed_area:
        raise ValueError(f"Chamber {chamber.cell} target {final_area:.4g} must exceed seed area {seed_area:.4g}.")
    target = target_area(seed_area, final_area, p)
    adhesion = adhesion_field(labels > 0, p)
    n = p.snapshots_per_chamber - 1
    u, steps, max_error = integrate(
        u, chamber.allowed, chamber.aperture,
        diffusion=p.curvature_coefficient,
        reaction=lambda u, window: reaction(u, window.crop(adhesion), p),
        target=target,
        save_times=[p.formation_time * i / n for i in range(1, n + 1)],
        dt_max=stable_dt(h, p.curvature_coefficient, reaction_rate_bound(p), p.max_dt),
        h=h,
        margin=math.ceil(WINDOW_MARGIN * p.interface_width / h),
        on_snapshot=lambda u, t: snapshot(record, labels, chamber, u, t, target, start_time, h))
    record.max_diffuse_error = max(record.max_diffuse_error, max_error)
    return u, steps


def freeze(labels, chamber, u, final_area, steps, x, y, p, record):
    """Commit the grown chamber to labels and report its contacts."""
    h = p.spacing
    occupied = labels > 0
    mask = crisp_chamber(u, chamber.allowed, chamber.aperture, final_area, h)
    # Checked only here: the snapshots crisped in grow() are just for display.
    if not is_connected(mask):
        raise RuntimeError("The new chamber disconnected; reduce growth/adhesion or revise the aperture.")
    contacts = sorted(int(v) for v in torch.unique(labels[dilate(mask) & occupied]) if v > 0)
    if chamber.parent not in contacts:
        raise RuntimeError("A new chamber lost contact with its parent.")
    labels[mask] = chamber.cell
    frozen_area = h**2*int(mask.sum())
    record.reports.append({"chamber": chamber.cell, "parent": chamber.parent, "target_area": final_area,
        "diffuse_final_area": h**2*float(u.sum()), "frozen_pixel_area": frozen_area,
        "area_rounding_error": abs(frozen_area - final_area),
        "contacts_with_earlier_chambers": contacts, "aperture_pixels": int(chamber.aperture.sum()),
        "aperture_center": [float(x[chamber.anchor]), float(y[chamber.anchor])], "steps": steps})


def simulate_colony(parameters=None, chamber_areas=None, device="cpu"):
    """Grow the colony on `device` (e.g. "cuda")."""
    p = parameters or ColonyParameters()
    chamber_areas = chamber_targets(p, chamber_areas)
    coords = torch.arange(-p.half_width, p.half_width + p.spacing/2, p.spacing,
                          dtype=DTYPE, device=device)
    x, y = torch.meshgrid(coords, coords, indexing="xy")
    labels, seed_reports = make_seed(p, x, y)
    initial_labels = labels.clone()
    seed_chambers = len(seed_reports)
    record = Record(frames=[{"labels": labels.clone(), "chamber": seed_chambers, "time": 0.0}])
    for index, final_area in enumerate(chamber_areas):
        chamber = place_chamber(labels, index, seed_chambers, x, y, p)
        u, steps = grow(labels, chamber, final_area, index*p.formation_time, x, y, p, record)
        freeze(labels, chamber, u, final_area, steps, x, y, p, record)
    for frame in record.frames:
        frame["labels"] = frame["labels"].cpu().numpy()
    return {"parameters": p, "coordinates": x[0].cpu().numpy(), "frames": record.frames,
        "labels": labels.cpu().numpy(), "history": torch.tensor(record.history, dtype=DTYPE).numpy(),
        "reports": record.reports, "seed_chambers": seed_chambers, "seed_reports": seed_reports,
        "initial_labels": initial_labels.cpu().numpy(),
        "max_diffuse_area_error_all_steps": record.max_diffuse_error,
        "pixel_area_tolerance": p.spacing**2/2, "growth_label": "Illustrative sigmoid for each chamber"}
