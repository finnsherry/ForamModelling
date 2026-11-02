"""Growth law and the simulation driver: for each new chamber, place → grow → freeze.

Everything runs as torch tensors on `device`, in parameters.DTYPE. Results are
converted to numpy only at the end, for output.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .morphology import dilate, is_connected
from .parameters import (DTYPE, EDGE_THRESHOLD, WINDOW_MARGIN, WINDOW_SPILL, WINDOW_TAIL,
                         ColonyParameters, SEED_STYLES)
from .phase_field import (Window, check_phase, corrected_phase_area, crisp_chamber, edge_values,
                          phase_rhs, stable_dt)
from .placement import adhesion_field, make_seed, nucleate, place_chamber


@dataclass
class Record:
    """What the simulation accumulates for output."""
    frames: list = field(default_factory=list)
    history: list = field(default_factory=list)
    reports: list = field(default_factory=list)
    max_diffuse_error: float = 0.0


def demo_growth(initial_area, final_area, p):
    """Illustrative normalized sigmoid over p.formation_time, NOT a fit to foraminiferal observations."""
    if final_area <= initial_area:
        raise ValueError("Final area must exceed the seed area.")
    end_time = p.formation_time
    rate, midpoint = p.growth_rate, p.growth_midpoint_fraction * end_time
    def sigmoid(t):
        return 1.0 / (1.0 + math.exp(-rate * (t - midpoint)))
    f0, f1 = sigmoid(0.0), sigmoid(end_time)
    scale = (final_area - initial_area) / (f1 - f0)
    def area(t):
        return initial_area + scale * (sigmoid(t) - f0)
    def rate_of_area(t):
        f = sigmoid(t)
        return scale * rate * f * (1.0 - f)
    return area, rate_of_area


def validate(p, chamber_areas):
    """Check parameters; return the target area of each new chamber."""
    if p.seed_mode not in SEED_STYLES:
        raise ValueError(f"seed_mode must be one of {sorted(SEED_STYLES)}.")
    positive = [p.central_radius, p.spacing, p.half_width, p.first_chamber_area,
                p.chamber_area_ratio, p.formation_time, p.curvature_coefficient,
                p.interface_width, p.adhesion_range, p.aperture_radius, p.max_dt,
                p.second_piece_radius_ratio, p.seed_separation_fraction,
                p.first_piece_radius_fraction, p.guide_start_radius, p.nucleus_radius,
                p.nucleus_support_radius, p.growth_rate]
    if not all(math.isfinite(v) and v > 0 for v in positive):
        raise ValueError("Geometric, area and numerical scales must be positive and finite.")
    nonnegative = [p.guide_radius_increment, p.nucleus_offset, p.growth_midpoint_fraction]
    if not all(math.isfinite(v) and v >= 0 for v in nonnegative):
        raise ValueError("Guide increment, nucleus offset and growth midpoint must be nonnegative and finite.")
    if p.new_chambers < 1 or p.snapshots_per_chamber < 2 or p.adhesion_strength < 0:
        raise ValueError("Invalid chamber count, snapshot count or adhesion strength.")
    if p.seed_mode == "bilocular" and p.seed_separation_fraction >= 1:
        raise ValueError("Bilocular generating discs must overlap: seed_separation_fraction < 1.")
    if chamber_areas is None:
        chamber_areas = [p.first_chamber_area * p.chamber_area_ratio**i for i in range(p.new_chambers)]
    chamber_areas = [float(a) for a in chamber_areas]
    if len(chamber_areas) != p.new_chambers or not all(math.isfinite(a) and a > 0 for a in chamber_areas):
        raise ValueError("Supply one finite positive target area per new chamber.")
    return chamber_areas


def snapshot(record, labels, chamber, u, t, target, start_time, h):
    """Store a crisp view of the growing chamber for the animation and history."""
    binary = crisp_chamber(u, chamber.allowed, chamber.aperture, target(t), h)
    image = labels.clone()
    image[binary] = chamber.cell
    record.frames.append({"labels": image, "chamber": chamber.cell, "time": start_time + t})
    record.history.append([chamber.cell, t, start_time + t, target(t),
                           h**2*float(u.sum()), h**2*int(binary.sum())])


def grow(labels, chamber, final_area, start_time, x, y, p, record):
    """Evolve u from nucleation to final_area; returns the final phase field and step count.

    The time stepping runs on a window around the growing chamber, not the whole
    grid. The window's outer ring is held at u = 0, like the grid's, so the step
    functions work on it unchanged. At every snapshot the window is refitted to
    the chamber. If u reached a side of the window during an interval, the window
    was too small: that side is widened and the interval is repeated. A side
    that lies on the grid border cannot be widened; there the usual edge check
    applies instead.
    """
    h = p.spacing
    u_full = nucleate(labels, chamber, x, y, p)
    seed_area = h**2 * float(u_full.sum())
    if final_area <= seed_area:
        raise ValueError(f"Chamber {chamber.cell} target {final_area:.4g} must exceed seed area {seed_area:.4g}.")
    target, _ = demo_growth(seed_area, final_area, p)
    adhesion_full = adhesion_field(labels > 0, p)
    dt_cap = stable_dt(p)
    margin = math.ceil(WINDOW_MARGIN * p.interface_width / h)
    t, steps = 0.0, 0
    max_error = torch.zeros((), device=u_full.device, dtype=DTYPE)
    snapshot(record, labels, chamber, u_full, t, target, start_time, h)
    window = Window.around(u_full > WINDOW_TAIL, margin)
    n = p.snapshots_per_chamber - 1
    for save_time in (p.formation_time * i / n for i in range(1, n + 1)):
        t_start, steps_start = t, steps
        while True:
            allowed = window.crop(chamber.allowed).clone()
            allowed[[0, -1], :] = False
            allowed[:, [0, -1]] = False
            aperture, adhesion = window.crop(chamber.aperture), window.crop(adhesion_full)
            u = window.crop(u_full) * allowed
            # Nothing in this loop reads back from the device; problems are only
            # checked at the snapshot, from running maxima kept on the device.
            interval_error = torch.zeros((), device=u.device, dtype=DTYPE)
            edges = torch.zeros(4, device=u.device, dtype=DTYPE)
            while t < save_time - 1e-12:
                dt = min(dt_cap, save_time - t)
                u, residual = corrected_phase_area(u + dt*phase_rhs(u, adhesion, p), target(t + dt),
                                                   allowed, aperture, h)
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
        snapshot(record, labels, chamber, u_full, t, target, start_time, h)
        window = Window.around(u_full > WINDOW_TAIL, margin)
    record.max_diffuse_error = max(record.max_diffuse_error, float(max_error))
    return u_full, steps


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
    chamber_areas = validate(p, chamber_areas)
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
