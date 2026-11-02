"""Growth law and the simulation driver: for each new chamber, place → grow → freeze."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .parameters import ColonyParameters, SEED_STYLES
from .phase_field import check_phase, corrected_phase_area, crisp_chamber, phase_rhs, stable_dt
from .placement import adhesion_field, make_seed, nucleate, place_chamber


@dataclass
class Record:
    """What the simulation accumulates for output."""
    frames: list = field(default_factory=list)
    history: list = field(default_factory=list)
    reports: list = field(default_factory=list)
    max_diffuse_error: float = 0.0


def demo_growth(initial_area, final_area, end_time):
    """Illustrative normalized sigmoid, NOT a fit to foraminiferal observations."""
    if final_area <= initial_area:
        raise ValueError("Final area must exceed the seed area.")
    rate, midpoint = 0.65, end_time * 0.48
    def sigmoid(t):
        return 1.0 / (1.0 + np.exp(-rate * (t - midpoint)))
    f0, f1 = sigmoid(0.0), sigmoid(end_time)
    scale = (final_area - initial_area) / (f1 - f0)
    def area(t):
        return initial_area + scale * (sigmoid(t) - f0)
    def rate_of_area(t):
        f = sigmoid(t)
        return scale * rate * f * (1.0 - f)
    return area, rate_of_area


def make_grid(p):
    coords = np.arange(-p.half_width, p.half_width + p.spacing/2, p.spacing)
    return np.meshgrid(coords, coords)


def validate(p, chamber_areas):
    """Check parameters; return the target area of each new chamber."""
    if p.seed_mode not in SEED_STYLES:
        raise ValueError(f"seed_mode must be one of {sorted(SEED_STYLES)}.")
    positive = [p.central_radius, p.spacing, p.half_width, p.first_chamber_area,
                p.chamber_area_ratio, p.formation_time, p.curvature_coefficient,
                p.interface_width, p.adhesion_range, p.aperture_radius, p.max_dt,
                p.second_piece_radius_ratio, p.seed_separation_fraction]
    if not all(np.isfinite(v) and v > 0 for v in positive):
        raise ValueError("Geometric, area and numerical scales must be positive and finite.")
    if p.new_chambers < 1 or p.snapshots_per_chamber < 2 or p.adhesion_strength < 0:
        raise ValueError("Invalid chamber count, snapshot count or adhesion strength.")
    if p.seed_mode == "bilocular" and p.seed_separation_fraction >= 1:
        raise ValueError("Bilocular generating discs must overlap: seed_separation_fraction < 1.")
    if chamber_areas is None:
        chamber_areas = p.first_chamber_area * p.chamber_area_ratio**np.arange(p.new_chambers)
    chamber_areas = np.asarray(chamber_areas, dtype=float)
    if len(chamber_areas) != p.new_chambers or np.any(chamber_areas <= 0) or not np.isfinite(chamber_areas).all():
        raise ValueError("Supply one finite positive target area per new chamber.")
    return chamber_areas


def snapshot(record, labels, chamber, u, t, target, start_time, h):
    """Store a crisp view of the growing chamber for the animation and history."""
    binary = crisp_chamber(u, chamber.allowed, chamber.aperture, float(target(t)), h)
    image = labels.copy()
    image[binary] = chamber.cell
    record.frames.append({"labels": image, "chamber": chamber.cell, "time": start_time + t})
    record.history.append([chamber.cell, t, start_time + t, float(target(t)),
                           h**2*u.sum(), h**2*binary.sum()])


def grow(labels, chamber, final_area, start_time, x, y, p, record):
    """Evolve u from nucleation to final_area; returns the final phase field and step count."""
    h = p.spacing
    u = nucleate(labels, chamber, x, y, p)
    seed_area = h**2 * u.sum()
    if final_area <= seed_area:
        raise ValueError(f"Chamber {chamber.cell} target {final_area:.4g} must exceed seed area {seed_area:.4g}.")
    target, _ = demo_growth(seed_area, final_area, p.formation_time)
    adhesion = adhesion_field(labels > 0, p)
    dt_cap = stable_dt(p)
    t, steps = 0.0, 0
    snapshot(record, labels, chamber, u, t, target, start_time, h)
    for save_time in np.linspace(0.0, p.formation_time, p.snapshots_per_chamber)[1:]:
        while t < save_time - 1e-12:
            dt = min(dt_cap, save_time - t)
            candidate = np.clip(u + dt*phase_rhs(u, adhesion, p), 0, 1) * chamber.allowed
            u = corrected_phase_area(candidate, float(target(t + dt)), chamber.allowed, chamber.aperture, h)
            t += dt
            steps += 1
            record.max_diffuse_error = max(record.max_diffuse_error, abs(h**2*u.sum() - float(target(t))))
            check_phase(u)
        snapshot(record, labels, chamber, u, t, target, start_time, h)
    return u, steps


def freeze(labels, chamber, u, final_area, steps, x, y, p, record):
    """Commit the grown chamber to labels and report its contacts."""
    h = p.spacing
    occupied = labels > 0
    mask = crisp_chamber(u, chamber.allowed, chamber.aperture, final_area, h)
    contacts = sorted(int(v) for v in np.unique(labels[ndimage.binary_dilation(mask) & occupied]) if v > 0)
    if chamber.parent not in contacts:
        raise RuntimeError("A new chamber lost contact with its parent.")
    labels[mask] = chamber.cell
    record.reports.append({"chamber": chamber.cell, "parent": chamber.parent, "target_area": float(final_area),
        "diffuse_final_area": float(h**2*u.sum()), "frozen_pixel_area": float(h**2*mask.sum()),
        "area_rounding_error": float(abs(h**2*mask.sum() - final_area)),
        "contacts_with_earlier_chambers": contacts, "aperture_pixels": int(chamber.aperture.sum()),
        "aperture_center": [float(x[chamber.anchor]), float(y[chamber.anchor])], "steps": steps})


def simulate_colony(parameters=None, chamber_areas=None):
    p = parameters or ColonyParameters()
    chamber_areas = validate(p, chamber_areas)
    x, y = make_grid(p)
    labels, seed_reports = make_seed(p, x, y)
    initial_labels = labels.copy()
    seed_chambers = len(seed_reports)
    record = Record(frames=[{"labels": labels.copy(), "chamber": seed_chambers, "time": 0.0}])
    for index, final_area in enumerate(chamber_areas):
        chamber = place_chamber(labels, index, seed_chambers, x, y, p)
        u, steps = grow(labels, chamber, final_area, index*p.formation_time, x, y, p, record)
        freeze(labels, chamber, u, final_area, steps, x, y, p, record)
    return {"parameters": p, "coordinates": x[0], "frames": record.frames, "labels": labels,
        "history": np.asarray(record.history), "reports": record.reports,
        "seed_chambers": seed_chambers, "seed_reports": seed_reports,
        "initial_labels": initial_labels,
        "max_diffuse_area_error_all_steps": record.max_diffuse_error,
        "pixel_area_tolerance": p.spacing**2/2, "growth_label": "Illustrative sigmoid for each chamber"}
