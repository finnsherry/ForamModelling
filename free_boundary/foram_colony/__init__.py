"""Sequential area-constrained chambers attached to a central seed.

Existing chambers are immobile, excluded regions. New chambers attach to a
fixed aperture on their immediate predecessor. A short-range adhesion field
favours growing near any existing chamber. This is a grid phase-field model.

Modules:
    parameters   ColonyParameters and per-seed-mode labels
    phase_field  Allen–Cahn step, area projection, crisping (pure numerics)
    placement    seed geometry, aperture choice, nucleation, adhesion (model rules)
    colony       growth law and the place → grow → freeze driver
    output       saving results and rendering figures

Run: python -m foram_colony --seed-mode bilocular   (writes output/bilocular/)
"""
from .parameters import ColonyParameters, SEED_STYLES
from .colony import simulate_colony
from .output import save_colony, render_colony

__all__ = ["ColonyParameters", "SEED_STYLES", "simulate_colony", "save_colony", "render_colony"]
