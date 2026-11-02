"""Sequential area-constrained chambers attached to a central seed.

Existing chambers are immobile, excluded regions. New chambers attach to a
fixed aperture on their immediate predecessor. A short-range adhesion field
favours growing near any existing chamber. This is a grid phase-field model.

Modules:
    parameters   ColonyParameters (validated on creation), constants, per-seed-mode labels
    layout       the initial seed, and where each new chamber attaches
    growth       how one chamber grows: reaction term, adhesion, nucleus, target area
    solver       model-free time integration with the area constraint
    morphology   dilation, connectivity and distance transform on masks
    colony       the place → grow → freeze driver
    output       saving results and rendering figures

Run: python -m foram_colony --seed-mode bilocular   (writes output/bilocular/)
"""
from .parameters import ColonyParameters, SEED_STYLES
from .colony import simulate_colony
from .output import save_colony, render_colony

__all__ = ["ColonyParameters", "SEED_STYLES", "simulate_colony", "save_colony", "render_colony"]
