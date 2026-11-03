"""Sequential area-constrained chambers attached to a central seed.

Existing chambers are immobile, excluded regions. New chambers attach to a
fixed aperture on their immediate predecessor. A short-range adhesion field
favours growing near any existing chamber. This is a grid phase-field model.
README.md describes the modules.

Run: python -m foram_colony --seed-mode bilocular   (writes output/bilocular/)
"""
from .parameters import ColonyParameters, ImplementationParameters, SEED_STYLES
from .colony import simulate_colony
from .output import save_colony, render_colony

__all__ = [
    "ColonyParameters",
    "ImplementationParameters",
    "SEED_STYLES",
    "simulate_colony",
    "save_colony",
    "render_colony",
]
