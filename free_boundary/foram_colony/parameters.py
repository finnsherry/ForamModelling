from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

# Precision of every floating-point tensor in the simulation.
DTYPE = torch.float32

# Newton steps per area projection. One is exact to rounding unless pixels hit
# the 0/1 clamp during the step; check_phase raises if the residual shows that
# it was not enough. A second step costs about 25% of the run time.
NEWTON_STEPS = 1

# The time-stepping window is the bounding box of u > WINDOW_TAIL plus a margin
# of WINDOW_MARGIN interface widths. If u exceeds WINDOW_SPILL next to the
# window border during an interval, the window is widened and the interval redone.
WINDOW_TAIL = 1e-6
WINDOW_SPILL = 1e-3
WINDOW_MARGIN = 8

# Safety factor on the explicit-Euler stability limit of the time step.
DT_SAFETY = 0.35

# Accepted area-projection residual, in multiples of DTYPE's machine epsilon
# times the area: a float sum is only accurate to about eps relative.
AREA_TOLERANCE_EPS = 64

# The adhesion field is computed out to this many adhesion ranges and is zero
# beyond; at 5 it has fallen below 0.7% of its value at the wall.
ADHESION_CUTOFF_RANGES = 5

# Growth "reached the computational edge" when u exceeds this next to the grid border.
EDGE_THRESHOLD = 0.5

# Fewest pixels a seed compartment may have and still count as resolved.
MIN_SEED_PIXELS = 12

# The flood fill checks for convergence once every this many dilations.
FLOOD_CHECK_EVERY = 8


@dataclass
class ColonyParameters:
    new_chambers: int = 6
    central_radius: float = 0.30
    seed_mode: Literal["unilocular", "bilocular"] = "bilocular"
    second_piece_radius_ratio: float = 1.10
    seed_separation_fraction: float = 0.60
    seed_orientation_degrees: float = 0.0
    spacing: float = 0.025
    half_width: float = 2.2
    first_chamber_area: float = 0.28
    chamber_area_ratio: float = 1.16
    turning_angle_degrees: float = 65.0
    formation_time: float = 2.5
    curvature_coefficient: float = 0.003
    interface_width: float = 0.04
    adhesion_strength: float = 1.0
    adhesion_range: float = 0.14
    aperture_radius: float = 0.075
    snapshots_per_chamber: int = 18
    max_dt: float = 0.006
    # Bilocular seed: radius of the first generating disc, as a fraction of central_radius.
    first_piece_radius_fraction: float = 0.70
    # Aperture placement: the guide point for new chamber i (from 0) lies at
    # distance guide_start_radius + i * guide_radius_increment from the origin.
    guide_start_radius: float = 0.55
    guide_radius_increment: float = 0.075
    # Nucleation: a soft disc of nucleus_radius, centred nucleus_offset outward
    # from the aperture and cut off beyond nucleus_support_radius.
    nucleus_offset: float = 0.07
    nucleus_radius: float = 0.13
    nucleus_support_radius: float = 0.32
    # Growth law: a sigmoid in time with this rate, centred at this fraction of formation_time.
    growth_rate: float = 0.65
    growth_midpoint_fraction: float = 0.48


# Everything that differs between the seed modes, apart from the geometry
# constructed in placement.make_seed.
SEED_STYLES = {
    "unilocular": {
        "names": ["Initial chamber (proloculus)"],
        "colors": None,
        "animation_title": "Unilocular seed",
        "first_frame": "chamber 1",
        "seed_title": "Unilocular central chamber",
        "summary_title": "Central chamber 1; sequential attachments",
    },
    "bilocular": {
        "names": ["Initial chamber (proloculus/protoconch)",
                  "Second chamber (deuteroloculus/deuteroconch)"],
        "colors": ["#553b83", "#a25db3"],
        "animation_title": "Two-chamber seed",
        "first_frame": "initial pair",
        "seed_title": "Two distinct connected initial chambers",
        "summary_title": "Initial chambers 1 and 2; later attachments",
    },
}
