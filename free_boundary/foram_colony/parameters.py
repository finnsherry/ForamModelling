import math
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
# of WINDOW_MARGIN interface widths, refitted at every snapshot. If u exceeds
# WINDOW_SPILL next to the window border, the run stops with an error.
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

# Fewest pixels a seed compartment may have and still count as resolved.
MIN_SEED_PIXELS = 12

# The flood fill checks for convergence once every this many dilations.
FLOOD_CHECK_EVERY = 8


@dataclass
class ColonyParameters:
    """Parameters of the seed, the chambers, the growth model and the grid; validated on creation.

    The docstring under each field documents it, and is the help text of its
    command-line option.
    """

    new_chambers: int = 6
    """Number of chambers grown after the seed."""

    central_radius: float = 0.30
    """Radius of the unilocular seed disc; the bilocular discs are scaled from it."""

    seed_mode: Literal["unilocular", "bilocular"] = "bilocular"
    """unilocular: the seed is one disc. bilocular: two adjacent compartments."""

    second_piece_radius_ratio: float = 1.10
    """Bilocular: radius of the second generating disc relative to the first."""

    seed_separation_fraction: float = 0.60
    """Bilocular: distance between the disc centres, as a fraction of the sum of their radii (< 1)."""

    seed_orientation_degrees: float = 0.0
    """Direction from the first seed disc to the second, counterclockwise from +x.
    The aperture guide angles are measured from it."""

    spacing: float = 0.025
    """Grid spacing."""

    half_width: float = 2.2
    """The grid covers [-half_width, half_width] in x and in y."""

    first_chamber_area: float = 0.28
    """Final area of the first new chamber, unless chamber_areas is given."""

    chamber_area_ratio: float = 1.16
    """Final area of each new chamber relative to the previous one, unless chamber_areas is given."""

    chamber_areas: tuple[float, ...] | None = None
    """Final area of each new chamber, one per new chamber. Overrides first_chamber_area
    and chamber_area_ratio."""

    turning_angle_degrees: float = 65.0
    """Angle by which the aperture guide point turns from one new chamber to the next."""

    formation_time: float = 2.5
    """Time a chamber takes to grow from its nucleus to its final area."""

    curvature_coefficient: float = 0.003
    """Diffusion coefficient of the phase field; sets how strongly curvature is smoothed."""

    interface_width: float = 0.04
    """Width of the diffuse chamber interface."""

    adhesion_strength: float = 1.0
    """Strength of the attraction towards existing chambers; 0 turns it off."""

    adhesion_range: float = 0.14
    """Decay length of the adhesion with distance from existing chambers."""

    aperture_radius: float = 0.075
    """Radius of the aperture, the patch of parent wall a new chamber grows from."""

    snapshots_per_chamber: int = 18
    """Frames recorded per chamber, including its nucleus."""

    max_dt: float = 0.006
    """Largest time step; the explicit-Euler stability limit can make it smaller."""

    first_piece_radius_fraction: float = 0.70
    """Bilocular: radius of the first generating disc, as a fraction of central_radius."""

    guide_start_radius: float = 0.55
    """The aperture of new chamber i (from 0) is the exposed parent wall nearest a guide
    point at distance guide_start_radius + i * guide_radius_increment from the origin."""

    guide_radius_increment: float = 0.075
    """Increase of the guide point's distance from the origin per new chamber."""

    nucleus_offset: float = 0.07
    """Distance of the nucleus centre outward from the aperture centre."""

    nucleus_radius: float = 0.13
    """Radius of the soft disc a new chamber starts from."""

    nucleus_support_radius: float = 0.32
    """The nucleus is cut off beyond this distance from its centre."""

    growth_rate: float = 0.65
    """Rate of the sigmoid in time that the target area follows."""

    growth_midpoint_fraction: float = 0.48
    """Midpoint of the growth sigmoid, as a fraction of formation_time."""

    def __post_init__(self):
        if self.seed_mode not in SEED_STYLES:
            raise ValueError(f"seed_mode must be one of {sorted(SEED_STYLES)}.")

        positive = [
            self.central_radius,
            self.spacing,
            self.half_width,
            self.first_chamber_area,
            self.chamber_area_ratio,
            self.formation_time,
            self.curvature_coefficient,
            self.interface_width,
            self.adhesion_range,
            self.aperture_radius,
            self.max_dt,
            self.second_piece_radius_ratio,
            self.seed_separation_fraction,
            self.first_piece_radius_fraction,
            self.guide_start_radius,
            self.nucleus_radius,
            self.nucleus_support_radius,
            self.growth_rate,
        ]
        if not all(math.isfinite(v) and v > 0 for v in positive):
            raise ValueError("Geometric, area and numerical scales must be positive and finite.")

        nonnegative = [
            self.guide_radius_increment,
            self.nucleus_offset,
            self.growth_midpoint_fraction,
        ]
        if not all(math.isfinite(v) and v >= 0 for v in nonnegative):
            raise ValueError(
                "Guide increment, nucleus offset and growth midpoint must be nonnegative and finite."
            )

        if self.new_chambers < 1 or self.snapshots_per_chamber < 2 or self.adhesion_strength < 0:
            raise ValueError("Invalid chamber count, snapshot count or adhesion strength.")
        if self.seed_mode == "bilocular" and self.seed_separation_fraction >= 1:
            raise ValueError("Bilocular generating discs must overlap: seed_separation_fraction < 1.")

        if self.chamber_areas is not None:
            self.chamber_areas = tuple(float(a) for a in self.chamber_areas)
            if (
                len(self.chamber_areas) != self.new_chambers
                or not all(math.isfinite(a) and a > 0 for a in self.chamber_areas)
            ):
                raise ValueError("chamber_areas needs one finite positive area per new chamber.")

    def final_areas(self):
        """The final area of each new chamber, in formation order."""
        if self.chamber_areas is not None:
            return list(self.chamber_areas)
        return [self.first_chamber_area * self.chamber_area_ratio**i for i in range(self.new_chambers)]


# Per seed mode: the names of the seed compartments, and their plot colours
# (None: use the same colour scale as the later chambers).
SEED_STYLES = {
    "unilocular": {
        "names": [
            "Initial chamber (proloculus)",
        ],
        "colors": None,
    },
    "bilocular": {
        "names": [
            "Initial chamber (proloculus/protoconch)",
            "Second chamber (deuteroloculus/deuteroconch)",
        ],
        "colors": ["#553b83", "#a25db3"],
    },
}
