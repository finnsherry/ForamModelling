import math
from dataclasses import dataclass
from typing import Literal

import torch

@dataclass
class ColonyParameters:
    """Parameters of the seed, the chambers and the growth model; validated on creation.

    Lengths are in units of the grid's half-width: the grid covers [-1, 1] in x
    and in y. Times are in formation times: each chamber grows over one time unit.

    The docstring under each field documents it, and is the help text of its
    command-line option.
    """

    new_chambers: int = 6
    """Number of chambers grown after the seed."""

    central_radius: float = 0.0955
    """Radius of the unilocular seed disc, or of the first bilocular generating disc."""

    seed_mode: Literal["unilocular", "bilocular"] = "bilocular"
    """unilocular: the seed is one disc. bilocular: two adjacent compartments."""

    second_piece_radius_ratio: float = 1.10
    """Bilocular: radius of the second generating disc relative to the first."""

    seed_separation_fraction: float = 0.60
    """Bilocular: distance between the disc centres, as a fraction of the sum of their radii (< 1)."""

    first_chamber_area: float = 0.0579
    """Final area of the first new chamber, unless chamber_areas is given."""

    chamber_area_ratio: float = 1.16
    """Final area of each new chamber relative to the previous one, unless chamber_areas is given."""

    chamber_areas: tuple[float, ...] | None = None
    """Final area of each new chamber, one per new chamber. Overrides first_chamber_area
    and chamber_area_ratio."""

    turning_angle_degrees: float = 65.0
    """Angle by which the aperture guide point turns from one new chamber to the next."""

    curvature_coefficient: float = 0.00155
    """Diffusion coefficient of the phase field; sets how strongly curvature is smoothed."""

    interface_width: float = 0.0182
    """Width of the diffuse chamber interface."""

    adhesion_strength: float = 2.5
    """Strength of the attraction towards existing chambers; 0 turns it off."""

    adhesion_range: float = 0.0636
    """Decay length of the adhesion with distance from existing chambers."""

    aperture_radius: float = 0.0341
    """Radius of the aperture, the patch of parent wall a new chamber grows from."""

    snapshots_per_chamber: int = 18
    """Frames recorded per chamber, including its nucleus."""

    guide_start_radius: float = 0.25
    """The aperture of new chamber i (from 0) is the exposed parent wall nearest a guide
    point at distance guide_start_radius + i * guide_radius_increment from the origin."""

    guide_radius_increment: float = 0.0341
    """Increase of the guide point's distance from the origin per new chamber."""

    nucleus_offset: float = 0.0318
    """Distance of the nucleus centre outward from the aperture centre."""

    nucleus_radius: float = 0.0591
    """Radius of the soft disc a new chamber starts from."""

    growth_rate: float = 1.625
    """Rate of the sigmoid in time that the target area follows."""

    growth_midpoint_fraction: float = 0.48
    """Midpoint of the growth sigmoid, as a fraction of the formation time."""

    def __post_init__(self):
        if self.seed_mode not in SEED_STYLES:
            raise ValueError(f"seed_mode must be one of {sorted(SEED_STYLES)}.")

        positive = [
            self.central_radius,
            self.first_chamber_area,
            self.chamber_area_ratio,
            self.curvature_coefficient,
            self.interface_width,
            self.adhesion_range,
            self.aperture_radius,
            self.second_piece_radius_ratio,
            self.seed_separation_fraction,
            self.guide_start_radius,
            self.nucleus_radius,
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


@dataclass
class ImplementationParameters:
    """Numerical settings that don't define the model; validated on creation.

    The docstring under each field documents it, and is the help text of its
    command-line option.
    """

    grid_points: int = 177
    """Grid samples per side; the grid covers [-1, 1] in x and in y."""

    dtype: Literal["float32", "float64"] = "float32"
    """Precision of every floating-point tensor in the simulation."""

    newton_steps: int = 1
    """Newton steps per area projection. One is exact to rounding unless pixels hit the
    0/1 clamp during the step; the run stops if the residual shows it was not enough.
    A second step costs about 25% of the run time."""

    window_tail: float = 1e-6
    """The time-stepping window is the bounding box of u > window_tail plus a margin,
    refitted at every snapshot."""

    window_margin: float = 8.0
    """Margin of the time-stepping window, in interface widths."""

    window_spill: float = 1e-3
    """The run stops if u exceeds this next to the window border."""

    area_tolerance_eps: float = 64.0
    """Accepted area-projection residual, in multiples of the dtype's machine epsilon times
    the area: a float sum is only accurate to about eps relative."""

    adhesion_cutoff_ranges: float = 5.0
    """The adhesion field is computed out to this many adhesion ranges and is zero beyond;
    at 5 it has fallen below 0.7% of its value at the wall."""

    nucleus_cutoff_widths: float = 4.75
    """The nucleus is cut off beyond this many interface widths outside its radius; at
    4.75 its profile has fallen below 0.9%."""

    min_seed_pixels: int = 12
    """Fewest pixels a seed compartment may have and still count as resolved."""

    def __post_init__(self):
        if self.grid_points < 3 or self.newton_steps < 1 or self.min_seed_pixels < 1:
            raise ValueError("grid_points must be at least 3, newton_steps and min_seed_pixels at least 1.")
        positive = [
            self.window_tail,
            self.window_margin,
            self.window_spill,
            self.area_tolerance_eps,
            self.adhesion_cutoff_ranges,
            self.nucleus_cutoff_widths,
        ]
        if not all(math.isfinite(v) and v > 0 for v in positive):
            raise ValueError("Window, tolerance and cutoff settings must be positive and finite.")

    @property
    def spacing(self):
        return 2 / (self.grid_points - 1)

    @property
    def torch_dtype(self):
        return getattr(torch, self.dtype)


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
