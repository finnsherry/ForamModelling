from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


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


# Everything that differs between the seed modes, apart from the geometry
# constructed in placement.make_seed.
SEED_STYLES = {
    "unilocular": {
        "names": ["Original one-chamber central seed"],
        "colors": None,
        "animation_title": "Central seed",
        "first_frame": "chamber 1",
        "seed_title": "Original unilocular central chamber",
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
