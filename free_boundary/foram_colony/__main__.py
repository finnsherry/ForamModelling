"""Command line: python -m foram_colony --seed-mode bilocular"""
import argparse
import json
from pathlib import Path

import torch

from .parameters import ColonyParameters, SEED_STYLES
from .colony import simulate_colony
from .output import save_colony, render_colony


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",help="Output directory (default: output/<seed-mode>)")
    parser.add_argument("--chambers",type=int,default=6,help="Number of NEW chambers")
    parser.add_argument("--adhesion",type=float,default=1.,help="Attraction toward fixed older walls")
    parser.add_argument("--turn-angle",type=float,default=65.,help="Coiling guide angle increment in degrees")
    parser.add_argument("--areas",type=float,nargs="+",help="Target areas of NEW chambers, in formation order")
    parser.add_argument("--spacing",type=float,default=.025)
    parser.add_argument("--seed-mode",choices=sorted(SEED_STYLES),default="bilocular")
    parser.add_argument("--second-radius-ratio",type=float,default=1.10)
    parser.add_argument("--seed-separation",type=float,default=.60)
    parser.add_argument("--seed-angle",type=float,default=0.)
    parser.add_argument("--device",default="cpu",help="Torch device for the phase-field loop, e.g. cuda")
    parser.add_argument("--dtype",choices=["float32","float64"],default="float32")
    args=parser.parse_args()
    p=ColonyParameters(new_chambers=args.chambers,adhesion_strength=args.adhesion,
                       turning_angle_degrees=args.turn_angle,spacing=args.spacing,
                       seed_mode=args.seed_mode,second_piece_radius_ratio=args.second_radius_ratio,
                       seed_separation_fraction=args.seed_separation,seed_orientation_degrees=args.seed_angle)
    output=args.output or Path("output")/args.seed_mode
    result=simulate_colony(p,args.areas,device=args.device,dtype=getattr(torch,args.dtype))
    metrics=save_colony(result,output)
    render_colony(result,output)
    print(json.dumps(metrics,indent=2))


if __name__ == "__main__":
    main()
