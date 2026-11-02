"""Command line: python -m foram_colony --seed-mode bilocular

Every field of ColonyParameters is an option, e.g. --new-chambers 3 or --adhesion-strength 0.
"""
import argparse
import ast
import inspect
import json
from dataclasses import fields
from itertools import pairwise
from pathlib import Path

from .parameters import ColonyParameters, SEED_STYLES
from .colony import simulate_colony
from .output import save_colony, render_colony


def field_docs(cls):
    """The docstring under each field of dataclass cls, by field name.

    Python keeps no record of these attribute docstrings at run time (unlike
    function docstrings), so they are read from the source.
    """
    body = ast.parse(inspect.getsource(cls)).body[0].body
    docs = {}
    for statement, following in pairwise(body):
        if (
            isinstance(statement, ast.AnnAssign)
            and isinstance(following, ast.Expr)
            and isinstance(following.value, ast.Constant)
            and isinstance(following.value.value, str)
        ):
            docs[statement.target.id] = inspect.cleandoc(following.value.value)
    return docs


def main():
    parser = argparse.ArgumentParser(
        description="Grow a colony chamber by chamber and write its results and figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", help="Output directory; None means output/<seed-mode>")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cuda")
    parser.add_argument(
        "--animation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write the growth animation (chamber_growth.gif)",
    )

    docs = field_docs(ColonyParameters)
    for f in fields(ColonyParameters):
        flag = "--" + f.name.replace("_", "-")
        if f.name == "seed_mode":
            parser.add_argument(flag, choices=sorted(SEED_STYLES), default=f.default, help=docs[f.name])
        elif f.name == "chamber_areas":
            parser.add_argument(flag, type=float, nargs="+", help=docs[f.name])
        else:
            parser.add_argument(flag, type=type(f.default), default=f.default, help=docs[f.name])
    args = parser.parse_args()

    p = ColonyParameters(**{f.name: getattr(args, f.name) for f in fields(ColonyParameters)})
    output = args.output or Path("output") / p.seed_mode
    result = simulate_colony(p, device=args.device)
    metrics = save_colony(result, output)
    render_colony(result, output, animation=args.animation)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
