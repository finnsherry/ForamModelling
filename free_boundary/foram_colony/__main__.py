"""Command line: python -m foram_colony --seed-mode bilocular

Every field of ColonyParameters and ImplementationParameters is an option, e.g.
--new-chambers 3, --adhesion-speed 0 or --grid-points 353.
"""
import argparse
import ast
import inspect
import json
from dataclasses import fields
from itertools import pairwise
from pathlib import Path
from typing import Literal, get_args, get_origin

from .parameters import ColonyParameters, ImplementationParameters
from .colony import simulate_colony
from .output import save_colony, render_colony


def field_docs(cls):
    """The docstring under each field of dataclass cls, by field name.

    Python keeps no record of these attribute docstrings at run time (unlike
    function docstrings), so they are read from the source. "%" is escaped, as
    argparse treats help text as a format string.
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
            docs[statement.target.id] = inspect.cleandoc(following.value.value).replace("%", "%%")
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

    for cls in (ColonyParameters, ImplementationParameters):
        group = parser.add_argument_group(cls.__name__)
        docs = field_docs(cls)
        for f in fields(cls):
            flag = "--" + f.name.replace("_", "-")
            if get_origin(f.type) is Literal:
                group.add_argument(flag, choices=get_args(f.type), default=f.default, help=docs[f.name])
            elif f.name == "chamber_areas":
                group.add_argument(flag, type=float, nargs="+", help=docs[f.name])
            else:
                group.add_argument(flag, type=type(f.default), default=f.default, help=docs[f.name])
    args = parser.parse_args()

    p, q = (
        cls(**{f.name: getattr(args, f.name) for f in fields(cls)})
        for cls in (ColonyParameters, ImplementationParameters)
    )
    output = args.output or Path("output") / p.seed_mode
    result = simulate_colony(p, q, device=args.device)
    metrics = save_colony(result, output)
    render_colony(result, output, animation=args.animation)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
