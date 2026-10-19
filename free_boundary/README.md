# Latest update: two distinct connected initial chambers

The new default central seed is bilocular: initial compartments 1 and 2, followed by growth of chambers 3 onward. See [BILOCULAR_SEED.md](BILOCULAR_SEED.md). Run `python foram_growth.py --multi --seed-mode bilocular --output bilocular_output`.

The original one-piece central geometry remains available with `--seed-mode single`, and the original single-chamber solver is unchanged. Prior results and information are preserved below and in their original output folders.

---

# Updated: central seed and connected chambers

Run `python foram_growth.py --multi --output central_output`, or open the central-model cells at the top of `Foraminiferal_Chamber_Growth.ipynb`.

See [CENTRAL_MODEL.md](CENTRAL_MODEL.md) for the multi-chamber geometry, attachment and adhesion rules, inputs, numerical tolerances, and limitations. The completed animation is in `central_output/central_chamber_growth.gif`.

The original single-chamber model is documented below.

---

# Prescribed-area chamber growth

This runnable Python example evolves one planar chamber from a fixed aperture.
The aperture endpoints and existing straight wall stay fixed. A freely moving
curve encloses the chamber together with the segment across the aperture.
It can develop overhangs; it is not restricted to a graph y = f(x).

The included growth law is an **illustrative sigmoid**, not measured
foraminiferal data or a species-specific fit. You can substitute a CSV of
time-resolved chamber areas. This is an exploratory geometric model, not a
validated biological mechanism.

## Run

Requires Python 3.10 or later. Extract the ZIP, open a terminal in its folder,
and run:

```bash
python -m pip install -r requirements.txt
python foram_growth.py --output output
```

Or open `Foraminiferal_Chamber_Growth.ipynb` in Jupyter in the extracted folder
and run its cells. The notebook imports the accompanying `foram_growth.py`.
Its simulation and plotting take several seconds on a typical computer.

The supplied `output/` folder already contains the animation, figure, numerical
results, and validation metrics from a completed run.

## Mathematical model

The moving curve is Gamma(t); its endpoints are (-a,0) and (a,0). The domain
between it and the fixed base is Omega(t). Its area is prescribed as A*(t).

The geometric PDE, in outward normal velocity form, is

    V_n = m(s,t) [lambda(t) - gamma kappa],
    |Omega(t)| = A*(t).

Here kappa is positive for a convex circular arc, gamma >= 0 smooths the
boundary, and m(s,t) = sin(pi s / L(t))^2 is a specified mobility, where s is
arclength from the left endpoint and L is total curve length. This mobility
vanishes at the two endpoints, so they remain pinned. It is a modelling choice
and affects shape, not just a numerical convenience. Away from the endpoints,
the boundary is driven by the scalar lambda and smoothed by curvature.

The area-growth identity gives

    lambda = [A*'(t) + gamma integral_Gamma m kappa ds]
             / integral_Gamma m ds.

The code uses the exact gradient of discrete polygon area to form the discrete
counterpart of this multiplier. Every timestep is followed by equal-arclength
remeshing and a scalar normal-displacement correction to enforce the requested
polygon area. The correction retains the endpoints. The area tolerance does
not establish that the evolving shape is exact; spatial and temporal errors
still need assessment.

No pressure, nutrient diffusion, mineralization chemistry, actin mechanics,
wall thickness, or placement of the next chamber is modelled. Increasing total
area does not imply that every boundary point moves outward. Arbitrary fixed
obstacles are not implemented in this explicit-curve version. The curve must
remain simple and above the fixed wall; crossing checks run at saved times and
half-plane checks run at every step. Touching/tangent intersections are not
detected by the proper-crossing test.

## Replace the growth curve

Prepare a CSV with two columns, `time,area`, with at least two rows, increasing
times, positive areas, and nondecreasing area values. For example:

```csv
time,area
0,0.14
0.5,0.18
1.0,0.25
```

Those numbers are only an input-format example. Use consistent length/time
units for the geometry and curve. The curvature coefficient gamma has units
length^2/time; its supplied value is dimensionless in the demonstration.

```bash
python foram_growth.py --growth-csv my_growth.csv --output measured_output
```

The code shifts the first observation time to zero, interpolates using a
monotone cubic PCHIP interpolant, and evolves only over the observed interval.
The initial semicircular seed is stretched vertically to match the first
observed area without moving its endpoints. In the Python notebook, set
`growth_csv="my_growth.csv"` in the `simulate` call. The aperture width is an
independent parameter; for physical data set it to the measured width.

If observations are **3D volumes**, they cannot be imposed directly as planar
areas. With a specified constant thickness h, use area = volume/h. With an
explicit geometric-similarity assumption, use area proportional to volume^(2/3).
The proportionality factor must be supplied/calibrated. This script does not
implement an axisymmetric volume model.

Volumes indexed only by completed chamber number do not supply within-chamber
formation times. They can determine target final sizes; a growth schedule and
a separate next-aperture rule would still be needed for a multi-chamber model.

## Numerical settings

```bash
python foram_growth.py --points 121 --max-dt 0.002 --gamma 0.012
python foram_growth.py --aperture-half-width 0.3
```

The default run uses 81 curve points and adaptive timesteps capped at 0.004.
Refine points and timesteps before interpreting shapes quantitatively.
The included refinement comparison changes both point count and timestep cap,
so it is a sensitivity check, not a formal convergence rate.

## Output files

- `chamber_growth.gif`: animation with pinned aperture endpoints.
- `growth_summary.png`: chamber shapes and prescribed versus simulated area.
- `area_history.csv`: saved time, area, target area, and absolute area error.
- `simulation.npz`: curve coordinates and time/area arrays.
- `validation.json`: all-step area and endpoint errors and proper-crossing check.
- `refinement_check.json`: shape comparison against a refined run.

## References and biological interpretation

For measurements and growth estimation:
J. Hohenegger and A. Briguglio (2014), *Methods for Estimating Individual Growth
of Foraminifera Based on Chamber Volumes*, pp. 29-54.
https://doi.org/10.1007/978-4-431-54388-6_3

For volume-preserving curvature flow with obstacles:
J. Jang, *Existence of weak solutions to volume-preserving mean curvature flow
with obstacles*, arXiv:2501.03455. https://arxiv.org/abs/2501.03455
That paper concerns preserved volume and a phase-field formulation. It does
not validate the growing-volume, pinned-curve construction implemented here.
