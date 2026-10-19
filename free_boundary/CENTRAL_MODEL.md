# Preserved one-piece central model

This document records the earlier one-piece central model. Use `seed_mode="single"` (CLI `--seed-mode single`) to reproduce it. The new default two-piece initial geometry is documented in `BILOCULAR_SEED.md`. Earlier figures and numerical results remain in `central_output/`.

---

# Central seed and connected successive chambers

The update adds `foram_colony.py` and a central-growth mode to `foram_growth.py`.
The original single-chamber model is still available.

## Run the central model

Extract the ZIP and run these commands in its folder:

```bash
python -m pip install -r requirements.txt
python foram_growth.py --multi --seed-mode single --output central_output
```

Or run the new central-model cells near the top of
`Foraminiferal_Chamber_Growth.ipynb`.

The completed default simulation is supplied in `central_output/`.
Chamber 1 is the fixed central seed. Chambers 2 through 7 grow one after another.

## What connects to what?

Every new chamber is guaranteed to attach to its immediate predecessor:
2 to 1, 3 to 2, and so on. It grows from a fixed small aperture patch at an
exposed part of that predecessor's boundary. Earlier chambers are immobile and
cannot be overwritten. A new chamber cannot overlap their interiors.

A short-range adhesion field also favours growth near **all** older chamber
walls. This can produce extra contacts with chambers other than the immediate
predecessor. The default result has such extra contacts. It does not require a
new chamber to touch every previous chamber; that can be geometrically
impossible. These are geometric contacts and a designated attachment patch,
not a simulation of cytoplasmic transport through apertures.

The next aperture is selected on the previous chamber's exposed wall using a
coiling guide angle around the central seed. That is a prescribed placement
rule, not a biological law derived from the growth PDE.

## Parameters

```bash
python foram_growth.py --multi --seed-mode single --chambers 6 --adhesion 1.0 --turn-angle 65
python foram_colony.py --seed-mode single --chambers 6 --adhesion 0.0 --output no_adhesion
python foram_colony.py --seed-mode single --spacing 0.02 --output finer_grid
```

In the notebook:

```python
parameters = ColonyParameters(
    seed_mode="single",
    new_chambers=6,
    central_radius=0.30,
    first_chamber_area=0.28,
    chamber_area_ratio=1.16,
    turning_angle_degrees=65.0,
    adhesion_strength=1.0,
    adhesion_range=0.14,
    aperture_radius=0.075,
    spacing=0.025,
)
result = simulate_colony(parameters)
render_colony(result, "central_output")
```

Increasing `adhesion_strength` favours growth near old walls; zero turns off
that attraction while retaining the mandatory parent attachment.
`adhesion_range` sets its distance scale. `turning_angle_degrees` changes
aperture placement and therefore the coiling arrangement.

`chamber_area_ratio` describes an illustrative increase in completed chamber
size, not an estimated foraminiferal parameter. Each chamber follows an
illustrative normalized sigmoid over `formation_time`. Timing, chamber sizes,
and all lengths are dimensionless in the default example.

## Use specified final chamber sizes

```bash
python foram_colony.py --seed-mode single --chambers 3 --areas 0.30 0.39 0.48
```

Or:

```python
result = simulate_colony(ColonyParameters(new_chambers=3, seed_mode="single"),
                        chamber_areas=[0.30, 0.39, 0.48])
```

These are the areas of the NEW chambers, excluding the central chamber. Each
must be larger than its initialized seed and aperture. Volumes require an
explicit, justified conversion to planar areas (see `README.md`). Within-
chamber time-resolved data are supported by the original single-chamber mode;
the central mode currently accepts final sizes and illustrative timing.

## Numerical model

To handle curved earlier walls and contacts, the multi-chamber routine uses a
grid phase field u(x,y,t), rather than the original open explicit curve.

On the space outside existing chambers, the provisional evolution is

    u_t = gamma Laplacian(u)
          + (2 gamma/ell^2) u(1-u)(u-1/2)
          + adhesion_strength exp(-d_old/adhesion_range) u(1-u)
          + lambda(t) u(1-u).

Here d_old is the distance to any previously completed chamber. The positive
adhesion field raises the local drive near old walls; it is a phenomenological
attachment preference, not a force inferred from experiments. The scalar
lambda is applied through a numerical correction of each time step so that

    spacing^2 * sum(u) = prescribed_area(t).

Inside earlier chambers and at the outer computational edge u=0. On the
chosen aperture patch, immediately outside the parent wall, u=1. These values
are held fixed during the current chamber's growth. A compact seed prevents
spontaneous distant nucleation. No existing chamber interior is reassigned.

The displayed/frozen chamber is formed by selecting the highest phase values
to match the nearest achievable pixel area, including the aperture. Thus:

- diffuse area is enforced to floating-point tolerance;
- frozen/displayed area differs from its requested value by at most
  spacing^2/2;
- every displayed chamber is checked for 4-neighbour connectedness;
- each final chamber is checked for contact with its parent;
- the assembled organism is checked for connectedness;
- older labels are checked for immobility.

The central chamber's radius is rasterized on the grid; its area is not itself
an empirical constraint. The saved `central_validation.json` reports per-cell
contacts and area errors. The default grid has pixel-area tolerance 0.0003125.

## Limits

This remains an exploratory geometric construction. It does not model
calcification, wall thickness, chemistry, mechanics, or biological aperture
selection. Attraction strength, coiling angle, sizes, and timing are inputs.
Strong attraction or confined geometry can make the constraints incompatible;
the routine stops if a chamber disconnects or reaches the computational edge.
Grid refinement and sensitivity checks remain necessary for quantitative use.

The recorded additional checks compare adhesion with zero adhesion, check
custom target sizes, and test a smaller grid spacing. They demonstrate the
implemented behaviour, not a convergence proof or biological validation.
