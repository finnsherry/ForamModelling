# Sequential foraminiferal chamber growth

An exploratory 2D phase-field model of a foraminifer growing chamber by
chamber. Growth starts from a fixed central seed. Each new chamber grows from
an aperture on its immediate predecessor to a prescribed area. Older chambers
are immobile and act as obstacles. A short-range adhesion field favours growth
along existing walls.

Chamber sizes and timing use an **illustrative sigmoid**, not measured
foraminiferal data or a species-specific fit. All lengths, areas and times are
dimensionless. This is an exploratory geometric construction, not a validated
biological mechanism.

## Run

Requires Python 3.10 or later.

```bash
python -m pip install -r requirements.txt
python -m foram_colony --seed-mode bilocular    # writes output/bilocular/
python -m foram_colony --seed-mode unilocular   # writes output/unilocular/
```

Results go to `output/<seed-mode>/` unless `--output` is given.

Or open `Foraminiferal_Chamber_Growth.ipynb` and run its cells. The notebook
imports the `foram_colony` package from the same folder. A default run takes a few
seconds.

From Python, simulating, saving results and rendering figures are separate
steps:

```python
from foram_colony import ColonyParameters, simulate_colony, save_colony, render_colony

result = simulate_colony(ColonyParameters(seed_mode="bilocular", new_chambers=6))
metrics = save_colony(result, "output/bilocular")   # csv, json, npz
render_colony(result, "output/bilocular")           # gif, png (needs matplotlib)
```

### GPU

The simulation runs in PyTorch and can run on a GPU:

```python
result = simulate_colony(parameters, device="cuda")
```

```bash
python -m foram_colony --device cuda
```

The default is `device="cpu"`. All computation is in float32 (`DTYPE` in
`parameters.py`), which meets the area constraint to about 1e-7. CPU and GPU
round differently, so their colonies can differ by a few pixels.

A GPU pays off on fine grids. On the default 177×177 grid, per-step overhead
dominates and the CPU is nearly as fast.

## Code layout

The `foram_colony/` package:

- `parameters.py`: `ColonyParameters` and the per-seed-mode labels and colours.
- `morphology.py`: binary dilation (boolean shifts), connectivity by flood fill, and a
  truncated min-plus distance transform, all on torch masks.
- `phase_field.py`: pure numerics, i.e. the Allen–Cahn step, the area projection,
  crisping a phase field to pixels, and the stable timestep.
- `placement.py`: the model rules, i.e. seed geometry, aperture choice,
  nucleation of each new chamber, and the adhesion field.
- `colony.py`: the growth law and the driver. For each new chamber it runs
  place → grow → freeze.
- `output.py`: `save_colony` and `render_colony`.
- `__main__.py`: the command line.

## Central seed

`seed_mode` selects the fixed initial structure. `new_chambers` and any
supplied `chamber_areas` count only the chambers grown afterwards, never the
seed compartments.

### Bilocular seed (default)

The seed has **two distinct connected chambers**, labelled 1 and 2. New
chambers start at 3, and chamber 3 grows from chamber 2. Both initial
compartments stay fixed throughout.

They represent the initial chamber (proloculus/protoconch) plus the following
chamber (deuteroloculus/deuteroconch) of one foraminifer. Their relative sizes
and geometry vary between taxa (see References), which is why the seed is
adjustable.

The two generating discs have radii

    r1 = first_piece_radius_fraction * central_radius,   (default 0.70)
    r2 = r1 * second_piece_radius_ratio,

and centre separation

    d = seed_separation_fraction * (r1 + r2).

The centres sit at opposite ends of a line through the origin, rotated by
`seed_orientation_degrees`. Compartment 1 keeps its whole generating disc.
Compartment 2 is the part of its own disc that lies outside compartment 1. So
the interiors don't overlap, and the shared interface connects the two. The
second compartment is usually crescent-like rather than a full disc, so the
radius ratio is **not** an area ratio. The actual compartment areas are
reported in `validation.json`. With the defaults they are 0.140625 and
0.125625.

`seed_separation_fraction` must be between zero and one. Extreme separations or
radius ratios can leave a compartment missing or under-resolved, and the code
rejects such cases. The grid must resolve both compartments and their contact.
The pair is initialized all at once, and the shared interface stands in for the
connection between them.

```python
parameters = ColonyParameters(
    seed_mode="bilocular",
    central_radius=0.30,
    second_piece_radius_ratio=1.10,
    seed_separation_fraction=0.60,
    seed_orientation_degrees=0.0,
    new_chambers=6,
)
```

```bash
python -m foram_colony --seed-mode bilocular --second-radius-ratio 0.85 \
  --seed-separation 0.60 --seed-angle 30 --output output/rotated_pair
```

### Unilocular seed

`seed_mode="unilocular"` uses one disc of radius `central_radius` as chamber 1.
Chambers 2 onward grow from it.

## What connects to what?

Every new chamber is guaranteed to attach to its immediate predecessor (3 to 2,
4 to 3, and so on). It grows from a fixed small aperture patch on an exposed
part of that predecessor's boundary. Earlier chambers are immobile and cannot
be overwritten, and a new chamber cannot overlap their interiors.

A short-range adhesion field also favours growth near **all** older chamber
walls. This can produce extra contacts with chambers other than the immediate
predecessor, and the default result has some. A new chamber is not required to
touch every previous chamber, which can be geometrically impossible.

The next aperture is placed on the previous chamber's exposed wall, nearest a
guide point that turns by `turning_angle_degrees` around the seed for each new
chamber. The guide point spirals outward: for new chamber i (counting from 0)
it lies at distance `guide_start_radius + i * guide_radius_increment` from the
origin.

## Parameters

```bash
python -m foram_colony --chambers 6 --adhesion 1.0 --turn-angle 65
python -m foram_colony --adhesion 0.0 --output output/no_adhesion
python -m foram_colony --spacing 0.02 --output output/finer_grid
```

```python
parameters = ColonyParameters(
    seed_mode="unilocular",
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
```

- `adhesion_strength` sets how strongly growth is drawn towards old walls. Zero
  switches off the attraction but keeps the mandatory parent attachment.
- `adhesion_range` sets the distance scale of the attraction.
- `turning_angle_degrees` changes aperture placement and therefore the coiling
  arrangement.
- `chamber_area_ratio` sets the growth in completed chamber size from one
  chamber to the next.
- Each chamber follows an illustrative normalized sigmoid over `formation_time`,
  with rate `growth_rate` and centred at `growth_midpoint_fraction` of it.
- Each new chamber starts as a soft disc of `nucleus_radius`, centred
  `nucleus_offset` outward from its aperture and cut off beyond
  `nucleus_support_radius`.

Numerical settings that don't define the model (precision, tolerances, the
time-stepping window) are constants at the top of `parameters.py`.

## Specified final chamber sizes

```bash
python -m foram_colony --chambers 3 --areas 0.30 0.39 0.48
```

```python
result = simulate_colony(ColonyParameters(new_chambers=3),
                         chamber_areas=[0.30, 0.39, 0.48])
```

These are the areas of the NEW chambers only. Each must be larger than its
initial seed and aperture.

If observations are **3D volumes**, they can't be imposed directly as planar
areas. With a specified constant thickness h, use area = volume/h. With an
explicit geometric-similarity assumption, use area proportional to
volume^(2/3), with the proportionality factor supplied or calibrated. Volumes
indexed only by completed chamber number give target final sizes. They don't
give within-chamber formation times, which stay illustrative here.

## Numerical model

The domain is a square grid of spacing `spacing`. Each new chamber is
represented by a phase field u(x,y,t) in [0, 1]: a smoothed indicator that is
about 1 inside the growing chamber and about 0 outside. On the space outside
existing chambers, the provisional evolution is

    u_t = gamma Laplacian(u)
          + (2 gamma/ell^2) u(1-u)(u-1/2)
          + adhesion_strength exp(-d_old/adhesion_range) u(1-u)
          + lambda(t) u(1-u),

where gamma is `curvature_coefficient` and ell is `interface_width`. The first
two terms are the Allen–Cahn equation, which moves the interface approximately
by curvature. d_old is the distance to any previously completed chamber. It is
computed exactly up to 5 adhesion ranges; beyond that the adhesion term is set
to zero, since it has already fallen below 0.7% of its value at the wall. The
positive adhesion term raises the local drive near old walls.
The scalar lambda is applied as a numerical correction after each time step, so
that

    spacing^2 * sum(u) = prescribed_area(t).

u = 0 inside earlier chambers and at the outer computational edge. u = 1 on the
chosen aperture patch, just outside the parent wall. These values are held
fixed while the current chamber grows. A compactly supported initial seed
prevents spontaneous nucleation far away. Time stepping is explicit, with a
timestep cap set by the diffusive and reaction stability limits.
Each step is computed only on a window around the growing chamber, refitted at
every snapshot. If the chamber reaches the window border, the window is widened
and the interval repeated, so the result is the same as on the full grid.

When a chamber is frozen, it takes the pixels with the highest phase values,
up to the nearest achievable pixel area (aperture included). Thus:

- the diffuse area is enforced to floating-point tolerance;
- the frozen/displayed area differs from the requested value by at most
  spacing^2/2 (0.0003125 on the default grid);
- every frozen chamber is checked for 4-neighbour connectedness;
- each frozen chamber is checked for contact with its parent;
- older labels are immobile by construction: a new chamber can only claim
  pixels outside every existing chamber.

`validation.json` reports per-chamber contacts and area errors.

## Output files

`save_colony` writes:

- `area_history.csv`: chamber, local and global time, target, diffuse
  and frozen pixel area at each snapshot.
- `validation.json`: parameters, seed and per-chamber reports, and area
  errors.
- `simulation.npz`: final and initial label grids, coordinates and
  history.

`render_colony` writes:

- `chamber_growth.gif`: animation of the whole growth sequence.
- `seed.png`: the initial seed compartments.
- `growth_summary.png`: final labelled colony and target versus diffuse
  area.

## Limits

This is an exploratory geometric construction, not a validated biological
mechanism. It does not model calcification, wall thickness, septa, chemistry,
mechanics, cytoplasmic transport or biological aperture selection. Aperture
placement, adhesion strength, chamber sizes and timing are inputs, not
quantities derived from data. The adhesion term is a phenomenological
attachment preference, not a measured force. The bilocular seed is one
adjustable two-disc construction, not a description of any particular taxon.

Strong attraction or confined geometry can make the constraints incompatible;
the routine stops if a chamber disconnects or reaches the computational edge.
Grid refinement and sensitivity checks remain necessary for quantitative use.

## References

Bilocular embryonic chambers:
Takagi, Kurasawa and Kimoto (2020) describe two-chambered offspring of
Globigerinita uvula, with the second chamber commonly smaller than the
proloculus: https://doi.org/10.1093/plankt/fbaa033.
Bilocular embryonic structures in some larger benthic taxa have different
relative sizes and enclosure geometry; see the orthophragminid study at
https://doi.org/10.1080/09853111.2015.1026795.

For measurements and growth estimation:
J. Hohenegger and A. Briguglio (2014), *Methods for Estimating Individual Growth
of Foraminifera Based on Chamber Volumes*, pp. 29-54.
https://doi.org/10.1007/978-4-431-54388-6_3

For volume-preserving curvature flow with obstacles:
J. Jang, *Existence of weak solutions to volume-preserving mean curvature flow
with obstacles*, arXiv:2501.03455. https://arxiv.org/abs/2501.03455
That paper concerns preserved volume, not the growing area used here.
