# Two-piece initial seed

The default central model now begins with **two distinct connected chambers**,
labelled 1 and 2. Subsequent chambers start at 3 and grow from chamber 2 onward.
Both initial compartments remain fixed throughout subsequent growth. Earlier
single-chamber and one-piece-central models, results, and documentation are
retained in this package.

The intended interpretation is the initial chamber (proloculus/protoconch) plus
the following chamber (deuteroloculus/deuteroconch). They are compartments of
one foraminifer, not two separate organisms. Terminology and initial geometry
depend on the taxon. This change is not a claim that one two-disc construction
describes most foraminifera.

For example, Takagi, Kurasawa and Kimoto (2020) describe two-chambered offspring
of Globigerinita uvula, with the second chamber commonly smaller than the
proloculus: https://doi.org/10.1093/plankt/fbaa033.
Bilocular embryonic structures in some larger benthic taxa can have different
relative sizes and enclosure geometry; see the orthophragminid study at
https://doi.org/10.1080/09853111.2015.1026795.
These references motivate adjustable initial compartments; they do not supply
the model's example parameter values or validate its subsequent growth law.

## Run

```bash
python foram_growth.py --multi --seed-mode bilocular --output bilocular_output
```

Or open the **Two-piece initial seed** section at the top of
`Foraminiferal_Chamber_Growth.ipynb`.

To recover the previous one-piece central model:

```bash
python foram_growth.py --multi --seed-mode single --output central_output
```

The previous single-seed geometry was checked against the stored preceding
result and matches it exactly on the grid. The original single free-boundary
chamber model is still run without `--multi`.

## Adjustable initial geometry

```python
parameters = ColonyParameters(
    seed_mode="bilocular",
    central_radius=0.30,
    second_piece_radius_ratio=1.10,
    seed_separation_fraction=0.60,
    seed_orientation_degrees=0.0,
    new_chambers=6,
)
result = simulate_colony(parameters)
render_colony(result, "bilocular_output")
```

The two generating discs have radii

    r1 = 0.70 * central_radius,
    r2 = r1 * second_piece_radius_ratio.

Their centre separation is

    d = seed_separation_fraction * (r1 + r2).

The centres are at opposite ends of a line through the origin, rotated by
`seed_orientation_degrees`. The first compartment retains its generating
disc; the second occupies the portion of its generating disc outside the
first. Thus their interiors do not overlap, while their shared interface
connects the two compartments. The second compartment is generally crescent-
like, rather than a complete disc. The radius ratio therefore is **not** an
area ratio. Actual compartment areas are reported in `central_validation.json`.

`seed_separation_fraction` must be between zero and one. Extreme values or
extreme radius ratios can leave a compartment missing or under-resolved; the
code rejects such cases. The grid must resolve both compartments and their
contact. This initial geometry has no explicit septal thickness or transport
through an internal foramen; its interface represents a prescribed connection.

CLI equivalents:

```bash
python foram_colony.py --seed-mode bilocular --second-radius-ratio 0.85 \
  --seed-separation 0.60 --seed-angle 30 --output rotated_pair
```

The example default pair has rasterized areas 0.140625 and 0.125625, in the
demonstration's dimensionless units. Those are generated geometry, not measured
foraminiferal chamber volumes. Changing the parameters changes these areas.

## What is preserved?

- The single moving-curve chamber solver, its CSV timing input and outputs.
- The earlier one-piece central model through `seed_mode="single"`.
- The earlier central animation, figure, validation and numerical outputs.
- Prescribed new-chamber areas, subsequent attachment rules, and adhesion.
- The fixed-old-wall and non-overlap constraints.
- Original documentation, plus this update's instructions and validation.

`new_chambers` counts only subsequently grown chambers. It excludes BOTH
initial pieces in bilocular mode. Supplied `chamber_areas` likewise refers only
to the newly grown chambers. The central pair is initialized simultaneously;
the code does not reconstruct the developmental formation of the pair itself.

## Verification

The default six-new-chamber bilocular simulation was run. Both initial labels
stay fixed, new chamber 3 contacts initial chamber 2, and every later chamber
contacts its predecessor. The assembled organism remains connected. Diffuse
areas match their prescribed targets to floating-point tolerance, and frozen
areas satisfy the existing half-pixel-area tolerance. A rotated unequal pair
was also tested. These checks establish software behaviour, not biological
calibration or mathematical convergence.
