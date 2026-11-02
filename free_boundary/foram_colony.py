"""Sequential area-constrained chambers attached to a central seed.

Existing chambers are immobile, excluded regions. New chambers attach to a
fixed aperture on their immediate predecessor. A short-range adhesion field
favours growing near any existing chamber. This is a grid phase-field model.

Run: python foram_colony.py --seed-mode bilocular   (writes output/bilocular/)
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import numpy as np
from scipy import ndimage
from scipy.special import expit


@dataclass
class ColonyParameters:
    new_chambers: int = 6
    central_radius: float = 0.30
    seed_mode: Literal["single", "bilocular"] = "bilocular"
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
# constructed in make_seed.
SEED_STYLES = {
    "single": {
        "names": ["Original one-chamber central seed"],
        "colors": None,
        "animation_title": "Central seed",
        "first_frame": "chamber 1",
        "seed_title": "Original single central chamber",
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


def demo_growth(initial_area, final_area, end_time):
    """Illustrative normalized sigmoid, NOT a fit to foraminiferal observations."""
    if final_area <= initial_area:
        raise ValueError("Final area must exceed the seed area.")
    rate, midpoint = 0.65, end_time * 0.48
    def sigmoid(t):
        return 1.0 / (1.0 + np.exp(-rate * (t - midpoint)))
    f0, f1 = sigmoid(0.0), sigmoid(end_time)
    scale = (final_area - initial_area) / (f1 - f0)
    def area(t):
        return initial_area + scale * (sigmoid(t) - f0)
    def rate_of_area(t):
        f = sigmoid(t)
        return scale * rate * f * (1.0 - f)
    return area, rate_of_area


def corrected_phase_area(phase, desired, allowed, aperture, h):
    """Scalar multiplier correction of diffuse area, with fixed wall constraints."""
    u = np.clip(phase, 0.0, 1.0) * allowed
    u[aperture] = 1.0
    weights = u * (1.0 - u) * allowed
    weights[aperture] = 0.0
    for _ in range(15):
        error = desired / h**2 - u.sum()
        if abs(error) * h**2 < 5e-12:
            return u
        active = (u < 1.0) if error > 0 else (u > 0.0)
        denominator = weights[active].sum()
        if denominator < 1e-12:
            raise RuntimeError("Insufficient interface to enforce the requested area.")
        u = np.clip(u + (error / denominator) * weights * active, 0.0, 1.0)
        u *= allowed
        u[aperture] = 1.0
    raise RuntimeError("Diffuse area correction failed; decrease the timestep.")


def crisp_chamber(phase, allowed, aperture, area, h):
    """Threshold to the nearest pixel-area target while retaining the aperture."""
    count = int(round(area / h**2))
    candidates = np.flatnonzero(allowed & ~aperture)
    needed = count - int(aperture.sum())
    if needed <= 0 or needed > len(candidates):
        raise ValueError("Target area incompatible with aperture or remaining space.")
    chosen = candidates[np.argpartition(phase.flat[candidates], -needed)[-needed:]]
    mask = aperture.copy()
    mask.flat[chosen] = True
    _, components = ndimage.label(mask)
    if components != 1:
        raise RuntimeError("The new chamber disconnected; reduce growth/adhesion or revise the aperture.")
    return mask


def select_aperture(labels, parent, x, y, angle, radius, guide_index=None):
    """Choose an exposed parent wall nearest a prescribed coiling guide point."""
    parent_mask = labels == parent
    candidates = ndimage.binary_dilation(parent_mask) & (labels == 0)
    candidates[[0, -1], :] = False
    candidates[:, [0, -1]] = False
    rows, columns = np.nonzero(candidates)
    if len(rows) == 0:
        raise RuntimeError("The previous chamber has no exposed boundary.")
    theta = np.deg2rad(angle)
    # This aperture-placement rule is a geometric input, not derived biology.
    guide_radius = 0.55 + 0.075 * ((parent - 1) if guide_index is None else guide_index)
    guide = guide_radius * np.array([np.cos(theta), np.sin(theta)])
    scores = (x[rows, columns] - guide[0])**2 + (y[rows, columns] - guide[1])**2
    selected = np.argmin(scores)
    cy, cx = int(rows[selected]), int(columns[selected])
    distance = np.hypot(x - x[cy, cx], y - y[cy, cx])
    aperture = candidates & (distance <= radius)
    # The outside pixel layer of a curved wall can be diagonally connected.
    parts, _ = ndimage.label(aperture, structure=np.ones((3, 3), dtype=int))
    aperture = parts == parts[cy, cx]
    return aperture, (cy, cx)


def make_seed(p, x, y):
    """Two distinct adjacent compartments, or the original one-disc seed.

    For a bilocular seed, two overlapping generating discs are partitioned:
    compartment 1 retains its disc; compartment 2 occupies the second disc
    outside compartment 1. The resulting compartments have disjoint interiors
    and a common interface. This is geometry, not a universal embryonic shape.
    """
    labels = np.zeros_like(x, dtype=np.int16)
    if p.seed_mode == "single":
        labels[np.hypot(x, y) <= p.central_radius] = 1
    else:
        r1 = 0.70 * p.central_radius
        r2 = r1 * p.second_piece_radius_ratio
        distance = p.seed_separation_fraction * (r1 + r2)
        axis = np.array([np.cos(np.deg2rad(p.seed_orientation_degrees)),
                         np.sin(np.deg2rad(p.seed_orientation_degrees))])
        c1, c2 = -0.5 * distance * axis, 0.5 * distance * axis
        first = np.hypot(x - c1[0], y - c1[1]) <= r1
        second = np.hypot(x - c2[0], y - c2[1]) <= r2
        labels[first] = 1
        labels[second & ~first] = 2
    reports = []
    for cell, name in enumerate(SEED_STYLES[p.seed_mode]["names"], start=1):
        mask = labels == cell
        _, components = ndimage.label(mask)
        if mask.sum() < 12 or components != 1:
            raise ValueError("An initial compartment is missing, disconnected or under-resolved. Refine the grid or adjust seed geometry.")
        reports.append({"chamber":cell,"name":name,"pixel_area":float(mask.sum()*p.spacing**2),
                        "centroid":[float(x[mask].mean()),float(y[mask].mean())]})
    _, components = ndimage.label(labels > 0)
    if components != 1:
        raise ValueError("The central seed pieces do not connect; adjust their separation or grid spacing.")
    return labels, reports


def simulate_colony(parameters=None, chamber_areas=None):
    p = parameters or ColonyParameters()
    if p.seed_mode not in SEED_STYLES:
        raise ValueError(f"seed_mode must be one of {sorted(SEED_STYLES)}.")
    positive = [p.central_radius, p.spacing, p.half_width, p.first_chamber_area,
                p.chamber_area_ratio, p.formation_time, p.curvature_coefficient,
                p.interface_width, p.adhesion_range, p.aperture_radius, p.max_dt,
                p.second_piece_radius_ratio, p.seed_separation_fraction]
    if not all(np.isfinite(v) and v > 0 for v in positive):
        raise ValueError("Geometric, area and numerical scales must be positive and finite.")
    if p.new_chambers < 1 or p.snapshots_per_chamber < 2 or p.adhesion_strength < 0:
        raise ValueError("Invalid chamber count, snapshot count or adhesion strength.")
    if p.seed_mode == "bilocular" and p.seed_separation_fraction >= 1:
        raise ValueError("Bilocular generating discs must overlap: seed_separation_fraction < 1.")
    h = p.spacing
    coords = np.arange(-p.half_width, p.half_width + h/2, h)
    x, y = np.meshgrid(coords, coords)
    labels, seed_reports = make_seed(p, x, y)
    seed_chambers = len(seed_reports)
    initial_labels = labels.copy()
    if chamber_areas is None:
        chamber_areas = p.first_chamber_area * p.chamber_area_ratio**np.arange(p.new_chambers)
    chamber_areas = np.asarray(chamber_areas, dtype=float)
    if len(chamber_areas) != p.new_chambers or np.any(chamber_areas <= 0) or not np.isfinite(chamber_areas).all():
        raise ValueError("Supply one finite positive target area per new chamber.")
    allowed_edge = np.ones_like(labels, dtype=bool)
    allowed_edge[[0, -1], :] = False
    allowed_edge[:, [0, -1]] = False
    k = 2 * p.curvature_coefficient / p.interface_width**2
    dt_cap = min(p.max_dt, 0.35 / (4*p.curvature_coefficient/h**2 + 3*k + p.adhesion_strength))
    save_times = np.linspace(0.0, p.formation_time, p.snapshots_per_chamber)
    frames = [{"labels": labels.copy(), "chamber": seed_chambers, "time": 0.0}]
    rows, chamber_reports = [], []
    max_diffuse_error = 0.0
    for index, final_area in enumerate(chamber_areas):
        parent, cell = index + seed_chambers, index + seed_chambers + 1
        old = labels.copy()
        occupied = old > 0
        allowed = ~occupied & allowed_edge
        aperture, (cy, cx) = select_aperture(old, parent, x, y,
            index*p.turning_angle_degrees + p.seed_orientation_degrees,
            p.aperture_radius, guide_index=index)
        if (aperture & ~allowed).any():
            raise RuntimeError("Selected aperture intersects the computational boundary.")
        parent_rows, parent_cols = np.nonzero(old == parent)
        center = np.array([x[parent_rows, parent_cols].mean(), y[parent_rows, parent_cols].mean()])
        anchor = np.array([x[cy, cx], y[cy, cx]])
        outward = anchor - center
        outward /= max(np.linalg.norm(outward), 1e-12)
        seed_center = anchor + 0.07*outward
        distance = np.hypot(x-seed_center[0], y-seed_center[1])
        u = expit(-(distance-0.13)/p.interface_width) * allowed
        # Compact support prevents phase nucleation far from the seeded chamber.
        u[distance > 0.32] = 0.0
        u[aperture] = 1.0
        seed_area = h**2*u.sum()
        if final_area <= seed_area:
            raise ValueError(f"Chamber {cell} target {final_area:.4g} must exceed seed area {seed_area:.4g}.")
        target, _ = demo_growth(seed_area, final_area, p.formation_time)
        distance_old = ndimage.distance_transform_edt(~occupied)*h
        adhesion = p.adhesion_strength*np.exp(-distance_old/p.adhesion_range)
        t, steps = 0.0, 0
        def save():
            binary = crisp_chamber(u, allowed, aperture, float(target(t)), h)
            image_labels = old.copy()
            image_labels[binary] = cell
            frames.append({"labels": image_labels, "chamber": cell,
                           "time": index*p.formation_time + t})
            rows.append([cell, t, index*p.formation_time+t, float(target(t)),
                         h**2*u.sum(), h**2*binary.sum()])
        save()
        for save_time in save_times[1:]:
            while t < save_time - 1e-12:
                dt = min(dt_cap, save_time-t)
                lap = (np.roll(u,1,0)+np.roll(u,-1,0)+np.roll(u,1,1)+np.roll(u,-1,1)-4*u)/h**2
                rhs = p.curvature_coefficient*lap + k*u*(1-u)*(u-0.5) + adhesion*u*(1-u)
                candidate = np.clip(u+dt*rhs,0,1)*allowed
                u = corrected_phase_area(candidate,float(target(t+dt)),allowed,aperture,h)
                t += dt
                steps += 1
                error = abs(h**2*u.sum()-float(target(t)))
                max_diffuse_error = max(max_diffuse_error,error)
                if not np.isfinite(u).all():
                    raise RuntimeError("Nonfinite phase field.")
                if max(u[1,:].max(),u[-2,:].max(),u[:,1].max(),u[:,-2].max()) > .5:
                    raise RuntimeError("Growth reached the computational edge; increase half_width.")
            save()
        new_mask = crisp_chamber(u,allowed,aperture,final_area,h)
        labels[new_mask] = cell
        neighbors = ndimage.binary_dilation(new_mask) & occupied
        contacts = sorted(int(v) for v in np.unique(old[neighbors]) if v > 0)
        if parent not in contacts:
            raise RuntimeError("A new chamber lost contact with its parent.")
        chamber_reports.append({"chamber":cell,"parent":parent,"target_area":float(final_area),
            "diffuse_final_area":float(h**2*u.sum()),"frozen_pixel_area":float(h**2*new_mask.sum()),
            "area_rounding_error":float(abs(h**2*new_mask.sum()-final_area)),
            "contacts_with_earlier_chambers":contacts,"aperture_pixels":int(aperture.sum()),
            "aperture_center":[float(x[cy,cx]),float(y[cy,cx])],"steps":steps})
    return {"parameters":p,"coordinates":coords,"frames":frames,"labels":labels,
        "history":np.asarray(rows),"reports":chamber_reports,
        "seed_chambers":seed_chambers,"seed_reports":seed_reports,
        "initial_labels":initial_labels,
        "max_diffuse_area_error_all_steps":max_diffuse_error,
        "pixel_area_tolerance":h**2/2,"growth_label":"Illustrative sigmoid for each chamber"}


def save_colony(result, output):
    """Write numerical results and validation metrics; returns the metrics."""
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    p=result["parameters"]; history=result["history"]
    with (output/"area_history.csv").open("w",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(["chamber","formation_time","global_time","target_area","diffuse_area","pixel_area"])
        writer.writerows(history)
    metrics={"parameters":asdict(p),"max_diffuse_area_error_all_steps":result["max_diffuse_area_error_all_steps"],
        "pixel_area_tolerance":result["pixel_area_tolerance"],
        "seed_chambers":result["seed_chambers"],"seed_reports":result["seed_reports"],
        "reports":result["reports"],"growth_label":result["growth_label"]}
    (output/"validation.json").write_text(json.dumps(metrics,indent=2))
    np.savez_compressed(output/"simulation.npz",labels=result["labels"],initial_labels=result["initial_labels"],
                        coordinates=result["coordinates"],history=history)
    return metrics


def render_colony(result, output):
    """Write the growth animation, seed figure and summary figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.animation import FuncAnimation, PillowWriter
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    p=result["parameters"]; frames=result["frames"]; coords=result["coordinates"]
    style=SEED_STYLES[p.seed_mode]
    extent=[coords[0]-p.spacing/2,coords[-1]+p.spacing/2]*2
    total_chambers=p.new_chambers+result["seed_chambers"]
    colors=[plt.cm.viridis(v) for v in np.linspace(.08,.92,total_chambers)]
    if style["colors"]:
        colors[:len(style["colors"])]=style["colors"]
    cmap=ListedColormap(["#ffffff"]+colors)
    norm=BoundaryNorm(np.arange(-.5,total_chambers+1),cmap.N)
    where=np.argwhere(result["labels"]>0)
    lo=coords[where.min(axis=0)]-.25; hi=coords[where.max(axis=0)]+.25
    def setup(ax):
        ax.set(xlim=(lo[1],hi[1]),ylim=(lo[0],hi[0]),xlabel="x",ylabel="y")
        ax.set_aspect("equal"); ax.spines[["top","right"]].set_visible(False)
    fig,ax=plt.subplots(figsize=(6.3,6.0),layout="constrained"); setup(ax)
    image=ax.imshow(frames[0]["labels"],origin="lower",extent=extent,cmap=cmap,norm=norm,interpolation="nearest")
    title=ax.set_title("")
    ax.text(.02,.02,"Fixed older chambers; illustrative growth",transform=ax.transAxes,fontsize=9)
    def update(j):
        f=frames[j]; image.set_data(f["labels"])
        stage=style["first_frame"] if j==0 else f"chamber {f['chamber']}"
        title.set_text(f"{style['animation_title']} + successive chambers\n{stage}   t = {f['time']:.2f}")
        return image,title
    sequence=[0]*12+list(range(len(frames)))+[len(frames)-1]*12
    anim=FuncAnimation(fig,update,frames=sequence,interval=100)
    anim.save(output/"chamber_growth.gif",writer=PillowWriter(fps=12))
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(5.0,4.3),layout="constrained")
    initial=result["initial_labels"]
    initial_where=np.argwhere(initial>0)
    seed_lo=coords[initial_where.min(axis=0)]-.10
    seed_hi=coords[initial_where.max(axis=0)]+.10
    ax.imshow(initial,origin="lower",extent=extent,cmap=cmap,norm=norm,interpolation="nearest")
    ax.set(xlim=(seed_lo[1],seed_hi[1]),ylim=(seed_lo[0],seed_hi[0]),xlabel="x",ylabel="y")
    ax.set_aspect("equal")
    for report in result["seed_reports"]:
        ax.text(*report["centroid"],str(report["chamber"]),ha="center",va="center",color="white",weight="bold",fontsize=15)
    ax.set_title(style["seed_title"])
    fig.savefig(output/"seed.png",dpi=170);plt.close(fig)
    fig,(ax,graph)=plt.subplots(1,2,figsize=(11,5.4),layout="constrained"); setup(ax)
    ax.imshow(result["labels"],origin="lower",extent=extent,cmap=cmap,norm=norm,interpolation="nearest")
    for cell in range(1,total_chambers+1):
        ys,xs=np.nonzero(result["labels"]==cell)
        ax.text(coords[xs].mean(),coords[ys].mean(),str(cell),ha="center",va="center",color="white",weight="bold")
    ax.set_title(style["summary_title"])
    history=result["history"]
    for cell in range(result["seed_chambers"]+1,total_chambers+1):
        points=history[history[:,0]==cell]
        color=cmap(cell)
        graph.plot(points[:,2],points[:,3],color=color,lw=2)
        graph.plot(points[:,2],points[:,4],"--",color="black",lw=.7)
    graph.set(xlabel="global time",ylabel="new chamber area",title="Targets (colour) and diffuse areas (dashes)")
    fig.savefig(output/"growth_summary.png",dpi=170);plt.close(fig)


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
    args=parser.parse_args()
    p=ColonyParameters(new_chambers=args.chambers,adhesion_strength=args.adhesion,
                       turning_angle_degrees=args.turn_angle,spacing=args.spacing,
                       seed_mode=args.seed_mode,second_piece_radius_ratio=args.second_radius_ratio,
                       seed_separation_fraction=args.seed_separation,seed_orientation_degrees=args.seed_angle)
    output=args.output or Path("output")/args.seed_mode
    result=simulate_colony(p,args.areas)
    metrics=save_colony(result,output)
    render_colony(result,output)
    print(json.dumps(metrics,indent=2))


if __name__=="__main__":
    main()
