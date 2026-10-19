"""Pinned 2D chamber growth with prescribed area (an exploratory geometric model).

Run: python foram_growth.py --output output
Measured growth: python foram_growth.py --growth-csv growth.csv --output output
The CSV needs columns time,area, with increasing times and nondecreasing areas.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator


@dataclass
class Parameters:
    aperture_half_width: float = 0.30
    gamma: float = 0.012
    points: int = 81
    end_time: float = 12.0
    final_area: float = 1.80
    max_dt: float = 0.004
    snapshots: int = 61


def polygon_area(curve):
    """Area between a clockwise open arc and its fixed straight closing segment."""
    x, y = curve.T
    return -0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))


def area_gradient(curve):
    x, y = curve.T
    return np.column_stack((
        -0.5 * (np.roll(y, -1) - np.roll(y, 1)),
        -0.5 * (np.roll(x, 1) - np.roll(x, -1)),
    ))


def geometry(curve):
    lengths = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    if lengths.min() < 1e-10:
        raise RuntimeError("Curve nodes collapsed; reduce the timestep.")
    s = np.r_[0.0, np.cumsum(lengths)]
    tangent = np.gradient(curve, s, axis=0, edge_order=2)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    # Left normal is outward for the clockwise arc.
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    curvature_vector = np.gradient(tangent, s, axis=0, edge_order=2)
    curvature = -np.sum(curvature_vector * normal, axis=1)
    # A model mobility that goes to zero at the pinned aperture endpoints.
    mobility = np.sin(np.pi * s / s[-1]) ** 2
    mobility[[0, -1]] = 0.0
    return s, normal, curvature, mobility


def redistribute(curve):
    """Piecewise-linear equal-arclength remeshing; area is corrected afterwards."""
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
    grid = np.linspace(0.0, s[-1], len(curve))
    return np.column_stack([np.interp(grid, s, curve[:, k]) for k in (0, 1)])


def project_area(curve, target):
    """Enforce polygon area by a small scalar displacement in a normal direction."""
    _, normal, _, mobility = geometry(curve)
    direction = mobility[:, None] * normal
    corrected = curve.copy()
    for _ in range(12):
        residual = target - polygon_area(corrected)
        if abs(residual) <= 2e-13 * max(1.0, target):
            return corrected
        slope = np.sum(area_gradient(corrected) * direction)
        if slope <= 1e-12:
            raise RuntimeError("Area correction is singular for this geometry.")
        corrected += (residual / slope) * direction
    raise RuntimeError("Area correction did not converge; reduce the timestep.")


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


def measured_growth(csv_path):
    with open(csv_path, newline="") as stream:
        rows = list(csv.DictReader(stream))
    times = np.array([float(row["time"]) for row in rows])
    areas = np.array([float(row["area"]) for row in rows])
    if len(times) < 2 or not np.isfinite(times).all() or not np.isfinite(areas).all():
        raise ValueError("Provide at least two finite time,area observations.")
    if np.any(np.diff(times) <= 0) or np.any(np.diff(areas) < 0) or np.any(areas <= 0):
        raise ValueError("Times must increase and positive areas must not decrease.")
    times = times - times[0]
    fitted = PchipInterpolator(times, areas, extrapolate=False)
    return fitted, fitted.derivative(), float(times[-1]), float(areas[0])


def self_intersects(curve):
    """Check proper crossings of nonadjacent edges, including the fixed base."""
    starts = curve
    ends = np.roll(curve, -1, axis=0)
    def cross(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    for i in range(len(curve)):
        candidates = np.arange(i + 2, len(curve))
        if i == 0:
            candidates = candidates[candidates != len(curve) - 1]
        if not len(candidates):
            continue
        a, b = starts[i], ends[i]
        c, d = starts[candidates], ends[candidates]
        o1, o2 = cross(b - a, c - a), cross(b - a, d - a)
        o3, o4 = cross(d - c, a - c), cross(d - c, b - c)
        if np.any((o1 * o2 < -1e-20) & (o3 * o4 < -1e-20)):
            return True
    return False


def simulate(parameters=None, growth_csv=None):
    p = parameters or Parameters()
    if p.points < 15 or p.gamma < 0 or p.max_dt <= 0 or p.end_time <= 0:
        raise ValueError("Invalid numerical parameters.")
    if p.aperture_half_width <= 0 or p.snapshots < 2:
        raise ValueError("Aperture must be positive; at least two snapshots are required.")
    angles = np.linspace(np.pi, 0.0, p.points)
    curve = p.aperture_half_width * np.column_stack((np.cos(angles), np.sin(angles)))
    curve[[0, -1], 1] = 0.0
    if growth_csv:
        target, target_rate, end_time, initial_area = measured_growth(growth_csv)
        curve[:, 1] *= initial_area / polygon_area(curve)
        growth_label = "User-supplied time/area data (monotone cubic interpolation)"
    else:
        end_time = p.end_time
        target, target_rate = demo_growth(polygon_area(curve), p.final_area, end_time)
        growth_label = "Illustrative sigmoid; no biological calibration"
    endpoints = curve[[0, -1]].copy()
    save_times = np.linspace(0.0, end_time, p.snapshots)
    curves, times, areas, targets = [curve.copy()], [0.0], [polygon_area(curve)], [float(target(0.0))]
    t, steps, max_error, max_endpoint_error = 0.0, 0, 0.0, 0.0
    for save_time in save_times[1:]:
        while t < save_time - 1e-12:
            s, normal, curvature, mobility = geometry(curve)
            grad = area_gradient(curve)
            direction = mobility[:, None] * normal
            weights = np.sum(grad * direction, axis=1)
            # Discrete counterpart of lambda=(A' + gamma*integral(m*kappa))/integral(m).
            multiplier = (float(target_rate(t)) + p.gamma * np.dot(weights, curvature)) / weights.sum()
            velocity = direction * (multiplier - p.gamma * curvature)[:, None]
            spacing = np.diff(s).min()
            diffusion_dt = 0.15 * spacing**2 / max(p.gamma, 1e-15)
            motion_dt = 0.12 * spacing / max(np.linalg.norm(velocity, axis=1).max(), 1e-15)
            dt = min(p.max_dt, diffusion_dt, motion_dt, save_time - t)
            if dt < 1e-10:
                raise RuntimeError("Timestep collapsed: refine or revise the growth/shape model.")
            candidate = redistribute(curve + dt * velocity)
            curve = project_area(candidate, float(target(t + dt)))
            curve[[0, -1]] = endpoints
            t += dt
            steps += 1
            if not np.isfinite(curve).all() or curve[:, 1].min() < -1e-9:
                raise RuntimeError("The curve left the allowed half-plane or became nonfinite.")
            max_error = max(max_error, abs(polygon_area(curve) - float(target(t))))
            max_endpoint_error = max(max_endpoint_error, float(np.abs(curve[[0, -1]] - endpoints).max()))
        if self_intersects(curve):
            raise RuntimeError("Boundary self-intersection: this curve representation is no longer valid.")
        curves.append(curve.copy())
        times.append(t)
        areas.append(polygon_area(curve))
        targets.append(float(target(t)))
    return {
        "curves": np.array(curves), "time": np.array(times),
        "area": np.array(areas), "target_area": np.array(targets),
        "steps": steps, "max_area_error_all_steps": max_error,
        "max_endpoint_error_all_steps": max_endpoint_error,
        "growth_label": growth_label, "parameters": p,
    }


def render(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    curves, times = result["curves"], result["time"]
    a = result["parameters"].aperture_half_width
    limit = max(float(np.abs(curves[:, :, 0]).max()) * 1.15, a * 2)
    height = float(curves[:, :, 1].max()) * 1.12
    def axes_wall(ax):
        ax.plot([-limit, -a], [0, 0], color="#303943", lw=3)
        ax.plot([a, limit], [0, 0], color="#303943", lw=3)
        ax.scatter([-a, a], [0, 0], color="#e76542", s=35, zorder=5)
        ax.set(xlim=(-limit, limit), ylim=(-height * .06, height), xlabel="x", ylabel="y")
        ax.set_aspect("equal")
        ax.spines[["top", "right"]].set_visible(False)
    plt.rcParams.update({"font.size": 11, "figure.facecolor": "white"})
    fig, (ax, graph) = plt.subplots(1, 2, figsize=(10.5, 4.8), layout="constrained")
    axes_wall(ax)
    for j in np.linspace(0, len(curves) - 1, 6).astype(int):
        ax.plot(*curves[j].T, color=plt.cm.viridis(j / (len(curves) - 1)), lw=2, label=f"t={times[j]:.1f}")
    ax.set_title("One chamber; aperture endpoints fixed")
    ax.legend(fontsize=8, loc="upper right")
    graph.plot(times, result["target_area"], color="#e76542", lw=3, label="Prescribed area")
    graph.plot(times, result["area"], "--", color="#264b73", lw=1.8, label="Simulated area")
    graph.set(xlabel="time", ylabel="area", title="Growth constraint")
    graph.legend()
    graph.text(.03, .03, "Illustrative growth, not a biological fit" if "Illustrative" in result["growth_label"] else "User-supplied growth data", transform=graph.transAxes, fontsize=9)
    fig.savefig(output / "growth_summary.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.4, 5.0), layout="constrained")
    axes_wall(ax)
    line, = ax.plot(*curves[0].T, color="#267b91", lw=2.5)
    title = ax.set_title("")
    annotation = ax.text(.03, .03, "Illustrative growth; dimensionless units", transform=ax.transAxes, fontsize=9)
    if "Illustrative" not in result["growth_label"]:
        annotation.set_text("User-supplied time/area data")
    def update(index):
        line.set_data(*curves[index].T)
        title.set_text(f"t = {times[index]:.2f}    area = {result['area'][index]:.3f}")
        return line, title
    animation = FuncAnimation(fig, update, frames=len(curves), interval=90)
    animation.save(output / "chamber_growth.gif", writer=PillowWriter(fps=12))
    plt.close(fig)
    np.savez_compressed(output / "simulation.npz", curves=curves, time=times, area=result["area"], target_area=result["target_area"])
    with (output / "area_history.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time", "simulated_area", "target_area", "absolute_error"])
        writer.writerows(zip(times, result["area"], result["target_area"], np.abs(result["area"] - result["target_area"])))
    metrics = {key: result[key] for key in ("steps", "max_area_error_all_steps", "max_endpoint_error_all_steps", "growth_label")}
    metrics.update(initial_area=float(result["area"][0]), final_area=float(result["area"][-1]), all_saved_curves_without_proper_crossings=all(not self_intersects(c) for c in curves))
    (output / "validation.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="output")
    parser.add_argument("--growth-csv")
    parser.add_argument("--gamma", type=float, default=0.012)
    parser.add_argument("--points", type=int, default=81)
    parser.add_argument("--max-dt", type=float, default=0.004)
    parser.add_argument("--aperture-half-width", type=float, default=0.30)
    parser.add_argument("--multi", action="store_true", help="Grow sequential chambers from a central seed")
    parser.add_argument("--chambers", type=int, default=6, help="Number of NEW chambers in multi mode")
    parser.add_argument("--adhesion", type=float, default=1.0, help="Attraction toward fixed older walls in multi mode")
    parser.add_argument("--turn-angle", type=float, default=65.0, help="Coiling guide angle increment in degrees")
    parser.add_argument("--areas", type=float, nargs="+", help="One target final area per NEW chamber")
    parser.add_argument("--seed-mode", choices=["single", "bilocular"], default="bilocular")
    parser.add_argument("--second-radius-ratio", type=float, default=1.10)
    parser.add_argument("--seed-separation", type=float, default=0.60)
    parser.add_argument("--seed-angle", type=float, default=0.)
    args = parser.parse_args()
    if args.multi:
        if args.growth_csv:
            parser.error("Multi mode takes final chamber areas via --areas; --growth-csv is for single-chamber timing.")
        from foram_colony import ColonyParameters, simulate_colony, render_colony
        p = ColonyParameters(new_chambers=args.chambers, adhesion_strength=args.adhesion,
                             turning_angle_degrees=args.turn_angle,seed_mode=args.seed_mode,
                             second_piece_radius_ratio=args.second_radius_ratio,
                             seed_separation_fraction=args.seed_separation,seed_orientation_degrees=args.seed_angle)
        print(json.dumps(render_colony(simulate_colony(p, args.areas), args.output), indent=2))
        return
    result = simulate(Parameters(gamma=args.gamma, points=args.points, max_dt=args.max_dt, aperture_half_width=args.aperture_half_width), args.growth_csv)
    print(json.dumps(render(result, args.output), indent=2))


if __name__ == "__main__":
    main()
