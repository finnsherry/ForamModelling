"""Saving results (csv, json, npz) and rendering figures (gif, png)."""

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .parameters import SEED_STYLES


def save_colony(result, output):
    """Write numerical results and validation metrics; returns the metrics."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    p = result["parameters"]
    history = result["history"]

    with (output/"area_history.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "chamber",
            "formation_time",
            "global_time",
            "target_area",
            "diffuse_area",
            "pixel_area",
        ])
        writer.writerows(history)

    metrics = {
        "parameters": asdict(p),
        "implementation": asdict(result["implementation"]),
        "max_diffuse_area_error_all_steps": result["max_diffuse_area_error_all_steps"],
        "seed_reports": result["seed_reports"],
        "reports": result["reports"],
    }
    (output/"validation.json").write_text(json.dumps(metrics, indent=2))

    np.savez_compressed(
        output/"simulation.npz",
        labels=result["labels"],
        initial_labels=result["frames"][0]["labels"],
        coordinates=result["coordinates"],
        history=history,
    )
    return metrics


def render_colony(result, output, animation=True):
    """Write the summary figure and, if `animation`, the growth animation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from PIL import Image

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    p = result["parameters"]
    frames = result["frames"]
    coords = result["coordinates"]
    seed_chambers = len(result["seed_reports"])
    seed_name = f"{p.seed_mode.capitalize()} seed"
    total_chambers = len(p.final_areas()) + seed_chambers
    h = result["implementation"].spacing
    extent = [coords[0] - h/2, coords[-1] + h/2] * 2

    # One colour per chamber, with white for the background.
    colors = [plt.cm.viridis(v) for v in np.linspace(0.08, 0.92, total_chambers)]
    seed_colors = SEED_STYLES[p.seed_mode]["colors"]
    if seed_colors:
        colors[:len(seed_colors)] = seed_colors
    cmap = ListedColormap(["#ffffff"] + colors)
    norm = BoundaryNorm(np.arange(-0.5, total_chambers + 1), cmap.N)

    def show(ax, labels):
        return ax.imshow(
            labels,
            origin="lower",
            extent=extent,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )

    # The colony figures are zoomed to the final colony plus a border.
    where = np.argwhere(result["labels"] > 0)
    lo = coords[where.min(axis=0)] - 0.1
    hi = coords[where.max(axis=0)] + 0.1

    def setup(ax):
        ax.set(xlim=(lo[1], hi[1]), ylim=(lo[0], hi[0]), xlabel="x", ylabel="y")
        ax.set_aspect("equal")
        ax.spines[["top", "right"]].set_visible(False)

    if animation:
        fig, ax = plt.subplots(figsize=(6.3, 6.0), layout="constrained")
        setup(ax)
        ax.set_axis_off()
        image = show(ax, frames[0]["labels"])
        # A two-line placeholder, so that the layout leaves room for the title.
        # A fixed y stops matplotlib from moving the title while it is hidden.
        title = ax.set_title("\n", y=1.0)

        # Blitting: draw everything that stays the same once, then redraw only
        # the image and the title for each frame.
        fig.canvas.draw()
        fig.set_layout_engine("none")
        image.set_visible(False)
        title.set_visible(False)
        fig.canvas.draw()
        background = fig.canvas.copy_from_bbox(fig.bbox)
        image.set_visible(True)
        title.set_visible(True)

        pictures = []
        for j, f in enumerate(frames):
            image.set_data(f["labels"])
            stage = "seed" if j == 0 else f"chamber {f['chamber']}"
            title.set_text(f"{seed_name} + successive chambers\n{stage}   t = {f['time']:.2f}")
            fig.canvas.restore_region(background)
            ax.draw_artist(image)
            ax.draw_artist(title)
            picture = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
            pictures.append(picture.quantize(method=Image.Quantize.FASTOCTREE))
        plt.close(fig)

        # 12 frames per second, with the first and last frames held for two seconds.
        durations = [2000] + [1000 / 12] * (len(pictures) - 2) + [2000]
        pictures[0].save(
            output/"chamber_growth.gif",
            save_all=True,
            append_images=pictures[1:],
            duration=durations,
            loop=0,
        )

    # Summary figure: the final colony, and the area history of each new chamber.
    fig, (ax, graph) = plt.subplots(1, 2, figsize=(11, 5.4), layout="constrained")
    setup(ax)
    show(ax, result["labels"])
    for cell in range(1, total_chambers + 1):
        ys, xs = np.nonzero(result["labels"] == cell)
        ax.text(
            coords[xs].mean(),
            coords[ys].mean(),
            str(cell),
            ha="center",
            va="center",
            color="white",
            weight="bold",
        )
    seed_cells = "s 1–" if seed_chambers > 1 else " "
    ax.set_title(f"{seed_name} (chamber{seed_cells}{seed_chambers}) and later chambers")

    history = result["history"]
    for cell in range(seed_chambers + 1, total_chambers + 1):
        points = history[history[:, 0] == cell]
        graph.plot(points[:, 2], points[:, 3], color=cmap(cell), lw=2)
        graph.plot(points[:, 2], points[:, 4], "--", color="black", lw=0.7)
    graph.set(
        xlabel="global time",
        ylabel="new chamber area",
        title="Targets (colour) and diffuse areas (dashes)",
    )
    fig.savefig(output/"growth_summary.png", dpi=150)
    plt.close(fig)
