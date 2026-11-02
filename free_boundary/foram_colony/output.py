"""Saving results (csv, json, npz) and rendering figures (gif, png)."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .parameters import SEED_STYLES


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
