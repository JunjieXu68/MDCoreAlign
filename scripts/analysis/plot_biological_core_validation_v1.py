#!/usr/bin/env python3
"""Render frozen biological-validation tables and real molecular structures.

This script does not estimate cores, alter scientific configuration, or choose
frames. It uses the full-scope residue tables and preselected superpositions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[2]
PYMOL = Path("/home/jjxu/miniconda3/envs/pymol/bin/python")
COLORS = {
    "MDCoreAlign_native60": "#7955A1",
    "MDCoreAlign_matched_k": "#AF78B5",
    "MDCoreAlign": "#7955A1",
    "MDLovoFit": "#148A9D",
    "all_CA": "#777777",
    "literature_CORE": "#3377AA",
    "CORE": "#3377AA",
    "NMP": "#EE9944",
    "LID": "#BB5566",
    "framework_A": "#3377AA",
    "framework_B": "#CC7733",
}
LABELS = {
    "MDCoreAlign_native60": "MDCoreAlign (native 60%)",
    "MDCoreAlign_matched_k": "MDCoreAlign (matched k)",
    "MDCoreAlign": "MDCoreAlign",
    "MDLovoFit": "MDLovoFit",
    "all_CA": "All Cα Kabsch",
    "literature_CORE": "Literature CORE Kabsch",
    "literature_framework_A": "Framework A",
    "literature_framework_B": "Framework B",
    "framework_A": "Framework A",
    "framework_B": "Framework B",
}
LABELS.update(
    {"portal_26_35": "Portal 26–35", "portal_53_56": "Portal 53–56", "portal_73_76": "Portal 73–76"}
)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def read_tsv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vals(rows, key):
    return np.asarray([float(row[key]) for row in rows])


def label(ax, text):
    ax.set_title(text, loc="left", fontweight="bold", pad=10)


def spans(x, mask):
    result = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        if start is not None and (i == len(mask) - 1 or not mask[i + 1]):
            result.append((float(x[start]) - 0.5, float(x[i]) + 0.5))
            start = None
    return result


def get_case(run, config, name):
    directory = run / "cases" / name / "analysis"
    case = next(c for c in config["cases"] if c["case_id"] == name)
    summary = json.loads((directory / "case_summary.json").read_text())
    rows = [r for r in read_tsv(directory / "residue_results.tsv") if r["scope"] == "full"]
    regions = {k: np.asarray(v, dtype=bool) for k, v in case["regions"].items()}
    return {
        "case_id": name,
        "directory": directory,
        "config": case,
        "summary": summary,
        "rows": rows,
        "x": vals(rows, "source_resid"),
        "regions": regions,
        "geometry": read_tsv(directory / "geometry_metrics.tsv"),
        "random": read_tsv(directory / "random_geometry.tsv"),
        "region_metrics": read_tsv(directory / "region_metrics.tsv"),
        "profiles": dict(np.load(directory / "alignment_profiles.npz")),
    }


def region_tracks(ax, case, y=-0.20, step=0.14, annotate=True):
    transform = ax.get_xaxis_transform()
    for j, (name, mask) in enumerate(case["regions"].items()):
        color = COLORS.get(name, "#CE946B")
        intervals = spans(case["x"], mask)
        for left, right in intervals:
            ax.plot(
                [left, right],
                [y - j * step] * 2,
                lw=5,
                color=color,
                transform=transform,
                clip_on=False,
                solid_capstyle="butt",
            )
            if annotate and right - left >= 12:
                ax.text(
                    (left + right) / 2,
                    y - j * step - 0.075,
                    LABELS.get(name, name),
                    transform=transform,
                    color=color,
                    fontsize=8,
                    ha="center",
                    va="top",
                )


def sequence_panel(ax, case, letter="C", framework=False):
    rows, x = case["rows"], case["x"]
    w = vals(rows, "MDCoreAlign_weight")
    for name, mask in case["regions"].items():
        if name == "portal_combined":
            continue
        if framework:
            continue
        for a, b in spans(x, mask):
            ax.axvspan(a, b, color=COLORS.get(name, "#CE946B"), alpha=0.09, lw=0)
    ax.plot(x, w, color=COLORS["MDCoreAlign"], lw=1.25, zorder=3)
    core = vals(rows, "MDCoreAlign_native60").astype(bool)
    ax.scatter(x[core], w[core], s=6, color=COLORS["MDCoreAlign"], zorder=4)
    ax.set(
        xlim=(x.min() - 0.5, x.max() + 0.5),
        ylim=(0, max(1.18, float(w.max()) * 1.18)),
        ylabel="Fitting weight",
    )
    ax.set_xlabel("Mapped author residue number", labelpad=31 if framework else 8)
    if framework:
        for j, (name, mask) in enumerate(case["regions"].items()):
            yy = -0.14 - j * 0.16
            for a, b in spans(x, mask):
                ax.plot(
                    [a, b],
                    [yy, yy],
                    transform=ax.get_xaxis_transform(),
                    lw=5,
                    color=COLORS.get(name, "#CE946B"),
                    clip_on=False,
                    solid_capstyle="butt",
                )
            ax.text(
                1.006,
                yy,
                LABELS.get(name, name),
                transform=ax.transAxes,
                color=COLORS.get(name, "#CE946B"),
                fontsize=8,
                va="center",
            )
    label(ax, f"{letter}  Fitting weights; dots mark the native 60% report set")
    if not framework:
        handles = [
            plt.Line2D(
                [0], [0], lw=6, color=COLORS.get(n, "#CE946B"), alpha=0.5, label=LABELS.get(n, n)
            )
            for n in case["regions"]
            if n != "portal_combined"
        ]
        ax.legend(
            handles=handles,
            loc="lower right",
            bbox_to_anchor=(1, 1.02),
            borderaxespad=0,
            frameon=False,
            ncol=min(3, len(handles)),
        )


def frequency_panel(ax, case, letter="D"):
    x, rows = case["x"], case["rows"]
    freq = vals(rows, "MDLovoFit_frequency")
    ax.fill_between(x, 0, freq, step="mid", color=COLORS["MDLovoFit"], alpha=0.75, lw=0)
    ax.set(
        xlim=(x.min() - 0.5, x.max() + 0.5),
        ylim=(-0.04, 1.08),
        yticks=[0, 0.5, 1],
        ylabel="Selection frequency",
        xlabel="Mapped author residue number",
    )
    kind = case["summary"]["MDLovoFit_core_kind"].replace("_", " ")
    label(ax, f"{letter}  MDLovoFit occupancy ({kind})")


def geometry_panel(ax, case, letter="E", literature=True):
    keep = ["MDCoreAlign_native60", "MDCoreAlign_matched_k", "MDLovoFit"]
    if literature:
        keep += (
            ["literature_CORE"]
            if case["case_id"] == "ADK"
            else [f"literature_{n}" for n in case["regions"]]
        )
    rows = [
        r
        for name in keep
        for r in case["geometry"]
        if r["method"] == name and r.get("G_A") not in (None, "", "None", "nan")
    ]
    for i, row in enumerate(rows):
        k = int(row["core_size"])
        random = vals([r for r in case["random"] if int(r["cardinality"]) == k], "G_A")
        parts = ax.violinplot(
            [random], positions=[i], vert=False, widths=0.64, showmedians=True, showextrema=False
        )
        for body in parts["bodies"]:
            body.set_facecolor("#C3C9CD")
            body.set_edgecolor("#808A91")
            body.set_alpha(0.65)
        parts["cmedians"].set_color("#667078")
        color = COLORS.get(
            row["method"], COLORS.get(row["method"].replace("literature_", ""), "#CC7733")
        )
        ax.scatter(float(row["G_A"]), i, s=44, color=color, edgecolor="white", lw=0.7, zorder=5)
    names = []
    for row in rows:
        method = row["method"]
        name = LABELS.get(method, method)
        if method == "literature_CORE":
            name = "Literature CORE"
        names.append(f"{name}\n(k={row['core_size']})")
    ax.set(yticks=range(len(rows)), yticklabels=names, xlabel="Held-out internal-distance G (Å)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.15, zorder=0)
    label(ax, f"{letter}  Frozen training sets on held-out coordinates")
    footnote = "Point: selected set. Grey: 1,000 equal-size random sets per row."
    if literature:
        footnote += (
            "\nLiterature sets have their own size reference; compare to each row's control."
        )
    ax.text(0, -0.20, footnote, transform=ax.transAxes, fontsize=8, va="top")


def rmsf_panel(ax, case, letter="F"):
    for name, profile in case["profiles"].items():
        ax.plot(
            case["x"],
            profile,
            lw=1.1,
            color=COLORS.get(name, "black"),
            label=LABELS.get(name, name),
        )
    ax.set_ylim(0, max(float(np.max(p)) for p in case["profiles"].values()) * 1.28)
    ax.set(
        xlim=(case["x"].min(), case["x"].max()),
        ylabel="RMSF about each fitted mean (Å)",
        xlabel="Mapped author residue number",
    )
    ax.legend(frameon=False, fontsize=8, ncol=2)
    label(ax, f"{letter}  The analysis reference frame changes RMSF")


PYMOL_SCRIPT = r"""
import json,sys,math
from pathlib import Path
import pymol
from pymol import cmd, cgo
pymol.finish_launching(['pymol','-cq'])
payload=json.loads(Path(sys.argv[1]).read_text())
cmd.set('max_threads',1)
cmd.bg_color('white')
cmd.set('ray_shadows',0)
cmd.set('antialias',2)
cmd.set('ambient',.55)
cmd.set('specular',.15)
cmd.set('orthoscopic',1)
cmd.set('ray_opaque_background',1)

def rgb(h):return [int(h[i:i+2],16)/255 for i in (1,3,5)]
def models(path):
    found=[];current=[]
    for line in Path(path).read_text().splitlines():
        if line.startswith('ATOM') and line[12:16].strip()=='CA':
            current.append([float(line[a:b]) for a,b in [(30,38),(38,46),(46,54)]])
        elif line[:6].strip() in ('END','ENDMDL') and current:
            found.append(current);current=[]
    if current:found.append(current)
    return found
def trace(name,xyz,colors,radius=.13):
    obj=[]
    for i in range(len(xyz)-1):
        if payload['residues'][i]['chain'] != payload['residues'][i+1]['chain']:continue
        if payload['residues'][i+1]['sequence_position']-payload['residues'][i]['sequence_position'] != 1:continue
        obj += [cgo.CYLINDER,*xyz[i],*xyz[i+1],radius,*colors[i],*colors[i+1]]
    cmd.load_cgo(obj,name)
def save(path):
    cmd.png(path,1200,900,ray=1,quiet=1)
cmd.delete('all')
full=payload['full_protein']
if full:
    cmd.load(full,'protein')
    cmd.remove('not polymer.protein')
    cmd.hide('everything')
    cmd.show('cartoon','protein')
    cmd.set('cartoon_loop_radius',.20)
    cmd.orient('protein')
    cmd.zoom('protein',4)
else:
    xyz=models(payload['reference'])[0]
    trace('protein',xyz,[[.7,.7,.7]]*len(xyz))
    cmd.orient('protein');cmd.zoom('protein',4)
view=cmd.get_view()
for job in payload['maps']:
    if full:
        cmd.color('gray75','protein')
        for i,(residue,col) in enumerate(zip(payload['residues'],job['colors'])):
            name='col'+str(i)
            cmd.set_color(name,col)
            cmd.color(name,'protein and resi '+str(residue['source_resid'])+residue.get('icode',''))
    else:
        cmd.delete('protein')
        trace('protein',xyz,job['colors'])
    cmd.set_view(view)
    save(job['path'])
for job in payload.get('overlays',[]):
    cmd.delete('all')
    frames=models(job['pdb'])
    if len(frames)!=len(job['colors']):raise ValueError('superposition state count mismatch')
    for i,(xyz,col) in enumerate(zip(frames,job['colors'])):
        trace('frame'+str(i),xyz,[col]*len(xyz),.095)
    if 'overlayview' not in globals():
        cmd.orient('all');cmd.zoom('all',4);overlayview=cmd.get_view()
    else:cmd.set_view(overlayview)
    save(job['path'])
cmd.quit()
"""


def render_structures(run, case, figdir):
    molecular = figdir / "structure_renders"
    molecular.mkdir(exist_ok=True)
    scripts = figdir / "pymol"
    scripts.mkdir(exist_ok=True)
    source = run / "cases" / case["case_id"] / "reference_full_protein.pdb"
    residues = case["config"]["residues"]
    if source.exists():
        ca = [
            line
            for line in source.read_text().splitlines()
            if line.startswith("ATOM") and line[12:16].strip() == "CA"
        ]
        observed = [(int(line[22:26]), line[26].strip(), line[17:20].strip()) for line in ca]
        expected = [(int(r["source_resid"]), r.get("icode", ""), r["resname"]) for r in residues]
        if observed != expected:
            raise ValueError(
                f"{case['case_id']}: full-protein render residue mapping does not match frozen residue metadata"
            )
    default = np.tile([0.74, 0.76, 0.77], (len(residues), 1))
    maps = []
    cmap = plt.get_cmap("coolwarm")
    w = vals(case["rows"], "MDCoreAlign_weight")
    norm = Normalize(float(w.min()), float(w.max()))
    if case["case_id"] == "ADK":
        colors = default.copy()
        for name, mask in case["regions"].items():
            colors[mask] = matplotlib.colors.to_rgb(COLORS[name])
        maps.append({"name": "domains", "colors": colors.tolist()})
        maps.append({"name": "weights", "colors": cmap(norm(w))[:, :3].tolist()})
    elif case["case_id"] == "3nci_A":
        allmasks = list(case["regions"].values())
        shared = np.logical_and.reduce(allmasks)
        for name, mask in case["regions"].items():
            colors = default.copy()
            colors[mask] = matplotlib.colors.to_rgb(COLORS.get(name, "#3377AA"))
            colors[shared] = matplotlib.colors.to_rgb("#AA55AA")
            maps.append({"name": name, "colors": colors.tolist()})
    for job in maps:
        job["path"] = str(molecular / f"{case['case_id']}_{job['name']}.png")
    overlays = []
    if case["case_id"] == "ADK":
        frames = case["summary"]["superposition_frame_indices"]
        colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.92, len(frames)))[:, :3].tolist()
        for method in ["MDCoreAlign", "literature_CORE"]:
            overlays.append(
                {
                    "name": method,
                    "pdb": str(case["directory"] / f"superposition_{method}.pdb"),
                    "path": str(molecular / f"ADK_overlay_{method}.png"),
                    "colors": colors,
                }
            )
    payload = {
        "full_protein": str(source) if source.exists() else None,
        "reference": str(case["directory"] / "reference.pdb"),
        "residues": residues,
        "maps": maps,
        "overlays": overlays,
    }
    payload_path = scripts / f"{case['case_id']}_render.json"
    payload_path.write_text(json.dumps(payload, indent=2) + "\n")
    script = scripts / "render_real_structures.py"
    script.write_text(PYMOL_SCRIPT)
    completed = False
    if PYMOL.exists() and maps:
        proc = subprocess.run(
            [str(PYMOL), str(script), str(payload_path)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        (scripts / f"{case['case_id']}.log").write_text(proc.stdout + "\n" + proc.stderr)
        completed = proc.returncode == 0 and all(
            Path(job["path"]).exists() for job in maps + overlays
        )
    return {
        "completed": completed,
        "representation": "full-protein cartoon" if source.exists() else "Cα trace",
        "full_protein_residue_mapping_verified": bool(source.exists()),
        "maps": {job["name"]: Path(job["path"]) for job in maps},
        "overlays": {job["name"]: Path(job["path"]) for job in overlays},
        "script": str(script),
        "payload": str(payload_path),
        "norm": norm,
    }


def structure_panel(ax, render, key, title):
    path = render["maps"].get(key)
    if render["completed"] and path and path.exists():
        ax.imshow(plt.imread(path))
    else:
        ax.text(
            0.5,
            0.5,
            "Structure render unavailable.\nReproducible PyMOL script retained.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.axis("off")
    label(ax, title)
    if render["representation"] == "Cα trace":
        ax.text(
            0.5,
            -0.035,
            render["representation"],
            transform=ax.transAxes,
            ha="center",
            fontsize=8,
            color="#555555",
        )


def save_figure(fig, figdir, stem):
    outputs = []
    for ext in ["pdf", "png"]:
        path = figdir / f"{stem}.{ext}"
        fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.12)
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def adk_main(run, case, figdir, render):
    fig = plt.figure(figsize=(14, 13.2))
    gs = fig.add_gridspec(4, 2, height_ratios=[2.4, 1.25, 0.95, 2.4], hspace=0.70, wspace=0.72)
    fig.subplots_adjust(bottom=0.18)
    a = fig.add_subplot(gs[0, 0])
    b = fig.add_subplot(gs[0, 1])
    structure_panel(a, render, "domains", "A  External structural domains")
    structure_panel(b, render, "weights", "B  Automatically estimated fitting weights")
    a.legend(
        handles=[
            plt.Line2D([0], [0], lw=6, color=COLORS[n], label=n) for n in ["CORE", "NMP", "LID"]
        ],
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.10),
    )
    cb = fig.colorbar(
        ScalarMappable(norm=render["norm"], cmap="coolwarm"), ax=b, fraction=0.035, pad=0.01
    )
    cb.set_label("Fitting weight (higher = more weight)", fontsize=8)
    sequence_panel(fig.add_subplot(gs[1, :]), case)
    frequency_panel(fig.add_subplot(gs[2, :]), case)
    geometry_panel(fig.add_subplot(gs[3, 0]), case)
    rmsf_panel(fig.add_subplot(gs[3, 1]), case)
    fig.suptitle(
        "ADK: external domain annotation, automatic reference frames and held-out geometry",
        y=0.99,
        fontweight="bold",
        fontsize=15,
    )
    counts = []
    for method in ["MDCoreAlign_native60", "MDLovoFit"]:
        row = next(
            r
            for r in case["region_metrics"]
            if r["scope"] == "full" and r["region"] == "CORE" and r["method"] == method
        )
        counts.append(
            f"{'MDCoreAlign' if method.startswith('MDCoreAlign') else 'MDLovoFit'} {row['overlap']}/{row['core_size']}"
        )
    fig.text(
        0.5,
        0.955,
        "Full-trajectory report sets within literature CORE: " + "; ".join(counts) + ".",
        ha="center",
        fontsize=10,
    )
    n = case["summary"]["n_frames"]
    nt = len(case["summary"]["train_indices"])
    nv = len(case["summary"]["test_indices"])
    fig.text(
        0.5,
        0.932,
        f"A–D and F: full trajectory ({n:,} frames). E: first {nt:,} frames define sets → last {nv:,} frames evaluate geometry.",
        ha="center",
        fontsize=9,
    )
    fig.text(
        0.5,
        0.015,
        "Full-trajectory weights and RMSF are descriptive; time-block geometry uses sets frozen from the training half.\nLiterature CORE fitting is a knowledge-guided reference. Its own low fitting residual is not independent validation.",
        ha="center",
        fontsize=9,
    )
    return save_figure(fig, figdir, "adk_main")


def time_scale(run, case):
    c = case["config"]
    for key in ["frame_times_ps", "times_ps"]:
        if key in c:
            return np.asarray(c[key], dtype=float) / 1000, "Time (ns)"
    for key in ["frame_interval_ps", "dt_ps"]:
        if key in c:
            return np.arange(case["summary"]["n_frames"]) * float(c[key]) / 1000, "Time (ns)"
    source = run / "sources/local_inventory.json"
    if source.exists():
        inventory = json.loads(source.read_text()).get("cases", {}).get(case["case_id"], {})

        def seek(d):
            if isinstance(d, dict):
                for key in ["frame_interval_ps", "dt_ps", "trajectory_dt_ps"]:
                    if key in d and isinstance(d[key], (int, float)):
                        return float(d[key])
                for value in d.values():
                    x = seek(value)
                    if x is not None:
                        return x
            return None

        dt = seek(inventory)
        if dt is not None:
            return np.arange(case["summary"]["n_frames"]) * dt / 1000, "Time (ns)"
    return np.arange(case["summary"]["n_frames"]), "Frame index (0-based)"


def adk_motion(run, case, figdir, render):
    rows = read_tsv(case["directory"] / "domain_motion_timeseries.tsv")
    time, xlabel = time_scale(run, case)
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.7], hspace=0.45, wspace=0.28)
    a = fig.add_subplot(gs[0, 0])
    b = fig.add_subplot(gs[0, 1])
    metrics = list(dict.fromkeys(row["metric"] for row in rows))
    for name in metrics:
        selected = [r for r in rows if r["metric"] == name]
        y = vals(selected, "value")
        idx = vals(selected, "frame_index").astype(int)
        if name.endswith("distance_A"):
            a.plot(
                time[idx],
                y,
                lw=0.6,
                alpha=0.85,
                label=name.replace("_centroid_distance_A", "").replace("_", "–"),
            )
        else:
            b.plot(time[idx], y, lw=0.6, color=COLORS["LID"])
    a.set(xlabel=xlabel, ylabel="Domain-centroid distance (Å)")
    a.legend(frameon=False, ncol=3)
    b.set(xlabel=xlabel, ylabel="LID–CORE–NMP centroid angle (°)")
    label(a, "A  Relative domain separation without alignment")
    label(b, "B  Relative domain angle without alignment")
    for col, name in enumerate(["MDCoreAlign", "literature_CORE"]):
        ax = fig.add_subplot(gs[1, col])
        path = render["overlays"].get(name)
        if render["completed"] and path and path.exists():
            ax.imshow(plt.imread(path))
        else:
            ax.text(0.5, 0.5, "Render unavailable; PyMOL script retained", ha="center")
        ax.axis("off")
        label(ax, f"{'C' if col == 0 else 'D'}  {LABELS[name]}: seven fixed frames (Cα traces)")
    frames = case["summary"]["superposition_frame_indices"]
    fig.text(
        0.5,
        0.035,
        "Same preselected frame indices: "
        + ", ".join(map(str, frames))
        + ".\nTrace colour progresses from early (purple) to late (yellow); no extra fitting is applied during rendering.",
        ha="center",
        fontsize=9,
    )
    fig.suptitle(
        "ADK: sampled domain motion and the resulting analysis reference frames",
        fontweight="bold",
        fontsize=15,
    )
    fig.text(
        0.5,
        0.94,
        f"All {case['summary']['n_frames']:,} frames for distance/angle profiles; the seven displayed frames span the full trajectory.",
        ha="center",
        fontsize=10,
    )
    return save_figure(fig, figdir, "adk_motion")


def ifabp(run, case, figdir):
    fig = plt.figure(figsize=(13, 10.8))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 0.85, 1.65], hspace=0.7, wspace=0.7)
    fig.subplots_adjust(bottom=0.20)
    sequence_panel(fig.add_subplot(gs[0, :]), case, "A")
    frequency_panel(fig.add_subplot(gs[1, :]), case, "B")
    ax = fig.add_subplot(gs[2, 0])
    weights = vals(case["rows"], "MDCoreAlign_weight")
    names = [n for n in case["regions"] if n != "portal_combined"]
    boxes = [weights[case["regions"][n]] for n in names]
    ticklabels = []
    for name in names:
        selected = [
            r for r in case["region_metrics"] if r["scope"] == "full" and r["region"] == name
        ]
        md = next(r for r in selected if r["method"] == "MDCoreAlign_native60")
        lo = next(r for r in selected if r["method"] == "MDLovoFit")
        ticklabels.append(
            f"{LABELS.get(name, name)}\nMDCA {md['overlap']}/{md['region_size']}\nMDLovo {lo['overlap']}/{lo['region_size']}"
        )
    ax.boxplot(
        boxes,
        labels=ticklabels,
        widths=0.45,
        patch_artist=True,
        boxprops={"facecolor": "#DABFAA", "alpha": 0.6},
        medianprops={"color": "#7955A1"},
    )
    for i, name in enumerate(names, 1):
        mask = case["regions"][name]
        y = weights[mask]
        ax.scatter(
            np.linspace(i - 0.13, i + 0.13, len(y)),
            y,
            s=13,
            color=COLORS["MDCoreAlign"],
            alpha=0.65,
        )
    ax.set(ylabel="Fitting weights in experimentally annotated regions")
    label(ax, "C  Regional weights and report-set inclusion")
    geometry_panel(fig.add_subplot(gs[2, 1]), case, "D", literature=False)
    fig.suptitle(
        "I-FABP: experimental portal-region annotations and limited local sampling",
        fontweight="bold",
        fontsize=15,
    )
    n = case["summary"]["n_frames"]
    nt = len(case["summary"]["train_indices"])
    nv = len(case["summary"]["test_indices"])
    fig.text(
        0.5,
        0.94,
        f"A–C: full trajectory ({n} frames); inclusion counts are full-trajectory counts. D: first {nt} frames define sets → last {nv} frames test geometry.",
        ha="center",
        fontsize=9,
    )
    fig.text(
        0.5,
        0.025,
        "Rat I-FABP; local trajectory ≈500 ps (0.5 ns); original ligand state unresolved.\nThe apo NMR regions are dynamic annotations. Other residues are not experimentally certified stable.\nNMR mobility and Cα weights differ; this is exploratory. Four-residue regions have no eligible internal pairs\nafter excluding sequence separations ≤3, so their G is not evaluable (it is not zero).",
        ha="center",
        fontsize=9,
    )
    return save_figure(fig, figdir, "ifabp_supplement")


def rb69(run, case, figdir, render):
    fig = plt.figure(figsize=(14.4, 14))
    gs = fig.add_gridspec(4, 2, height_ratios=[2.2, 1.2, 0.9, 2.1], hspace=0.8, wspace=0.8)
    fig.subplots_adjust(bottom=0.18)
    names = list(case["regions"])
    for j, name in enumerate(names):
        structure_panel(
            fig.add_subplot(gs[0, j]),
            render,
            name,
            f"{'A' if j == 0 else 'B'}  {LABELS.get(name, name)}; shared residues in purple",
        )
    sequence_panel(fig.add_subplot(gs[1, :]), case, "C", framework=True)
    frequency_panel(fig.add_subplot(gs[2, :]), case, "D")
    ax = fig.add_subplot(gs[3, 0])
    metrics = case["region_metrics"]
    for j, method in enumerate(["MDCoreAlign_native60", "MDLovoFit"]):
        y = [
            float(
                next(
                    r
                    for r in metrics
                    if r["scope"] == "full" and r["region"] == n and r["method"] == method
                )["coverage"]
            )
            for n in names
        ]
        ax.bar(
            np.arange(len(names)) + (j - 0.5) * 0.28,
            y,
            width=0.28,
            color=COLORS[method],
            label=LABELS[method],
        )
    ax.set(
        xticks=range(len(names)),
        xticklabels=[LABELS.get(n, n) for n in names],
        ylim=(0, 1.1),
        ylabel="Fraction of each framework in report set",
    )
    ax.legend(frameon=False, fontsize=8)
    label(ax, "E  Both frameworks remain visible")
    geometry_panel(fig.add_subplot(gs[3, 1]), case, "F")
    fig.suptitle(
        "RB69 polymerase: two external structural frameworks, evaluated separately",
        fontweight="bold",
        fontsize=15,
    )
    fig.text(
        0.5,
        0.94,
        "A–E: full R1 trajectory. F: cores defined using R1 → geometry evaluated only on R2 (10,001 frames each).",
        ha="center",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.013,
        "Training: ATLAS R1 (100 ns). Geometry evaluation: R2 (100 ns), a separate simulation with the same construction.\nThe 231- and 318-residue frameworks share 29 residues and can move relative to one another. Their union is not a unique core.\nD222A/D327A construct; both full TPR files lack DNA, dCTP and Ca. These simulations do not test translocation or proofreading.",
        ha="center",
        fontsize=9,
    )
    return save_figure(fig, figdir, "3nci_frameworks")


CAPTIONS = {
    "ADK": """**adk_main. ADK stable-reference-frame case.** Panels A–B map the externally defined domains and production-mode continuous fitting weights onto the real first-frame protein structure, using one camera view. CORE, NMP and LID boundaries follow Li et al. (2015), Figure 1, and were frozen independently of algorithm output. Panel C shows full-trajectory fitting weights; dots mark the native 60% report set, which is a descriptive discretisation rather than the weights used internally for fitting. Panel D shows MDLovoFit's observed membership frequency. Panel E evaluates training-defined sets on the held-out time block. Each grey distribution contains 1,000 random sets of the same size as its own row; nearby same-chain pairs with sequence separation ≤3 are excluded. These draws are geometric controls, not biological replicates. Panel F reports RMSF about each method's own aligned temporal mean. All four fits use the same input, reference frame and residue ordering. CORE-only fitting is a knowledge-guided reference. [Li et al., 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4572606/).

**adk_motion. ADK domain movement without using an alignment residual as evidence.** Domain-centroid distances and the LID–CORE–NMP angle are calculated directly from internal coordinates. The lower panels use the same seven uniformly spaced frames, already chosen by the analysis script, displayed as real Cα traces after MDCoreAlign or literature CORE fitting. Rendering applies no further alignment. The domain angle is the explicitly defined three-centroid angle and is not asserted to equal any separately published catalytic-angle definition. Time follows the verified 240-ps stored-frame spacing.""",
    "I-FABP": """**ifabp_supplement. Limited I-FABP region correspondence.** The highlighted regions are the independently reported apo NMR dynamic regions 26–35, 53–56 and 73–76. All full-trajectory weights and MDLovoFit memberships are displayed, including high core inclusion where the external annotation reports dynamics. Region counts show membership divided by the number of residues in that region. The boxplots describe residue-level weights and are not uncertainty intervals; individual residues are shown. Other residues have no verified stable-region annotation. Held-out geometry uses the second half of this approximately 500-ps trajectory and cores estimated only from the first half. Grey distributions are 1,000 equal-size random sets. The local protein is rat I-FABP, but the ligand state of the original simulation remains unresolved. No residue-level experimental values were invented, and NMR order/relaxation measurements are not the same physical quantity as Cα alignment weights. [Hodsdon and Cistola, 1997](https://pubmed.ncbi.nlm.nih.gov/9047330/).""",
    "3nci_A": """**3nci_frameworks. Two separate RB69 polymerase frameworks.** Ren (2016), Figure 3 identifies a 231-residue palm-dominated framework and a 318-residue N-terminal/exonuclease framework. Their 29 shared residues are purple in both structural maps and remain represented in both sequence tracks. We retain both frameworks and do not define their union as a unique correct core. Maps use the real first-frame protein structure; weights and memberships use all of ATLAS R1. Core sets estimated from R1 are evaluated on R2, a separate 100-ns simulation of the same construct. Each set is accompanied by its own equal-size random reference, so different-sized literature sets do not define direct method wins/losses. Both complete production topologies contain the D222A/D327A protein, water and Na/Cl, but no DNA, dCTP or Ca. This figure supports only structural-region and sampled-geometry interpretation; it does not validate proofreading or DNA translocation. [Ren, 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC5009745/).""",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--case",
        choices=["ADK", "I-FABP", "3nci_A"],
        help="Optional single ready case; --run alone renders every completed case",
    )
    args = parser.parse_args()
    run = args.run.resolve()
    config = json.loads((run / "analysis_config.json").read_text())
    figdir = run / "figures"
    figdir.mkdir(exist_ok=True)
    manifest_path = figdir / "figure_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {"figures": {}, "pending_cases": []}
    )
    manifest["pending_cases"] = []
    for name in ["ADK", "I-FABP", "3nci_A"]:
        if not (run / "cases" / name / "analysis/case_summary.json").exists():
            manifest["pending_cases"].append(name)
            print(f"{name}: pending case_summary.json", flush=True)
            continue
        if args.case and name != args.case:
            continue
        case = get_case(run, config, name)
        outputs = []
        render = None
        if name != "I-FABP":
            render = render_structures(run, case, figdir)
        if name == "ADK":
            outputs += adk_main(run, case, figdir, render) + adk_motion(run, case, figdir, render)
        elif name == "I-FABP":
            outputs += ifabp(run, case, figdir)
        else:
            outputs += rb69(run, case, figdir, render)
        inputs = [
            case["directory"] / p
            for p in [
                "residue_results.tsv",
                "region_metrics.tsv",
                "geometry_metrics.tsv",
                "random_geometry.tsv",
                "alignment_profiles.npz",
                "case_summary.json",
            ]
        ]
        full_structure = run / "cases" / name / "reference_full_protein.pdb"
        if full_structure.exists():
            inputs.append(full_structure)
        inputs.extend(sorted(case["directory"].glob("superposition_*.pdb")))
        record = {
            "outputs": [{"path": p, "sha256": sha(p)} for p in outputs],
            "inputs": [{"path": str(p), "sha256": sha(p)} for p in inputs],
            "configuration_sha256": sha(run / "analysis_config.json"),
            "source_scope": "full trajectory descriptive profiles; frozen training sets for held-out geometry",
            "frame_selection": "unchanged indices from case_summary.json",
        }
        if render:
            record["structure_render"] = {
                k: v
                for k, v in render.items()
                if k
                in [
                    "completed",
                    "representation",
                    "script",
                    "payload",
                    "full_protein_residue_mapping_verified",
                ]
            }
        manifest["figures"][name] = record
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{name}: wrote {', '.join(Path(p).name for p in outputs)}", flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (figdir / "figure_captions.md").write_text(
        "\n\n".join(CAPTIONS[n] for n in ["ADK", "I-FABP", "3nci_A"] if n in manifest["figures"])
        + "\n"
    )


if __name__ == "__main__":
    main()
