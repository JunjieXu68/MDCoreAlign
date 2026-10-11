#!/usr/bin/env python3
"""Plot frozen per-residue disagreements without changing scientific analyses.

All plotted sets retain their selection and evaluation sources. Structure and
rank panels use original full-trajectory sets; sensitivity uses training data.
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
from matplotlib.colors import ListedColormap, Normalize

PYMOL = Path("/home/jjxu/miniconda3/envs/pymol/bin/python")
GROUPS = ["shared", "MCA_only", "MLF_only", "neither"]
COLORS = {"shared": "#7B8D96", "MCA_only": "#D17A24", "MLF_only": "#247AB5", "neither": "#DDDEE0"}
LABELS = {"shared": "Shared", "MCA_only": "MCA only", "MLF_only": "MLF only", "neither": "Neither"}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "svg.fonttype": "none",
    }
)


def read_tsv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def truth(value):
    return value is True or str(value).lower() in {"1", "true"}


def number(value):
    return float(value) if value not in (None, "", "NA", "None", "nan") else np.nan


def atom_index(row):
    return int(row["atom_index"])


def residue_label(row):
    return f"{row['resname']}{row['source_resid']}{row.get('icode', '')}"


def title(ax, text):
    ax.set_title(text, loc="left", fontweight="bold", pad=10)


def fallback_full_rows(case):
    """Read preserved full-scope table for early structure rendering only."""
    path = Path(case["source_case_directory"]) / "residue_results.tsv"
    old = [r for r in read_tsv(path) if r["scope"] == "full"]
    weights = np.asarray([float(r["MDCoreAlign_weight"]) for r in old])
    k = sum(int(r["MDLovoFit_core"]) for r in old)
    order = np.lexsort((np.arange(len(old)), -weights))
    ranks = np.empty(len(old), int)
    ranks[order] = np.arange(1, len(old) + 1)
    rows = []
    for i, row in enumerate(old):
        mca, mlf = int(row["MDCoreAlign_matched_k"]), int(row["MDLovoFit_core"])
        assert mca == int(ranks[i] <= k)
        group = "shared" if mca and mlf else "MCA_only" if mca else "MLF_only" if mlf else "neither"
        rows.append(
            {
                **row,
                "case": case["case_id"],
                "group": group,
                "frozen_weight": weights[i],
                "rank": int(ranks[i]),
                "k": k,
                "rank_minus_k": int(ranks[i]) - k,
                "cutoff_weight": float(weights[order[k - 1]]),
                "atom_index": i,
            }
        )
    return rows


def full_rows(run, case):
    old = fallback_full_rows(case)
    path = run / "residue_membership.tsv"
    if not path.exists():
        return old
    rows = [r for r in read_tsv(path) if r["case"] == case["case_id"]]
    if case["case_id"] == "ADK":
        rows = [r for r in rows if r["context"] == "ADK_full_descriptive"]
    else:
        rows = [r for r in rows if r["context"] == "3nci_A_R1_to_R2"]
    rows = sorted(rows, key=atom_index)
    assert len(rows) == len(old)
    assert all(
        r["group"] == o["group"]
        and int(r["rank"]) == int(o["rank"])
        and float(r["frozen_weight"]) == float(o["frozen_weight"])
        for r, o in zip(rows, old, strict=True)
    )
    return rows


RENDER_SCRIPT = r"""
import json,sys
from pathlib import Path
import pymol
from pymol import cmd
pymol.finish_launching(['pymol','-cq'])
p=json.loads(Path(sys.argv[1]).read_text())
cmd.set('max_threads',1)
cmd.bg_color('white')
cmd.set('ray_shadows',0)
cmd.set('antialias',2)
cmd.set('ambient',.55)
cmd.set('specular',.15)
cmd.set('orthoscopic',1)
cmd.set('ray_opaque_background',1)
cmd.load(p['reference_full_protein'],'protein')
cmd.remove('not polymer.protein')
cmd.hide('everything')
cmd.show('cartoon','protein')
cmd.set('cartoon_transparency',.2)
cmd.set('cartoon_loop_radius',.18)
for name,color in p['colors'].items():cmd.set_color(name,[int(color[i:i+2],16)/255 for i in (1,3,5)])
cmd.color('neither','protein')
for r in p['residues']:
    selection='protein and resi '+str(r['source_resid'])+r.get('icode','')
    cmd.color(r['group'],selection)
    if r['group'] in ('MCA_only','MLF_only'):
        ca=selection+' and name CA'
        cmd.show('spheres',ca)
        cmd.set('sphere_scale',.50,ca)
        if p['case']=='ADK':cmd.label(ca,repr(r['label']))
        else:cmd.label(ca,repr(str(r['figure_index'])))
cmd.set('label_color','black')
cmd.set('label_outline_color','white')
cmd.set('label_size',14 if p['case']=='ADK' else 12)
cmd.set('label_position',(1.0,1.0,0))
cmd.orient('protein')
cmd.zoom('protein',5)
cmd.hide('labels')
cmd.png(p['output'],1600,1200,ray=1,quiet=1)
cmd.save(p['session'])
Path(p['render_audit']).write_text(json.dumps({'unique_CA_spheres':cmd.count_atoms('protein and name CA and (resi '+ '+'.join(str(r['source_resid']) for r in p['residues'] if r['group'] in ('MCA_only','MLF_only'))+')'),'view':list(cmd.get_view()),'field_of_view':float(cmd.get('field_of_view')),'image_width':1600,'image_height':1200,'pymol_version':list(cmd.get_version()),'frame_source':'unchanged historical first-frame full protein'},indent=2)+'\n')
cmd.quit()
"""


def render_structure(run, case, rows):
    figdir = run / "figures"
    scripts = figdir / "pymol"
    scripts.mkdir(parents=True, exist_ok=True)
    source = Path(case["reference_full_protein"])
    ca = [
        line
        for line in source.read_text().splitlines()
        if line.startswith("ATOM") and line[12:16].strip() == "CA"
    ]
    expected = [(int(r["source_resid"]), r.get("icode", ""), r["resname"]) for r in rows]
    actual = [(int(line[22:26]), line[26].strip(), line[17:20].strip()) for line in ca]
    assert actual == expected
    difference = [r for r in rows if r["group"] in {"MCA_only", "MLF_only"}]
    indices = {atom_index(r): i + 1 for i, r in enumerate(difference)}
    index_path = figdir / f"{case['case_id']}_disagreement_index.tsv"
    fields = [
        "figure_index",
        "case",
        "atom_index",
        "chain",
        "source_resid",
        "icode",
        "resname",
        "group",
        "rank",
        "k",
        "frozen_weight",
        "literature_regions",
        "context",
        "selection_source",
        "selection_frame_indices",
        "evaluation_source",
        "evaluation_frame_indices",
        "core_definition",
        "reference_source",
        "run_id",
    ]
    with index_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for r in difference:
            writer.writerow(
                {k: indices[atom_index(r)] if k == "figure_index" else r.get(k, "") for k in fields}
            )
    output = figdir / f"{case['case_id']}_structure_membership.png"
    audit = scripts / f"{case['case_id']}_render_audit.json"
    payload = {
        "render_script_sha256": hashlib.sha256(RENDER_SCRIPT.encode()).hexdigest(),
        "case": case["case_id"],
        "reference_full_protein": str(source),
        "colors": COLORS,
        "output": str(output),
        "session": str(scripts / f"{case['case_id']}_membership.pse"),
        "render_audit": str(audit),
        "residues": [
            {
                "source_resid": r["source_resid"],
                "icode": r.get("icode", ""),
                "group": r["group"],
                "label": residue_label(r),
                "figure_index": indices.get(atom_index(r)),
            }
            for r in rows
        ],
    }
    payload_file = scripts / f"{case['case_id']}_render.json"
    script = scripts / "render_disagreement_structures.py"
    old_payload = payload_file.read_text() if payload_file.exists() else None
    old_script = script.read_text() if script.exists() else None
    serialized = json.dumps(payload, indent=2) + "\n"
    payload_file.write_text(serialized)
    script.write_text(RENDER_SCRIPT)
    if not (
        old_payload == serialized
        and old_script == RENDER_SCRIPT
        and output.exists()
        and audit.exists()
    ):
        proc = subprocess.run(
            [str(PYMOL), str(script), str(payload_file)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        (scripts / f"{case['case_id']}.log").write_text(proc.stdout + "\n" + proc.stderr)
        if proc.returncode:
            raise RuntimeError(f"PyMOL rendering failed; see {scripts}")
    audit_data = json.loads(audit.read_text())
    assert audit_data["unique_CA_spheres"] == len(difference)
    v = np.asarray(audit_data["view"])
    rot = v[:9].reshape(3, 3)
    coordinates = np.asarray(
        [[float(line[a:b]) for a, b in [(30, 38), (38, 46), (46, 54)]] for line in ca]
    )
    # PyMOL stores its OpenGL rotation column-major. This orthoscopic projection
    # places labels on the actual rendered CAs, without moving atoms or the camera.
    eye = (coordinates - v[12:15]) @ rot + v[9:12]
    half_height = -v[11] * np.tan(np.radians(audit_data["field_of_view"] / 2))
    projected = np.column_stack(
        [800 + eye[:, 0] / half_height * 600, 600 - eye[:, 1] / half_height * 600]
    )
    result = {
        "image": output,
        "index_table": index_path,
        "payload": payload_file,
        "script": script,
        "audit": audit,
        "unique_count": len(difference),
        "source": source,
        "projected": projected,
    }
    fig, ax = plt.subplots(figsize=(10, 7.5))
    annotated_structure(ax, result, rows, case["case_id"], large=True)
    labelled = []
    for ext in ["png", "svg"]:
        path = figdir / f"{case['case_id']}_structure_membership_labeled.{ext}"
        fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.12)
        labelled.append(path)
    plt.close(fig)
    result["labelled_outputs"] = labelled
    return result


def annotated_structure(ax, render, rows, case_id, large=False):
    ax.imshow(plt.imread(render["image"]))
    different = [r for r in rows if r["group"] in {"MCA_only", "MLF_only"}]
    indices = {atom_index(r): i + 1 for i, r in enumerate(different)}
    for group, side in [("MCA_only", "left"), ("MLF_only", "right")]:
        subset = sorted(
            [r for r in different if r["group"] == group],
            key=lambda r: render["projected"][atom_index(r), 1],
        )
        if case_id == "ADK":
            py = np.asarray([render["projected"][atom_index(r), 1] for r in subset])
            yvalues = np.linspace(float(py.min()) - 110, float(py.max()) + 110, len(subset))
        else:
            yvalues = np.linspace(55, 1145, len(subset))
        for row, y in zip(subset, yvalues, strict=True):
            p = render["projected"][atom_index(row)]
            text = (
                residue_label(row)
                if case_id == "ADK"
                else f"{indices[atom_index(row)]}  {residue_label(row)}"
            )
            ax.annotate(
                text,
                xy=p,
                xytext=(80 if side == "left" else 1520, y),
                textcoords="data",
                ha="left" if side == "left" else "right",
                va="center",
                fontsize=(10 if case_id == "ADK" else 8)
                if large
                else (8 if case_id == "ADK" else 5.3),
                color=COLORS[group],
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.2},
                arrowprops={
                    "arrowstyle": "-",
                    "color": COLORS[group],
                    "lw": 0.6 if large else 0.35,
                    "alpha": 0.8,
                },
            )
    ax.set_xlim(0, 1600)
    ax.set_ylim(1200, 0)
    ax.axis("off")


def rank_curve(ax, rows, case_id):
    ordered = sorted(rows, key=lambda r: int(r["rank"]))
    x = np.asarray([int(r["rank"]) for r in ordered])
    w = np.asarray([float(r["frozen_weight"]) for r in ordered])
    k = int(rows[0]["k"])
    cut = float(rows[0]["cutoff_weight"])
    ax.plot(x, w, color="#8E969B", lw=1.5)
    for group in ["MCA_only", "MLF_only"]:
        d = [r for r in rows if r["group"] == group]
        ax.scatter(
            [int(r["rank"]) for r in d],
            [float(r["frozen_weight"]) for r in d],
            s=30,
            c=COLORS[group],
            edgecolor="white",
            lw=0.4,
            zorder=4,
            label=f"{LABELS[group]} (n={len(d)})",
        )
    ax.axvline(k, ls="--", lw=1, color="black", label=f"Main cutoff k={k}")
    ax.axhline(cut, ls=":", lw=0.8, color="#555555")
    ax.set(
        xlabel="MDCoreAlign rank (1 = highest actual frozen weight)",
        ylabel="Actual frozen fitting weight",
        xlim=(1, len(rows)),
    )
    ax.legend(frameon=False, loc="lower left" if case_id == "ADK" else "upper right")
    title(ax, "B  Original full-source ranks; every distinct member is marked")
    if case_id == "ADK":
        different = sorted(
            [r for r in rows if r["group"] in {"MCA_only", "MLF_only"}],
            key=lambda r: int(r["rank"]),
        )
        for i, r in enumerate(different):
            ax.annotate(
                residue_label(r),
                (int(r["rank"]), float(r["frozen_weight"])),
                xytext=(0.36 + (i % 4) * 0.16, 0.95 - (i // 4) * 0.12),
                textcoords="axes fraction",
                fontsize=8,
                color=COLORS[r["group"]],
                ha="center",
                arrowprops={"arrowstyle": "-", "lw": 0.5, "color": COLORS[r["group"]]},
            )
    ax.text(
        0,
        -0.24,
        f"Only the report set is truncated. Nonselected sites retain their actual weights.\nCutoff weight = {cut:.6f}; all residue identities and ranks appear below / in the index table.",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
    )


def geometry_panel(ax, geometry, context, title_text):
    selected = [r for r in geometry if r["context"] == context]
    for pos, group in enumerate(["shared", "MCA_only", "MLF_only"]):
        rows = [
            r for r in selected if r["group"] == group and np.isfinite(number(r.get("h_common_A")))
        ]
        values = np.asarray([number(r["h_common_A"]) for r in rows])
        if not len(values):
            continue
        jitter = np.linspace(-0.19, 0.19, len(values))
        # A deterministic spread exposes individual observations, not a random resample.
        ax.scatter(
            pos + jitter,
            values,
            s=10 if group == "shared" else 24,
            color=COLORS[group],
            alpha=0.60 if group == "shared" else 0.95,
            lw=0,
            zorder=3,
        )
        ax.hlines(np.median(values), pos - 0.27, pos + 0.27, lw=2, color="#303B43", zorder=4)
        ax.text(
            pos,
            1.015,
            f"n={len(values)}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            fontsize=8,
        )
    ax.set(
        xticks=range(3),
        xticklabels=[LABELS[g] for g in ["shared", "MCA_only", "MLF_only"]],
        ylabel="hᵢ(common core) (Å)",
    )
    ax.grid(axis="y", alpha=0.15)
    title(ax, title_text)


def tracked_rows(full, sensitivity):
    different = [r for r in full if r["group"] in {"MCA_only", "MLF_only"}]
    old = set(atom_index(r) for r in different)
    baseline = {atom_index(r): r for r in sensitivity}
    additional = [
        i
        for i, r in baseline.items()
        if truth(r.get("tracked_old_train_disagreement")) and i not in old
    ]
    indexed = {atom_index(r): r for r in full}
    return different + [indexed[i] for i in sorted(additional)]


def detailed_panel(fig, spec, full, sensitivity, cutoffs, case_id):
    """Every full-source difference, plus all training-only differences, gets a row."""
    tracked = tracked_rows(full, sensitivity)
    n = len(tracked)
    k = int(full[0]["k"])
    seeds = sorted({int(r["seed"]) for r in sensitivity})
    data = {(atom_index(r), int(r["seed"])): r for r in sensitivity}
    gs = spec.subgridspec(1, 3, width_ratios=[1.6, 1.05, 1.05], wspace=0.3)
    a = fig.add_subplot(gs[0, 0])
    b = fig.add_subplot(gs[0, 1])
    c = fig.add_subplot(gs[0, 2])
    full_unique = [r for r in full if r["group"] in {"MCA_only", "MLF_only"}]
    figure_ids = {atom_index(r): i + 1 for i, r in enumerate(full_unique)}
    yy = np.arange(n)
    labels = []
    for r in tracked:
        prefix = str(figure_ids[atom_index(r)]) if atom_index(r) in figure_ids else "T"
        labels.append(f"{prefix:>2}  {residue_label(r)}")
    for i, r in enumerate(tracked):
        rank = int(r["rank"])
        color = COLORS[r["group"]]
        a.plot([k, rank], [i, i], color=color, lw=1)
        a.scatter(
            rank,
            i,
            s=25,
            color=color,
            edgecolor="#555555" if r["group"] in {"shared", "neither"} else "none",
            lw=0.4,
        )
    a.axvline(k, color="black", ls="--", lw=0.8)
    a.set(yticks=yy, yticklabels=labels, xlabel="Full-source rank (dashed = k)")
    a.tick_params(axis="y", labelsize=8 if n < 20 else 6.7)
    a.set_ylim(n - 0.5, -0.5)
    original_count = len(full_unique)
    additional_count = len(tracked) - original_count
    title(
        a,
        f"E  All {original_count} original differences: exact ranks"
        + ("; T = training only" if additional_count else ""),
    )
    if not seeds:
        b.text(
            0.5,
            0.5,
            "Representative-frame sensitivity unavailable",
            ha="center",
            va="center",
            transform=b.transAxes,
        )
        b.axis("off")
        c.axis("off")
        return
    ranks = np.array(
        [[int(data[(atom_index(r), seed)]["rank"]) for seed in seeds] for r in tracked]
    )
    included = np.array(
        [[truth(data[(atom_index(r), seed)]["included"]) for seed in seeds] for r in tracked]
    )
    offsets = ranks - k
    limit = max(1, int(np.max(np.abs(offsets))))
    b.imshow(offsets, aspect="auto", cmap="coolwarm", norm=Normalize(-limit, limit))
    for i in range(n):
        for j in range(len(seeds)):
            b.text(
                j,
                i,
                str(ranks[i, j]),
                ha="center",
                va="center",
                fontsize=7 if n < 20 else 5.8,
                color="black" if abs(offsets[i, j]) < 0.65 * limit else "white",
                fontweight="bold" if included[i, j] else "normal",
            )
    b.set(
        xticks=range(len(seeds)),
        xticklabels=[str(s) for s in seeds],
        yticks=[],
        xlabel="Training-source seed; bold = included",
    )
    b.tick_params(axis="x", labelrotation=45, labelsize=7)
    title(b, "F  Five training runs: actual ranks")
    # Report cutoffs apply to original full frozen weights, independently of the
    # five training-source estimations. Both operations are labelled separately.
    context = "ADK_full_descriptive" if case_id == "ADK" else "3nci_A_R1_to_R2"
    selected = [r for r in cutoffs if r.get("context") == context]
    newks = sorted({int(r["new_k"]) for r in selected})
    cutmap = {(atom_index(r), int(r["new_k"])): truth(r["included_at_new_k"]) for r in selected}
    matrix = np.asarray([[int(r["rank"]) <= newk for newk in newks] for r in tracked], dtype=int)
    for r in tracked:
        for newk in newks:
            if (atom_index(r), newk) in cutmap:
                assert cutmap[(atom_index(r), newk)] == (int(r["rank"]) <= newk)
    c.imshow(matrix, aspect="auto", cmap=ListedColormap(["#F1F1F1", "#486B8C"]), vmin=0, vmax=1)
    for i in range(n):
        for j in range(len(newks)):
            c.text(
                j,
                i,
                "●" if matrix[i, j] else "–",
                ha="center",
                va="center",
                fontsize=7 if n < 20 else 5.8,
                color="white" if matrix[i, j] else "#8B8B8B",
            )
        c.text(
            len(newks) - 0.25,
            i,
            f" {int(included[i].sum())}/5",
            ha="left",
            va="center",
            fontsize=7 if n < 20 else 5.8,
        )
    c.set(
        xticks=range(len(newks)),
        xticklabels=[str(v) for v in newks],
        yticks=[],
        xlabel="Full-source report k (weights unchanged)",
    )
    c.tick_params(axis="x", labelsize=8)
    title(c, "G  Cutoff inclusion | five-run count")


def save(fig, run, stem):
    paths = []
    for ext in ["png", "svg"]:
        path = run / "figures" / f"{stem}.{ext}"
        fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.14)
        paths.append(path)
    plt.close(fig)
    return paths


def combination(run, case, full, render, geometry, sensitivity, cutoffs):
    adk = case["case_id"] == "ADK"
    ntracked = len(tracked_rows(full, sensitivity))
    bottom = max(3.0, ntracked * 0.135)
    fig = plt.figure(figsize=(16, 7.8 + bottom))
    gs = fig.add_gridspec(3, 2, height_ratios=[3.1, 2.1, bottom], hspace=0.64, wspace=0.35)
    fig.subplots_adjust(left=0.12, right=0.91, bottom=0.19, top=0.90)
    a = fig.add_subplot(gs[0, 0])
    annotated_structure(a, render, full, case["case_id"])
    title(a, "A  Original full-source membership on the real first frame")
    counts = {g: sum(r["group"] == g for r in full) for g in GROUPS}
    a.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=COLORS[g],
                label=f"{LABELS[g]} ({counts[g]})",
            )
            for g in GROUPS
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.035),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    rank_curve(fig.add_subplot(gs[0, 1]), full, case["case_id"])
    if adk:
        geometry_panel(
            fig.add_subplot(gs[1, 0]),
            geometry,
            "ADK_full_descriptive",
            "C  Full-trajectory groups and geometry: descriptive",
        )
        geometry_panel(
            fig.add_subplot(gs[1, 1]),
            geometry,
            "ADK_train_heldout",
            "D  Training-defined groups, held-out geometry",
        )
    else:
        geometry_panel(
            fig.add_subplot(gs[1, 0]),
            geometry,
            "3nci_A_R1_to_R2",
            "C  R1-defined groups, R2 geometry",
        )
        ax = fig.add_subplot(gs[1, 1])
        selected = [
            r
            for r in geometry
            if r["context"] == "3nci_A_R1_to_R2" and r["group"] in {"MCA_only", "MLF_only"}
        ]
        for group in ["MCA_only", "MLF_only"]:
            rows = [r for r in selected if r["group"] == group]
            ax.scatter(
                [number(r["h_framework_A_A"]) for r in rows],
                [number(r["h_framework_B_A"]) for r in rows],
                color=COLORS[group],
                s=24,
                label=LABELS[group],
                alpha=0.8,
            )
        ax.set(xlabel="hᵢ(framework A) on R2 (Å)", ylabel="hᵢ(framework B) on R2 (Å)")
        ax.legend(frameon=False)
        title(ax, "D  Both external frameworks retained")
    detailed_panel(fig, gs[2, :], full, sensitivity, cutoffs, case["case_id"])
    case_label = "ADK" if adk else "RB69 polymerase (3nci_A)"
    fig.suptitle(
        f"{case_label}: who differs, where they rank, and how sensitive the report is",
        fontweight="bold",
        fontsize=15,
        y=0.985,
    )
    source = "full 4,187-frame ADK trajectory" if adk else "full 10,001-frame R1 trajectory"
    evaluation = "first 2,093 frames → last 2,094 frames" if adk else "R1 → separate R2 simulation"
    geometry_letters = "D" if adk else "C–D"
    fig.text(
        0.5,
        0.956,
        f"A, B, E, G: original {source}. {geometry_letters} / held-out geometry: {evaluation}.",
        ha="center",
        fontsize=9,
    )
    fig.text(
        0.5,
        0.935,
        "F and five-run counts: training-source representative-frame perturbations; all five seeds retained. These are not biological replicates.",
        ha="center",
        fontsize=9,
    )
    fig.text(
        0.5,
        0.02,
        "Rank panels list every original distinct member. T marks additional training-set differences; these may be shared/neither in the full-source comparison.\nGrey/white structure membership is not a biological truth label. hᵢ measures distance stability relative to a fixed reference set, not local disorder.\nChanging report k does not change any frozen fitting weight. The 5-run count is an empirical inclusion count, not a calibrated probability.",
        ha="center",
        fontsize=8.5,
    )
    stem = "adk_core_disagreement" if adk else "3nci_core_disagreement"
    return save(fig, run, stem)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--structures-only", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    config = json.loads((run / "analysis_config.json").read_text())
    (run / "figures").mkdir(exist_ok=True)
    manifest = {
        "run_id": config["run_id"],
        "configuration_sha256": sha(run / "analysis_config.json"),
        "plot_script": {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)},
        "figures": {},
    }
    required = [
        "residue_membership.tsv",
        "geometry_by_residue.tsv",
        "cutoff_sensitivity.tsv",
        "sensitivity_by_residue.tsv",
    ]
    tables = {name: read_tsv(run / name) for name in required if (run / name).exists()}
    for case in config["cases"]:
        rows = full_rows(run, case)
        render = render_structure(run, case, rows)
        print(
            case["case_id"],
            "structure ready; all distinct sites:",
            render["unique_count"],
            flush=True,
        )
        if args.structures_only:
            continue
        missing = [name for name in required if name not in tables]
        if missing:
            raise FileNotFoundError("Analysis tables not ready: " + ", ".join(missing))
        geom = [r for r in tables["geometry_by_residue.tsv"] if r["case"] == case["case_id"]]
        sensitivity = [
            r for r in tables["sensitivity_by_residue.tsv"] if r["case"] == case["case_id"]
        ]
        cutoff = [r for r in tables["cutoff_sensitivity.tsv"] if r["case"] == case["case_id"]]
        outputs = (
            combination(run, case, rows, render, geom, sensitivity, cutoff)
            + render["labelled_outputs"]
        )
        inputs = [run / name for name in required] + [
            Path(case["reference_full_protein"]),
            Path(case["source_case_directory"]) / "residue_results.tsv",
        ]
        manifest["figures"][case["case_id"]] = {
            "outputs": [{"path": str(p), "sha256": sha(p)} for p in outputs],
            "inputs": [{"path": str(p), "sha256": sha(p)} for p in inputs],
            "all_full_source_distinct_residues_displayed": render["unique_count"],
            "index_table": str(render["index_table"]),
            "index_table_sha256": sha(render["index_table"]),
            "render_script": str(render["script"]),
            "render_script_sha256": sha(render["script"]),
            "render_payload_sha256": sha(render["payload"]),
            "render_audit": {"path": str(render["audit"]), "sha256": sha(render["audit"])},
            "structure_source": "unchanged historical first-frame full protein; all distinct CAs rendered and labelled",
            "scope": "original full-source membership/weight ranks and cutoff changes; held-out geometry only for training-defined groups; five-seed estimates on training sources",
        }
        print(case["case_id"], "combo ready", flush=True)
    if not args.structures_only:
        (run / "figures/figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (run / "figures/figure_captions.md").write_text(
            "**ADK core disagreement.** Structure/rank panels identify every original full-trajectory disagreement, while the two geometry panels explicitly distinguish full-source descriptive measurements from training-defined groups evaluated on the held-out half. All full-source unique residues and any additional training-only disagreements have individually labelled rows. Five-run ranks and inclusion counts use only training-source representative-frame perturbations; these do not repeat the full-trajectory analysis. Report-cutoff membership changes use unchanged full-source weights.\n\n"
            "**3nci_A core disagreement.** All 52 original distinct members are shown as coloured Cα spheres and indexed; the complete residue index accompanies the figure. R1 defines membership, and R2 supplies distance variances. The framework scatter retains both independently annotated frameworks. Rows identify every distinct residue, its original rank, its five training-source ranks and inclusion count, and its inclusion under each predefined report cutoff. The geometry and sensitivity are descriptive; neither method supplies biological truth labels.\n\n"
            "**Scope of sensitivity rows.** The displayed rows track original method disagreements (and any additional training-only disagreements), not every residue whose membership changes across seeds. The full-sequence sensitivity_by_residue.tsv retains every residue and every seed, including variable sites outside the original disagreements. Bold ranks indicate inclusion in the training-source matched-k report set. The right-hand inclusion count uses those same five training runs; the adjacent cutoff grid instead uses unchanged original full-source weights.\n"
        )


if __name__ == "__main__":
    main()
