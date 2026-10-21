#!/usr/bin/env python3
"""RQ1 — figura de confusao entre localizados, L0 vs L3 (le ladder_confusion.csv). Uso: rq1_confusion_fig.py <csv> <saida.pdf>"""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SP, COLS = ["MD", "MV", "MC", "MF"], ["MD", "MV", "MC", "MF", "outra"]
v = {(r["arm"], r["gt"], r["pred"]): float(r["frac_dos_localizados"] or 0) for r in csv.DictReader(open(sys.argv[1]))}
fig, axs = plt.subplots(1, 2, figsize=(3.35, 1.75), sharey=True)
for ax, arm, name in zip(axs, ["L0", "L3"], ["L0", "L3 (full)"]):
    m = [[100 * v[(arm, g, p)] for p in COLS] for g in SP]
    im = ax.imshow(m, vmin=0, vmax=100, cmap="Blues")
    ax.set_xticks(range(5), ["MD", "MV", "MC", "MF", "non-\ntarget"], fontsize=7)
    ax.set_yticks(range(4), SP, fontsize=7)
    ax.set_title(name, fontsize=8)
    ax.set_xlabel("predicted", fontsize=7)
    for i in range(4):
        for j in range(5):
            ax.text(j, i, f"{m[i][j]:.0f}", ha="center", va="center", fontsize=7, color="white" if m[i][j] > 55 else "black")
axs[0].set_ylabel("annotated", fontsize=7)
fig.savefig(sys.argv[2], bbox_inches="tight", pad_inches=0.02)
