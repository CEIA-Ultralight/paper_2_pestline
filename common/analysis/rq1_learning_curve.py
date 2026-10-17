#!/usr/bin/env python3
"""RQ1 B4 — curva de aprendizado acc(N) = a - b*N^-c nos degraus L0..L3 (TEST, media de 3 seeds, TEST fixo v9.0.5).
L2ORG (composicao diferente, 1497 imgs) fica fora do ajuste. 4 pontos / 3 parametros: ajuste exploratorio; so descreve
a tendencia (extrapolacao nao e evidencia). Uso: python rq1_learning_curve.py <ablation_ladder.csv> <OUT.csv>"""
import csv, sys
from collections import defaultdict
import numpy as np

N = {"L0": 103, "L1": 244, "L2": 869, "L3": 1738}  # imagens de treino (contagem em datasets/rq1_ladder/*/images/train)
f = lambda n, a, b, c: a - b * n ** (-c)  # noqa: E731

acc = defaultdict(list)
for r in csv.DictReader(open(sys.argv[1])):
    if r["arm"] in N:
        acc[r["arm"]].append(float(r["acc_especie"]))
arms = list(N)
x = np.array([N[a] for a in arms], float)
y = np.array([np.mean(acc[a]) for a in arms])
best = None  # grade em c; para cada c, (a, b) por minimos quadrados lineares (b>=0, a<=1)
for c in np.arange(0.05, 3.0, 0.005):
    A = np.stack([np.ones_like(x), -x ** (-c)], 1)
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    if b < 0 or a > 1.0:
        continue
    sse = float(((A @ [a, b] - y) ** 2).sum())
    if best is None or sse < best[0]:
        best = (sse, a, b, c)
p = best[1:]
rows = [["split", "n_imgs", "acc_obs_media", "acc_ajustada"]] + [["TEST", int(n), f"{o:.4f}", f"{f(n, *p):.4f}"] for n, o in zip(x, y)]
rows += [["ajuste", "a(assintota)", f"{p[0]:.4f}", ""], ["ajuste", "b", f"{p[1]:.4f}", ""], ["ajuste", "c", f"{p[2]:.4f}", ""],
         ["ganho_previsto_2x", "N=3476", f"{f(3476, *p) - f(1738, *p):+.4f}", "exploratorio"]]
csv.writer(open(sys.argv[2], "w", newline="")).writerows(rows)
print(rows)
