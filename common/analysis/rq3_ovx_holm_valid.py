#!/usr/bin/env python3
"""RQ3 campanha (a) (L40S): Holm recalculado so com as comparacoes nao afetadas pela unidade da loss (CPU).

Le rq3_tiny_object/results/v9_ovx_paired.csv (p bootstrap bilateral por braco vs E0, rq3_v9_analysis.py) e reaplica Holm
excluindo os bracos cuja loss usou NWD/GCD em unidades de grid (PAPER_FACTS §12.1): E14_nwd_rfla, E15_nwd_loss (NWD saturado)
e E11_gcd (GCD com clamp de 1 celula = 8/16/32 px; refeito em pixels como H7 na cadeia H). Validos: P2, WIoU (invariante a
escala), 2560, NWD so no assigner (recebe pixels) e YOLO11m.
Uso: python3 rq3_ovx_holm_valid.py [RESULTS_DIR]   Saida: v9_ovx_paired_valid.csv
"""
import csv
import os
import sys

RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                                         "rq3_tiny_object", "results")
EXCLUDED = {"E14_nwd_rfla", "E15_nwd_loss", "E11_gcd"}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, run_max = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run_max = max(run_max, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = run_max
    return out


def main():
    with open(os.path.join(RES, "v9_ovx_paired.csv")) as f:
        rows = [r for r in csv.DictReader(f) if r["arm_a"] not in EXCLUDED]
    adj = holm([float(r["p_boot"]) for r in rows])
    cols = ["split", "arm_a", "arm_b", "n_seeds", "delta_acc_a_menos_b", "ic95_lo", "ic95_hi", "p_boot"]
    with open(os.path.join(RES, "v9_ovx_paired_valid.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols + ["p_holm_validos", "m_holm", "p_holm_antigo_m8", "delta_por_seed"])
        for r, p in zip(rows, adj):
            w.writerow([r[c] for c in cols] + [f"{p:.4f}", len(rows), r["p_holm"], r["delta_por_seed"]])
            print(r["arm_a"], r["delta_acc_a_menos_b"], r["p_boot"], f"holm_validos={p:.4f}", f"(antigo {r['p_holm']})")


if __name__ == "__main__":
    main()
