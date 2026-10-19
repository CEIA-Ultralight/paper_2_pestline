"""Controle de mesmo tamanho da escada RQ1: acuracia de especie (TEST) das subamostras EQ<N> do L3 vs. L1/L2 (ablation_ladder_summary.csv).
Le <RUNS>/EQ<N>_s<seed>/test_eval/summary.json; escreve rq1_data/results/equalsize_summary.csv.
Uso: python rq1_equalsize.py [RUNS_ROOT]"""
import json, sys, pathlib, statistics as st, csv
runs = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path.home()/'pestline_rq3/runs')
out = pathlib.Path(__file__).resolve().parents[2]/'rq1_data/results/equalsize_summary.csv'
rows = []
prior = {}   # execucoes anteriores (nó ovx, L40S), valores rastreados em PAPER_FACTS §11b
pf = pathlib.Path(__file__).resolve().parents[2]/'rq1_data/results/equalsize_prior_ovx.csv'
if pf.exists():
    for r in csv.DictReader(open(pf)): prior[(r['arm'], int(r['seed']))] = float(r['species_acc'])
for n in (103, 244, 869):
    accs = []
    for s in (0, 1, 2):
        f = runs/f'EQ{n}_s{s}/test_eval/summary.json'
        if f.exists():
            d = json.load(open(f)); k = next(iter(d)); accs.append((s, d[k]['single/species_accuracy_all_gt'], 'h100'))
        elif (f'EQ{n}', s) in prior: accs.append((s, prior[(f'EQ{n}', s)], 'ovx'))
    if accs:
        v = [a for _, a, _ in accs]
        rows.append([f'EQ{n}', n, len(v), round(st.mean(v), 4), round(st.stdev(v), 4) if len(v) > 1 else '', '|'.join(f's{s}={a:.4f}({h})' for s, a, h in accs), 'TEST', 'multi-seed' if len(v) == 3 else 'parcial'])
with open(out, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['arm', 'n_train', 'n_seeds', 'acc_mean', 'acc_std', 'per_seed', 'split', 'seeds']); w.writerows(rows)
for r in rows: print(r)
