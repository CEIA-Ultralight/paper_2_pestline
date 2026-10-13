"""RQ1: diff entre versões do DS-F2 por conteúdo (md5 da imagem). Imagens originais (sem aug) de train/val/test;
para imagens comuns compara rótulos (n caixas, classes). Uso: python -I rq1_version_diff.py <ROOT_VERSOES> <V9_ROOT> <OUT_DIR>"""
import sys, csv, hashlib, pathlib, collections, re
root, v9, out = map(pathlib.Path, sys.argv[1:4])
vers = {d.name.replace("DS-F2_",""): d for d in sorted(root.glob("DS-F2_v*.x/DS-F2_v*"))}; vers["v9.0.5"] = v9
aug = re.compile(r"-instance|-[0-9a-f]{8}-[0-9a-f]{4}-")
def lab(p):
    if not p.exists(): return []
    return sorted((int(l.split()[0]),)+tuple(round(float(x),3) for x in l.split()[1:5]) for l in p.read_text().splitlines() if l.strip())
idx = {}
for v,d in vers.items():
    idx[v] = {}
    for s in ("train","val","test"):
        for im in (d/"images"/s).iterdir():
            if aug.search(im.stem): continue
            h = hashlib.md5(im.read_bytes()).hexdigest()
            idx[v][h] = (s, lab(d/"labels"/s/(im.stem+".txt")))
vl = list(vers); rows = []
for a,b in zip(vl, vl[1:]):
    A,B = idx[a],idx[b]; common = set(A)&set(B)
    same = sum(A[h][1]==B[h][1] for h in common)
    ndiff = sum(len(A[h][1])!=len(B[h][1]) for h in common)
    cdiff = sum(sorted(x[0] for x in A[h][1])!=sorted(x[0] for x in B[h][1]) for h in common)
    split_moved = sum(A[h][0]!=B[h][0] for h in common)
    rows.append([a,b,len(A),len(B),len(common),len(set(A)-set(B)),len(set(B)-set(A)),same,ndiff,cdiff,split_moved])
with open(out/"rq1_versions_diff.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["vA","vB","orig_A","orig_B","common","only_A","only_B","labels_identical","n_boxes_differ","class_multiset_differ","split_changed"]); w.writerows(rows)
# toda versão vs v9.0.5
rows=[]
for a in vl:
    A,B=idx[a],idx["v9.0.5"]; c=set(A)&set(B)
    rows.append([a,len(A),len(c),sum(A[h][1]==B[h][1] for h in c),sum(A[h][0]!=B[h][0] for h in c)])
with open(out/"rq1_versions_vs_v9.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["version","orig","in_v9","labels_identical_to_v9","split_differs_from_v9"]); w.writerows(rows)
