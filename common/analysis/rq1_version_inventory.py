"""RQ1: inventário das versões do DS-F2 (v4..v9.0.5): imagens/instâncias por split x origem x aumento x classe.
Origem = prefixo do nome (DS_F2-v3, DS-F2_v5.1, DS-ORG, DS-F2_v6.0, v7, v8, v9). Aumento = sufixo -<uuid> (aug de imagem)
e -instance-<uuid> (aug de instância). Uso: python -I rq1_version_inventory.py <ROOT_VERSOES> <V9_ROOT> <OUT_DIR>"""
import re, sys, csv, collections, pathlib
root, v9, out = map(pathlib.Path, sys.argv[1:4])
UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
def parse(stem):
    aug = "instance" if "-instance" in stem else ("image" if re.search(r"-" + UUID, stem) else "none")
    base = re.sub(r"-instance.*$", "", stem)
    base = re.split(r"-" + UUID, base)[0]
    origin = re.sub(r"_?\d{6}$", "", base)
    return origin, aug, base
vers = {}
for d in sorted(root.glob("DS-F2_v*.x/DS-F2_v*")):
    vers[d.name.replace("DS-F2_", "")] = d
vers["v9.0.5"] = v9
names = ["MD","MV","MC","MF","INS","NOISE","MAR"]
rows_split, rows_origin, rows_cls, bases = [], [], [], {}
for v, d in vers.items():
    bases[v] = {}
    for s in ("train","val","test"):
        imgs = sorted((d/"images"/s).iterdir()); n_inst = 0; n_empty = 0
        co = collections.Counter(); cc = collections.Counter(); ic = collections.Counter()
        for im in imgs:
            o, a, b = parse(im.stem); bases[v].setdefault(s, set()).add(b)
            lb = d/"labels"/s/(im.stem+".txt"); ls = []
            if lb.exists():
                ls = [l.split() for l in lb.read_text().splitlines() if l.strip()]
            if not ls: n_empty += 1
            n_inst += len(ls); co[(o,a)] += 1
            for l in ls:
                c = int(l[0]); cc[(a,c)] += 1; ic[c] += 1
        rows_split.append([v,s,len(imgs),n_inst,n_empty])
        for (o,a),n in sorted(co.items()): rows_origin.append([v,s,o,a,n])
        for c in range(max(7, max(ic) + 1 if ic else 0)):
            rows_cls.append([v,s,c,names[c] if c < 7 else "?",ic[c],cc[("none",c)],cc[("image",c)],cc[("instance",c)]])
def w(f,h,r):
    with open(out/f,"w",newline="") as fh: cw=csv.writer(fh); cw.writerow(h); cw.writerows(r)
w("rq1_versions_splits.csv",["version","split","images","instances","empty_images"],rows_split)
w("rq1_versions_origin.csv",["version","split","origin","aug","images"],rows_origin)
w("rq1_versions_classes.csv",["version","split","cls_id","cls","inst_total","inst_orig","inst_aug_image","inst_aug_instance"],rows_cls)
# sobreposição de imagens-base entre versões (por split)
vl = list(vers); ov = []
for s in ("train","val","test"):
    for a in vl:
        for b in vl:
            A, B = bases[a].get(s,set()), bases[b].get(s,set())
            ov.append([s,a,b,len(A),len(B),len(A&B)])
w("rq1_versions_overlap.csv",["split","vA","vB","baseA","baseB","shared_base"],ov)
# vazamento: base de test de X presente em train de X ou no train de v9
lk = []
for v in vl:
    lk.append([v,"test_in_own_train",len(bases[v]["test"]&bases[v]["train"])])
    lk.append([v,"val_in_own_train",len(bases[v]["val"]&bases[v]["train"])])
    lk.append([v,"test_in_v9_train",len(bases[v]["test"]&bases["v9.0.5"]["train"])])
w("rq1_versions_leakage.csv",["version","check","n_base"],lk)

# sobreposição por CONTEÚDO (md5 das imagens): test/val de cada versão vs train/test das demais
import hashlib
def md5(p): return hashlib.md5(p.read_bytes()).hexdigest()
H = {v:{s:{md5(p) for p in (d/"images"/s).iterdir()} for s in ("train","val","test")} for v,d in vers.items()}
rows=[]
for a in vl:
    for b in vl:
        rows.append([a,b,len(H[a]["test"]),len(H[a]["test"]&H[b]["test"]),len(H[a]["test"]&H[b]["train"]),len(H[a]["test"]&H[b]["val"])])
w("rq1_versions_test_content_overlap.csv",["vA_test","vB","nA_test","in_B_test","in_B_train","in_B_val"],rows)
