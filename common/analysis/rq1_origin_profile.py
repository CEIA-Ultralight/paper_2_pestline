"""RQ1: perfil por origem (lote de coleta) dentro de uma versão: imgs, inst/img, mix de classes, √wh médio (px, 1920x1080 assumido).
Uso: python -I rq1_origin_profile.py <VERSION_DIR> <OUT_CSV>"""
import sys, re, csv, pathlib, collections, math
d, out = pathlib.Path(sys.argv[1]), sys.argv[2]
aug = re.compile(r"-instance|-[0-9a-f]{8}-[0-9a-f]{4}-")
acc = collections.defaultdict(lambda: [0,0,[0]*8,0.0])
for s in ("train","val","test"):
    for lb in (d/"labels"/s).glob("*.txt"):
        if aug.search(lb.stem): continue
        o = re.sub(r"_?\d{6}$","",lb.stem); a = acc[(s,o)]; a[0]+=1
        for l in lb.read_text().splitlines():
            p=l.split()
            if len(p)<5: continue
            a[1]+=1; a[2][int(p[0])]+=1; a[3]+=math.sqrt(float(p[3])*1920*float(p[4])*1080)
with open(out,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["split","origin","orig_images","instances","inst_per_img","mean_sqrt_wh_px","MD","MV","MC","MF","INS","NOISE","MAR"])
    for (s,o),a in sorted(acc.items()):
        w.writerow([s,o,a[0],a[1],round(a[1]/a[0],1),round(a[3]/max(a[1],1),1)]+a[2][:7])
