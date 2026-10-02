"""Inventário dos runs W&B da entity pestline (CPU, read-only). Saída: _scratch/wandb_inventory.csv"""
import os, sys, csv
import wandb

api = wandb.Api(timeout=60)
entity = os.environ.get("WANDB_ENTITY", "pestline")
rows = []
projects = list(api.projects(entity))
print(f"{len(projects)} projetos em {entity}:")
for p in projects:
    runs = list(api.runs(f"{entity}/{p.name}", per_page=200))
    print(f"  {p.name}: {len(runs)} runs")
    for r in runs:
        s = r.summary._json_dict if hasattr(r.summary, "_json_dict") else dict(r.summary)
        c = r.config
        rows.append({
            "project": p.name,
            "name": r.name,
            "id": r.id,
            "state": r.state,
            "created": str(r.created_at)[:16],
            "tags": ";".join(r.tags),
            "job_type": r.job_type or "",
            "seed": c.get("seed", ""),
            "model": c.get("model", ""),
            "imgsz": c.get("imgsz", ""),
            "epochs": c.get("epochs", ""),
            "epoch_done": s.get("epoch", ""),
            "mAP50": s.get("metrics/mAP50(B)", ""),
            "mAP50-95": s.get("metrics/mAP50-95(B)", ""),
            "precision": s.get("metrics/precision(B)", ""),
            "recall": s.get("metrics/recall(B)", ""),
            "runtime_s": s.get("_runtime", ""),
            "notes": (r.notes or "").replace("\n", " ")[:120],
            "url": r.url,
        })
out = sys.argv[1] if len(sys.argv) > 1 else "_scratch/wandb_inventory.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"{len(rows)} runs → {out}")
