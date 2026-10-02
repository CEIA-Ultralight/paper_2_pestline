"""Exporta config+summary dos runs pestline/fly-species para _scratch/fly_species_detail.json (só leitura)."""
import json, os, sys
import wandb

os.environ.setdefault("WANDB_DIR", "/raid/user_marcospaulo/.cache/wandb")
api = wandb.Api(timeout=60)
out = []
for r in api.runs("pestline/fly-species", per_page=200):
    cfg = {k: v for k, v in r.config.items() if not k.startswith("_")}
    summ = {}
    for k, v in r.summary.items():
        if k.startswith("_") and k not in ("_runtime", "_step"):
            continue
        if isinstance(v, (int, float, str, bool)) or v is None:
            summ[k] = v
    out.append({
        "name": r.name, "id": r.id, "state": r.state, "created": str(r.created_at),
        "job_type": r.job_type, "tags": r.tags, "notes": r.notes,
        "config": cfg, "summary": summ,
    })
json.dump(out, open(sys.argv[1], "w"), indent=1, default=str)
print(len(out), "runs")
