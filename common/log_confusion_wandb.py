#!/usr/bin/env python3
"""Gera matriz de confusão no TEST para cada run do classificador e loga na run
W&B correspondente (resume=must, id=<run_id>), para comparação lado a lado.

Uso:
  python log_confusion_wandb.py --crops /raid/.../DS-F2_crops_pad75
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).parent))
from ensemble_tta import build_model, infer_model_name

# run_dir -> (wandb_id, model_dir_name)
RUNS = {
    "E3_swin":        "ettsirmw",
    "E3b_convnext":   "l2ys9lsg",
    "E3c_focal":      "453bepak",
    "E3d_convnext_s": "iegv21jl",
    "E4_convnext_sam":"wrjfm0w0",
    "E5_convnext_512":"skfyw3du",
}


def confusion_for(run_dir: Path, crops: Path, device):
    ck_path = run_dir / "best.pt"
    if not ck_path.is_file():
        return None
    ck = torch.load(ck_path, map_location="cpu")
    classes = ck["classes"]
    img_sz = int(ck.get("img_size", 384))
    name = infer_model_name(str(ck_path), ck)
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tf = transforms.Compose([transforms.Resize((img_sz, img_sz)), transforms.ToTensor(), norm])
    ds = datasets.ImageFolder(crops / "test", tf)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=8)
    m = build_model(name, len(classes)).to(device).eval()
    m.load_state_dict(ck["model"])
    yt, yp = [], []
    with torch.no_grad():
        for x, y in dl:
            p = m(x.to(device)).argmax(1).cpu()
            yt.extend(y.tolist()); yp.extend(p.tolist())
    return np.array(yt), np.array(yp), classes, name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", required=True)
    ap.add_argument("--runs-root", default="/raid/user_marcospaulo/flydet_runs")
    ap.add_argument("--project", default="fly-species")
    ap.add_argument("--entity", default="pestline")
    args = ap.parse_args()

    if not os.environ.get("WANDB_API_KEY"):
        env = Path("/raid/user_marcospaulo/secrets/wandb.env")
        for line in env.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1); os.environ.setdefault(k, v)

    import wandb
    device = "cuda" if torch.cuda.is_available() else "cpu"
    crops = Path(args.crops)

    for run_name, wb_id in RUNS.items():
        run_dir = Path(args.runs_root) / run_name
        try:
            res = confusion_for(run_dir, crops, device)
        except Exception as e:
            print(f"[{run_name}] ERRO: {e}")
            continue
        if res is None:
            print(f"[{run_name}] sem best.pt, pulando")
            continue
        yt, yp, classes, arch = res
        acc = float((yt == yp).mean())
        # loga na run existente
        with wandb.init(project=args.project, entity=args.entity, id=wb_id,
                        resume="allow", name=run_name) as run:
            cm = wandb.plot.confusion_matrix(
                preds=yp.tolist(), y_true=yt.tolist(), class_names=list(classes))
            run.log({"test/confusion_matrix": cm, "test/acc_recomputed": acc})
        print(f"[{run_name}] ({arch}) acc={acc:.4f} -> W&B run {wb_id}")


if __name__ == "__main__":
    main()
