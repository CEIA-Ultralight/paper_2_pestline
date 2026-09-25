#!/usr/bin/env python3
"""Ensemble + TTA (Test-Time Augmentation) dos classificadores de espécies.

Combina as probabilidades (soft-voting) de N modelos e, opcionalmente, aplica
TTA (flips + rotações) em cada um antes de fazer a média. Sem treino — só inferência.

Uso:
  python ensemble_tta.py --data /raid/.../DS-F2_crops_pad75 \
      --ckpts E3_swin/best.pt E3b_convnext/best.pt E3c_focal/best.pt \
      --tta --out /raid/.../flydet_runs/ensemble
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def build_model(name, num_classes):
    if name == "swin_t":
        m = models.swin_t(weights=None); m.head = torch.nn.Linear(m.head.in_features, num_classes)
    elif name == "swin_s":
        m = models.swin_s(weights=None); m.head = torch.nn.Linear(m.head.in_features, num_classes)
    elif name == "convnext_t":
        m = models.convnext_tiny(weights=None); m.classifier[2] = torch.nn.Linear(m.classifier[2].in_features, num_classes)
    elif name == "convnext_s":
        m = models.convnext_small(weights=None); m.classifier[2] = torch.nn.Linear(m.classifier[2].in_features, num_classes)
    elif name == "vit_b16":
        m = models.vit_b_16(weights=None); m.heads.head = torch.nn.Linear(m.heads.head.in_features, num_classes)
    else:
        raise ValueError(name)
    return m


def infer_model_name(ckpt_path: str, state: dict) -> str:
    """Deduz a arquitetura. Prioridade: campo 'arch' no checkpoint > nome do path
    > inspecao das chaves do state_dict (robusto a nomes de pasta sem o modelo)."""
    if isinstance(state, dict) and state.get("arch"):
        return state["arch"]
    p = ckpt_path.lower()
    if "convnext_s" in p: return "convnext_s"
    if "convnext" in p: return "convnext_t"
    if "swin_s" in p: return "swin_s"
    if "swin" in p: return "swin_t"
    if "vit" in p: return "vit_b16"
    # fallback: inspeciona as chaves do state_dict
    sd = state.get("model", {}) if isinstance(state, dict) else {}
    keys = list(sd.keys())
    if any(k.startswith("features") and "relative_position" in k for k in keys) or any("attn.qkv" in k for k in keys):
        return "swin_t"
    if any("classifier.2" in k for k in keys):
        return "convnext_t"
    if any(k.startswith("encoder.layers") for k in keys):
        return "vit_b16"
    raise ValueError(f"Nao foi possivel inferir a arquitetura de {ckpt_path}")


def tta_views(x):
    """Gera vistas TTA: identidade, hflip, vflip, rot90/180/270."""
    return [
        x,
        torch.flip(x, dims=[3]),      # hflip
        torch.flip(x, dims=[2]),      # vflip
        torch.rot90(x, 1, dims=[2, 3]),
        torch.rot90(x, 2, dims=[2, 3]),
        torch.rot90(x, 3, dims=[2, 3]),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--img-size", type=int, default=384,
                    help="Tamanho de fallback caso o checkpoint nao traga img_size.")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # Carrega em tamanho base (maior img_size entre os modelos); cada modelo
    # redimensiona p/ o SEU img_size na inferencia (Swin 224 != ConvNeXt 384).
    tf = transforms.Compose([transforms.ToTensor()])  # so tensor; resize por modelo

    # Para saber num_classes e img_size de cada modelo antes de montar o loader,
    # lemos os checkpoints primeiro.
    ckpts_info = []
    for ck in args.ckpts:
        state = torch.load(ck, map_location="cpu")
        name = infer_model_name(ck, state)
        img_sz = int(state.get("img_size", args.img_size))
        ckpts_info.append((name, ck, state, img_sz))
    class_names = ckpts_info[0][2]["classes"]
    base_size = max(info[3] for info in ckpts_info)

    tf_eval = transforms.Compose([transforms.Resize((base_size, base_size)), transforms.ToTensor(), norm])
    ds = datasets.ImageFolder(Path(args.data) / "test", tf_eval)
    assert list(ds.classes) == list(class_names), f"classes divergem: {ds.classes} vs {class_names}"
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    # carrega modelos, cada um com o SEU img_size (evita mismatch do Swin 224 vs 384)
    model_list = []
    for name, ck, state, img_sz in ckpts_info:
        m = build_model(name, len(class_names)).to(device).eval()
        m.load_state_dict(state["model"])
        model_list.append((name, m, img_sz))
    print(f"ensemble de {len(model_list)} modelos: {[(n, s) for n, _, s in model_list]} | TTA={'on' if args.tta else 'off'}")

    def probs_of(m, x, img_sz):
        if x.shape[-1] != img_sz:
            x = F.interpolate(x, size=(img_sz, img_sz), mode="bilinear", align_corners=False)
        views = tta_views(x) if args.tta else [x]
        p = None
        for v in views:
            q = F.softmax(m(v), dim=1)
            p = q if p is None else p + q
        return p

    # inferência com ensemble (+TTA)
    all_true, all_pred = [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            probs_sum = None
            for _, m, img_sz in model_list:
                p = probs_of(m, x, img_sz)
                probs_sum = p if probs_sum is None else probs_sum + p
            pred = probs_sum.argmax(1)
            all_pred.extend(pred.tolist()); all_true.extend(y.tolist())

    all_true = np.array(all_true); all_pred = np.array(all_pred)
    acc = float((all_true == all_pred).mean())
    per_class = {}
    for i, c in enumerate(class_names):
        mask = all_true == i
        per_class[c] = float((all_pred[mask] == i).mean()) if mask.sum() else 0.0

    print(f"ENSEMBLE{'+TTA' if args.tta else ''} acc={acc:.4f}")
    print("por classe:", json.dumps({k: round(v, 4) for k, v in per_class.items()}))
    summary = {"acc": acc, "per_class_acc": per_class, "tta": args.tta,
               "models": [(n, s) for n, _, s in model_list], "classes": class_names,
               "base_size": base_size}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # W&B
    try:
        import os
        if os.environ.get("WANDB_API_KEY"):
            import wandb
            wandb.init(project=os.environ.get("WANDB_PROJECT", "fly-species"),
                       entity=os.environ.get("WANDB_ENTITY", "pestline"),
                       name=os.environ.get("WANDB_RUN_NAME", "ensemble_tta"), config=summary)
            wandb.log({"test_acc": acc, **{f"test_acc/{k}": v for k, v in per_class.items()}})
            try:
                cm = wandb.plot.confusion_matrix(preds=all_pred.tolist(), y_true=all_true.tolist(), class_names=class_names)
                wandb.log({"confusion_matrix": cm})
            except Exception:
                pass
            wandb.finish()
    except Exception as e:
        print("wandb off:", e)


if __name__ == "__main__":
    main()
