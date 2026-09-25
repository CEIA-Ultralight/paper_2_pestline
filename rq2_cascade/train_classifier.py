#!/usr/bin/env python3
"""E3 — Classificador de espécies (Swin-T) sobre crops das detecções.

Estágio 2 do pipeline 2 estágios: recebe crops (ImageFolder) e classifica.
Mede acurácia GERAL e acurácia por espécie (MD/MV/MC/MF) — alvo 0.95.

Uso (dentro do container, via Slurm):
  python train_classifier.py --data /raid/.../DS-F2_crops --epochs 40 \
      --model swin_t --img-size 224 --batch 64 --out /raid/.../flydet_runs/E3_swin
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def build_model(name: str, num_classes: int):
    if name == "swin_t":
        m = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        m.head = nn.Linear(m.head.in_features, num_classes)
    elif name == "swin_s":
        m = models.swin_s(weights=models.Swin_S_Weights.IMAGENET1K_V1)
        m.head = nn.Linear(m.head.in_features, num_classes)
    elif name == "convnext_t":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    elif name == "convnext_s":
        m = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    elif name == "vit_b16":
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
    else:
        raise ValueError(name)
    return m


def evaluate(model, loader, device, class_names):
    model.eval()
    correct = total = 0
    per_class_correct = Counter_zeros(len(class_names))
    per_class_total = Counter_zeros(len(class_names))
    all_pred, all_true = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
            for t, p in zip(y.tolist(), pred.tolist()):
                per_class_total[t] += 1
                if t == p:
                    per_class_correct[t] += 1
            all_pred.extend(pred.tolist()); all_true.extend(y.tolist())
    acc = correct / max(1, total)
    per_class = {class_names[i]: (per_class_correct[i] / per_class_total[i] if per_class_total[i] else 0.0)
                 for i in range(len(class_names))}
    return acc, per_class, np.array(all_true), np.array(all_pred)


def Counter_zeros(n):
    return {i: 0 for i in range(n)}


class FocalLoss(nn.Module):
    """Focal Loss multi-classe: down-weights exemplos fáceis, foca nos difíceis."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def compute_class_weights(dataset, num_classes, device):
    """Peso = 1/freq normalizado — classes raras/dificeis recebem mais peso."""
    counts = np.zeros(num_classes)
    for _, y in dataset.samples:
        counts[y] += 1
    counts = np.maximum(counts, 1)
    w = counts.sum() / (num_classes * counts)
    return torch.tensor(w, dtype=torch.float32, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="swin_t",
                    choices=["swin_t", "swin_s", "convnext_t", "convnext_s", "vit_b16"])
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=12,
                    help="Early stopping: para se val_acc nao melhora por N epocas (0=desliga).")
    ap.add_argument("--focal", action="store_true",
                    help="Usa Focal Loss (gamma=2) em vez de CrossEntropy — foca em classes dificeis (ex.: MD).")
    ap.add_argument("--class-weights", action="store_true",
                    help="Pondera a loss pela frequencia inversa da classe (ajuda classes minoritarias/dificeis).")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Path(args.data)

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tf_train = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),          # moscas ficam em qualquer orientacao na placa
        transforms.RandomRotation(30),            # rotacao livre (postura variada)
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.02),
        transforms.ToTensor(), norm,
    ])
    tf_eval = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(), norm,
    ])

    ds_train = datasets.ImageFolder(data / "train", tf_train)
    ds_val = datasets.ImageFolder(data / "val", tf_eval)
    ds_test = datasets.ImageFolder(data / "test", tf_eval)
    class_names = ds_train.classes
    num_classes = len(class_names)
    print("classes:", class_names)

    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    dl_test = DataLoader(ds_test, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    model = build_model(args.model, num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # Loss: focal e/ou class-weighting (ajuda classes dificeis como MD)
    cw = compute_class_weights(ds_train, num_classes, device) if args.class_weights else None
    if args.focal:
        crit = FocalLoss(gamma=2.0, weight=cw)
        print(f"loss: FocalLoss(gamma=2){' + class_weights' if cw is not None else ''}")
    else:
        crit = nn.CrossEntropyLoss(label_smoothing=0.1, weight=cw)
        print(f"loss: CrossEntropy{' + class_weights' if cw is not None else ''}")

    # W&B opcional
    wandb = None
    try:
        import os
        if os.environ.get("WANDB_API_KEY"):
            import wandb as _wb
            _wb.init(project=os.environ.get("WANDB_PROJECT", "fly-species"),
                     entity=os.environ.get("WANDB_ENTITY", "pestline"),
                     name=os.environ.get("WANDB_RUN_NAME", f"E3_{args.model}"),
                     config=vars(args))
            wandb = _wb
    except Exception as e:
        print("wandb off:", e)

    best_val = 0.0
    epochs_no_improve = 0
    t0 = time.time()
    stopped_early = False
    for ep in range(1, args.epochs + 1):
        model.train()
        run_loss = run_n = 0
        for x, y in dl_train:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward(); opt.step()
            run_loss += loss.item() * y.numel(); run_n += y.numel()
        sched.step()
        val_acc, val_pc, _, _ = evaluate(model, dl_val, device, class_names)
        msg = f"ep {ep}/{args.epochs} loss={run_loss/max(1,run_n):.4f} val_acc={val_acc:.4f}"
        print(msg, flush=True)
        if wandb:
            wandb.log({"epoch": ep, "train_loss": run_loss/max(1,run_n), "val_acc": val_acc,
                       **{f"val_acc/{k}": v for k, v in val_pc.items()}})
        if val_acc > best_val:
            best_val = val_acc
            epochs_no_improve = 0
            torch.save({"model": model.state_dict(), "classes": class_names, "img_size": args.img_size,
                        "arch": args.model},
                       out / "best.pt")
        else:
            epochs_no_improve += 1
            if args.patience > 0 and epochs_no_improve >= args.patience:
                print(f"EARLY STOP: sem melhora por {args.patience} epocas (best val={best_val:.4f})", flush=True)
                stopped_early = True
                break

    # test final com o melhor
    ck = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ck["model"])
    test_acc, test_pc, yt, yp = evaluate(model, dl_test, device, class_names)
    elapsed = (time.time() - t0) / 3600
    print(f"TEST acc={test_acc:.4f} | por classe: {json.dumps({k: round(v,4) for k,v in test_pc.items()})}")
    print(f"tempo: {elapsed:.2f} h")

    summary = {"model": args.model, "test_acc": test_acc, "per_class_acc": test_pc,
               "val_best": best_val, "epochs": args.epochs, "epochs_ran": ep,
               "stopped_early": stopped_early, "time_hours": elapsed,
               "focal": args.focal, "class_weights": args.class_weights,
               "classes": class_names}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if wandb:
        wandb.log({"test_acc": test_acc, **{f"test_acc/{k}": v for k, v in test_pc.items()}})
        try:
            import wandb as _w
            cm = _w.plot.confusion_matrix(preds=yp.tolist(), y_true=yt.tolist(), class_names=class_names)
            wandb.log({"confusion_matrix": cm})
        except Exception:
            pass
        wandb.finish()


if __name__ == "__main__":
    main()
