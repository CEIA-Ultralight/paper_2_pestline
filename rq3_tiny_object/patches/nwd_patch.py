"""NWD (Normalized Wasserstein Distance) patch para Ultralytics YOLO26 — experimento E10.

Motivacao (literatura tiny objects):
  - IoU e hiper-sensivel a deslocamento de 1-2 px em caixas pequenas (~40 px no
    DS-F2, mediana sqrt(w*h)=39 px) — pune desproporcionalmente e desestabiliza
    o label assignment (TaskAlignedAssigner).
  - NWD (Wang et al., ISPRS J. 2022, arXiv:2206.13996; +4.3 AP em AI-TOD-v2)
    modela cada bbox como gaussiana 2D e usa a distancia de Wasserstein-2:
        W2^2 = ||c_a - c_b||^2 + ||wh_a/2 - wh_b/2||^2
        NWD  = exp(-sqrt(W2^2) / C)
    C = tamanho medio absoluto dos objetos do dataset (aqui: 44 px, medido nos
    labels de treino do DS-F2_v8.1.1 — media sqrt(w*h) = 44.4 px).

Estrategia do patch (mudanca unica vs E0, config fixa do E10):
  - BboxLoss.forward:        iou = (1-w) * CIoU + w * NWD        (w = 0.5)
  - TaskAlignedAssigner:     overlap = (1-w) * CIoU + w * NWD    (w = 0.5)
  O patch e de CLASSE, entao cobre os dois heads do E2EDetectLoss
  (one2many + one2one) usados pelo YOLO26.

Configuracao por variaveis de ambiente (com defaults do E10):
  FLYDET_NWD_W    (default 0.5)   — peso do NWD na mistura com CIoU
  FLYDET_NWD_C    (default 44.0)  — constante de escala (px)
  FLYDET_NWD_MODE (default nwd)   — "nwd" (C fixo) ou "gcd" (escala local,
                                    invariante a escala; arXiv:2510.27649)

Compativel com ultralytics 8.4.142 (assinaturas verificadas no container
flydet-train.sif em 2026-09-21). Se a assinatura mudar, apply_nwd_patch
falha ruidosamente (fail-fast) em vez de treinar silenciosamente sem NWD.
"""

import inspect
import os

import torch
import torch.nn.functional as F
from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import TaskAlignedAssigner, bbox2dist

from gcd_patch import gcd_similarity

NWD_W = float(os.environ.get("FLYDET_NWD_W", "0.5"))
NWD_C = float(os.environ.get("FLYDET_NWD_C", "44.0"))
NWD_MODE = os.environ.get("FLYDET_NWD_MODE", "nwd").strip().lower()

_PATCHED = False


def nwd_similarity(boxes1: torch.Tensor, boxes2: torch.Tensor, constant: float) -> torch.Tensor:
    """NWD entre pares de caixas xyxy (pixels), shape (..., 4) -> (..., 1) em (0, 1].

    boxes1/boxes2 devem estar alinhados par a par (broadcast simples).
    """
    c1 = (boxes1[..., :2] + boxes1[..., 2:]) / 2.0
    c2 = (boxes2[..., :2] + boxes2[..., 2:]) / 2.0
    wh1 = (boxes1[..., 2:] - boxes1[..., :2]).clamp(min=0.0)
    wh2 = (boxes2[..., 2:] - boxes2[..., :2]).clamp(min=0.0)
    center_dist = ((c1 - c2) ** 2).sum(-1, keepdim=True)
    wh_dist = (((wh1 - wh2) / 2.0) ** 2).sum(-1, keepdim=True)
    w2 = (center_dist + wh_dist).clamp(min=1e-12)
    return torch.exp(-torch.sqrt(w2) / constant)


def tiny_similarity(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Despacha entre NWD (C fixo) e GCD (escala local) conforme NWD_MODE."""
    if NWD_MODE == "gcd":
        return gcd_similarity(boxes1, boxes2)
    if NWD_MODE == "nwd":
        return nwd_similarity(boxes1, boxes2, NWD_C)
    raise RuntimeError(f"[nwd_patch] FLYDET_NWD_MODE invalido: {NWD_MODE!r} (use 'nwd' ou 'gcd')")


def _patched_bbox_loss_forward(
    self,
    pred_dist: torch.Tensor,
    pred_bboxes: torch.Tensor,
    anchor_points: torch.Tensor,
    target_bboxes: torch.Tensor,
    target_scores: torch.Tensor,
    target_scores_sum: torch.Tensor,
    fg_mask: torch.Tensor,
    imgsz: torch.Tensor,
    stride: torch.Tensor,
):
    """BboxLoss.forward do ultralytics 8.4.142 com IoU hibrido CIoU+NWD.

    Corpo identico ao original; unica mudanca: a metrica de similaridade
    `iou` passa a ser a media ponderada (1-w)*CIoU + w*NWD.
    """
    weight = target_scores[fg_mask].sum(-1, keepdim=True)
    iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
    nwd = tiny_similarity(pred_bboxes[fg_mask], target_bboxes[fg_mask])
    iou = (1.0 - NWD_W) * iou + NWD_W * nwd
    loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

    # DFL loss (copiado sem alteracao do original)
    if self.dfl_loss:
        target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
        loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
        loss_dfl = loss_dfl.sum() / target_scores_sum
    else:
        target_ltrb = bbox2dist(anchor_points, target_bboxes)
        # normalize ltrb by image size
        target_ltrb = target_ltrb * stride
        target_ltrb[..., 0::2] /= imgsz[1]
        target_ltrb[..., 1::2] /= imgsz[0]
        pred_dist = pred_dist * stride
        pred_dist[..., 0::2] /= imgsz[1]
        pred_dist[..., 1::2] /= imgsz[0]
        loss_dfl = (
            F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
        )
        loss_dfl = loss_dfl.sum() / target_scores_sum

    return loss_iou, loss_dfl


def _patched_iou_calculation(self, gt_bboxes: torch.Tensor, pd_bboxes: torch.Tensor) -> torch.Tensor:
    """TaskAlignedAssigner.iou_calculation com overlap hibrido CIoU+NWD."""
    iou = bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)
    nwd = tiny_similarity(gt_bboxes, pd_bboxes).squeeze(-1).clamp_(0)
    return (1.0 - NWD_W) * iou + NWD_W * nwd


def _check_signature(fn, expected_params: list[str], name: str) -> None:
    params = list(inspect.signature(fn).parameters)
    if params != expected_params:
        raise RuntimeError(
            f"[nwd_patch] Assinatura de {name} mudou nesta versao do ultralytics: {params} "
            f"(esperado {expected_params}). Recusando aplicar patch para nao treinar errado."
        )


def apply_nwd_patch() -> dict:
    """Aplica o monkeypatch NWD nas classes do ultralytics. Idempotente.

    Retorna dict com a config efetiva (para logging/rastreabilidade).
    """
    global _PATCHED
    cfg = {"nwd": True, "nwd_w": NWD_W, "nwd_c": NWD_C, "nwd_mode": NWD_MODE}
    if NWD_MODE == "off":  # E0 puro / multi-seed do baseline: sem patch
        print("[nwd_patch] FLYDET_NWD_MODE=off — treino SEM patch (baseline)")
        return {"nwd": False, "nwd_mode": "off"}
    if _PATCHED:
        return cfg

    # Fail-fast se o ultralytics mudou (evita treino silencioso sem NWD).
    _check_signature(
        BboxLoss.forward,
        ["self", "pred_dist", "pred_bboxes", "anchor_points", "target_bboxes",
         "target_scores", "target_scores_sum", "fg_mask", "imgsz", "stride"],
        "BboxLoss.forward",
    )
    _check_signature(
        TaskAlignedAssigner.iou_calculation,
        ["self", "gt_bboxes", "pd_bboxes"],
        "TaskAlignedAssigner.iou_calculation",
    )

    BboxLoss.forward = _patched_bbox_loss_forward
    TaskAlignedAssigner.iou_calculation = _patched_iou_calculation
    _PATCHED = True
    print(f"[nwd_patch] modo={NWD_MODE} w={NWD_W} C={NWD_C}px "
          f"(BboxLoss.forward + TaskAlignedAssigner.iou_calculation)")
    return cfg


def is_patched() -> bool:
    return _PATCHED
