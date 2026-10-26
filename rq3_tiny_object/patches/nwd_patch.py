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
  FLYDET_NWD_SCOPE (default both) — onde aplicar: "loss" (so BboxLoss.forward),
                                    "assigner" (so TaskAlignedAssigner.iou_calculation)
                                    ou "both" (E10). Ablacao E10-loss / E10-assigner.
  FLYDET_NWD_LOSS_UNITS (default grid) — unidade das caixas no NWD da LOSS:
                                    "grid" = LEGADO (E10/E11/E12/ablacoes ate 2026-10-16): o
                                             ultralytics passa ao BboxLoss.forward caixas em
                                             unidades de grid (pred_bboxes e target/stride), entao
                                             C=44 vira 44 celulas (352/704/1408 px em P3/P4/P5) e o
                                             NWD da loss satura (~0,99). Mantido so p/ reproduzir
                                             os runs antigos.
                                    "px"   = CORRETO (E15+): multiplica as caixas pelo stride antes
                                             do NWD, igual ao assigner (que ja recebe pixels).
                                    A CIoU e o assigner nao mudam (CIoU e invariante a escala).
                                    Sem default silencioso: com modo nwd/gcd e scope loss/both, apply_nwd_patch
                                    aborta se FLYDET_NWD_LOSS_UNITS nao estiver definido (2026-10-26; a
                                    cadeia G rodou em grid por omissao).
  FLYDET_REQUIRE_PX (default 0)   — "1" aborta se a loss nao estiver em px (cadeias novas).
  FLYDET_NWD_DIAG (default 0)     — N>0 imprime, nas N primeiras chamadas da loss, o max do
                                    target como chega e a media do NWD (diagnostico de unidade).

Compativel com ultralytics 8.4.142 (assinaturas verificadas no container
flydet-train.sif em 2026-09-21). Se a assinatura mudar, apply_nwd_patch
falha ruidosamente (fail-fast) em vez de treinar silenciosamente sem NWD.
"""

import inspect
import os

import torch
import torch.nn.functional as F
from gcd_patch import gcd_similarity
from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import TaskAlignedAssigner, bbox2dist

NWD_W = float(os.environ.get("FLYDET_NWD_W", "0.5"))
NWD_C = float(os.environ.get("FLYDET_NWD_C", "44.0"))
NWD_MODE = os.environ.get("FLYDET_NWD_MODE", "nwd").strip().lower()
VALID_MODES = ("nwd", "gcd", "wiou", "off")
VALID_SCOPES = ("loss", "assigner", "both")
VALID_LOSS_UNITS = ("grid", "px")
NWD_LOSS_UNITS = "grid"
NWD_DIAG = 0

_PATCHED = False
_diag_calls = 0

# Wise-IoU v3 (Tong et al., arXiv:2301.10051): L = r * R_WIoU * (1 - IoU), r = beta / (delta * alpha**(beta - delta)),
# beta = L_IoU / media_movel(L_IoU) (outlier degree, sem gradiente). So a loss de caixa (scope=loss).
WIOU_ALPHA = float(os.environ.get("FLYDET_WIOU_ALPHA", "1.9"))
WIOU_DELTA = float(os.environ.get("FLYDET_WIOU_DELTA", "3.0"))
WIOU_MOMENTUM = float(os.environ.get("FLYDET_WIOU_MOMENTUM", "0.01"))
_WIOU_STATE = {"mean": 1.0}


def wiou_v3_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Loss WIoU v3 por caixa (xyxy, pixels), shape (N, 1). Atualiza a media movel de L_IoU."""
    iou = bbox_iou(pred, target, xywh=False, CIoU=False)
    l_iou = 1.0 - iou
    c1, c2 = (pred[..., :2] + pred[..., 2:]) / 2.0, (target[..., :2] + target[..., 2:]) / 2.0
    lt = torch.minimum(pred[..., :2], target[..., :2])
    rb = torch.maximum(pred[..., 2:], target[..., 2:])
    diag = ((rb - lt) ** 2).sum(-1, keepdim=True).detach().clamp(min=1e-9)
    r_wiou = torch.exp(((c1 - c2) ** 2).sum(-1, keepdim=True) / diag)
    with torch.no_grad():
        _WIOU_STATE["mean"] = (1 - WIOU_MOMENTUM) * _WIOU_STATE["mean"] + WIOU_MOMENTUM * float(l_iou.mean())
        beta = l_iou.detach() / max(_WIOU_STATE["mean"], 1e-9)
        r = beta / (WIOU_DELTA * WIOU_ALPHA ** (beta - WIOU_DELTA))
    return r * r_wiou * l_iou


def resolve_mode() -> str:
    """Le FLYDET_NWD_MODE no momento da aplicacao do patch e sincroniza o global NWD_MODE."""
    global NWD_MODE
    mode = os.environ.get("FLYDET_NWD_MODE", "nwd").strip().lower()
    if mode not in VALID_MODES:
        raise RuntimeError(
            f"[nwd_patch] FLYDET_NWD_MODE invalido: {mode!r} (use {'|'.join(VALID_MODES)})"
        )
    NWD_MODE = mode
    return mode


def resolve_scope() -> str:
    """Le FLYDET_NWD_SCOPE no momento da aplicacao do patch (default: both)."""
    scope = os.environ.get("FLYDET_NWD_SCOPE", "both").strip().lower()
    if scope not in VALID_SCOPES:
        raise RuntimeError(
            f"[nwd_patch] FLYDET_NWD_SCOPE invalido: {scope!r} (use {'|'.join(VALID_SCOPES)})"
        )
    return scope


def resolve_loss_units() -> str:
    """Le FLYDET_NWD_LOSS_UNITS e FLYDET_NWD_DIAG e sincroniza os globais (default: grid = legado)."""
    global NWD_LOSS_UNITS, NWD_DIAG
    units = os.environ.get("FLYDET_NWD_LOSS_UNITS", "grid").strip().lower()
    if units not in VALID_LOSS_UNITS:
        raise RuntimeError(
            f"[nwd_patch] FLYDET_NWD_LOSS_UNITS invalido: {units!r} (use {'|'.join(VALID_LOSS_UNITS)})"
        )
    NWD_LOSS_UNITS = units
    NWD_DIAG = int(os.environ.get("FLYDET_NWD_DIAG", "0"))
    return units


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

    pred_bboxes e target_bboxes chegam em unidades de grid (o v8DetectionLoss divide pelo
    stride antes de chamar); `stride` tem shape (N_anchors, 1) e faz broadcast com (B, N, 4).
    Com FLYDET_NWD_LOSS_UNITS=px o NWD e calculado em pixels, como no assigner.
    """
    global _diag_calls
    weight = target_scores[fg_mask].sum(-1, keepdim=True)
    if NWD_MODE == "wiou":  # WIoU so usa razoes (IoU, dist/diag): invariante a escala, grid ou px tanto faz
        loss_iou = (wiou_v3_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask]) * weight).sum() / target_scores_sum
    else:
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        if NWD_LOSS_UNITS == "px":
            nwd = tiny_similarity((pred_bboxes * stride)[fg_mask], (target_bboxes * stride)[fg_mask])
        else:
            nwd = tiny_similarity(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        if _diag_calls < NWD_DIAG:
            _diag_calls += 1
            msg = (f"[nwd_diag] call={_diag_calls} loss_units={NWD_LOSS_UNITS} "
                   f"target_max_in={target_bboxes[fg_mask].max().item():.1f} "
                   f"target_max_px={(target_bboxes * stride)[fg_mask].max().item():.1f} "
                   f"nwd_mean={nwd.mean().item():.4f} nwd_min={nwd.min().item():.4f} "
                   f"ciou_mean={iou.mean().item():.4f} n_fg={int(fg_mask.sum())}")
            print(msg, flush=True)
            with open("nwd_diag.log", "a", encoding="utf-8") as fh:  # cwd = WORK do run (gate do DONE)
                fh.write(msg + "\n")
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
    mode = resolve_mode()
    if mode == "off":  # E0 puro / multi-seed do baseline: sem patch
        print("[nwd_patch] FLYDET_NWD_MODE=off — treino SEM patch (baseline)")
        return {"nwd": False, "nwd_mode": "off"}
    scope = "loss" if mode == "wiou" else resolve_scope()  # WIoU nao tem analogo no assigner
    units = resolve_loss_units()
    if mode in ("nwd", "gcd") and scope != "assigner":
        if not os.environ.get("FLYDET_NWD_LOSS_UNITS", "").strip():
            raise RuntimeError(
                "[nwd_patch] FLYDET_NWD_LOSS_UNITS nao definido: use px (correto) ou grid (so para reproduzir runs legados)"
            )
        if os.environ.get("FLYDET_REQUIRE_PX", "0") == "1" and units != "px":
            raise RuntimeError(f"[nwd_patch] FLYDET_REQUIRE_PX=1 mas loss_units={units!r}")
    cfg = {"nwd": True, "nwd_w": NWD_W, "nwd_c": NWD_C, "nwd_mode": mode, "nwd_scope": scope,
           "nwd_loss_units": units}
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

    applied = []
    if scope in ("loss", "both"):
        BboxLoss.forward = _patched_bbox_loss_forward
        applied.append("BboxLoss.forward")
    if scope in ("assigner", "both"):
        TaskAlignedAssigner.iou_calculation = _patched_iou_calculation
        applied.append("TaskAlignedAssigner.iou_calculation")
    _PATCHED = True
    legacy = " (LEGADO: NWD da loss em grid, saturado)" if mode == "nwd" and units == "grid" and scope != "assigner" else ""
    print(f"[nwd_patch] modo={mode} scope={scope} w={NWD_W} C={NWD_C}px loss_units={units}{legacy} "
          f"({' + '.join(applied)})")
    return cfg


def is_patched() -> bool:
    return _PATCHED
