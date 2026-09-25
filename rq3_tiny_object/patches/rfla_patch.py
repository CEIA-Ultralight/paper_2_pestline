"""RFLA-style candidate selection para o TaskAlignedAssigner — experimento E14.

Motivacao (RFLA, ECCV 2022, arXiv:2208.08738, +4.0 AP em AI-TOD):
  o criterio padrao ("centro da ancora dentro da caixa GT") pode nao atribuir
  NENHUMA ancora positiva a objetos tiny — para uma mosca de ~40 px vista no
  stride 32, o centro geometrico pode nao cair em nenhum ponto de grade.
  Este patch amplia o pool de candidatos para objetos pequenos:
    - lado >= 2*stride: criterio padrao (centro dentro da caixa);
    - lado <  2*stride: ancora dentro de um circulo de raio max(2*stride, lado/2)
      centrado no centro da caixa (GARFLA simplificado: garante positivos).
  Complementar ao NWD (que mede similaridade), nao conflita.

Ativado por FLYDET_RFLA=1. Compativel com ultralytics 8.4.142.
"""

import os

import torch
from ultralytics.utils.tal import TaskAlignedAssigner

RFLA_ENABLED = os.environ.get("FLYDET_RFLA", "0") == "1"

_PATCHED = False


def _patched_select_candidates_in_gts(self, xy_centers, gt_bboxes, mask_gt, eps=1e-9):
    """select_candidates_in_gts com pool ampliado para caixas tiny.

    Para cada GT, ancora e candidata se:
      - caixa grande (lado >= 2*stride): centro da ancora dentro da caixa (padrao); ou
      - caixa tiny  (lado <  2*stride): ancora dentro de raio max(2*stride, lado/2)
        do centro da caixa.
    Retorna mascara booleana (b, n_boxes, h*w) — mesma assinatura do original.
    """
    # caixa em xyxy -> centro e lados
    x1, y1, x2, y2 = gt_bboxes.unbind(-1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = (x2 - x1).clamp(min=0.0)
    h = (y2 - y1).clamp(min=0.0)
    side = torch.minimum(w, h)  # (b, n_boxes)

    # criterio padrao: centro da ancora dentro da caixa
    lt = torch.stack([x1, y1], -1).unsqueeze(2)  # (b, n, 1, 2)
    rb = torch.stack([x2, y2], -1).unsqueeze(2)
    in_box = (
        (xy_centers[:, 0] - lt[..., 0] > eps)
        & (xy_centers[:, 1] - lt[..., 1] > eps)
        & (rb[..., 0] - xy_centers[:, 0] > eps)
        & (rb[..., 1] - xy_centers[:, 1] > eps)
    )

    # criterio tiny: dentro do circulo de raio max(2*stride, lado/2)
    # stride_val e escalar (int) no ultralytics 8.4.142
    stride = float(self.stride_val)
    ax = xy_centers[:, 0].view(1, 1, -1)
    ay = xy_centers[:, 1].view(1, 1, -1)
    dist2 = (ax - cx.unsqueeze(-1)) ** 2 + (ay - cy.unsqueeze(-1)) ** 2
    radius = torch.clamp(torch.maximum(side / 2.0, torch.full_like(side, 2.0 * stride)), min=eps)
    in_circle = dist2 <= (radius.unsqueeze(-1) ** 2)

    is_tiny = (side < 2.0 * stride).unsqueeze(-1)  # (b, n, 1)
    mask = torch.where(is_tiny, in_circle, in_box)
    return mask


def apply_rfla_patch() -> dict:
    global _PATCHED
    cfg = {"rfla": True}
    if _PATCHED:
        return cfg
    if not hasattr(TaskAlignedAssigner, "select_candidates_in_gts"):
        raise RuntimeError("[rfla_patch] select_candidates_in_gts nao existe nesta versao")
    # stride_val e atributo de INSTANCIA (definido em __init__/forward), nao de classe.
    probe = TaskAlignedAssigner()
    if not hasattr(probe, "stride_val"):
        raise RuntimeError("[rfla_patch] stride_val nao existe nesta versao")
    TaskAlignedAssigner.select_candidates_in_gts = _patched_select_candidates_in_gts
    _PATCHED = True
    print("[rfla_patch] candidate selection ampliada para caixas tiny (raio max(2*stride, lado/2))")
    return cfg
