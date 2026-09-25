"""GCD (Gaussian Combined Distance) — variante do NWD com invariancia de escala.

Experimento E10b. O NWD usa a constante C fixa (tamanho medio do dataset),
o que torna o valor dependente da escala absoluta das caixas. GCD normaliza
a distancia pela escala dos proprios objetos (raiz da media dos produtos
w1*h1 e w2*h2), tornando a metrica menos sensivel a variacao de tamanho —
o paper (GRSL 2025, arXiv:2510.27649) reporta SOTA em AI-TOD-v2.

Formula implementada (mesma estrutura do NWD, constante -> escala local):
    W2^2 = ||c1 - c2||^2 + ||wh1/2 - wh2/2||^2
    C_local = sqrt( (w1*h1 + w2*h2) / 2 )   (clamp min 1 px)
    GCD   = exp(-sqrt(W2^2) / C_local)

Aplicacao: mesmo monkeypatch do nwd_patch (BboxLoss.forward +
TaskAlignedAssigner.iou_calculation), configurado via FLYDET_NWD_MODE.
"""

import torch


def gcd_similarity(boxes1: torch.Tensor, boxes2: torch.Tensor, min_scale: float = 1.0) -> torch.Tensor:
    """GCD entre pares de caixas xyxy (pixels), shape (..., 4) -> (..., 1) em (0, 1]."""
    c1 = (boxes1[..., :2] + boxes1[..., 2:]) / 2.0
    c2 = (boxes2[..., :2] + boxes2[..., 2:]) / 2.0
    wh1 = (boxes1[..., 2:] - boxes1[..., :2]).clamp(min=0.0)
    wh2 = (boxes2[..., 2:] - boxes2[..., :2]).clamp(min=0.0)
    center_dist = ((c1 - c2) ** 2).sum(-1, keepdim=True)
    wh_dist = (((wh1 - wh2) / 2.0) ** 2).sum(-1, keepdim=True)
    w2 = (center_dist + wh_dist).clamp(min=1e-12)
    scale = torch.sqrt(((wh1.prod(-1, keepdim=True) + wh2.prod(-1, keepdim=True)) / 2.0).clamp(min=min_scale**2))
    return torch.exp(-torch.sqrt(w2) / scale)
