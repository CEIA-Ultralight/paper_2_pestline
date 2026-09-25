"""ConvNeXt-T and DINOv2 ViT-B/14 ablations without extra annotations or
label-dependent inference.

Inputs are ImageNet-normalized RGB tensors; no resizing is performed here.
Configure TORCH_HOME under /raid/user_marcospaulo before requesting pretrained
ConvNeXt weights. The DINOv2 backbone never downloads: pretrained=True loads
the pinned local checkpoint (sha256-verified) and raises if it is absent or
corrupt. Run training and computational tests only inside a Slurm allocation.
"""

import hashlib
import logging
import math
import os
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

__all__ = ["build_model", "arcface_logits", "supcon_loss", "attention_diversity"]

_VARIANTS = frozenset({
    "baseline", "supcon", "arcface", "parts", "hires",
    "parts_supcon", "hires_supcon", "hires_parts", "dinov2",
})
_EMBEDDING_DIM = 128

_DINOV2_SHA256 = "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"
_DINOV2_DEFAULT_CHECKPOINT = Path(
    "/raid/user_marcospaulo/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth"
)
_DINOV2_PATCH = 14
_DINOV2_DIM = 768
_DINOV2_DEPTH = 12
_DINOV2_HEADS = 12


def _dinov2_checkpoint_path() -> Path:
    """Local checkpoint path; FLYDET_DINOV2_CHECKPOINT overrides (tests only)."""
    return Path(os.environ.get("FLYDET_DINOV2_CHECKPOINT", _DINOV2_DEFAULT_CHECKPOINT))


def _load_dinov2_state_dict() -> dict:
    """Offline, sha256-pinned load of the official DINOv2 ViT-B/14 backbone.

    Never downloads. Raises FileNotFoundError if the pinned file is absent and
    ValueError if its sha256 differs from the official release. The extra
    ``mask_token`` key exists only for DINO self-distillation pretraining and
    is dropped so the remaining keys load strictly.
    """
    path = _dinov2_checkpoint_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"DINOv2 checkpoint missing: {path}. This project runs offline: place "
            "dinov2_vitb14_pretrain.pth there manually; no automatic download."
        )
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    if digest.hexdigest() != _DINOV2_SHA256:
        raise ValueError(
            f"DINOv2 checkpoint sha256 mismatch: {path} has {digest.hexdigest()}, "
            f"expected {_DINOV2_SHA256}; refusing to load an unverified file"
        )
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(f"DINOv2 checkpoint at {path} is not a state dict")
    state = dict(state)
    state.pop("mask_token", None)
    return state


def _work_tensor(value: Tensor) -> Tensor:
    """Keep float64 for gradcheck, but promote low precision for reductions."""
    return value.float() if value.dtype in (torch.float16, torch.bfloat16) else value


def _normalize(value: Tensor, dim: int = 1) -> Tensor:
    return F.normalize(_work_tensor(value), dim=dim, eps=1e-8)


def _check_labels(labels: Tensor, count: int, device: torch.device) -> None:
    if labels.ndim != 1 or labels.shape[0] != count:
        raise ValueError("labels must have shape [N]")
    if labels.dtype != torch.long or labels.device != device:
        raise ValueError("labels must be torch.long on the input device")


def _positive_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _cosine_logits(embedding: Tensor, weight: Tensor) -> Tensor:
    # An enclosing AMP context must not downcast this matrix multiplication.
    with torch.autocast(device_type=embedding.device.type, enabled=False):
        return (_normalize(embedding) @ _normalize(weight).T).clamp(-1.0, 1.0)


def arcface_logits(
    embedding: Tensor,
    weight: Tensor,
    labels: Tensor,
    scale: float = 30.0,
    margin: float = 0.3,
) -> Tensor:
    """Training-only angular margin; inference uses scaled cosine scores.

    embedding: [N,D], weight: [C,D], labels: int64 [N]. Both vectors and
    class weights are normalized internally. The conventional monotonic
    fallback is used beyond pi-margin; the sine has a finite-gradient floor
    at cosine +/-1. Non-target scores are unchanged.
    """
    _positive_finite(scale, "scale")
    if not math.isfinite(margin) or not 0 <= margin < math.pi / 2:
        raise ValueError("margin must be finite and in [0, pi/2)")
    if embedding.ndim != 2 or weight.ndim != 2:
        raise ValueError("embedding and weight must be matrices")
    if embedding.shape[1] != weight.shape[1] or weight.shape[0] == 0:
        raise ValueError("weight must have shape [C,D], C > 0")
    if weight.device != embedding.device:
        raise ValueError("embedding and weight must share a device")
    _check_labels(labels, embedding.shape[0], embedding.device)
    if ((labels < 0) | (labels >= weight.shape[0])).any():
        raise ValueError("labels contain an invalid class index")
    cosine = _cosine_logits(embedding, weight)
    if margin == 0:
        return cosine * scale
    sine = (1.0 - cosine.square()).clamp_min(torch.finfo(cosine.dtype).eps).sqrt()
    phi = cosine * math.cos(margin) - sine * math.sin(margin)
    phi = torch.where(
        cosine > math.cos(math.pi - margin),
        phi,
        cosine - math.sin(math.pi - margin) * margin,
    )
    target = F.one_hot(labels, num_classes=weight.shape[0]).bool()
    return torch.where(target, phi, cosine) * scale


def supcon_loss(
    embeddings: Tensor, labels: Tensor, temperature: float = 0.1
) -> Tensor:
    """Mean supervised contrastive loss over anchors with positives.

    Input is [N,D], not [batch,views,D]; flatten multiple views and repeat
    their labels before calling. Uses normalized cosine similarity, excludes
    self-pairs from numerator and denominator, and averages every positive
    log-probability per anchor. No positives returns a graph-connected zero.
    """
    _positive_finite(temperature, "temperature")
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [N,D]")
    count = embeddings.shape[0]
    _check_labels(labels, count, embeddings.device)
    diagonal = torch.eye(count, dtype=torch.bool, device=embeddings.device)
    positives = labels[:, None].eq(labels[None, :]) & ~diagonal
    positive_count = positives.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        # Empty sum avoids overflow even when ignored embeddings are large.
        return _work_tensor(embeddings).reshape(-1)[:0].sum()
    with torch.autocast(device_type=embeddings.device.type, enabled=False):
        normalized = _normalize(embeddings)
        scores = (normalized @ normalized.T) / temperature
        scores = scores.masked_fill(diagonal, -torch.inf)
        log_prob = scores - torch.logsumexp(scores, dim=1, keepdim=True)
        positive_log_prob = torch.where(positives, log_prob, 0.0).sum(dim=1)
        return -(positive_log_prob[valid] / positive_count[valid]).mean()


def attention_diversity(attention: Tensor) -> Tensor:
    """Mean squared off-diagonal cosine overlap of spatial part maps.

    attention is [N,K,H,W], normally nonnegative spatial softmax maps.
    Minimize this penalty to discourage identical parts (no part labels).
    Disjoint maps score zero; identical nonzero maps score one. For K < 2
    or an empty batch the result is a graph-connected zero.
    """
    if attention.ndim != 4 or min(attention.shape[2:]) < 1:
        raise ValueError("attention must have shape [N,K,H,W], H,W > 0")
    count, parts = attention.shape[:2]
    if count == 0 or parts < 2:
        return _work_tensor(attention).reshape(-1)[:0].sum()
    with torch.autocast(device_type=attention.device.type, enabled=False):
        maps = _normalize(attention.flatten(2), dim=2)
        overlap = maps @ maps.transpose(1, 2)
        diagonal = torch.eye(parts, dtype=torch.bool, device=attention.device)
        return overlap.square().masked_fill(diagonal, 0).sum() / (
            count * parts * (parts - 1)
        )


class _DinoAttention(nn.Module):
    """Multi-head self-attention with the official fused qkv layout."""

    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(_DINOV2_DIM, _DINOV2_DIM * 3)
        self.proj = nn.Linear(_DINOV2_DIM, _DINOV2_DIM)

    def forward(self, x: Tensor) -> Tensor:
        count, tokens, dim = x.shape
        head_dim = dim // _DINOV2_HEADS
        qkv = self.qkv(x).reshape(count, tokens, 3, _DINOV2_HEADS, head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        # Explicit matmul: deterministic under torch.use_deterministic_algorithms.
        attention = (query * head_dim**-0.5) @ key.transpose(-2, -1)
        x = (attention.softmax(dim=-1) @ value).transpose(1, 2).reshape(count, tokens, dim)
        return self.proj(x)


class _LayerScale(nn.Module):
    """Per-channel residual scaling (DINOv2 ls1/ls2, init 1e-5)."""

    def __init__(self) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((_DINOV2_DIM,), 1e-5))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class _DinoMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(_DINOV2_DIM, _DINOV2_DIM * 4)
        self.fc2 = nn.Linear(_DINOV2_DIM * 4, _DINOV2_DIM)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class _DinoBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(_DINOV2_DIM)
        self.attn = _DinoAttention()
        self.ls1 = _LayerScale()
        self.norm2 = nn.LayerNorm(_DINOV2_DIM)
        self.mlp = _DinoMlp()
        self.ls2 = _LayerScale()

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class _PatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            3, _DINOV2_DIM, kernel_size=_DINOV2_PATCH, stride=_DINOV2_PATCH
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class _DinoV2Backbone(nn.Module):
    """Minimal ViT-B/14 whose state-dict keys match the official DINOv2 release.

    pos_embed keeps the checkpoint layout [1, 1+G*G, D] (cls row first, G=37).
    For inputs other than the pretraining grid the patch rows are resized with
    bicubic interpolation, as DINOv2 does for non-native resolutions.
    """

    def __init__(self, pos_tokens: int = 1 + 37 * 37) -> None:
        super().__init__()
        self.patch_embed = _PatchEmbed()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, _DINOV2_DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, pos_tokens, _DINOV2_DIM))
        self.blocks = nn.ModuleList(_DinoBlock() for _ in range(_DINOV2_DEPTH))
        self.norm = nn.LayerNorm(_DINOV2_DIM)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self._pos_resize_cache: dict[tuple[int, int, int], Tensor] = {}

    def _pos_resize_matrix(self, grid: int, tokens_h: int, tokens_w: int) -> Tensor:
        # Bicubic resize is linear in its input; materializing its matrix once
        # keeps the graph on matmul, whose backward is deterministic (the
        # bicubic backward kernel is not, and training enforces determinism).
        key = (grid, tokens_h, tokens_w)
        weights = self._pos_resize_cache.get(key)
        if weights is None:
            basis = torch.eye(grid * grid).reshape(grid * grid, 1, grid, grid)
            with torch.no_grad():
                weights = F.interpolate(
                    basis, size=(tokens_h, tokens_w), mode="bicubic",
                    align_corners=False,
                ).reshape(grid * grid, tokens_h * tokens_w).T.contiguous()
            self._pos_resize_cache[key] = weights
        return weights

    def _interpolated_pos(self, tokens_h: int, tokens_w: int) -> Tensor:
        cls_pos, patch_pos = self.pos_embed[:, :1], self.pos_embed[:, 1:]
        grid = round(patch_pos.shape[1] ** 0.5)
        if grid * grid != patch_pos.shape[1]:
            raise ValueError("DINOv2 pos_embed patch grid must be square")
        if (tokens_h, tokens_w) == (grid, grid):
            return self.pos_embed
        weights = self._pos_resize_matrix(grid, tokens_h, tokens_w)
        if weights.device != patch_pos.device or weights.dtype != patch_pos.dtype:
            weights = weights.to(device=patch_pos.device, dtype=patch_pos.dtype)
            self._pos_resize_cache[(grid, tokens_h, tokens_w)] = weights
        return torch.cat((cls_pos, (weights @ patch_pos[0]).unsqueeze(0)), dim=1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # Kernel-14/stride-14 conv uses floor: non-multiple borders are dropped
        # (384 px -> 27x27 patches covering 378 px), matching the conv output.
        tokens_h, tokens_w = x.shape[-2] // _DINOV2_PATCH, x.shape[-1] // _DINOV2_PATCH
        tokens = self.patch_embed(x)
        count, _, dim = tokens.shape
        x = torch.cat((self.cls_token.expand(count, -1, -1), tokens), dim=1)
        x = x + self._interpolated_pos(tokens_h, tokens_w)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls, patches = x[:, 0], x[:, 1:]
        feature_map = patches.transpose(1, 2).reshape(count, dim, tokens_h, tokens_w)
        return cls, feature_map


class _DinoV2Model(nn.Module):
    """DINOv2 ViT-B/14 with the same forward contract as _ArchitectureModel.

    Representation is the final LayerNorm-ed cls token; feature_map is the
    post-norm patch tokens reshaped to [N,768,H/14,W/14] (27x27 at 384 px).
    """

    def __init__(self, num_classes: int, pretrained: bool) -> None:
        super().__init__()
        self.variant = "dinov2"
        self.backbone = _DinoV2Backbone()
        self.feature_dim = _DINOV2_DIM
        self.embedding_dim = _EMBEDDING_DIM
        self.num_parts = 0
        self.scale = 30.0
        checkpoint_path = None
        if pretrained:
            checkpoint_path = str(_dinov2_checkpoint_path())
            self.backbone.load_state_dict(_load_dinov2_state_dict(), strict=True)
            logging.getLogger(__name__).info(
                "dinov2: strict-loaded local checkpoint %s", checkpoint_path
            )
        self.embedding_head = nn.Linear(self.feature_dim, self.embedding_dim)
        self.head = nn.Linear(self.feature_dim, num_classes)
        self.architecture_metadata = {
            "variant": self.variant,
            "backbone": "dinov2_vitb14",
            "pretrained": pretrained,
            # No conv stem: keep the ConvNeXt metadata keys present but null.
            "stem_original_stride": None,
            "stem_stride": None,
            "stem_weights_preserved": None,
            "nominal_output_stride": _DINOV2_PATCH,
            "patch_size": _DINOV2_PATCH,
            "embed_dim": _DINOV2_DIM,
            "pos_embed_grid_pretrained": (37, 37),
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": _DINOV2_SHA256 if pretrained else None,
            "test_campaign_adapter": (
                "pending: test_campaign_heads._validate_arch_summary currently "
                "asserts backbone == torchvision.convnext_tiny; extend it separately"
            ),
        }

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        cls, feature_map = self.backbone(x)
        embedding = _normalize(self.embedding_head(cls))
        return {
            "logits": self.head(cls),
            "embedding": embedding,
            "feature_map": feature_map,
        }


class _ArchitectureModel(nn.Module):
    def __init__(self, variant: str, num_classes: int, pretrained: bool) -> None:
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = convnext_tiny(weights=weights)
        self.variant = variant
        self.features = backbone.features
        self.feature_norm = backbone.classifier[0]
        self.feature_dim = backbone.classifier[2].in_features
        self.embedding_dim = _EMBEDDING_DIM
        mechanisms = variant.split("_")
        self.num_parts = 4 if "parts" in mechanisms else 0
        self.scale = 30.0
        stem = self.features[0][0]
        original_stride = tuple(stem.stride)
        if "hires" in mechanisms:
            # Mutate only stride: retain pretrained kernel/bias and padding.
            stem.stride = (2, 2)
            logging.getLogger(__name__).info(
                "hires: ConvNeXt-T stem stride %s -> %s; weights unchanged",
                original_stride,
                stem.stride,
            )
        self.architecture_metadata = {
            "variant": variant,
            "backbone": "torchvision.convnext_tiny",
            "pretrained": pretrained,
            "stem_original_stride": original_stride,
            "stem_stride": tuple(stem.stride),
            "stem_weights_preserved": True,
            "nominal_output_stride": 16 if "hires" in mechanisms else 32,
        }
        if self.num_parts:
            self.attention_head = nn.Conv2d(self.feature_dim, self.num_parts, 1)
            self.fusion = nn.Sequential(
                nn.Linear((self.num_parts + 1) * self.feature_dim, self.feature_dim),
                nn.GELU(),
            )
        self.embedding_head = nn.Linear(self.feature_dim, self.embedding_dim)
        if variant == "arcface":
            self.metric_weight = nn.Parameter(torch.empty(num_classes, self.embedding_dim))
            nn.init.xavier_uniform_(self.metric_weight)
        else:
            self.head = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        feature_map = self.features(x)
        pooled = F.adaptive_avg_pool2d(feature_map, 1)
        attention = None
        if self.num_parts:
            attention_scores = self.attention_head(feature_map)
            attention = _work_tensor(attention_scores).flatten(2).softmax(dim=-1)
            local = torch.bmm(
                feature_map.flatten(2), attention.to(feature_map.dtype).transpose(1, 2)
            )
            # Shared channel LayerNorm for the global vector and each part.
            vectors = torch.cat((pooled.flatten(2), local), dim=2).unsqueeze(-1)
            vectors = self.feature_norm(vectors).squeeze(-1).transpose(1, 2)
            representation = self.fusion(vectors.flatten(1))
            attention = attention.reshape_as(attention_scores)
        else:
            representation = self.feature_norm(pooled).flatten(1)
        embedding = _normalize(self.embedding_head(representation))
        if self.variant == "arcface":
            logits = self.scale * _cosine_logits(embedding, self.metric_weight)
        else:
            logits = self.head(representation)
        outputs = {"logits": logits, "embedding": embedding, "feature_map": feature_map}
        if attention is not None:
            outputs["attention"] = attention
        return outputs


def build_model(
    variant: str, num_classes: int, pretrained: bool = True
) -> nn.Module:
    """Build the ConvNeXt ablations, their combinations, or DINOv2.

    forward(x) takes no labels and returns logits [N,C], L2-normalized
    embedding [N,128], and raw feature_map [N,768,H,W] (ConvNeXt) or
    [N,768,H/14,W/14] (dinov2); parts variants also return attention
    [N,4,H,W], each map summing to one spatially. Baseline and supcon have
    identical architectures; their training objectives differ. Combinations
    reuse the same heads and initialization order; hires variants change only
    the stem stride, and parts variants use four attention maps. Only arcface
    owns metric_weight [C,128]. dinov2 is a plain linear-head classifier over
    a minimal ViT-B/14 backbone; pretrained=True strict-loads the pinned,
    sha256-verified local DINOv2 checkpoint (offline, no download) and raises
    if it is missing or corrupt. External training code must select/combine
    losses; this module does not train or change datasets.

    Save variant/architecture_metadata with checkpoints and reconstruct via
    this factory: convolution strides are not serialized in state_dict.
    """
    if variant not in _VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {sorted(_VARIANTS)}")
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError("num_classes must be a positive integer")
    if variant == "dinov2":
        return _DinoV2Model(num_classes, pretrained)
    return _ArchitectureModel(variant, num_classes, pretrained)