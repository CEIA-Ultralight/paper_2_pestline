"""Backbones foundation para o classificador da cascata: DINOv3 ViT-B/16 (timm) e BioCLIP 2 (ViT-L/14, open_clip).

Pesos lidos de arquivos locais em /raid/user_marcospaulo/models/{dinov3,bioclip2} (sem download em tempo de treino;
o trainer redefine HF_HOME/TORCH_HOME). Bibliotecas extras (timm, open_clip, huggingface_hub) vivem em
/raid/user_marcospaulo/pylibs, anexado ao FIM do sys.path para nao sobrepor o torch do container.
Mesmo contrato de forward de models._ArchitectureModel: dict(logits, embedding, feature_map).
Fine-tuning completo (lr baixo, sem LP-FT): usar --lr 2e-5 e, para bioclip2, --img-size 224.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F

PYLIBS = "/raid/user_marcospaulo/pylibs"
if PYLIBS not in sys.path:
    sys.path.append(PYLIBS)

MODELS = Path("/raid/user_marcospaulo/models")
_IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
_CLIP = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


class _Foundation(nn.Module):
    def __init__(self, variant: str, num_classes: int, pretrained: bool) -> None:
        super().__init__()
        self.variant = variant
        self.num_parts = 0
        self.scale = 30.0
        self.embedding_dim = 128
        self.register_buffer("_shift", torch.zeros(1, 3, 1, 1), persistent=False)
        self.register_buffer("_scale", torch.ones(1, 3, 1, 1), persistent=False)
        if variant == "dinov3":
            import timm
            self.backbone = timm.create_model("vit_base_patch16_dinov3", pretrained=False, num_classes=0,
                                              dynamic_img_size=True)
            self.feature_dim = 768
            path = MODELS / "dinov3" / "model.safetensors"
            name = "timm/vit_base_patch16_dinov3.lvd1689m"
            patch = 16
        elif variant == "bioclip2":
            from open_clip.model import CLIPVisionCfg, _build_vision_tower
            cfg = json.loads((MODELS / "bioclip2" / "open_clip_config.json").read_text())
            vc = cfg["model_cfg"]["vision_cfg"]
            self.backbone = _build_vision_tower(cfg["model_cfg"]["embed_dim"], CLIPVisionCfg(**vc))
            self.feature_dim = cfg["model_cfg"]["embed_dim"]
            path = MODELS / "bioclip2" / "open_clip_model.safetensors"
            name = "imageomics/bioclip-2"
            patch = vc["patch_size"]
            # trainer normaliza com ImageNet; BioCLIP espera estatisticas CLIP -> reajuste exato dentro do modelo
            im_m = torch.tensor(_IMAGENET[0]).view(1, 3, 1, 1); im_s = torch.tensor(_IMAGENET[1]).view(1, 3, 1, 1)
            cl_m = torch.tensor(_CLIP[0]).view(1, 3, 1, 1); cl_s = torch.tensor(_CLIP[1]).view(1, 3, 1, 1)
            self._scale.copy_(im_s / cl_s); self._shift.copy_((im_m - cl_m) / cl_s)
        else:
            raise ValueError(variant)
        loaded = None
        if pretrained:
            from safetensors.torch import load_file
            state = load_file(str(path))
            if variant == "bioclip2":
                state = {k[len("visual."):]: v for k, v in state.items() if k.startswith("visual.")}
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            bad_missing = [k for k in missing if not k.startswith(("head", "fc_norm.", "attn_pool"))]
            if bad_missing or unexpected:
                raise RuntimeError(f"{variant}: pesos incompativeis missing={bad_missing[:5]} unexpected={list(unexpected)[:5]}")
            loaded = str(path)
            logging.getLogger(__name__).info("%s: pesos locais %s", variant, loaded)
        self.embedding_head = nn.Linear(self.feature_dim, self.embedding_dim)
        self.head = nn.Linear(self.feature_dim, num_classes)
        self.architecture_metadata = {
            "variant": variant, "backbone": name, "pretrained": pretrained, "patch_size": patch,
            "checkpoint_path": loaded, "stem_original_stride": None, "stem_stride": None,
            "stem_weights_preserved": None, "nominal_output_stride": patch,
        }

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        if self.variant == "bioclip2":
            x = x * self._scale + self._shift
        feats = self.backbone(x)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        embedding = F.normalize(self.embedding_head(feats).float(), dim=1)
        return {"logits": self.head(feats), "embedding": embedding, "feature_map": feats[:, :, None, None]}
