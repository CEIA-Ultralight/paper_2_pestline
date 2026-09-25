#!/usr/bin/env python3
"""Entry point do experimento E10: aplica o patch NWD e delega ao trainer fly-det.

Uso (dentro do container flydet-train.sif):
    python train_nwd.py <mesmos argumentos do fly-det-train>

Grava nwd_config.json no diretorio corrente (WORK) para rastreabilidade.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nwd_patch import apply_nwd_patch


def main() -> int:
    # Seed opt-in via FLYDET_SEED (multi-seed do E10/E0). O trainer chama
    # init_seeds(seed+1+RANK) do config; sobrescrevemos DEPOIS do patch para
    # que o nosso valor prevaleca na pratica (re-seed imediato).
    seed = os.environ.get("FLYDET_SEED", "").strip()
    if seed:
        from ultralytics.utils.torch_utils import init_seeds

        init_seeds(int(seed), deterministic=True)
        print(f"[train_nwd] FLYDET_SEED={seed} aplicada (deterministic=True)")
    cfg = apply_nwd_patch()
    # RFLA (amplia pool de ancoras positivas p/ tiny) e opt-in via FLYDET_RFLA=1.
    if os.environ.get("FLYDET_RFLA", "0") == "1":
        from rfla_patch import apply_rfla_patch

        cfg.update(apply_rfla_patch())
    with open("nwd_config.json", "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)

    # Import tardio: o patch precisa estar ativo ANTES de qualquer instancia
    # de BboxLoss/TaskAlignedAssigner ser criada pelo trainer.
    from fly_det.trainer import main as fly_det_main

    return fly_det_main()


if __name__ == "__main__":
    sys.exit(main())
