#!/usr/bin/env python3
"""Entry point dos experimentos E0/E10/E10b/E14: aplica o patch NWD e delega ao trainer fly-det.

Uso (dentro do container flydet-train-b200.sif):
    python train_nwd.py <mesmos argumentos do fly-det-train>

Variaveis de ambiente lidas aqui (alem das do nwd_patch/rfla_patch):
    FLYDET_SEED         seed inteiro -> Ultralytics ``seed`` (+ ``deterministic=True``)
    FLYDET_SAVE_PERIOD  inteiro -> Ultralytics ``save_period`` (default do yolo.yaml: 1)
    FLYDET_RESUME       "1" -> Ultralytics ``resume=True`` (o ``--model`` deve ser o last.pt)
    FLYDET_NWD_MODE     nwd|gcd|off (off = baseline E0 sem patch)
    FLYDET_NWD_SCOPE    loss|assigner|both (ver nwd_patch.py)
    FLYDET_RFLA         "1" -> patch RFLA

Mecanismo do seed (fix F01-01): o CLI do fly_det nao tem ``--seed``; o unico caminho para
``model.train(**kwargs)`` sem editar o submodule e o ``--yolo-config`` (YAML que sobrescreve o
``fly_det/config/yolo.yaml`` empacotado). Este script grava ``nwd_yolo_config.yaml`` no diretorio
corrente (WORK) com ``seed``/``deterministic``/``save_period``/``resume`` e injeta
``--yolo-config`` no argv. O ``BaseTrainer.__init__`` do Ultralytics chama
``init_seeds(args.seed + 1 + RANK, deterministic=args.deterministic)`` e grava ``args.yaml``
com o seed efetivo — auditavel no run. (O antigo ``init_seeds`` manual antes do trainer era
sobrescrito por ``args.seed=0`` — por isso as seeds 0/1/2 da cadeia 33015 sairam identicas.)

Grava nwd_config.json no diretorio corrente (WORK) para rastreabilidade.
"""

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "patches"))

from nwd_patch import apply_nwd_patch

YOLO_CONFIG_NAME = "nwd_yolo_config.yaml"


def yolo_overrides(env) -> dict:
    """Chaves Ultralytics derivadas do ambiente (vazio quando nada foi pedido)."""
    overrides: dict = {}
    seed = (env.get("FLYDET_SEED") or "").strip()
    if seed:
        overrides["seed"] = int(seed)
        overrides["deterministic"] = True
    save_period = (env.get("FLYDET_SAVE_PERIOD") or "").strip()
    if save_period:
        overrides["save_period"] = int(save_period)
    if (env.get("FLYDET_RESUME") or "").strip() == "1":
        overrides["resume"] = True
    return overrides


def inject_yolo_config(argv, env, workdir: Path) -> list:
    """Retorna argv com ``--yolo-config`` apontando para um YAML que inclui os overrides do env.

    Um ``--yolo-config`` ja presente e lido e mesclado (o env prevalece).
    """
    import yaml

    overrides = yolo_overrides(env)
    argv = list(argv)
    if not overrides:
        return argv
    merged: dict = {}
    if "--yolo-config" in argv:
        idx = argv.index("--yolo-config")
        with open(argv[idx + 1], encoding="utf-8") as fh:
            merged.update(yaml.safe_load(fh) or {})
        del argv[idx : idx + 2]
    merged.update(overrides)
    out = workdir / YOLO_CONFIG_NAME
    with out.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(merged, fh, sort_keys=True)
    print(f"[train_nwd] --yolo-config {out} <- {overrides}")
    return argv + ["--yolo-config", str(out)]


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cfg = apply_nwd_patch()
    # RFLA (amplia pool de ancoras positivas p/ tiny) e opt-in via FLYDET_RFLA=1.
    if os.environ.get("FLYDET_RFLA", "0") == "1":
        from rfla_patch import apply_rfla_patch

        cfg.update(apply_rfla_patch())
    cfg.update(yolo_overrides(os.environ))
    with open("nwd_config.json", "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)

    argv = inject_yolo_config(argv, os.environ, Path.cwd())

    # Import tardio: o patch precisa estar ativo ANTES de qualquer instancia
    # de BboxLoss/TaskAlignedAssigner ser criada pelo trainer.
    from fly_det.trainer import main as fly_det_main

    return fly_det_main(argv)


if __name__ == "__main__":
    sys.exit(main())
