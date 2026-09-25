#!/usr/bin/env python3
"""
Exit codes:
  0 — existe pelo menos um run com display name igual a run_name em entity/project
  1 — nao existe (ou projeto/nome vazio)
  2 — nao foi possivel consultar a API (rede, auth, projeto inexistente); o caller deve seguir o treino
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    if len(sys.argv) != 4:
        print(
            "Usage: wandb_run_exists.py <entity> <project> <run_name>",
            file=sys.stderr,
        )
        return 2

    entity, project, run_name = (a.strip() for a in sys.argv[1:4])
    if not project or not run_name:
        return 1

    key = (os.environ.get("WANDB_API_KEY") or "").strip()
    if not key:
        return 2

    try:
        import wandb
    except ImportError:
        print("[wandb_run_exists] wandb not installed", file=sys.stderr)
        return 2

    try:
        api = wandb.Api(api_key=key)
        if not entity:
            entity = api.default_entity
        path = f"{entity}/{project}"
        for run in api.runs(path):
            if run.name == run_name:
                return 0
    except Exception as e:
        print(f"[wandb_run_exists] {e}", file=sys.stderr)
        return 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
