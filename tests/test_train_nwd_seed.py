"""F01-01 — a seed de FLYDET_SEED tem de chegar ao ``model.train(seed=...)`` do Ultralytics.

Bug original: train_nwd.py chamava ``init_seeds`` manualmente ANTES do trainer, mas o
``BaseTrainer.__init__`` do Ultralytics re-semeia com ``args.seed`` (default 0) — as seeds
0/1/2 produziam treinos identicos. Correcao: injetar ``seed``/``deterministic`` via
``--yolo-config`` (o unico caminho do CLI fly_det para ``model.train(**kwargs)``).

Testes CPU: ``fly_det.trainer.yolo`` e substituido por um fake que grava os kwargs de
``train`` e interrompe a execucao (nao ha GPU, dataset nem export).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import ClassVar

import pytest
import train_nwd  # rq3_tiny_object/ esta no sys.path via tests/conftest.py
import yaml


class _StopAfterTrain(Exception):
    """Sinaliza que model.train foi alcancado; o resto do trainer (export/val) nao interessa."""


class _FakeYOLO:
    """Substituto de ``ultralytics.YOLO`` que grava os kwargs de train()."""

    recorded: ClassVar[list[dict]] = []

    def __init__(self, model):
        self.model_name = model

    def to(self, device):
        return self

    def add_callback(self, *args, **kwargs):
        return None

    def train(self, **kwargs):
        _FakeYOLO.recorded.append(dict(kwargs))
        raise _StopAfterTrain()


def _run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, seed: str | None, extra_env=None,
         extra_argv=None) -> dict:
    """Roda train_nwd.main num cwd temporario com FLYDET_SEED=seed e devolve os kwargs de train."""
    from fly_det import trainer

    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FLYDET_NWD_MODE", "off")  # baseline E0: sem patch, teste foca no seed
    monkeypatch.delenv("FLYDET_RFLA", raising=False)
    monkeypatch.delenv("FLYDET_SAVE_PERIOD", raising=False)
    monkeypatch.delenv("FLYDET_RESUME", raising=False)
    if seed is None:
        monkeypatch.delenv("FLYDET_SEED", raising=False)
    else:
        monkeypatch.setenv("FLYDET_SEED", seed)
    for k, v in (extra_env or {}).items():
        monkeypatch.setenv(k, v)

    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("path: .\ntrain: images\nval: images\nnames: [MD]\n", encoding="utf-8")

    monkeypatch.setattr(trainer, "yolo", _FakeYOLO)
    monkeypatch.setattr(trainer.torch.cuda, "is_available", lambda: False)
    _FakeYOLO.recorded.clear()

    argv = [
        "--backend", "yolo",
        "--model", "yolo26m.pt",
        "--data", str(data_yaml),
        "--project", str(tmp_path / "runs"),
        "--epochs", "1",
        "--batch", "2",
        "--imgsz", "64",
        "--no-aug",
    ] + list(extra_argv or [])
    with pytest.raises(_StopAfterTrain):
        train_nwd.main(argv)
    assert len(_FakeYOLO.recorded) == 1, "model.train deve ser chamado exatamente uma vez"
    return _FakeYOLO.recorded[0]


@pytest.mark.parametrize("seed", ["1", "2"])
def test_seed_reaches_model_train(monkeypatch, tmp_path, seed):
    kwargs = _run(monkeypatch, tmp_path, seed)
    assert kwargs["seed"] == int(seed)
    assert kwargs["deterministic"] is True


def test_seeds_differ_between_runs(monkeypatch, tmp_path):
    k1 = _run(monkeypatch, tmp_path / "s1", "1")
    k2 = _run(monkeypatch, tmp_path / "s2", "2")
    assert k1["seed"] != k2["seed"]


def test_no_seed_env_leaves_ultralytics_default(monkeypatch, tmp_path):
    kwargs = _run(monkeypatch, tmp_path, None)
    assert "seed" not in kwargs  # Ultralytics usa default 0 — comportamento anterior preservado
    assert not (tmp_path / train_nwd.YOLO_CONFIG_NAME).exists()


def test_user_yolo_config_is_merged_not_dropped(monkeypatch, tmp_path):
    user_cfg = tmp_path / "user.yaml"
    user_cfg.write_text("cls: 2.5\nseed: 99\n", encoding="utf-8")
    kwargs = _run(monkeypatch, tmp_path, "7", extra_argv=["--yolo-config", str(user_cfg)])
    assert kwargs["cls"] == 2.5  # override do usuario preservado
    assert kwargs["seed"] == 7  # env prevalece sobre o YAML do usuario
    merged = yaml.safe_load((tmp_path / train_nwd.YOLO_CONFIG_NAME).read_text())
    assert merged == {"cls": 2.5, "seed": 7, "deterministic": True}


def test_save_period_and_resume_env(monkeypatch, tmp_path):
    kwargs = _run(monkeypatch, tmp_path, "0",
                  extra_env={"FLYDET_SAVE_PERIOD": "10", "FLYDET_RESUME": "1"})
    assert kwargs["save_period"] == 10
    assert kwargs["resume"] is True
    assert kwargs["seed"] == 0


def test_nwd_config_json_records_seed(monkeypatch, tmp_path):
    _run(monkeypatch, tmp_path, "2")
    cfg = json.loads((tmp_path / "nwd_config.json").read_text())
    assert cfg["seed"] == 2
    assert cfg["deterministic"] is True
    assert cfg["nwd"] is False  # FLYDET_NWD_MODE=off


def test_patches_dir_on_sys_path():
    patches = Path(train_nwd.__file__).resolve().parent / "patches"
    assert str(patches) in sys.path
    assert (patches / "nwd_patch.py").exists()
