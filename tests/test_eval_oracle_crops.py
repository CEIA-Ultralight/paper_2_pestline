"""Testes CPU de rq2_cascade/eval_oracle_crops.py com dataset sintetico e classificador dummy.

O dummy le a cor do crop: cada caixa GT e pintada com R = 35*k + 20, onde k e o indice da classe
na ordem do classificador (INS, MAR, MC, MD, MF, MV, NOISE). Assim o teste controla exatamente o
que o classificador "preve" para cada GT sem depender de pesos reais.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import eval_oracle_crops as O  # rq2_cascade/ no sys.path via tests/conftest.py
import pytest
import torch
from PIL import Image, ImageDraw
from torchvision import transforms

CLF_ORDER = ("INS", "MAR", "MC", "MD", "MF", "MV", "NOISE")
YOLO = O.YOLO_CLASSES  # ('MD','MV','MC','MF','INS','NOISE','MAR')
W, H = 640, 360


def color_for(clf_class: str) -> tuple[int, int, int]:
    return (35 * CLF_ORDER.index(clf_class) + 20, 0, 0)


class _ColorModel(torch.nn.Module):
    """logits one-hot na classe codificada pelo maximo do canal R do crop."""

    def forward(self, x):
        k = ((x[:, 0].amax(dim=(1, 2)) * 255 - 20) / 35).round().long().clamp(0, len(CLF_ORDER) - 1)
        return {"logits": torch.nn.functional.one_hot(k, len(CLF_ORDER)).float() * 10}


def _fake_load_classifier(path, torch_mod, transforms_mod):
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    return _ColorModel().eval(), transform, CLF_ORDER, {"path": str(path), "variant": "dummy",
                                                       "class_order": list(CLF_ORDER)}


def _yolo_line(cls: str, x1, y1, x2, y2) -> str:
    cx, cy, w, h = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H
    return f"{YOLO.index(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def _write_dataset(root: Path, split: str) -> dict:
    """2 imagens; devolve {imagem: [(gt_class, painted_as)]} na ordem das labels."""
    (root / "images" / split).mkdir(parents=True)
    (root / "labels" / split).mkdir(parents=True)
    # (gt_class, cor pintada = classe que o dummy vai prever, caixa)
    spec = {
        "a.png": [("MD", "MD", (40, 40, 80, 80)),        # certo
                  ("MV", "MD", (300, 60, 340, 100)),     # errado (especie trocada)
                  ("INS", "INS", (500, 200, 540, 240)),  # nao-especie: fora do CSV
                  ("MC", "MC", (100, 250, 140, 290))],   # certo
        "b.png": [("MF", "NOISE", (60, 60, 100, 100)),   # errado, fora das 4 especies
                  ("MF", "MF", (400, 200, 440, 240)),    # certo
                  ("MD", "MD", (250, 150, 250, 190))],   # caixa degenerada (w=0)
    }
    for name, boxes in spec.items():
        img = Image.new("RGB", (W, H), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        lines = []
        for gt_cls, painted, (x1, y1, x2, y2) in boxes:
            if x2 > x1 and y2 > y1:
                draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=color_for(painted))
            lines.append(_yolo_line(gt_cls, x1, y1, x2, y2))
        img.save(root / "images" / split / name)
        (root / "labels" / split / Path(name).with_suffix(".txt")).write_text("\n".join(lines) + "\n")
    return spec


@pytest.fixture
def oracle_out(tmp_path, monkeypatch):
    data = tmp_path / "ds"
    spec = _write_dataset(data, "val")
    ckpt = tmp_path / "clf_dummy" / "best.pt"
    ckpt.parent.mkdir()
    ckpt.write_bytes(b"dummy")
    monkeypatch.setattr(O, "load_classifier", _fake_load_classifier)
    out = tmp_path / "oracle"
    rc = O.main(["--classifier", str(ckpt), "--data", str(data), "--split", "val", "--out", str(out),
                 "--device", "cpu", "--limit-images", "2"])
    assert rc == 0
    return out / "clf_dummy", spec, out


def test_per_instance_one_row_per_species_gt(oracle_out):
    folder, spec, _ = oracle_out
    with (folder / "per_instance.csv").open(newline="") as fh:
        reader = csv.DictReader(fh)
        assert tuple(reader.fieldnames) == O.PER_INSTANCE_COLUMNS
        rows = list(reader)
    expected = [(img, i, g) for img, boxes in spec.items() for i, (g, _, _) in enumerate(boxes) if g in O.SPECIES]
    assert [(r["image"], int(r["gt_index"]), r["gt_class"]) for r in rows] == expected
    assert all(r["gt_class"] in O.SPECIES for r in rows)  # INS nunca entra


def test_correct_matches_dummy_prediction(oracle_out):
    folder, spec, _ = oracle_out
    with (folder / "per_instance.csv").open(newline="") as fh:
        rows = {(r["image"], int(r["gt_index"])): r for r in csv.DictReader(fh)}
    for img, boxes in spec.items():
        for i, (gt_cls, painted, (x1, y1, x2, y2)) in enumerate(boxes):
            if gt_cls not in O.SPECIES:
                continue
            r = rows[(img, i)]
            if x2 <= x1:  # degenerado
                assert (r["pred_class"], r["score"], r["correct"]) == ("", "0.0", "False")
                continue
            assert r["pred_class"] == painted
            assert r["correct"] == str(painted == gt_cls)
            assert float(r["score"]) > 0.99
            assert float(r["gt_size"]) == pytest.approx(40.0)


def test_metrics_json_macro_f1_and_confusion(oracle_out):
    folder, _, out = oracle_out
    m = json.loads((folder / "metrics.json").read_text())
    # 6 GT de especie: MD ok, MV->MD, MC ok, MF->NOISE, MF ok, MD degenerado
    assert m["species_support"] == 6 and m["species_correct"] == 3
    assert m["species_accuracy"] == pytest.approx(0.5)
    assert m["degenerate_crops"] == 1 and m["n_pred_outside_species"] == 1
    pc = m["per_class"]
    assert (pc["MD"]["recall"], pc["MD"]["precision"]) == (0.5, 0.5)  # 1/2 GT; predicted MD = 2 (1 TP)
    assert (pc["MV"]["recall"], pc["MV"]["precision"]) == (0.0, 0.0)
    assert (pc["MC"]["recall"], pc["MC"]["precision"]) == (1.0, 1.0)
    assert (pc["MF"]["recall"], pc["MF"]["precision"]) == (0.5, 1.0)
    f1 = {"MD": 0.5, "MV": 0.0, "MC": 1.0, "MF": 2 * 0.5 * 1.0 / 1.5}
    assert m["species_macro_f1"] == pytest.approx(sum(f1.values()) / 4)
    conf = m["confusion"]
    cols = conf["pred_columns"]
    assert cols == list(CLF_ORDER) + [""]
    assert conf["counts"]["MV"][cols.index("MD")] == 1
    assert conf["counts"]["MF"][cols.index("NOISE")] == 1
    assert conf["counts"]["MD"][cols.index("")] == 1
    summary = json.loads((out / "summary.json").read_text())
    assert summary["clf_dummy"]["species_macro_f1"] == pytest.approx(m["species_macro_f1"])
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["split"] == "val" and manifest["images"] == 2


def test_rows_from_probs_rejects_size_mismatch():
    gt = [{"class": YOLO.index("MD"), "box": [0, 0, 10, 10]}]
    with pytest.raises(ValueError):
        O.rows_from_probs("x.jpg", gt, [], CLF_ORDER)
    with pytest.raises(ValueError):
        O.rows_from_probs("x.jpg", gt, [[0.5, 0.5]], CLF_ORDER)


def test_cpu_device_requires_limit_images(tmp_path):
    with pytest.raises(ValueError, match="smoke-only"):
        O.main(["--classifier", "x.pt", "--data", str(tmp_path), "--out", str(tmp_path / "o"), "--device", "cpu"])
