"""Testes CPU do harness rq2_cascade/eval_cascade_matrix.py com deteccoes sinteticas."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import eval_cascade_matrix as M  # rq2_cascade/ no sys.path via tests/conftest.py
import pytest

C = M.YOLO_CLASSES  # ('MD','MV','MC','MF','INS','NOISE','MAR')


def prop(box, cls, conf, probs=None, fallback=None):
    return {"box": list(box), "class": C.index(cls), "confidence": conf, "old_class": C.index(cls),
            "yolo_confidence": conf, "probabilities": probs, "fallback": fallback}


def gt(cls, box):
    return {"class": C.index(cls), "box": list(box)}


def synthetic_image():
    """3 GT (MD, MV, MC) + 3 propostas: MD ok, MV com classe errada no YOLO (MD) mas certa
    na cascata (MV), MC nao detectado; 1 FP INS longe; 1 proposta abaixo de conf .25."""
    g = [gt("MD", (100, 100, 140, 140)), gt("MV", (300, 300, 350, 350)), gt("MC", (700, 700, 730, 730))]
    single = [
        prop((102, 102, 142, 142), "MD", 0.90),
        prop((301, 299, 352, 351), "MD", 0.60),          # YOLO erra a especie
        prop((1500, 800, 1540, 840), "INS", 0.80),       # FP
        prop((700, 700, 730, 730), "MC", 0.10),          # abaixo de conf_match -> ignorada
    ]
    # cascata: mesma geometria; probabilidades em ordem YOLO
    p_md = [0.9, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005]
    p_mv = [0.2, 0.7, 0.05, 0.02, 0.01, 0.01, 0.01]
    p_ins = [0.1, 0.1, 0.1, 0.1, 0.5, 0.05, 0.05]
    p_mc = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    cascade = [
        dict(single[0], probabilities=p_md),
        dict(single[1], **{"class": C.index("MV"), "probabilities": p_mv}),
        dict(single[2], probabilities=p_ins),
        dict(single[3], probabilities=p_mc),
    ]
    return g, single, cascade


def test_rows_for_image_pairs_single_and_cascade():
    g, single, cascade = synthetic_image()
    rows = M.rows_for_image("img_a.jpg", g, single, cascade, "product")
    assert len(rows) == 3  # 1 linha por GT
    by = {r["gt_class"]: r for r in rows}
    assert by["MD"]["matched"] and by["MD"]["pred_class_single"] == "MD" == by["MD"]["pred_class_cascade"]
    assert by["MD"]["correct_single"] and by["MD"]["correct_cascade"]
    assert by["MD"]["iou"] > 0.8
    # MV: single erra (MD), cascata acerta (MV) -> par discordante para McNemar
    assert by["MV"]["matched"] and by["MV"]["pred_class_single"] == "MD"
    assert by["MV"]["pred_class_cascade"] == "MV"
    assert (by["MV"]["correct_single"], by["MV"]["correct_cascade"]) == (False, True)
    assert by["MV"]["score_single"] == pytest.approx(0.60)
    assert by["MV"]["score_cascade"] == pytest.approx(0.60 * 0.7)  # product
    # MC: proposta com conf .10 < .25 -> nao detectado
    assert not by["MC"]["matched"] and by["MC"]["pred_class_single"] == "" and by["MC"]["score_single"] == 0.0
    assert by["MC"]["gt_size"] == pytest.approx(30.0)
    assert by["MD"]["gt_box"] == "100.00,100.00,140.00,140.00"


def test_label_only_keeps_yolo_score_and_no_cascade_branch():
    g, single, cascade = synthetic_image()
    rows = M.rows_for_image("x", g, single, cascade, "label_only")
    assert {r["gt_class"]: r["score_cascade"] for r in rows}["MV"] == pytest.approx(0.60)
    rows_single = M.rows_for_image("x", g, single, None)
    assert all(r["pred_class_cascade"] == "" and r["correct_cascade"] is False for r in rows_single)


def test_false_positives_and_prf():
    g, single, cascade = synthetic_image()
    fp = M.false_positives(g, single)
    assert fp["INS"] == 1 and sum(fp.values()) == 1
    rows = M.rows_for_image("x", g, single, cascade, "label_only")
    m_single = M.prf_from_rows(rows, fp, "single")
    m_cascade = M.prf_from_rows(rows, M.false_positives(g, cascade), "cascade")
    # single: MD tp=1 predicted=2 (a MV virou MD) -> precision .5, recall 1
    assert m_single["per_class"]["MD"]["precision"] == pytest.approx(0.5)
    assert m_single["per_class"]["MD"]["recall"] == pytest.approx(1.0)
    assert m_single["per_class"]["MV"]["recall"] == 0.0
    assert m_single["species_accuracy_all_gt"] == pytest.approx(1 / 3)
    assert m_single["species_accuracy_conditional_matched"] == pytest.approx(1 / 2)
    assert m_single["species_detection_recall"] == pytest.approx(2 / 3)
    assert m_cascade["species_accuracy_all_gt"] == pytest.approx(2 / 3)
    assert m_cascade["per_class"]["MD"]["precision"] == pytest.approx(1.0)
    assert m_cascade["per_class"]["INS"]["fp_unmatched"] == 1


def test_relabel_named_with_background_class_and_degenerate():
    p = prop((0, 0, 10, 10), "MD", 0.5)
    order = ("BG", "INS", "MAR", "MC", "MD", "MF", "MV", "NOISE")
    out = M.relabel_named(p, [0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.03, 0.02], order)
    assert out["class"] == C.index("MD") and out["fallback"] == "background"  # mantem YOLO
    out2 = M.relabel_named(p, [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.9, 0.0], order)
    assert out2["class"] == C.index("MV") and out2["fallback"] is None
    assert out2["probabilities"] == [0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0]  # ordem YOLO
    out3 = M.relabel_named(p, None, order)
    assert out3["fallback"] == "degenerate_crop" and out3["class"] == C.index("MD")
    assert M.apply_policy(out3, "product") == 0.5  # sem probs -> conf YOLO
    with pytest.raises(ValueError):
        M.relabel_named(p, [1.0], order)


def test_assemble_outputs_writes_csv_and_jsonl(tmp_path: Path):
    g, single, cascade = synthetic_image()
    recs = [
        {"image": "a.jpg", "gt": g, "single": single, "cascade": cascade,
         "seconds": {"detector": 0.05, "classifier": 0.01}},
        {"image": "b.jpg", "gt": [], "single": [], "cascade": [],
         "seconds": {"detector": 0.04, "classifier": 0.0}},
    ]
    rows, jsonl, metrics = M.assemble_outputs(recs, "product", True, 0.25, 0.5)
    folder = tmp_path / "E0_s0" / "clf_ce_s42" / "product"
    M.write_per_instance(rows, folder / "per_instance.csv")
    M.write_jsonl(jsonl, folder / "per_image_preds.jsonl")
    with (folder / "per_instance.csv").open() as fh:
        read = list(csv.DictReader(fh))
    assert [r["image"] for r in read] == ["a.jpg"] * 3
    assert list(read[0].keys()) == list(M.PER_INSTANCE_COLUMNS)
    assert {r["gt_class"] for r in read} == {"MD", "MV", "MC"}
    lines = [json.loads(line) for line in (folder / "per_image_preds.jsonl").read_text().splitlines()]
    assert len(lines) == 2 and len(lines[0]["proposals"]) == 4 and lines[1]["proposals"] == []
    assert lines[0]["proposals"][1]["cascade_class"] == "MV"
    assert metrics["images"] == 2 and metrics["gt_objects"] == 3
    assert metrics["paired_discordant"] == {"single_right_cascade_wrong": 0, "single_wrong_cascade_right": 1}
    assert metrics["latency"]["detector"]["n"] == 2
    assert metrics["latency"]["end_to_end"]["mean_s"] == pytest.approx(0.05)
    # per_instance.csv e consumivel por common/stats.py
    import stats

    cm = stats.confusion_from_per_instance(folder / "per_instance.csv")
    assert cm.loc["MV", "MV"] == 1 and cm.loc["MC", "BG"] == 1
    mc = stats.mcnemar_paired([r["correct_single"] == "True" for r in read],
                              [r["correct_cascade"] == "True" for r in read])
    assert (mc["b"], mc["c"]) == (0, 1)


def test_derive_name_and_list_split(tmp_path: Path):
    assert M.derive_name("/raid/u/flydet_runs/E0_s0/runs/yolo26m.pt/weights/best.pt") == "E0_s0"
    assert M.derive_name("/raid/u/flydet_runs/clf_convnext_t_ce_s42/best.pt") == "clf_convnext_t_ce_s42"
    (tmp_path / "images/test").mkdir(parents=True)
    (tmp_path / "labels/test").mkdir(parents=True)
    (tmp_path / "images/test/b.jpg").write_bytes(b"")
    (tmp_path / "images/test/a.png").write_bytes(b"")
    (tmp_path / "labels/test/a.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    pairs = M.list_split(tmp_path, "test")
    assert [p[0].name for p in pairs] == ["a.png", "b.jpg"]
    assert M.load_gt(pairs[0][1], 1920, 1080) == [{"class": 0, "box": [864.0, 486.0, 1056.0, 594.0]}]
    assert M.load_gt(pairs[1][1], 1920, 1080) == []  # label ausente = sem GT
    with pytest.raises(FileNotFoundError):
        M.list_split(tmp_path, "val")


def test_parser_defaults_and_partition_guard():
    args = M.build_parser().parse_args(["--detector", "d.pt", "--data", "/x", "--out", "/y"])
    assert args.split == "test" and args.policies == ["label_only", "product"]
    assert args.wandb_project == M.DEFAULT_WANDB_PROJECT
    assert "b200n1" in M.EV.GPU_PARTITIONS
