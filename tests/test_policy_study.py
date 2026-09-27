"""Unit tests for policy_study — stdlib only, no numpy/torch."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rq2_cascade'))
import policy_study as PS


def probs(winner, others=0.1):
    values = [others] * 7
    values[winner] = 1 - 6 * others
    return values


class FusionTests(unittest.TestCase):
    def test_label_only_edge(self):
        p = probs(2, 0.05)
        self.assertAlmostEqual(PS.geometric(0.4, p, 1.0, 0.0), 0.4)

    def test_product_edge(self):
        p = probs(2, 0.05)
        self.assertAlmostEqual(PS.geometric(0.4, p, 1.0, 1.0), 0.4 * 0.7)

    def test_temperature_keeps_normalized(self):
        p = [0.7, 0.1, 0.05, 0.05, 0.04, 0.04, 0.02]
        for t in PS.TEMPERATURES:
            scaled = [max(v, 1e-12) ** (1 / t) for v in p]
            self.assertAlmostEqual(sum(scaled) / sum(scaled), 1.0)
        # T>1 achata: max relativo cai
        flat = [v ** 0.25 for v in p]
        self.assertLess(max(flat) / sum(flat), max(p))

    def test_fallback_uses_yolo(self):
        self.assertAlmostEqual(PS.geometric(0.3, None, 0.5, 0.5), 0.3 ** 0.5)

    def test_grid_size(self):
        fusion = 25 - 5 + 5 * 4  # b=0 → 5 combos (T fixo); b>0 → 20 combos × 4 T = 80... reconta abaixo
        configs = PS.grid_configs()
        kinds = [c['kind'] for c in configs]
        self.assertEqual(kinds.count('fusion'), 5 + 20 * 4)
        self.assertEqual(kinds.count('selective'), 24)
        names = {PS.config_name('parts', c) for c in configs}
        self.assertEqual(len(names), len(configs))


class SelectiveTests(unittest.TestCase):
    def record(self, yolo_cls, conf, winner):
        pred = [{'box': [0, 0, 10, 10], 'class': yolo_cls, 'confidence': conf,
                 'yolo_confidence': conf, 'probabilities': None, 'fallback': None}]
        row = [{'box': [0, 0, 10, 10], 'class': yolo_cls, 'confidence': conf,
                'yolo_confidence': conf, 'probabilities': probs(winner), 'fallback': None}]
        return {'gt': [], 'pipelines': {'yolo': pred, f'parts_label_only': row}}

    def test_high_conf_keeps_yolo(self):
        r = self.record(1, 0.8, 3)  # YOLO disse MV, classificador diz MF
        out = PS.apply_config(r, 'parts', 'selective', tau=0.5)
        self.assertEqual(out[0]['class'], 1)

    def test_low_conf_reclassifies(self):
        r = self.record(1, 0.3, 3)
        out = PS.apply_config(r, 'parts', 'selective', tau=0.5)
        self.assertEqual(out[0]['class'], 3)

    def test_gate_blocks_other_classes(self):
        r = self.record(6, 0.1, 3)  # YOLO disse MAR (fora do trio): nunca reclassifica
        out = PS.apply_config(r, 'parts', 'selective', tau=0.9, gated=True)
        self.assertEqual(out[0]['class'], 6)


class SelectionTests(unittest.TestCase):
    def test_parity_failure(self):
        # simula regra: nenhum config com mAP >= yolo-0.005
        results = {'yolo': {'mAP50': 0.8, 'species_f1': 0.5},
                   'x': {'mAP50': 0.79, 'species_f1': 0.9}}
        yolo_map = results['yolo']['mAP50']
        best = None
        for name, m in results.items():
            if name == 'yolo' or m['mAP50'] < yolo_map - PS.PARITY:
                continue
            if best is None or m['species_f1'] > results[best]['species_f1']:
                best = name
        self.assertIsNone(best)  # 0.79 < 0.795 → rejeitado

    def test_selection_prefers_f1_within_parity(self):
        results = {'yolo': {'mAP50': 0.8, 'species_f1': 0.5},
                   'a': {'mAP50': 0.796, 'species_f1': 0.6},
                   'b': {'mAP50': 0.799, 'species_f1': 0.55}}
        best = None
        for name, m in results.items():
            if name == 'yolo' or m['mAP50'] < results['yolo']['mAP50'] - PS.PARITY:
                continue
            if best is None or m['species_f1'] > results[best]['species_f1']:
                best = name
        self.assertEqual(best, 'a')


class GuardTests(unittest.TestCase):
    def test_record_count_guard(self):
        with tempfile.TemporaryDirectory(dir='/raid/user_marcospaulo') as d:
            Path(d, 'x.json').write_text(json.dumps({'record': {'gt': [], 'pipelines': {}}}))
            with self.assertRaises(ValueError):
                PS.load_records(d)

    def test_slurm_guard(self):
        with self.assertRaises(ValueError):
            PS.guard()


if __name__ == '__main__':
    unittest.main()
