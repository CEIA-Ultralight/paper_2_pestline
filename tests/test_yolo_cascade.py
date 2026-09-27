"""Run with stdlib unittest; no torch/numpy imports outside explicit Slurm tests."""
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1] / 'rq2_cascade/eval_yolo_cascade.py'
SPEC = importlib.util.spec_from_file_location('eval_yolo_cascade', SOURCE)
cascade = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cascade)


def prediction(cls=0, score=.9, box=(10., 10., 20., 20.)):
    return {'class': cls, 'confidence': score, 'box': list(box), 'old_class': cls,
            'yolo_confidence': score, 'probabilities': None, 'fallback': None}


def record(gt, predictions):
    return {'gt': gt, 'pipelines': {name: predictions for name in cascade.PIPELINES}}


class GeometryAndContracts(unittest.TestCase):
    def test_guard_before_heavy_imports(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'SLURM_JOB_ID'):
                cascade.allocation_guard()
        for env in ({'SLURM_JOB_ID': '1', 'SLURM_JOB_PARTITION': 'h100n3', 'APPTAINER_NAME': 'test'},
                    {'SLURM_JOB_ID': '1', 'SLURM_JOB_PARTITION': 'h100n2'}):
            with patch.dict(os.environ, env, clear=True), self.assertRaises(ValueError):
                cascade.allocation_guard()

    def test_required_out_and_defaults(self):
        args = cascade.build_parser().parse_args(['--out', '/raid/user_marcospaulo/eval'])
        self.assertTrue(args.detector.endswith('E0_yolo26m_baseline/runs/yolo26m.pt/weights/best.pt'))
        self.assertEqual(cascade.PREDICT['conf'], .001)
        self.assertFalse(args.no_wandb)
        self.assertEqual(len(cascade.PIPELINES), 7)

    def test_fixed_class_remapping(self):
        self.assertEqual(cascade.REMAP, (4, 6, 2, 0, 3, 1, 5))
        cascade.class_names(dict(enumerate(cascade.YOLO_CLASSES)))
        with self.assertRaises(ValueError):
            cascade.class_names(cascade.CROP_CLASSES)
        for i, name in enumerate(cascade.CROP_CLASSES):
            probs = [0.]*7
            probs[i] = 1.
            p = cascade.relabel(prediction(), probs)
            self.assertEqual(cascade.YOLO_CLASSES[p['class']], name)
            self.assertEqual(p['probabilities'][p['class']], 1.)

    def test_padding_clipping_truncation_and_tiny(self):
        self.assertEqual(cascade.padded_box((10.9, 20.9, 14.9, 24.9)), (7, 17, 17, 27))
        self.assertEqual(cascade.padded_box((0, 0, 2, 2)), (0, 0, 3, 3))
        self.assertEqual(cascade.padded_box((1918, 1078, 1920, 1080)), (1916, 1076, 1920, 1080))
        self.assertEqual(cascade.padded_box((10, 10, 11, 11)), (9, 9, 11, 11))
        for box in ((1, 1, 1, 2), (5, 5, 4, 6), (-5, -5, -4, -4), (1.01, 1.01, 1.02, 1.02)):
            self.assertIsNone(cascade.padded_box(box))
        with self.assertRaises(ValueError):
            cascade.padded_box((0, 0, float('nan'), 4))

    def test_invalid_crop_preserves_original_under_both_policies(self):
        old = prediction(cls=6, score=.73, box=(1, 2, 1, 3))
        for policy in cascade.POLICIES:
            result = cascade.policy([cascade.relabel(old, None)], policy)
            self.assertEqual(len(result), 1)
            for key in ('box', 'class', 'confidence', 'old_class'):
                self.assertEqual(result[0][key], old[key])
            self.assertEqual(result[0]['fallback'], 'degenerate_crop')

    def test_label_only_and_product_ranking_class_swap(self):
        a = cascade.relabel(prediction(0, .9), [.1, .1, .1, .1, .1, .4, .1])
        b = cascade.relabel(prediction(1, .8), [0., 0., 0., .95, 0., .05, 0.])
        self.assertEqual((a['class'], b['class']), (1, 0))
        primary, secondary = cascade.policy([a, b], 'label_only'), cascade.policy([a, b], 'product')
        self.assertGreater(primary[0]['confidence'], primary[1]['confidence'])
        self.assertLess(secondary[0]['confidence'], secondary[1]['confidence'])
        self.assertEqual([p['box'] for p in primary], [p['box'] for p in secondary])
        with self.assertRaises(ValueError):
            cascade.relabel(prediction(), [float('nan')]*7)

    def test_normalized_gt_validation_no_silent_skip(self):
        gt = cascade.parse_gt('0 .5 .5 .1 .2\n6 .5 .5 .001 .001\n')
        self.assertEqual(len(gt), 2)
        self.assertEqual(gt[0]['box'], [864., 432., 1056., 648.])
        self.assertEqual(cascade.parse_gt(''), [])
        for line in ('8 .5 .5 .1 .1', '0 960 540 10 10', '0 .5 .5 0 .1', '0 nan .5 .1 .1',
                     '0 .5 .5 .1', '0 .01 .5 .2 .2', '0 .5 .5 .1 .1 extra', '1.5 .5 .5 .1 .1'):
            with self.subTest(line=line), self.assertRaises(ValueError):
                cascade.parse_gt(line)

    def test_matching_duplicate_predictions_and_gt(self):
        pairs = [(0, 0, .9), (0, 1, .8), (1, 0, .7)]
        self.assertEqual(cascade.greedy(pairs, .5), [(0, 0, .9)])
        # np.unique prediction IDs reorders matches before the unique-GT step.
        self.assertEqual(cascade.greedy([(0, 1, .9), (0, 0, .8)], .5), [(0, 0, .8)])
        self.assertEqual(cascade.greedy([], .5), [])
        self.assertEqual(cascade.greedy([(0, 0, .499)], .5), [])
        self.assertEqual(cascade.overlap([0, 0, 0, 1], [0, 0, 0, 1]), 0.)
        self.assertEqual(cascade.greedy([(0, 0, .5)], .5), [(0, 0, .5)])

    def test_confusion_rows_gt_columns_pred_background(self):
        gt = [{'class': 0, 'box': [10, 10, 20, 20]}, {'class': 6, 'box': [30, 30, 40, 40]}]
        predictions = [prediction(1, .25), prediction(5, .9, [50, 50, 60, 60]), prediction(6, .249, [30, 30, 40, 40])]
        matrix = cascade.confusion(gt, predictions)
        self.assertEqual(matrix[0][1], 1)
        self.assertEqual(matrix[6][7], 1)
        self.assertEqual(matrix[7][5], 1)
        self.assertEqual(sum(map(sum, matrix)), 3)
        self.assertEqual(cascade.confusion([], [prediction(score=0)]), [[0]*8 for _ in range(8)])
        self.assertEqual(cascade.candidates(gt[:1], predictions[:1], class_aware=True), [])

    def test_record_validation_source_hash_and_probabilities(self):
        entry = {'gt': []}
        old = prediction()
        r = {'input': entry, 'gt': [], 'fingerprint': 'test', 'seconds': dict.fromkeys(('detector', *cascade.HEADS), 1.),
                         'probability_class_order': list(cascade.YOLO_CLASSES),
             'pipelines': {'yolo': [old]}}
        for head in cascade.HEADS:
            for policy in cascade.POLICIES:
                r['pipelines'][f'{head}_{policy}'] = cascade.policy([cascade.relabel(old, [0., 0., 0., 1., 0., 0., 0.])], policy)
        cascade.validate_record(r, entry, 'test')
        with self.assertRaises(ValueError):
            cascade.validate_record(r, entry, 'changed-source')
        r['pipelines']['baseline_product'][0]['confidence'] = .1
        with self.assertRaises(ValueError):
            cascade.validate_record(r, entry, 'test')


HEAVY_ALLOWED = (os.environ.get('FLYDET_CASCADE_METRIC_TESTS') == '1' and
                 bool(os.environ.get('SLURM_JOB_ID')) and os.environ.get('SLURM_JOB_PARTITION') == 'h100n2' and
                 bool(os.environ.get('APPTAINER_CONTAINER') or os.environ.get('APPTAINER_NAME')))


@unittest.skipUnless(HEAVY_ALLOWED, 'Actual Ultralytics tests opt-in only inside h100n2 Apptainer CUDA allocation')
class ActualUltralyticsMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cascade.allocation_guard()
        # Opt-in tests inherit an already configured allocation; never let SDKs
        # create caches under home, even when only importing metric utilities.
        for key in ('XDG_CACHE_HOME', 'TORCH_HOME', 'YOLO_CONFIG_DIR', 'MPLCONFIGDIR'):
            if not os.environ.get(key):
                raise unittest.SkipTest(f'{key} must be configured under RAID')
            cascade.raid_path(os.environ[key])
        import torch
        if not torch.cuda.is_available():
            raise unittest.SkipTest('Requires CUDA allocation; no CPU fallback')
        import numpy as np
        from ultralytics.utils.metrics import ap_per_class
        cls.np, cls.ap = np, staticmethod(ap_per_class)

    def test_perfect_fixture_ap_near_one_all_thresholds(self):
        predictions = [prediction(i, .9, (30*i, 0, 30*i+10, 10)) for i in range(7)]
        gt = [{'class': p['class'], 'box': p['box']} for p in predictions]
        result = cascade.metrics([record(gt, predictions)], self.np, self.ap)
        for m in result['pipelines'].values():
            self.assertGreater(m['mAP50'], .99)
            self.assertGreater(m['mAP50_95'], .99)
            self.assertEqual(m['present_class_ids'], list(range(7)))
            self.assertEqual(m['species_accuracy_all_gt_including_misses'], 1.)
        self.assertTrue(all(c['recall'] == 1. for c in result['localization_ceiling_conf_001_class_agnostic']))

    def test_confidence_sorting_changes_ap(self):
        gt = [{'class': 0, 'box': [10, 10, 20, 20]}]
        good = [prediction(score=.9), prediction(score=.1, box=(100, 100, 110, 110))]
        bad = [prediction(score=.1), prediction(score=.9, box=(100, 100, 110, 110))]
        a = cascade.metrics([record(gt, good)], self.np, self.ap)['pipelines']['yolo']['mAP50']
        b = cascade.metrics([record(gt, bad)], self.np, self.ap)['pipelines']['yolo']['mAP50']
        self.assertGreater(a, b)
        # AP must sort by confidence internally, not by arrival order.
        c = cascade.metrics([record(gt, list(reversed(good)))], self.np, self.ap)['pipelines']['yolo']['mAP50']
        self.assertEqual(a, c)

    def test_absent_classes_misses_wrong_class_zero_predictions(self):
        gt = [{'class': 0, 'box': [10, 10, 20, 20]}]
        for predictions in ([], [prediction(1)], [prediction(0, 0.)]):
            m = cascade.metrics([record(gt, predictions)], self.np, self.ap)['pipelines']['yolo']
            self.assertEqual(m['present_class_ids'], [0])
            self.assertIsNone(m['per_class']['MAR']['AP50'])
            self.assertEqual(m['species_accuracy_all_gt_including_misses'], 0.)
        empty = cascade.metrics([record([], [])], self.np, self.ap)['pipelines']['yolo']
        self.assertEqual(empty['mAP50'], 0.)
        self.assertEqual(empty['present_class_ids'], [])

    def test_matching_matches_numpy_unique_order_untied(self):
        np = self.np
        pairs = [(0, 1, .9), (0, 0, .8), (1, 0, .7), (1, 2, .6)]
        matches = np.asarray(pairs)
        matches = matches[matches[:, 2].argsort()[::-1]]
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        self.assertEqual(cascade.greedy(pairs, .5), [tuple(row) for row in matches.tolist()])

    def test_matching_matches_installed_ultralytics_validator(self):
        import torch
        from ultralytics.engine.validator import BaseValidator
        validator = BaseValidator.__new__(BaseValidator)
        validator.iouv = torch.tensor(cascade.THRESHOLDS)
        iou = torch.tensor([[.91, .81, .1], [.71, .2, .61]])
        predicted = torch.tensor([0, 0, 0])
        target = torch.tensor([0, 0])
        expected = validator.match_predictions(predicted, target, iou).cpu().numpy()
        actual = self.np.zeros((3, 10), dtype=bool)
        pairs = [(g, p, float(iou[g, p])) for g in range(2) for p in range(3)]
        for t, threshold in enumerate(cascade.THRESHOLDS):
            for _, p, _ in cascade.greedy(pairs, threshold):
                actual[p, t] = True
        self.np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()