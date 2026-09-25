"""Run with stdlib unittest; no torch/numpy/PIL imports outside explicit Slurm runs."""
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1] / 'scripts/extract_proposal_crops.py'
SPEC = importlib.util.spec_from_file_location('extract_proposal_crops', SOURCE)
prop = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prop)


class FakeImage:
    """Duck-typed PIL stand-in: size + crop recording, no Pillow required."""

    def __init__(self, size=(1920, 1080)):
        self.size = size
        self.crops = []

    def crop(self, box):
        self.crops.append(box)
        return ('crop', box)


class LabelsAndMatching(unittest.TestCase):
    def test_class_order_exact(self):
        self.assertEqual(prop.CLASSES, ('BG', 'INS', 'MAR', 'MC', 'MD', 'MF', 'MV', 'NOISE'))
        self.assertEqual(len(prop.CLASSES), 8)
        self.assertEqual(tuple(sorted(prop.CLASSES)), prop.CLASSES)
        self.assertEqual(prop.SPLITS, ('train', 'val'))

    def test_bg_labeling_and_proposal_order(self):
        gt = [{'class': 0, 'box': [0., 0., 100., 100.]}]  # YOLO class 0 = MD
        boxes = [[0., 0., 100., 100.], [1000., 1000., 1100., 1100.]]
        self.assertEqual(prop.assign_labels(gt, boxes), ['MD', 'BG'])
        # Without GT everything is BG; detector class is irrelevant (never an input).
        self.assertEqual(prop.assign_labels([], boxes), ['BG', 'BG'])
        # A second, worse proposal over the same GT stays BG (unique-GT greedy).
        dup = [[0., 0., 100., 100.], [10., 10., 90., 90.]]
        self.assertEqual(prop.assign_labels(gt, dup), ['MD', 'BG'])

    def test_iou_threshold_edges(self):
        gt = [{'class': 2, 'box': [0., 0., 10., 10.]}]  # YOLO class 2 = MC
        # Intersection 30, union 100 -> IoU exactly 0.3 -> matched.
        self.assertEqual(prop.cascade.overlap((0, 0, 10, 10), (0, 0, 6, 5)), .3)
        self.assertEqual(prop.assign_labels(gt, [[0., 0., 6., 5.]]), ['MC'])
        # Intersection 29.9, union 100 -> IoU 0.299 -> BG.
        self.assertLess(prop.cascade.overlap((0, 0, 10, 10), (0, 0, 2.99, 10)), .3)
        self.assertEqual(prop.assign_labels(gt, [[0., 0., 2.99, 10.]]), ['BG'])

    def test_invalid_boxes_rejected(self):
        with self.assertRaises(ValueError):
            prop.assign_labels([], [[0., 0., float('nan'), 1.]])
        with self.assertRaises(ValueError):
            prop.assign_labels([], [[0., 0., 1.]])


class CropPlanning(unittest.TestCase):
    def test_skip_tiny_counting_and_ordinal_gaps(self):
        image = FakeImage()
        written = {}

        def writer(crop, target):
            written[str(target)] = crop

        gt = [{'class': 3, 'box': [100., 100., 200., 200.]}]  # YOLO class 3 = MF
        boxes = [[100., 100., 200., 200.],   # index 0 -> MF
                 [10., 10., 12., 12.],       # index 1 -> padded 5x5 < 8px -> skipped
                 [500., 500., 560., 560.]]   # index 2 -> BG (no GT overlap)
        result = prop.process_image(image, boxes, gt, Path('/out/train'), 'img', writer)
        self.assertEqual(result['proposals'], 3)
        self.assertEqual(result['skipped_tiny'], 1)
        self.assertEqual(result['crops'], 2)
        self.assertEqual(result['counts']['MF'], 1)
        self.assertEqual(result['counts']['BG'], 1)
        self.assertEqual(set(result['counts']), set(prop.CLASSES))
        self.assertIn('/out/train/MF/img_000.jpg', written)
        self.assertIn('/out/train/BG/img_002.jpg', written)
        self.assertNotIn('/out/train/BG/img_001.jpg', written)
        self.assertEqual(len(image.crops), 2)

    def test_degenerate_and_clipped_boxes(self):
        self.assertIsNone(prop.plan_crop((5., 5., 4., 6.)))
        self.assertIsNone(prop.plan_crop((10., 10., 11., 11.)))
        # Clipped at the corner but >=8px: kept (unlike eval_yolo_cascade, tiny is dropped here).
        self.assertEqual(prop.plan_crop((1900., 1060., 1920., 1080.)), (1885, 1045, 1920, 1080))
        # Clipped but <8px tall after padding: skipped by the MIN_SIZE filter.
        self.assertIsNone(prop.plan_crop((1918., 1078., 1920., 1080.)))
        with self.assertRaises(ValueError):
            prop.plan_crop((0., 0., float('inf'), 4.))

    def test_run_image_uses_predict_fn_once(self):
        image = FakeImage()
        calls = []

        def predict(img):
            calls.append(img)
            return [[0., 0., 100., 100.]]

        result = prop.run_image(predict, image, [], Path('/out/val'), 's', lambda c, t: None)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result['counts']['BG'], 1)
        self.assertEqual(result['skipped_tiny'], 0)


class ResumeAndManifest(unittest.TestCase):
    def test_marker_validation(self):
        identity = {'image_sha256': 'a', 'label_sha256': 'b', 'detector_sha256': 'c'}
        result = {'proposals': 1, 'crops': 1, 'skipped_tiny': 0,
                  'counts': {name: 0 for name in prop.CLASSES}}
        marker = {'schema': 1, 'contract': 'H', 'identity': identity,
                  'status': 'complete', 'result': result}
        self.assertTrue(prop.marker_valid(marker, identity, 'H'))
        self.assertFalse(prop.marker_valid(marker, dict(identity, image_sha256='x'), 'H'))
        self.assertFalse(prop.marker_valid(marker, identity, 'OTHER'))
        self.assertFalse(prop.marker_valid(dict(marker, status='running'), identity, 'H'))
        self.assertFalse(prop.marker_valid(dict(marker, schema=2), identity, 'H'))
        bad_counts = dict(marker, result=dict(result, counts={'BG': 1}))
        self.assertFalse(prop.marker_valid(bad_counts, identity, 'H'))
        self.assertFalse(prop.marker_valid(None, identity, 'H'))
        self.assertFalse(prop.marker_valid({}, identity, 'H'))

    def test_merge_stats_totals(self):
        stats = prop.new_split_stats()
        result = {'proposals': 3, 'crops': 2, 'skipped_tiny': 1,
                  'counts': {name: 0 for name in prop.CLASSES}}
        result['counts']['MD'] = result['counts']['BG'] = 1
        prop.merge_stats(stats, 'a', result)
        prop.merge_stats(stats, 'b', dict(result, skipped_tiny=0))
        self.assertEqual(stats['images'], 2)
        self.assertEqual(stats['proposals'], 6)
        self.assertEqual(stats['crops'], 4)
        self.assertEqual(stats['skipped_tiny'], 1)
        self.assertEqual(stats['per_class']['MD'], 2)
        self.assertEqual(stats['per_class']['BG'], 2)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            prop.merge_stats(stats, 'a', result)

    def test_manifest_fields_and_warnings(self):
        stats = prop.new_split_stats()
        result = {'proposals': 2, 'crops': 1, 'skipped_tiny': 1,
                  'counts': {name: 0 for name in prop.CLASSES}}
        result['counts']['BG'] = 1
        prop.merge_stats(stats, 'img', result)
        context = {'status': 'completed', 'error': None, 'source_commit': 'abc123',
                   'script_sha256': 's', 'detector_path': '/d/best.pt', 'detector_sha256': 'd',
                   'data_path': '/data', 'data_yaml_sha256': 'y'}
        manifest = prop.build_manifest({'train': stats, 'val': prop.new_split_stats()}, context)
        for key in ('schema', 'created_utc', 'status', 'source_commit', 'script_sha256',
                    'detector', 'data', 'classes', 'thresholds', 'splits', 'warnings'):
            self.assertIn(key, manifest)
        self.assertEqual(manifest['source_commit'], 'abc123')
        self.assertEqual(manifest['detector']['sha256'], 'd')
        self.assertEqual(manifest['data']['data_yaml_sha256'], 'y')
        self.assertEqual(manifest['classes'], list(prop.CLASSES))
        self.assertEqual(manifest['thresholds']['iou_match'], .3)
        self.assertEqual(manifest['thresholds']['pad'], .75)
        self.assertEqual(manifest['thresholds']['min_size'], 8)
        self.assertEqual(manifest['thresholds']['predict']['conf'], .001)
        self.assertEqual(manifest['thresholds']['predict']['imgsz'], 1920)
        train = manifest['splits']['train']
        self.assertEqual(train['skipped_tiny'], 1)
        self.assertEqual(train['per_class']['BG'], 1)
        self.assertIn('img', train['per_image'])
        joined = ' '.join(manifest['warnings'])
        self.assertIn('selection', joined)  # E0 val-selection leakage note.
        self.assertIn('differ', joined)     # Species counting differs from GT crops.
        self.assertIn('test', joined)       # Test split never read.


class CliGuard(unittest.TestCase):
    def test_module_is_import_safe_without_heavy_deps(self):
        for name in ('torch', 'numpy', 'PIL', 'ultralytics', 'cv2'):
            self.assertNotIn(name, vars(prop))

    def test_guard_refuses_outside_slurm(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'SLURM_JOB_ID'):
                prop.main(['--out', '/raid/user_marcospaulo/tmp/propcrops'])
        for env in ({'SLURM_JOB_ID': '1', 'SLURM_JOB_PARTITION': 'h100n3', 'APPTAINER_NAME': 'x'},
                    {'SLURM_JOB_ID': '1', 'SLURM_JOB_PARTITION': 'h100n2'}):
            with patch.dict(os.environ, env, clear=True), self.assertRaises(ValueError):
                prop.main(['--out', '/raid/user_marcospaulo/tmp/propcrops'])

    def test_out_env_override(self):
        with patch.dict(os.environ, {'FLYDET_PROP_CROPS_ROOT': '/raid/user_marcospaulo/tmp/pc'}):
            args = prop.build_parser().parse_args([])
        self.assertEqual(args.out, '/raid/user_marcospaulo/tmp/pc')
        with patch.dict(os.environ, {}, clear=True):
            args = prop.build_parser().parse_args([])
        self.assertTrue(args.out.endswith('datasets/DS-F2_crops_proposals'))
        self.assertTrue(args.detector.endswith('E0_yolo26m_baseline/runs/yolo26m.pt/weights/best.pt'))
        self.assertTrue(args.data.endswith('datasets/DS-F2_v8.1.1'))


if __name__ == '__main__':
    unittest.main()
