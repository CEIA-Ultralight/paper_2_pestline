"""CPU/stdlib-only campaign contract tests; never load models or submit jobs."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[1] / 'scripts/eval_test_campaign.py'
spec = importlib.util.spec_from_file_location('eval_test_campaign_under_test', SOURCE)
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


class MetricsTests(unittest.TestCase):
    def test_species_support_617_not_729_with_ins(self):
        matrix = [[0]*8 for _ in range(8)]
        # Real support regression: species 617; adding INS incorrectly gives729.
        for i, count in enumerate((100, 200, 150, 167, 112, 30, 20)):
            matrix[i][i] = count
        values = campaign.group_metrics(matrix, range(4))
        self.assertEqual(values['support'], 617)
        self.assertEqual(values['tp'], 617)
        self.assertEqual(values['indices'], [0, 1, 2, 3])
        self.assertEqual(campaign.group_metrics(matrix, range(5))['support'], 729)
        self.assertEqual(values['micro'], dict(precision=1., recall=1., f1=1.))

    def test_bg_other_classes_and_conditional_true_rows(self):
        m = [[0]*8 for _ in range(8)]
        m[0][0], m[0][1], m[0][4], m[0][7] = 10, 2, 3, 5
        m[1][1], m[1][0], m[4][0], m[7][0] = 20, 4, 6, 7
        r = campaign.group_metrics(m, range(4))
        self.assertEqual((r['tp'], r['fp'], r['fn'], r['support'], r['matched']), (30, 19, 14, 44, 39))
        self.assertAlmostEqual(r['micro']['precision'], 30/49)
        self.assertAlmostEqual(r['micro']['recall'], 30/44)
        self.assertAlmostEqual(r['micro']['f1'], 60/93)
        self.assertAlmostEqual(r['classification_accuracy_conditional_matched'], 30/39)
        self.assertAlmostEqual(r['macro']['recall'], (10/20+20/24)/4)

    def test_crop_seven_and_padded_eight_equivalent(self):
        matrix = [[0]*7 for _ in range(7)]
        matrix[0][0], matrix[0][4], matrix[4][0] = 3, 2, 1
        padded = [r+[0] for r in matrix] + [[0]*8]
        for indices in (range(4), range(7)):
            self.assertEqual(campaign.group_metrics(matrix, indices), campaign.group_metrics(padded, indices))
        rows = [dict(true=0, pred=0, probabilities=[1., 0, 0, 0, 0, 0, 0]),
                dict(true=0, pred=4, probabilities=[0, 0, 0, 0, 1., 0, 0]),
                dict(true=4, pred=4, probabilities=[0, 0, 0, 0, 1., 0, 0])]
        result = campaign.classification_metrics(rows)
        self.assertAlmostEqual(result['crop_accuracy_all7'], 2/3)
        self.assertEqual(result['crop_accuracy_species_true_rows'], .5)
        self.assertEqual(result['all7']['micro']['recall'], result['crop_accuracy_all7'])

    def test_zero_support_and_invalid_matrix(self):
        self.assertEqual(campaign.group_metrics([[0]*8 for _ in range(8)], range(4))['micro']['f1'], 0)
        for matrix, indices in (([[0]], [0]), ([[0]*8 for _ in range(8)], [4, 4]),
                                ([[0]*8 for _ in range(8)], [7])):
            with self.assertRaises(ValueError):
                campaign.group_metrics(matrix, indices)


class GeometryTests(unittest.TestCase):
    def test_padding_int_clip_no_small_discard(self):
        self.assertEqual(campaign.padded_box([10, 10, 20, 20], .15), (8, 8, 21, 21))
        self.assertEqual(campaign.padded_box([10, 10, 20, 20], .75), (2, 2, 27, 27))
        self.assertEqual(campaign.padded_box([0, 0, 1, 1], .15), (0, 0, 1, 1))
        self.assertEqual(campaign.padded_box([1918, 1078, 1920, 1080], .75), (1916, 1076, 1920, 1080))
        for box in ([1, 2, 1, 5], [-9, -9, -8, -8], [1921, 2, 1922, 3]):
            self.assertIsNone(campaign.padded_box(box, .15))

    def test_yolo_probabilities_not_alphabetic_mapping(self):
        base = [dict(box=[1, 2, 5, 9], **{'class': 4, 'confidence': .8, 'yolo_confidence': .8})]
        probs = [[.1, .6, .1, .05, .05, .05, .05]]
        result = campaign.score_policies(base, probs)
        self.assertEqual(result['label_only'][0]['class'], 1)  # MV, not alphabetic MAR.
        self.assertEqual(result['label_only'][0]['confidence'], .8)
        self.assertEqual(result['product'][0]['confidence'], .48)
        self.assertEqual(result['product'][0]['box'], base[0]['box'])
        self.assertEqual(base[0]['class'], 4)
        fallback = campaign.score_policies(base, [None])
        for rows in fallback.values():
            self.assertEqual(rows[0]['confidence'], .8)
            self.assertEqual(rows[0]['class'], 4)
            self.assertEqual(rows[0]['fallback'], 'invalid_crop')
        with self.assertRaises(ValueError):
            campaign.score_policies(base, [])

    def test_product_does_not_filter_low_scores_or_fp(self):
        base = [dict(box=[1, 2, 4, 5], **{'class': 0, 'confidence': .001, 'yolo_confidence': .001})] * 3
        result = campaign.score_policies(base, [[1/7]*7]*3)
        self.assertEqual(len(result['product']), 3)
        self.assertLess(result['product'][0]['confidence'], .001)
        result['product'][0]['box'] = [0, 0, 3, 3]
        with self.assertRaises(ValueError):
            campaign.validate_geometry(base, result)

    def test_grid_native_borders_and_coverage(self):
        boxes = campaign.tile_grid()
        self.assertEqual(len(boxes), 15)
        self.assertEqual(boxes[-1], (1408, 568, 1920, 1080))
        self.assertEqual(boxes, sorted(boxes, key=lambda b: (b[1], b[0])))
        for coordinate, extent in ((0, 1920), (1, 1080)):
            starts = sorted({b[coordinate] for b in boxes})
            self.assertEqual(starts[0], 0)
            self.assertEqual(starts[-1]+512, extent)
            self.assertTrue(all(b-a <= 384 for a, b in zip(starts, starts[1:])))
        self.assertEqual(campaign.tile_grid(100, 90), [(0, 0, 100, 90)])


class MembershipTests(unittest.TestCase):
    def setUp(self):
        self.entries = {'a_full_stem_123': {'image': '/test/a_full_stem_123.jpg', 'gt': [{'class': 0}, {'class': 4}]}}

    def test_exact_stem_ordinal_and_class(self):
        row = campaign.crop_membership('/test/MD/a_full_stem_123_000.jpg', self.entries)
        self.assertEqual(row, dict(image='/test/a_full_stem_123.jpg', gt_index=0, true=0))
        self.assertEqual(campaign.crop_membership('/test/INS/a_full_stem_123_001.jpg', self.entries)['true'], 4)

    def test_reject_train_prefix_bad_ordinal_and_relabel(self):
        for path in ('MD/a_full_stem_123_extra_000.jpg', 'MD/train_000.jpg', 'MD/a_full_stem_123_002.jpg',
                     'INS/a_full_stem_123_000.jpg', 'MD/a_full_stem_123_0.jpg', 'UNKNOWN/a_full_stem_123_000.jpg'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                campaign.crop_membership(path, self.entries)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        # Ephemeral outputs ONLY on RAID, never home/dataset; removed on cleanup.
        self.tmp = tempfile.TemporaryDirectory(prefix='campaign-stdlib-test-', dir=campaign.RAID)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.contract = dict(source='abc', detector='123', data='456', heads=['spec'],
                     tiles=[(0, 0, 512, 512)])
        self.store = campaign.Store(self.root, self.contract, False)

    def test_atomic_roundtrip_uuid_and_identity(self):
        self.store.put('stage/image.json', {'image': 'a', 'sha': 'x'}, {'predictions': [1, 2]})
        other = campaign.Store(self.root, self.contract, True)
        self.assertEqual(other.state['wandb_run_id'], self.store.state['wandb_run_id'])
        self.assertEqual(other.get('stage/image.json', {'image': 'a', 'sha': 'x'}), {'predictions': [1, 2]})
        for key in self.contract:
            changed = dict(self.contract, **{key: 'changed'})
            with self.assertRaises(ValueError):
                campaign.Store(self.root, changed, True)
        with self.assertRaises(ValueError):
            other.get('stage/image.json', {'image': 'a', 'sha': 'changed'})
        self.assertEqual(list((self.root / 'stage').glob('.*')), [])

    def test_corrupt_record_and_symlink_rejected(self):
        self.store.put('stage/a.json', 1, {'class': 2})
        path = self.root / 'stage/a.json'
        payload = json.loads(path.read_text())
        payload['value']['class'] = 5
        path.write_text(json.dumps(payload))
        with self.assertRaises(ValueError):
            self.store.get('stage/a.json', 1)
        (self.root / 'link.json').symlink_to(path)
        with self.assertRaises(ValueError):
            self.store.get('link.json', 1)
        with self.assertRaises(ValueError):
            self.store.path('../escape.json')

    def test_stage_resume_avoids_computation_and_checks_refs(self):
        engine = campaign.Campaign(self.store, [], {}, campaign.Stop())
        def compute(prefix):
            self.store.put(prefix+'/images/000.json', 'input', {'value': 1})
            return dict(population=1)
        first = engine.run_stage('detector/E0', 'detector', compute)
        other = Mock(side_effect=AssertionError('must not recompute'))
        self.assertEqual(engine.run_stage('detector/E0', 'detector', other), first)
        other.assert_not_called()
        path = next((self.root / 'stages').rglob('000.json'))
        path.write_text('{}')
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertIsNone(engine.run_stage('detector/E0', 'detector', other))
        self.assertEqual(self.store.state['stages']['detector/E0']['status'], 'failed')

    def test_stage_failure_does_not_block_next_stage(self):
        engine = campaign.Campaign(self.store, [], {}, campaign.Stop())
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertIsNone(engine.run_stage('bad', 'detector', Mock(side_effect=ValueError('bad stage'))))
        self.assertEqual(engine.run_stage('good', 'gt_crop', lambda _: {'population': 2}), {'population': 2})
        self.assertEqual(self.store.state['stages']['good']['status'], 'completed')

    def test_signal_commits_completed_batch_then_stops(self):
        stop = campaign.Stop()
        engine = campaign.Campaign(self.store, [], {}, stop)
        def compute():
            stop.signal(15, None)
            return {'complete': True}
        with self.assertRaises(InterruptedError):
            engine.checkpoint('batch.json', [1], compute)
        self.assertEqual(self.store.get('batch.json', [1]), {'complete': True})

    def test_partial_batch_is_never_committed(self):
        engine = campaign.Campaign(self.store, [], {}, campaign.Stop())
        with self.assertRaises(InterruptedError):
            engine.checkpoint('batch.json', [1], Mock(side_effect=InterruptedError('stop')))
        self.assertIsNone(self.store.get('batch.json', [1]))

    def test_tracking_failure_explicit_and_redacted(self):
        with patch.dict(os.environ, {'WANDB_API_KEY': 'unit-secret'}), contextlib.redirect_stderr(io.StringIO()):
            tracker = campaign.Tracking(self.store, True)
            tracker.fail('unit', RuntimeError('unit-secret upload failed'))
        self.assertEqual(self.store.state['wandb']['status'], 'failed')
        self.assertNotIn('unit-secret', (self.root / 'errors.json').read_text())
        self.assertIn('[REDACTED]', (self.root / 'errors.json').read_text())


    def test_tracking_never_logs_to_wrong_run(self):
        fake_run = Mock(id='old-training', entity='pestline', project='fly-species')
        wandb = Mock()
        wandb.init.return_value = fake_run
        with patch.dict(sys.modules, {'wandb': wandb}), \
                patch.dict(os.environ, {'WANDB_API_KEY': 'unit-secret', 'WANDB_MODE': 'online'}), \
                contextlib.redirect_stderr(io.StringIO()):
            tracker = campaign.Tracking(self.store, False)
            tracker.stage('test', {'metrics': {'accuracy': 1.}})
        self.assertIsNone(tracker.run)
        fake_run.log.assert_not_called()
        self.assertEqual(self.store.state['wandb']['status'], 'failed')

    def test_cached_diagnostic_checks_png(self):
        engine = campaign.Campaign(self.store, [], {}, campaign.Stop())
        def compute(prefix):
            self.store.path(prefix+'/cam.png').write_bytes(b'fake-png')
            return {'population': 1}
        engine.run_stage('cam/head', 'cam', compute)
        next((self.root / 'stages').rglob('cam.png')).write_bytes(b'changed')
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertIsNone(engine.run_stage('cam/head', 'cam', Mock()))

    def test_gt_batches_keep_sam_filename_and_resume(self):
        class Image:
            def __init__(self, path):
                self.filename = str(path)
            def load(self):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        crops = []
        for i in range(3):
            path = self.root / f'crop_{i}.jpg'
            path.write_bytes(b'placeholder; mocked PIL')
            crops.append(dict(campaign.fingerprint(path), image='native-test', gt_index=i, true=0))
        adapter = Mock()
        seen = []
        def predict(images):
            seen.extend(image.filename for image in images)
            return [[1., 0, 0, 0, 0, 0, 0] for _ in images]
        adapter.predict.side_effect = predict
        session = Mock(spec={'batch_size': 2, 'crop_root': 'sam'}, get=Mock(return_value=adapter))
        # Mock's `spec=` is reserved, so assign the adapter contract explicitly.
        session.spec = {'batch_size': 2, 'crop_root': 'sam'}
        runtime = {'Image': Mock(open=Image), 'torch': Mock()}
        engine = campaign.Campaign(self.store, [], runtime, campaign.Stop())
        result = engine.gt_crops(session, crops, 'gt-test')
        self.assertEqual(result['population'], 3)
        self.assertEqual(result['metrics']['crop_accuracy_all7'], 1.)
        self.assertEqual(result['timing']['n'], 2)
        self.assertEqual(seen, [crops[0]['path'], crops[1]['path'], crops[0]['path'], crops[1]['path'], crops[2]['path']])
        adapter.predict.reset_mock()
        self.assertEqual(engine.gt_crops(session, crops, 'gt-test'), result)
        adapter.predict.assert_not_called()


    def test_report_separates_diagnostics_population_and_accuracy(self):
        metrics = campaign.classification_metrics([
            dict(true=0, pred=0, probabilities=[1., 0, 0, 0, 0, 0, 0]),
            dict(true=0, pred=4, probabilities=[0, 0, 0, 0, 1., 0, 0])])
        self.store.state['stages'] = {
            'gt_crop/head1': dict(group='gt_crop', status='completed', metrics=metrics, population=2),
            'gt_crop/head2': dict(group='gt_crop', status='completed', metrics=metrics, population=2),
            'detector/E0': dict(group='detector', status='failed', error='detector unavailable'),
        }
        self.store.state['wandb'] = {'status': 'disabled_smoke'}
        contract = dict(selected_entries=[{'gt': [{'class': 0}, {'class': 0}]}], crops={'pad15': [0, 1]}, limit_images=1)
        campaign.write_report(self.store, contract)
        text = (self.root / 'report.md').read_text()
        summary = json.loads((self.root / 'summary.json').read_text())
        self.assertEqual(summary['evaluated_gt'], 2)
        self.assertIn('SMOKE', text)
        self.assertIn('não é uma avaliação cega', text)
        self.assertIn('E0 indisponível', text)
        table_start = text.index('## Diagnósticos')
        table_end = text.index('- Acurácia GT', table_start)
        table = text[table_start:table_end]
        self.assertIn('| gt_crop/head1 |', table)
        self.assertIn('| gt_crop/head2 |', table)
        self.assertIn('geral=0.5000', text)


class GuardTests(unittest.TestCase):
    def test_login_refused_before_runtime_or_writes(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(campaign, 'runtime') as runtime, \
                patch.object(Path, 'mkdir') as mkdir, contextlib.redirect_stderr(io.StringIO()):
            status = campaign.main(['--out', '/raid/user_marcospaulo/not-created'])
        self.assertEqual(status, 1)
        runtime.assert_not_called()
        mkdir.assert_not_called()

    def test_no_wandb_full_forbidden(self):
        args = campaign.parser().parse_args(['--out', '/raid/user_marcospaulo/not-created', '--no-wandb'])
        with patch.object(campaign.EV, 'allocation_guard'), patch.object(Path, 'mkdir') as mkdir:
            with self.assertRaisesRegex(ValueError, 'smoke-only'):
                campaign.prepare_environment(args)
        mkdir.assert_not_called()

    def test_wrong_partition_or_missing_apptainer_before_imports(self):
        for environment in (
                {'SLURM_JOB_ID': 'test', 'SLURM_JOB_PARTITION': 'h100n1', 'APPTAINER_NAME': 'test.sif'},
                {'SLURM_JOB_ID': 'test', 'SLURM_JOB_PARTITION': 'h100n2'}):
            with patch.dict(os.environ, environment, clear=True), patch.object(campaign, 'runtime') as runtime, \
                    patch.object(Path, 'mkdir') as mkdir, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(campaign.main(['--out', '/raid/user_marcospaulo/not-created']), 1)
            runtime.assert_not_called()
            mkdir.assert_not_called()

    def test_stdlib_import_does_not_import_heavy_modules(self):
        # Isolate from unrelated test suites that may already import numpy.
        code = ('import importlib.util,sys; '
                f's=importlib.util.spec_from_file_location("campaign", {str(SOURCE)!r}); '
                'm=importlib.util.module_from_spec(s); s.loader.exec_module(m); '
                'assert not ({"torch","torchvision","ultralytics","wandb","PIL","numpy"} & set(sys.modules))')
        subprocess.run([sys.executable, '-B', '-c', code], check=True, capture_output=True, text=True)

    def test_timing_percentiles(self):
        stats = campaign.statistics([3, 1, 2], 'seconds/image', 'warmup')
        self.assertEqual(stats['mean'], 2)
        self.assertEqual(stats['p50'], 2)
        self.assertAlmostEqual(stats['p95'], 2.9)
        self.assertIsNone(campaign.statistics([], 'batch', '')['p95'])


if __name__ == '__main__':
    unittest.main()