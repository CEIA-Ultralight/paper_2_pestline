"""Standard-library tests: no torch import, checkpoint deserialization or GPU.

Run with unittest discovery restricted to this filename and Python -B to avoid
bytecode writes. Real-RAID tests skip only when the campaign root is absent.
All mutation tests use in-memory copies/mocks; datasets/artifacts stay untouched.
"""

import ast
from contextlib import nullcontext
import copy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / 'rq2_cascade/test_campaign_heads.py'
IMPORT_SPEC = importlib.util.spec_from_file_location('campaign_heads_under_test', MODULE_PATH)
heads = importlib.util.module_from_spec(IMPORT_SPEC)
IMPORT_SPEC.loader.exec_module(heads)

DINOV2_IDS = {'dinov2-20260909/seed42', 'dinov2-20260909/seed84'}


def _dinov2_summary():
    """Minimal in-memory summary honoring the dinov2 architecture contract."""
    classes = list(heads.FOLDER_CLASSES)
    crop_root = str(heads.TRUSTED_ROOT / 'datasets/DS-F2_crops_pad75')
    return {
        'schema_version': 1,
        'export_complete': True,
        'exported_epoch': 12,
        'initial_checkpoint_fallback': False,
        'metadata': {
            'args': {'variant': 'dinov2', 'img_size': 384, 'data': crop_root,
                     'limit_train': None, 'limit_val': None},
            'preprocessing': {
                'input': 'pre-extracted pad75 RGB crops; caller guarantees padding',
                'dataset_files_modified': False,
                'resize': [384, 384], 'interpolation': 'bilinear', 'antialias': True,
                'normalization': copy.deepcopy(heads.NORMALIZATION),
                'val': 'resize + tensor + ImageNet normalization; no augmentation',
            },
            'architecture': {
                'variant': 'dinov2', 'backbone': 'dinov2_vitb14', 'pretrained': True,
                'stem_original_stride': None, 'stem_stride': None,
                'stem_weights_preserved': None, 'nominal_output_stride': 14,
                'patch_size': 14, 'embed_dim': 768,
            },
            'class_order': classes,
            'class_to_idx': {c: i for i, c in enumerate(classes)},
            'dataset_root': crop_root,
            'smoke': False,
        },
    }


class PureContractTests(unittest.TestCase):
    def test_module_scope_imports_are_stdlib_only(self):
        allowed = {'__future__', 'copy', 'gc', 'hashlib', 'json', 'os', 'pathlib',
                   're', 'shlex', 'types'}
        for node in ast.parse(MODULE_PATH.read_text()).body:
            if isinstance(node, ast.Import):
                self.assertTrue(all(n.name.split('.')[0] in allowed for n in node.names))
            elif isinstance(node, ast.ImportFrom):
                self.assertIn(node.module, allowed)

    def test_guard_runs_before_optional_imports(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch('builtins.__import__', side_effect=AssertionError('unexpected import')):
            with self.assertRaisesRegex(RuntimeError, 'SLURM_JOB_ID'):
                heads._runtime()
            with self.assertRaisesRegex(RuntimeError, 'SLURM_JOB_ID'):
                heads.load_head({})

    def test_partition_and_container_guards(self):
        cases = [({'SLURM_JOB_ID': '1'}, 'h100n2'),
                 ({'SLURM_JOB_ID': '1', 'SLURM_JOB_PARTITION': 'h100n3',
                   'APPTAINER_NAME': 'test.sif'}, 'h100n2'),
                 ({'SLURM_JOB_ID': '1', 'SLURM_JOB_PARTITION': 'h100n2'}, 'Apptainer')]
        for env, message in cases:
            with self.subTest(env=env), mock.patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(RuntimeError, message):
                    heads._guard()

    def test_guard_accepts_apptainer_markers_without_imports(self):
        for marker in ('APPTAINER_CONTAINER', 'APPTAINER_NAME'):
            env = {'SLURM_JOB_ID': '1', 'SLURM_JOB_PARTITION': 'h100n2', marker: 'test.sif'}
            with mock.patch.dict(os.environ, env, clear=True):
                heads._guard()  # Only tests environment validation; never calls _runtime.

    def test_runtime_requires_cuda_before_torchvision(self):
        fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with mock.patch.object(heads, '_guard'), \
                mock.patch('builtins.__import__', return_value=fake) as imports:
            with self.assertRaisesRegex(RuntimeError, 'CUDA is required'):
                heads._runtime()
        self.assertEqual([call.args[0] for call in imports.call_args_list], ['torch'])

    def test_exact_yolo_order(self):
        self.assertEqual(heads.YOLO_CLASSES, ('MD', 'MV', 'MC', 'MF', 'INS', 'NOISE', 'MAR'))
        self.assertEqual(heads._classes(heads.FOLDER_CLASSES), [3, 5, 2, 4, 0, 6, 1])
        self.assertEqual(heads._classes(heads.YOLO_CLASSES), list(range(7)))

    def test_invalid_classes(self):
        for classes in (None, [], ['MD'] * 7, list(heads.FOLDER_CLASSES)[:-1],
                        ['MD', 'MV', 'MC', 'MF', 'INS', 'NOISE', 'UNKNOWN']):
            with self.subTest(classes=classes), self.assertRaises(ValueError):
                heads._classes(classes)

    def test_dinov2_summary_contract_accepted(self):
        metadata = heads._validate_arch_summary(_dinov2_summary())
        self.assertEqual(metadata['architecture']['backbone'], 'dinov2_vitb14')
        self.assertEqual(metadata['args']['variant'], 'dinov2')
        self.assertEqual(metadata['class_order'], list(heads.FOLDER_CLASSES))

    def test_dinov2_summary_rejects_convnext_metadata_masquerade(self):
        # Full ConvNeXt metadata claiming the dinov2 variant.
        broken = _dinov2_summary()
        broken['metadata']['architecture'].update(
            backbone='torchvision.convnext_tiny', stem_stride=[4, 4],
            stem_original_stride=[4, 4], stem_weights_preserved=True,
            nominal_output_stride=32)
        with self.assertRaisesRegex(ValueError, 'Invalid architecture metadata'):
            heads._validate_arch_summary(broken)

    def test_convnext_variant_rejects_dinov2_metadata_masquerade(self):
        # DINOv2 metadata claiming a ConvNeXt variant.
        broken = _dinov2_summary()
        broken['metadata']['args']['variant'] = 'baseline'
        broken['metadata']['architecture']['variant'] = 'baseline'
        with self.assertRaisesRegex(ValueError, 'Invalid architecture metadata'):
            heads._validate_arch_summary(broken)

    def test_dinov2_summary_rejects_stem_and_stride_pollution(self):
        for key, value in (('stem_stride', [14, 14]),
                           ('stem_original_stride', [14, 14]),
                           ('stem_weights_preserved', True),
                           ('nominal_output_stride', 32),
                           ('backbone', 'torchvision.convnext_tiny')):
            broken = _dinov2_summary()
            broken['metadata']['architecture'][key] = value
            with self.subTest(key=key), \
                    self.assertRaisesRegex(ValueError, 'Invalid architecture metadata'):
                heads._validate_arch_summary(broken)

    def test_hash_is_canonical_and_covers_contract(self):
        first = {'id': 'a', 'pad': 0.75, 'size': 384, 'spec_hash': 'ignored'}
        second = {'size': 384, 'pad': 0.75, 'id': 'a'}
        self.assertEqual(heads.spec_hash(first), heads.spec_hash(second))
        second['pad'] = 0.15
        self.assertNotEqual(heads.spec_hash(first), heads.spec_hash(second))

    def test_metadata_normalization_preserves_values_and_input(self):
        metadata = {'stride': (4, 4), 'nested': [{'shape': (2, 3)}]}
        original = copy.deepcopy(metadata)
        canonical = heads._canonical_metadata(metadata)
        self.assertEqual(canonical, {'stride': [4, 4], 'nested': [{'shape': [2, 3]}]})
        self.assertEqual(metadata, original)
        for mismatch in ({'stride': [2, 2], 'nested': [{'shape': [2, 3]}]},
                         {'stride': [4, 4]},
                         {'stride': [4, 4], 'nested': [{'shape': [3, 2]}]}):
            self.assertNotEqual(canonical, heads._canonical_metadata(mismatch))

    def test_untrusted_and_relative_paths_rejected(self):
        for path in ('relative.pt', '/etc/passwd'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                heads._trusted(path)

    def test_changed_hash_rejected_before_artifact_access(self):
        with self.assertRaisesRegex(ValueError, 'Spec hash mismatch'):
            heads._validate_spec({'spec_hash': 'bad'})

    def test_ensemble_gradcam_explicitly_unsupported(self):
        adapter = heads._Head.__new__(heads._Head)
        adapter.spec = {'kind': 'ensemble'}
        adapter.closed = False
        with mock.patch.object(heads, '_guard'), \
                self.assertRaisesRegex(NotImplementedError, 'ensemble'):
            adapter.gradcam(object())

    def test_sam_rejects_anonymous_images_before_transform(self):
        adapter = heads._Head.__new__(heads._Head)
        adapter.supports_cascade = False
        with self.assertRaisesRegex(ValueError, 'filename'):
            adapter._image(object())

    def test_close_is_idempotent_and_clears_models(self):
        adapter = heads._Head.__new__(heads._Head)
        adapter.models = [object(), object()]
        adapter.closed = False
        adapter.torch = mock.Mock()
        adapter.close()
        adapter.close()
        self.assertEqual(adapter.models, [])
        self.assertTrue(adapter.closed)
        adapter.torch.cuda.empty_cache.assert_called_once_with()
        with mock.patch.object(heads, '_guard'), self.assertRaisesRegex(RuntimeError, 'closed'):
            adapter._ready()

    def test_predict_ensemble_arithmetic_views_resize_and_batching(self):
        # Tiny stdlib doubles exercise the real predict control flow, without
        # importing torch or doing tensor/GPU work. Probabilities are Python lists.
        class Probabilities:
            def __init__(self, rows):
                self.rows = rows

            def softmax(self, dim):
                self_dim = dim
                if self_dim != 1:
                    raise AssertionError('Wrong softmax dimension')
                return self

            def __add__(self, other):
                return Probabilities([[a + b for a, b in zip(x, y)]
                                      for x, y in zip(self.rows, other.rows)])

            def __truediv__(self, divisor):
                return Probabilities([[p / divisor for p in row] for row in self.rows])

            def __getitem__(self, key):
                rows, columns = key
                return Probabilities([[row[c] for c in columns] for row in self.rows[rows]])

            def cpu(self):
                return self

            def tolist(self):
                return self.rows

        class Input:
            def __init__(self, tag, count):
                self.tag, self.count = tag, count

            def to(self, **kwargs):
                return self

        member_probs = [[i / 28 for i in range(1, 8)], [0.1] * 6 + [0.4], [1 / 7] * 7]
        for tta in (False, True):
            with self.subTest(tta=tta):
                adapter = heads._Head.__new__(heads._Head)
                adapter.spec = {'kind': 'ensemble', 'size': 384, 'tta': tta,
                                'views': 6 if tta else 1,
                                'members': [{'size': s} for s in (224, 384, 384)]}
                adapter.batch_size = 2
                adapter.order = heads._classes(heads.FOLDER_CLASSES)
                adapter.closed = False
                adapter._image = lambda image: image
                resize = mock.Mock(side_effect=lambda x, **kw: Input('resize224', x.count))
                adapter.torch = SimpleNamespace(
                    inference_mode=nullcontext, autocast=lambda **kw: nullcontext(),
                    float32='float32', stack=lambda images: Input('base', len(images)),
                    nn=SimpleNamespace(functional=SimpleNamespace(interpolate=resize)),
                    flip=lambda x, dims: Input('h' if dims == [3] else 'v', x.count),
                    rot90=lambda x, k, dims: Input('r' + str(k), x.count),
                )
                adapter.models = [mock.Mock(side_effect=lambda x, row=row:
                                            Probabilities([row] * x.count)) for row in member_probs]
                with mock.patch.object(heads, '_guard'):
                    output = adapter.predict([object(), object(), object()])
                    self.assertEqual(adapter.predict([]), [])
                self.assertEqual(len(output), 3)
                expected = [sum(row[i] for row in member_probs) / 3 for i in adapter.order]
                for row in output:
                    self.assertAlmostEqual(sum(row), 1.0)
                    for actual, target in zip(row, expected):
                        self.assertAlmostEqual(actual, target)
                self.assertEqual(resize.call_count, 2)
                for call in resize.call_args_list:
                    self.assertEqual(call.kwargs, {'size': (224, 224), 'mode': 'bilinear',
                                                  'align_corners': False, 'antialias': False})
                for index, model in enumerate(adapter.models):
                    tags = ['resize224' if index == 0 else 'base']
                    if tta:
                        tags += ['h', 'v', 'r1', 'r2', 'r3']
                    self.assertEqual([c.args[0].tag for c in model.call_args_list], tags * 2)


@unittest.skipUnless(heads.RUNS_ROOT.is_dir(), 'Real RAID campaign is not mounted')
@unittest.skipUnless((heads.RUNS_ROOT / 'E3_swin').exists(),
                     'checkpoints reais do fly-det (flydet_runs/) ausentes nesta maquina')
class RealDiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Discovery is stdlib-only, reads bytes for hashes, never unpickles them.
        # allow_missing: the dinov2 campaign may still be training; its strict
        # behavior is covered by a dedicated test below.
        cls.specs = heads.discover_heads(allow_missing_campaigns=True)
        cls.by_id = {s['id']: s for s in cls.specs}
        cls.arch = [s for s in cls.specs if s['kind'] == 'architecture']
        cls.dinov2 = [s for s in cls.arch if s['id'] in DINOV2_IDS]

    def test_concrete_counts_and_all_ten_architectures(self):
        extra = len(self.dinov2)
        self.assertIn(extra, (0, 2))
        self.assertEqual(len(self.specs), 18 + extra)
        self.assertEqual(len(self.by_id), 18 + extra)
        self.assertEqual(sum(s['kind'] == 'legacy' for s in self.specs), 6)
        self.assertEqual(sum(s['kind'] == 'ensemble' for s in self.specs), 2)
        self.assertEqual(len(self.arch), 10 + extra)
        night = 'architecture-night-v1-20260907/crops/'
        follow = 'architecture-followup-v1-20260908/'
        expected = {night + v for v in ('arcface', 'baseline', 'hires', 'parts', 'supcon')}
        expected |= {follow + 'combos/' + v for v in ('hires_parts', 'hires_supcon', 'parts_supcon')}
        expected |= {follow + 'seed84/' + v for v in ('baseline', 'parts')}
        if extra:
            expected |= DINOV2_IDS
        self.assertEqual({s['id'] for s in self.arch}, expected)
        self.assertEqual(sum(s['seed'] == 84 for s in self.arch), 2 + extra // 2)

    def test_json_manifest_and_absolute_sources(self):
        self.assertEqual(json.loads(json.dumps(self.specs)), self.specs)
        for spec in self.specs:
            with self.subTest(head=spec['id']):
                self.assertEqual(spec['spec_hash'], heads.spec_hash(spec))
                for key in ('checkpoint', 'source', 'crop_root', 'summary'):
                    self.assertTrue(Path(spec[key]).is_absolute())
                    self.assertTrue(Path(spec[key]).exists())
                for record in spec['sources'] + spec['checkpoint_sources']:
                    self.assertRegex(record['sha256'], r'^[0-9a-f]{64}$')
                    self.assertEqual(Path(record['path']).stat().st_size, record['bytes'])
                self.assertEqual(spec['checkpoints'], [r['path'] for r in spec['checkpoint_sources']])

    def test_two_original_architecture_source_hashes(self):
        expected = {'ecda7f57a1bf3e88c0690e937054f659f006cf0766fee49188b7d7714f1c585b',
                    '5bab9c13ed721836513a682099e5217c55ba91911192f9e0f8901beebcd8be58'}
        convnext = [s for s in self.arch if s['id'] not in DINOV2_IDS]
        self.assertEqual({s['sources'][0]['sha256'] for s in convnext}, expected)
        for spec in self.arch:
            self.assertIn('/source/', spec['source'])
            self.assertEqual(Path(spec['source']).name, 'models.py')
            self.assertEqual(spec['sources'][0]['sha256'],
                             spec['metadata']['source']['file_sha256']['models.py'])

    def test_export_epochs_and_no_initial_fallback(self):
        expected = {'crops/arcface': 15, 'crops/baseline': 16, 'crops/hires': 9,
                    'crops/parts': 17, 'crops/supcon': 1, 'combos/hires_parts': 3,
                    'combos/hires_supcon': 4, 'combos/parts_supcon': 6,
                    'seed84/baseline': 7, 'seed84/parts': 3}
        for spec in self.arch:
            heads._validate_arch_summary(heads._json(spec['summary']))
            if spec['id'] in DINOV2_IDS:
                continue  # Exported epochs are known only once training finishes.
            self.assertEqual(spec['exported_epoch'], expected[spec['id'].split('/', 1)[1]])

    def test_batches_and_cascade_support(self):
        extra = len(self.dinov2)  # dinov2: batch 32, cascade-capable.
        self.assertEqual(sum(s['batch_size'] == 8 for s in self.specs), 3)
        self.assertEqual(sum(s['batch_size'] == 32 for s in self.specs), 15 + extra)
        self.assertEqual(sum(s['supports_cascade'] for s in self.specs), 17 + extra)
        sam = self.by_id['E4_convnext_sam']
        self.assertIsNone(sam['pad'])
        self.assertFalse(sam['supports_cascade'])
        self.assertTrue(sam['crop_root'].endswith('/DS-F2_crops_sam'))
        self.assertEqual(self.by_id['E3_swin']['pad'], 0.15)
        self.assertEqual(self.by_id['E3_swin']['size'], 224)
        self.assertEqual(self.by_id['E5_convnext_512']['size'], 512)

    def test_dinov2_specs_when_campaign_present(self):
        if not self.dinov2:
            self.skipTest('dinov2 campaign has not finished training yet')
        self.assertEqual({s['id'] for s in self.dinov2}, DINOV2_IDS)
        snapshot = str(heads.RUNS_ROOT / 'dinov2-20260909/source') + '/'
        for spec in self.dinov2:
            arch = spec['metadata']['architecture']
            self.assertEqual(arch['backbone'], 'dinov2_vitb14')
            self.assertEqual(arch['nominal_output_stride'], 14)
            for key in ('stem_stride', 'stem_original_stride', 'stem_weights_preserved'):
                self.assertIsNone(arch[key])
            self.assertEqual(spec['size'], 384)
            self.assertEqual(spec['batch_size'], 32)
            self.assertEqual(spec['pad'], 0.75)
            self.assertTrue(spec['supports_cascade'])
            self.assertEqual(spec['seed'], int(spec['id'].rsplit('seed', 1)[1]))
            self.assertTrue(spec['source'].startswith(snapshot))
            self.assertEqual(spec['sources'][0]['sha256'],
                             spec['metadata']['source']['file_sha256']['models.py'])

    def test_dinov2_missing_campaign_strict_vs_allow_missing(self):
        root = heads.RUNS_ROOT / 'dinov2-20260909'
        complete = root.is_dir() and all((root / seed / 'summary.json').is_file()
                                         for seed in ('seed42', 'seed84'))
        if complete:
            self.assertEqual(len(heads.discover_heads()), 20)
            self.assertEqual(len(heads.discover_heads(allow_missing_campaigns=True)), 20)
        else:
            with self.assertRaisesRegex(ValueError, 'dinov2'):
                heads.discover_heads()
            specs = heads.discover_heads(allow_missing_campaigns=True)
            self.assertEqual(len(specs), 18)
            self.assertEqual({s['id'] for s in specs if s['kind'] == 'architecture'} &
                             DINOV2_IDS, set())

    def test_campaign_roots_filter_is_deterministic(self):
        night = heads.RUNS_ROOT / 'architecture-night-v1-20260907'
        specs = heads.discover_heads(campaign_roots=[night], allow_missing_campaigns=True)
        arch = [s['id'] for s in specs if s['kind'] == 'architecture']
        self.assertEqual(len(arch), 5)
        self.assertEqual(arch, sorted(arch))
        self.assertTrue(all(a.startswith('architecture-night-v1-20260907/') for a in arch))
        dinov2 = heads.RUNS_ROOT / 'dinov2-20260909'
        if not dinov2.is_dir():
            specs = heads.discover_heads(campaign_roots=[dinov2],
                                         allow_missing_campaigns=True)
            self.assertEqual(sum(s['kind'] == 'architecture' for s in specs), 0)
            with self.assertRaisesRegex(ValueError, 'dinov2'):
                heads.discover_heads(campaign_roots=[dinov2])

    def test_original_ensemble_members_and_base384(self):
        for name, views in (('E3_ensemble', 1), ('E3_ensemble_tta', 6)):
            spec = self.by_id[name]
            self.assertEqual([m['id'] for m in spec['members']],
                             ['E3_swin', 'E3b_convnext', 'E3c_focal'])
            self.assertEqual([m['size'] for m in spec['members']], [224, 384, 384])
            self.assertEqual(spec['size'], 384)
            self.assertEqual(spec['pad'], 0.75)
            self.assertEqual(spec['views'], views)
            self.assertEqual(spec['tta'], views == 6)
        self.assertNotEqual(self.by_id['E3_ensemble']['spec_hash'],
                            self.by_id['E3_ensemble_tta']['spec_hash'])

    def test_legacy_schema_without_checkpoint_deserialization(self):
        for spec in (s for s in self.specs if s['kind'] == 'legacy'):
            ck = {'model': {}, 'arch': spec['variant'], 'img_size': spec['size'],
                  'classes': spec['classes']}
            heads._validate_checkpoint(spec, ck)
            for key in ('model', 'arch', 'img_size', 'classes'):
                broken = copy.deepcopy(ck)
                del broken[key]
                if key == 'arch' and spec['id'] in ('E3_swin', 'E3b_convnext', 'E3c_focal'):
                    original = copy.deepcopy(broken)
                    heads._validate_checkpoint(spec, broken)
                    self.assertEqual(broken, original)
                    continue
                with self.subTest(head=spec['id'], key=key), self.assertRaises(ValueError):
                    heads._validate_checkpoint(spec, broken)

    def test_legacy_wrong_or_null_arch_rejected_for_every_id(self):
        for spec in (s for s in self.specs if s['kind'] == 'legacy'):
            for arch in (None, '', 'swin_t' if spec['variant'] != 'swin_t' else 'convnext_t'):
                ck = {'model': {}, 'arch': arch, 'img_size': spec['size'],
                      'classes': spec['classes']}
                with self.subTest(head=spec['id'], arch=arch), \
                        self.assertRaisesRegex(ValueError, 'Legacy checkpoint'):
                    heads._validate_checkpoint(spec, ck)

    def test_missing_arch_requires_explicit_id_variant_and_source_provenance(self):
        for name in ('E3_swin', 'E3b_convnext', 'E3c_focal'):
            spec = self.by_id[name]
            script = next(r['path'] for r in spec['sources']
                          if Path(r['path']).name == heads._LEGACY[name][0])
            self.assertEqual(heads._script_options(script)['--model'], [spec['variant']])
            self.assertEqual(heads._json(spec['summary'])['model'], spec['variant'])
            ck = {'model': {}, 'img_size': spec['size'], 'classes': spec['classes']}
            broken = copy.deepcopy(spec)
            broken['variant'] = 'convnext_s'
            with self.subTest(head=name), self.assertRaisesRegex(ValueError, 'Legacy checkpoint'):
                heads._validate_checkpoint(broken, ck)
            for source in (script, spec['summary']):
                def changed(path):
                    content = (heads._script_options(script) if path == script
                               else heads._json(spec['summary']))
                    return dict(content, **({'--model': ['convnext_s']} if path == script
                                            else {'model': 'convnext_s'}))

                target = '_script_options' if source == script else '_json'
                original = getattr(heads, target)
                altered = changed(source)
                with mock.patch.object(heads, target, side_effect=lambda path:
                                       altered if str(path) == source else original(path)), \
                        self.subTest(head=name, source=source), \
                        self.assertRaisesRegex(ValueError, 'Legacy (model mismatch|script changed)'):
                    heads.discover_heads()

    def test_missing_arch_still_loads_state_dict_strictly(self):
        for name in ('E3_swin', 'E3b_convnext', 'E3c_focal'):
            spec = self.by_id[name]
            ck = {'model': {}, 'img_size': spec['size'], 'classes': spec['classes']}
            original = copy.deepcopy(ck)
            model = mock.MagicMock()
            model.head.in_features = 768
            model.classifier[2].in_features = 768
            model.load_state_dict.side_effect = RuntimeError('state mismatch')
            factories = SimpleNamespace(swin_t=mock.Mock(return_value=model),
                                        convnext_tiny=mock.Mock(return_value=model),
                                        convnext_small=mock.Mock(return_value=model))
            torch = mock.Mock()
            torch.load.return_value = ck
            with self.subTest(head=name), self.assertRaisesRegex(RuntimeError, 'state mismatch'):
                heads._build_single(spec, torch, factories)
            model.load_state_dict.assert_called_once_with(ck['model'], strict=True)
            model.to.assert_not_called()
            self.assertEqual(ck, original)

    def test_arch_checkpoint_schema_and_metadata_fail_closed(self):
        for spec in self.arch:
            ck = {'model': {}, 'metadata': spec['metadata'], 'epoch': spec['exported_epoch'],
                  'is_initial_fallback': False}
            heads._validate_checkpoint(spec, ck)
            mutations = [('epoch', 0), ('is_initial_fallback', True), ('metadata', {}),
                         ('model', None)]
            for key, value in mutations:
                broken = copy.deepcopy(ck)
                broken[key] = value
                with self.subTest(head=spec['id'], key=key), self.assertRaises(ValueError):
                    heads._validate_checkpoint(spec, broken)

    def test_arch_checkpoint_tuple_metadata_and_real_mismatches(self):
        for spec in self.arch:
            metadata = copy.deepcopy(spec['metadata'])
            for key in ('stem_stride', 'stem_original_stride'):
                value = metadata['architecture'][key]
                if value is not None:  # dinov2 stem fields are null by contract
                    metadata['architecture'][key] = tuple(value)
            metadata['preprocessing']['resize'] = tuple(metadata['preprocessing']['resize'])
            ck = {'model': {}, 'metadata': metadata, 'epoch': spec['exported_epoch'],
                  'is_initial_fallback': False}
            original = copy.deepcopy(ck)
            heads._validate_checkpoint(spec, ck)
            self.assertEqual(ck, original)
            for section, key, value in (('architecture', 'stem_stride', (1, 1)),
                                        ('args', 'seed', -1),
                                        ('source', 'file_sha256', {})):
                broken = copy.deepcopy(ck)
                broken['metadata'][section][key] = value
                with self.subTest(head=spec['id'], section=section), \
                        self.assertRaisesRegex(ValueError, 'Checkpoint/summary metadata mismatch'):
                    heads._validate_checkpoint(spec, broken)
            broken_spec = copy.deepcopy(spec)
            broken_spec['metadata']['args']['seed'] = -1
            with self.assertRaisesRegex(ValueError, 'Checkpoint/summary metadata mismatch'):
                heads._validate_checkpoint(broken_spec, ck)

    def test_reconstructed_architecture_normalizes_checkpoint_and_rejects_mismatch(self):
        for spec in self.arch:
            ck = {'model': {}, 'metadata': copy.deepcopy(spec['metadata']),
                  'epoch': spec['exported_epoch'], 'is_initial_fallback': False}
            architecture = ck['metadata']['architecture']
            for key in ('stem_stride', 'stem_original_stride'):
                if architecture[key] is not None:  # dinov2 stem fields are null
                    architecture[key] = tuple(architecture[key])
            original = copy.deepcopy(ck)
            for mismatch in (False, True):
                model = mock.Mock()
                model.architecture_metadata = copy.deepcopy(architecture)
                model.architecture_metadata['pretrained'] = False
                if mismatch:
                    model.architecture_metadata['stem_stride'] = (1, 1)
                model.load_state_dict.side_effect = RuntimeError('strict load reached')
                module = SimpleNamespace(build_model=mock.Mock(return_value=model))
                torch = mock.Mock()
                torch.load.return_value = ck
                # Keep real source reading/SHA and checkpoint schema validation,
                # but replace snapshot execution and all tensor/model operations.
                with mock.patch.object(heads.types, 'ModuleType', return_value=module), \
                        mock.patch('builtins.exec'), self.subTest(head=spec['id'], mismatch=mismatch):
                    if mismatch:
                        with self.assertRaisesRegex(ValueError, 'Reconstructed architecture differs'):
                            heads._build_single(spec, torch, None)
                        model.load_state_dict.assert_not_called()
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'strict load reached'):
                            heads._build_single(spec, torch, None)
                        model.load_state_dict.assert_called_once_with(ck['model'], strict=True)
                self.assertEqual(ck, original)

    def test_summary_export_preprocessing_and_stride_rejections(self):
        summary = heads._json(self.arch[0]['summary'])
        for key, value in (('export_complete', False), ('exported_epoch', 0),
                           ('initial_checkpoint_fallback', True)):
            broken = copy.deepcopy(summary)
            broken[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                heads._validate_arch_summary(broken)
        for section, key, value in (('preprocessing', 'antialias', False),
                                    ('preprocessing', 'resize', [224, 224]),
                                    ('preprocessing', 'normalization', {}),
                                    ('architecture', 'stem_stride', [1, 1]),
                                    ('args', 'variant', 'invented')):
            broken = copy.deepcopy(summary)
            broken['metadata'][section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                heads._validate_arch_summary(broken)

    def test_changed_source_hash_rejected(self):
        spec = copy.deepcopy(self.by_id['E3_swin'])
        spec['sources'][0]['sha256'] = '0' * 64
        spec['spec_hash'] = heads.spec_hash(spec)
        # Skip checkpoint bytes here; we only need to simulate their valid digest.
        digests = {r['path']: r['sha256'] for r in spec['checkpoint_sources']}
        real_sha = heads._sha
        def digest(path):
            return digests[str(path)] if str(path) in digests else real_sha(path)
        with mock.patch.object(heads, '_sha', side_effect=digest), \
                self.assertRaisesRegex(ValueError, 'Artifact changed'):
            heads._validate_spec(spec)

    def test_sam_rejects_real_uncleaned_crop(self):
        adapter = heads._Head.__new__(heads._Head)
        adapter.spec = self.by_id['E4_convnext_sam']
        adapter.supports_cascade = False
        source = Path(self.by_id['E3_swin']['crop_root']) / 'test/MD'
        image = mock.Mock(filename=str(next(p for p in source.iterdir() if p.is_file())))
        with self.assertRaisesRegex(ValueError, 'cleaned GT TEST'):
            adapter._image(image)

    def test_existing_cleaned_sam_crop_is_accepted(self):
        adapter = heads._Head.__new__(heads._Head)
        adapter.spec = self.by_id['E4_convnext_sam']
        adapter.supports_cascade = False
        adapter.transform = mock.Mock(return_value='tensor-placeholder')
        source = Path(adapter.spec['crop_root']) / 'test/MD'
        image = mock.Mock(filename=str(next(p for p in source.iterdir() if p.is_file())))
        self.assertEqual(adapter._image(image), 'tensor-placeholder')
        image.convert.assert_called_once_with('RGB')


if __name__ == '__main__':
    unittest.main()