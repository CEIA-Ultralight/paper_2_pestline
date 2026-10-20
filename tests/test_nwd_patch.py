"""Testes CPU do patch NWD (experimento E10).

Rodam sem GPU e sem dataset. Validam:
  - propriedades matematicas do NWD (identidade, monotonicidade, range)
  - vantagem do NWD sobre CIoU em caixas tiny deslocadas
  - patch aplica com fail-fast e e idempotente
  - BboxLoss.forward e iou_calculation patchados produzem valores finitos
"""

import os
import unittest

import torch

os.environ.setdefault("FLYDET_NWD_W", "0.5")
os.environ.setdefault("FLYDET_NWD_C", "44.0")

# sys.path (rq3_tiny_object/patches) e configurado em tests/conftest.py
from nwd_patch import apply_nwd_patch, is_patched, nwd_similarity


class TestNWDSimilarity(unittest.TestCase):
    C = 44.0

    def test_identical_boxes_give_one(self):
        b = torch.tensor([[100.0, 100.0, 140.0, 140.0]])
        self.assertAlmostEqual(nwd_similarity(b, b, self.C).item(), 1.0, places=6)

    def test_similarity_decreases_with_distance(self):
        b1 = torch.tensor([[100.0, 100.0, 140.0, 140.0]])
        near = nwd_similarity(b1, b1 + 2.0, self.C).item()
        far = nwd_similarity(b1, b1 + 400.0, self.C).item()
        self.assertGreater(near, far)
        self.assertGreaterEqual(near, 0.0)
        self.assertLessEqual(near, 1.0)
        self.assertGreaterEqual(far, 0.0)

    def test_tinier_penalty_than_ciou_on_small_box_shift(self):
        # caixa 40x40 deslocada 2 px: CIoU pune muito mais que NWD
        from ultralytics.utils.metrics import bbox_iou

        b1 = torch.tensor([[100.0, 100.0, 140.0, 140.0]])
        b2 = torch.tensor([[102.0, 102.0, 142.0, 142.0]])
        ciou = bbox_iou(b1, b2, xywh=False, CIoU=True).item()
        nwd = nwd_similarity(b1, b2, self.C).item()
        self.assertGreater(nwd, ciou)

    def test_batch_shapes(self):
        b1 = torch.rand(7, 4).clamp(0, 100)
        b2 = torch.rand(7, 4).clamp(0, 100)
        out = nwd_similarity(b1, b2, self.C)
        self.assertEqual(tuple(out.shape), (7, 1))
        self.assertTrue(torch.isfinite(out).all())


class TestNWDPatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        apply_nwd_patch()

    def test_patch_applied_and_idempotent(self):
        self.assertTrue(is_patched())
        cfg = apply_nwd_patch()  # segunda chamada nao levanta
        self.assertTrue(cfg["nwd"])
        self.assertAlmostEqual(cfg["nwd_w"], 0.5)
        self.assertAlmostEqual(cfg["nwd_c"], 44.0)

    def test_bbox_loss_forward_finite(self):
        from ultralytics.utils.loss import BboxLoss

        bl = BboxLoss(reg_max=16)
        B, N, R = 1, 8, 16
        li, ld = bl(
            torch.rand(B, N, 4 * R),
            torch.tensor([[[0.0, 0.0, 40.0, 40.0]] * N]).float(),
            torch.rand(N, 2) * 100,
            torch.tensor([[[1.0, 1.0, 41.0, 41.0]] * N]).float(),
            torch.rand(B, N, 7),
            torch.tensor(5.0),
            torch.ones(B, N, dtype=torch.bool),
            torch.tensor([1920.0, 1920.0]),
            torch.tensor([8.0]),
        )
        self.assertTrue(torch.isfinite(li))
        self.assertTrue(torch.isfinite(ld))

    def test_assigner_overlap_in_unit_range(self):
        from ultralytics.utils.tal import TaskAlignedAssigner

        ta = TaskAlignedAssigner()
        gt = torch.tensor([[0.0, 0.0, 40.0, 40.0], [100.0, 100.0, 140.0, 140.0]])
        pd = torch.tensor([[2.0, 2.0, 42.0, 42.0], [400.0, 400.0, 440.0, 440.0]])
        ov = ta.iou_calculation(gt, pd)
        self.assertTrue(torch.isfinite(ov).all())
        self.assertTrue(((ov >= 0) & (ov <= 1)).all())
        # caixa deslocada 2 px deve ter overlap alto; caixa longe, ~zero
        self.assertGreater(ov[0].item(), 0.8)
        self.assertLess(ov[1].item(), 0.01)


class TestNWDScope(unittest.TestCase):
    """FLYDET_NWD_SCOPE=loss|assigner|both escolhe quais metodos recebem o patch."""

    def setUp(self):
        import nwd_patch
        from ultralytics.utils.loss import BboxLoss
        from ultralytics.utils.tal import TaskAlignedAssigner

        self.mod = nwd_patch
        self.BboxLoss = BboxLoss
        self.TAA = TaskAlignedAssigner
        # estado a restaurar (outros testes dependem do patch "both")
        self._saved = (BboxLoss.forward, TaskAlignedAssigner.iou_calculation, nwd_patch._PATCHED)
        self._env = os.environ.get("FLYDET_NWD_SCOPE")

    def tearDown(self):
        self.BboxLoss.forward, self.TAA.iou_calculation, self.mod._PATCHED = self._saved
        if self._env is None:
            os.environ.pop("FLYDET_NWD_SCOPE", None)
        else:
            os.environ["FLYDET_NWD_SCOPE"] = self._env

    def _reset(self, scope):
        # sentinelas com a MESMA assinatura do ultralytics 8.4.142 (o patch faz fail-fast nela)
        def sentinel_loss(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                          target_scores, target_scores_sum, fg_mask, imgsz, stride):
            return "orig_loss"

        def sentinel_iou(self, gt_bboxes, pd_bboxes):
            return "orig_iou"

        self.BboxLoss.forward = sentinel_loss
        self.TAA.iou_calculation = sentinel_iou
        self.mod._PATCHED = False
        if scope is None:
            os.environ.pop("FLYDET_NWD_SCOPE", None)
        else:
            os.environ["FLYDET_NWD_SCOPE"] = scope
        return sentinel_loss, sentinel_iou

    def test_default_scope_is_both(self):
        self._reset(None)
        cfg = self.mod.apply_nwd_patch()
        self.assertEqual(cfg["nwd_scope"], "both")
        self.assertIs(self.BboxLoss.forward, self.mod._patched_bbox_loss_forward)
        self.assertIs(self.TAA.iou_calculation, self.mod._patched_iou_calculation)

    def test_scope_loss_only(self):
        _, orig_iou = self._reset("loss")
        cfg = self.mod.apply_nwd_patch()
        self.assertEqual(cfg["nwd_scope"], "loss")
        self.assertIs(self.BboxLoss.forward, self.mod._patched_bbox_loss_forward)
        self.assertIs(self.TAA.iou_calculation, orig_iou)

    def test_scope_assigner_only(self):
        orig_loss, _ = self._reset("assigner")
        cfg = self.mod.apply_nwd_patch()
        self.assertEqual(cfg["nwd_scope"], "assigner")
        self.assertIs(self.BboxLoss.forward, orig_loss)
        self.assertIs(self.TAA.iou_calculation, self.mod._patched_iou_calculation)

    def test_invalid_scope_fails_fast(self):
        self._reset("everything")
        with self.assertRaises(RuntimeError):
            self.mod.apply_nwd_patch()


class TestNWDLossUnits(unittest.TestCase):
    """O ultralytics passa ao BboxLoss.forward caixas em grid (target/stride); o NWD da loss
    precisa ser calculado em pixels (FLYDET_NWD_LOSS_UNITS=px) para C=44 px valer."""

    def setUp(self):
        import nwd_patch

        self.mod = nwd_patch
        self._saved = (nwd_patch.NWD_LOSS_UNITS, nwd_patch.NWD_MODE, nwd_patch.NWD_W, nwd_patch.NWD_C)
        nwd_patch.NWD_MODE, nwd_patch.NWD_W, nwd_patch.NWD_C = "nwd", 0.5, 44.0
        self._env = os.environ.get("FLYDET_NWD_LOSS_UNITS")

    def tearDown(self):
        (self.mod.NWD_LOSS_UNITS, self.mod.NWD_MODE, self.mod.NWD_W, self.mod.NWD_C) = self._saved
        if self._env is None:
            os.environ.pop("FLYDET_NWD_LOSS_UNITS", None)
        else:
            os.environ["FLYDET_NWD_LOSS_UNITS"] = self._env

    def _loss_iou(self, units, stride_val):
        """Chama o forward patchado como o v8DetectionLoss: caixas em grid, stride (N,1)."""
        from ultralytics.utils.loss import BboxLoss

        from nwd_patch import _patched_bbox_loss_forward

        self.mod.NWD_LOSS_UNITS = units
        n = 4
        gt_px = torch.tensor([[100.0, 100.0, 140.0, 140.0]] * n)
        pd_px = gt_px + torch.tensor([4.0, 2.8, 4.0, 2.8])  # W2 ~ 4,9 px
        stride = torch.full((n, 1), stride_val)
        li, _ = _patched_bbox_loss_forward(
            BboxLoss(reg_max=1),  # YOLO26: reg_max=1 (sem DFL)
            torch.zeros(1, n, 4),
            (pd_px / stride).unsqueeze(0),
            torch.full((n, 2), 15.0),
            (gt_px / stride).unsqueeze(0),
            torch.ones(1, n, 7) / 7,
            torch.tensor(float(n)),
            torch.ones(1, n, dtype=torch.bool),
            torch.tensor([1920.0, 1920.0]),
            stride,
        )
        return li.item(), pd_px, gt_px

    def test_px_mode_matches_pixel_nwd_on_every_level(self):
        from ultralytics.utils.metrics import bbox_iou

        for s in (8.0, 16.0, 32.0):
            li, pd_px, gt_px = self._loss_iou("px", s)
            ciou = bbox_iou(pd_px, gt_px, xywh=False, CIoU=True)
            nwd = nwd_similarity(pd_px, gt_px, 44.0)
            expected = (1.0 - (0.5 * ciou + 0.5 * nwd)).mean().item()
            self.assertAlmostEqual(li, expected, places=5, msg=f"stride={s}")
            self.assertLess(nwd.mean().item(), 0.92)  # NWD em px nao satura

    def test_grid_mode_reproduces_legacy_saturation(self):
        for s in (8.0, 16.0, 32.0):
            li_grid, pd_px, gt_px = self._loss_iou("grid", s)
            nwd_grid = nwd_similarity(pd_px / s, gt_px / s, 44.0).mean().item()
            self.assertGreater(nwd_grid, 0.98, msg=f"stride={s}")
            li_px, _, _ = self._loss_iou("px", s)
            self.assertLess(li_grid, li_px)  # legado: loss menor (termo NWD ~constante)

    def test_invalid_loss_units_fails_fast(self):
        os.environ["FLYDET_NWD_LOSS_UNITS"] = "metros"
        with self.assertRaises(RuntimeError):
            self.mod.resolve_loss_units()

    def test_default_loss_units_is_legacy_grid(self):
        os.environ.pop("FLYDET_NWD_LOSS_UNITS", None)
        self.assertEqual(self.mod.resolve_loss_units(), "grid")


if __name__ == "__main__":
    unittest.main()
