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


if __name__ == "__main__":
    unittest.main()
