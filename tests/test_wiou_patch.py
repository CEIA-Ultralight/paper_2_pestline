"""Testes CPU do Wise-IoU v3 (modo wiou do nwd_patch)."""

import os
import unittest

import torch

import nwd_patch
from nwd_patch import apply_nwd_patch, wiou_v3_loss


class TestWIoU(unittest.TestCase):
    def setUp(self):
        self._env = {k: os.environ.get(k) for k in ("FLYDET_NWD_MODE", "FLYDET_NWD_SCOPE")}
        self._state = (nwd_patch._PATCHED, nwd_patch.NWD_MODE, nwd_patch.BboxLoss.forward,
                       nwd_patch.TaskAlignedAssigner.iou_calculation)
        nwd_patch._WIOU_STATE["mean"] = 1.0

    def tearDown(self):
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        (nwd_patch._PATCHED, nwd_patch.NWD_MODE, nwd_patch.BboxLoss.forward,
         nwd_patch.TaskAlignedAssigner.iou_calculation) = self._state

    def test_identical_boxes_zero_loss(self):
        b = torch.tensor([[100.0, 100.0, 140.0, 140.0]])
        self.assertAlmostEqual(wiou_v3_loss(b, b).item(), 0.0, places=5)

    def test_finite_and_shape_and_grad(self):
        p = (torch.rand(9, 4) * 50 + torch.tensor([0, 0, 60, 60.0])).requires_grad_(True)
        t = torch.rand(9, 4) * 50 + torch.tensor([5, 5, 65, 65.0])
        out = wiou_v3_loss(p, t)
        self.assertEqual(tuple(out.shape), (9, 1))
        self.assertTrue(torch.isfinite(out).all())
        out.sum().backward()
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_worse_box_costs_more(self):
        t = torch.tensor([[100.0, 100.0, 140.0, 140.0]])
        near = wiou_v3_loss(t + 2.0, t).item()
        far = wiou_v3_loss(t + 15.0, t).item()
        self.assertGreater(far, near)

    def test_apply_patch_wiou_scope_loss(self):
        os.environ["FLYDET_NWD_MODE"] = "wiou"
        nwd_patch._PATCHED = False
        cfg = apply_nwd_patch()
        self.assertEqual(cfg["nwd_mode"], "wiou")
        self.assertEqual(cfg["nwd_scope"], "loss")


if __name__ == "__main__":
    unittest.main()
