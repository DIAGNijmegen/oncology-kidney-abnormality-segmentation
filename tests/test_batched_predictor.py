"""
Unit tests for BatchedNNUNetPredictor — verifies that batching sliding-window
patches and TTA mirror variants into single forward passes produces the same
numerical results as nnU-Net's stock one-patch-at-a-time / one-mirror-at-a-time
implementation. Batching should only change throughput, never results.

Requires nnunetv2 to be installed (it is not part of the smoke-test baseline
environment), so this whole module is skipped if it's unavailable.

Run with:
    pytest tests/test_batched_predictor.py -v
"""

import pytest

pytest.importorskip("nnunetv2")

from types import SimpleNamespace

import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from kidney_abnormality_segmentation.segmentation.batched_predictor import (
    BatchedNNUNetPredictor,
)


def _make_fake_predictor(sw_batch_size=4, use_mirroring=False, allowed_mirroring_axes=None,
                          num_segmentation_heads=2, patch_size=(4, 4, 4), network=None):
    """Build a BatchedNNUNetPredictor without going through __init__/model loading,
    setting only the attributes _internal_predict_sliding_window_return_logits and
    _internal_maybe_mirror_and_predict actually touch."""
    p = object.__new__(BatchedNNUNetPredictor)
    p.sw_batch_size = sw_batch_size
    p.network = network if network is not None else (lambda x: torch.cat([x, x * 2.0], dim=1))
    p.label_manager = SimpleNamespace(num_segmentation_heads=num_segmentation_heads)
    p.configuration_manager = SimpleNamespace(patch_size=patch_size)
    p.use_gaussian = True
    p.use_mirroring = use_mirroring
    p.allowed_mirroring_axes = allowed_mirroring_axes
    p.device = torch.device("cpu")
    p.verbose = False
    p.allow_tqdm = False
    p.perform_everything_on_device = True
    p.tile_step_size = 0.5
    return p


class TestPatchBatching:
    """sw_batch_size must change throughput, not the resulting logits."""

    def test_batched_matches_unbatched(self):
        data = torch.randn(1, 12, 12, 12)
        p_ref = _make_fake_predictor(sw_batch_size=1)
        slicers = p_ref._internal_get_sliding_window_slicers(data.shape[1:])
        assert len(slicers) > 1, "test setup should exercise multiple overlapping patches"

        logits_bs1 = p_ref._internal_predict_sliding_window_return_logits(data, slicers)

        p_bs3 = _make_fake_predictor(sw_batch_size=3)
        logits_bs3 = p_bs3._internal_predict_sliding_window_return_logits(data, slicers)

        p_bsN = _make_fake_predictor(sw_batch_size=len(slicers))
        logits_bsN = p_bsN._internal_predict_sliding_window_return_logits(data, slicers)

        assert torch.allclose(logits_bs1, logits_bs3, atol=1e-6)
        assert torch.allclose(logits_bs1, logits_bsN, atol=1e-6)
        assert logits_bs1.shape == (2, 12, 12, 12)

    def test_sw_batch_size_zero_rejected(self):
        with pytest.raises(ValueError):
            BatchedNNUNetPredictor.__init__(object.__new__(BatchedNNUNetPredictor), sw_batch_size=0)


class TestMirrorBatching:
    """Stacking all TTA mirror variants into one forward pass must match nnU-Net's
    stock sequential mirror loop exactly."""

    def test_matches_upstream_sequential_mirroring(self):
        # Position-sensitive fake network so a wrong unflip axis would be caught
        # (a flip-invariant network, e.g. plain scaling, wouldn't catch that bug).
        offset = torch.randn(1, 2, 6, 6, 6)

        def fake_network(x):
            b = x.shape[0]
            out = torch.cat([x, x * 2.0], dim=1)
            return out + offset.expand(b, -1, -1, -1, -1)

        x = torch.randn(3, 1, 6, 6, 6)

        p = _make_fake_predictor(use_mirroring=True, allowed_mirroring_axes=(0, 1, 2),
                                  network=fake_network)

        reference = nnUNetPredictor._internal_maybe_mirror_and_predict(p, x)
        batched = BatchedNNUNetPredictor._internal_maybe_mirror_and_predict(p, x)

        assert torch.allclose(reference, batched, atol=1e-5)

    def test_no_mirroring_falls_back_to_plain_forward(self):
        p = _make_fake_predictor(use_mirroring=False)
        x = torch.randn(2, 1, 4, 4, 4)
        assert torch.equal(p._internal_maybe_mirror_and_predict(x), p.network(x))


class TestPatchAndMirrorBatchingCombined:
    """End-to-end: sliding-window patch batching + TTA mirror batching together."""

    def test_combined_matches_reference(self):
        offset = torch.randn(1, 2, 4, 4, 4)

        def fake_network(x):
            b = x.shape[0]
            out = torch.cat([x, x * 2.0], dim=1)
            return out + offset.expand(b, -1, -1, -1, -1)

        data = torch.randn(1, 10, 10, 10)

        p_ref = _make_fake_predictor(sw_batch_size=1, use_mirroring=True,
                                      allowed_mirroring_axes=(0, 1, 2), network=fake_network)
        slicers = p_ref._internal_get_sliding_window_slicers(data.shape[1:])

        logits_ref = p_ref._internal_predict_sliding_window_return_logits(data, slicers)

        p_batched = _make_fake_predictor(sw_batch_size=4, use_mirroring=True,
                                          allowed_mirroring_axes=(0, 1, 2), network=fake_network)
        logits_batched = p_batched._internal_predict_sliding_window_return_logits(data, slicers)

        assert torch.allclose(logits_ref, logits_batched, atol=1e-5)
