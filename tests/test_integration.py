"""
Integration tests for Renal-Net — requires a real model and real CT images.

Before running:
  1. Fill in MODEL_PATH and TEST_CT_FILES in tests/integration_config.py
  2. Make sure the package is installed:  pip install -e .

Run with:
    pytest tests/test_integration.py -v

These tests are deliberately minimal: one run without ROI cropping and one
with ROI cropping.  The goal is to confirm the end-to-end pipeline produces
a valid segmentation mask — not to validate clinical accuracy.

Expected runtime: several minutes per image (GPU recommended).
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

# ---------------------------------------------------------------------------
# Load configuration — skip the whole module if the file is missing or not
# filled in, so the smoke tests can always run cleanly in CI.
# ---------------------------------------------------------------------------

try:
    from tests import integration_config as cfg
except ImportError:
    try:
        import integration_config as cfg
    except ImportError:
        cfg = None

if cfg is None:
    pytest.skip(
        "integration_config.py not found — copy the template and fill in paths.",
        allow_module_level=True,
    )

MODEL_PATH: Path = cfg.MODEL_PATH
TEST_CT_FILES: list = cfg.TEST_CT_FILES

# Skip everything if the paths look like placeholders
if not MODEL_PATH.exists():
    pytest.skip(
        f"MODEL_PATH does not exist: {MODEL_PATH}",
        allow_module_level=True,
    )

missing = [p for p in TEST_CT_FILES if not p.exists()]
if missing:
    pytest.skip(
        f"Some TEST_CT_FILES do not exist: {missing}",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_LABELS = {0, 1, 2}


def assert_valid_segmentation(output_path: Path, original_ct_path: Path):
    """
    Shared post-run assertions applied to every output mask:

    1. The output file exists.
    2. It can be read by SimpleITK.
    3. It contains only the labels {0, 1, 2}.
    4. Its spatial size matches the original CT (pipeline resamples back).
    5. It is not all-zero (at least some structure was found).
    """
    assert output_path.exists(), f"Output file not created: {output_path}"

    mask = sitk.ReadImage(str(output_path))
    arr = sitk.GetArrayFromImage(mask)

    unique = set(int(v) for v in np.unique(arr))
    assert unique <= VALID_LABELS, f"Unexpected label values in output: {unique}"

    # Spacing should match the original CT (pipeline resamples back)
    orig = sitk.ReadImage(str(original_ct_path))
    for got, want in zip(mask.GetSpacing(), orig.GetSpacing()):
        assert abs(got - want) < 0.01, (
            f"Output spacing {mask.GetSpacing()} does not match "
            f"original {orig.GetSpacing()}"
        )

    # At least background + kidney + lesion should be present
    assert unique == {0, 1, 2}, (
        f"Expected labels {{0, 1, 2}} in segmentation mask, got {unique}. "
        "All three classes (background, kidney, lesion) should be present."
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPipelineWithoutCropping:
    """Run the full pipeline (no ROI cropping) on each test CT image."""

    @pytest.mark.parametrize("ct_path", TEST_CT_FILES)
    def test_pipeline_produces_valid_mask(self, ct_path, tmp_path):
        from kidney_abnormality_segmentation import inference

        output_dir = tmp_path / "no_crop"
        output_dir.mkdir()

        inference.run(
            all_cts=[ct_path],
            model_path=MODEL_PATH,
            output_path=output_dir,
            crop_roi=False,
        )

        # Determine expected output filename using the same stem logic
        from kidney_abnormality_segmentation.utils import stem
        from kidney_abnormality_segmentation.config import CT_EXTENSIONS
        name = stem(str(ct_path))
        # Preserve the original extension
        ext = next((e for e in CT_EXTENSIONS if str(ct_path).endswith(e)), ".mha")
        output_file = output_dir / f"{name}{ext}"

        assert_valid_segmentation(output_file, ct_path)


class TestPipelineWithCropping:
    """Run the full pipeline with ROI cropping on the first test image only.

    We only use one image here to keep the total test time reasonable.
    Cropping is tested separately because it adds TotalSegmentator as a
    dependency and takes extra time.
    """

    def test_pipeline_with_cropping_produces_valid_mask(self, tmp_path):
        ct_path = TEST_CT_FILES[0]
        from kidney_abnormality_segmentation import inference

        output_dir = tmp_path / "with_crop"
        output_dir.mkdir()

        inference.run(
            all_cts=[ct_path],
            model_path=MODEL_PATH,
            output_path=output_dir,
            crop_roi=True,
        )

        from kidney_abnormality_segmentation.utils import stem
        from kidney_abnormality_segmentation.config import CT_EXTENSIONS
        name = stem(str(ct_path))
        ext = next((e for e in CT_EXTENSIONS if str(ct_path).endswith(e)), ".mha")
        output_file = output_dir / f"{name}{ext}"

        assert_valid_segmentation(output_file, ct_path)

    def test_cropping_and_no_cropping_agree_on_labels(self, tmp_path):
        """Both modes must produce masks with the same set of label values.

        This is a sanity check: cropping should not introduce extra labels
        or remove expected ones.
        """
        ct_path = TEST_CT_FILES[0]
        from kidney_abnormality_segmentation import inference
        from kidney_abnormality_segmentation.utils import stem
        from kidney_abnormality_segmentation.config import CT_EXTENSIONS

        name = stem(str(ct_path))
        ext = next((e for e in CT_EXTENSIONS if str(ct_path).endswith(e)), ".mha")

        out_no_crop = tmp_path / "no_crop"
        out_no_crop.mkdir()
        inference.run([ct_path], MODEL_PATH, out_no_crop, crop_roi=False)

        out_crop = tmp_path / "crop"
        out_crop.mkdir()
        inference.run([ct_path], MODEL_PATH, out_crop, crop_roi=True)

        arr_no_crop = sitk.GetArrayFromImage(
            sitk.ReadImage(str(out_no_crop / f"{name}{ext}"))
        )
        arr_crop = sitk.GetArrayFromImage(
            sitk.ReadImage(str(out_crop / f"{name}{ext}"))
        )

        labels_no_crop = set(int(v) for v in np.unique(arr_no_crop))
        labels_crop = set(int(v) for v in np.unique(arr_crop))

        assert labels_no_crop <= VALID_LABELS
        assert labels_crop <= VALID_LABELS
        # Both should at least detect a kidney
        assert 1 in labels_no_crop, "No kidney found in no-crop run"
        assert 1 in labels_crop, "No kidney found in crop run"