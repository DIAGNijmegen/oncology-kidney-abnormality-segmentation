"""
Integration tests for Renal-Net — requires a real model and real CT images.

Before running:
  1. cp tests/integration_config.py.example tests/integration_config.py
  2. Fill in MODEL_PATH and TEST_CT_FILES (Test CT files *must* depict kidney and tumours, this is explicitly tested)
     (for example data at https://zenodo.org/records/20719257)
  3. pip install -e ".[dev]"

  make smoke   # fast, no model needed
  make full    # smoke + integration, figures embedded in report
"""

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

try:
    from tests import integration_config as cfg
except ImportError:
    try:
        import integration_config as cfg
    except ImportError:
        cfg = None

if cfg is None:
    raise RuntimeError(
        "integration_config.py not found. "
        "Copy integration_config.py.example to integration_config.py "
        "and fill in the required paths."
    )

MODEL_PATH: Path = cfg.MODEL_PATH
TEST_CT_FILES: list = cfg.TEST_CT_FILES
TEST_MRI_FILES: list = cfg.TEST_MRI_FILES

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model path does not exist: {MODEL_PATH}")

missing_ct = [p for p in TEST_CT_FILES if not p.is_file()]
missing_mri = [p for p in TEST_MRI_FILES if not p.is_file()]

if missing_ct or missing_mri:
    problems = []

    if missing_ct:
        problems.append(f"Missing CT files: {missing_ct}")

    if missing_mri:
        problems.append(f"Missing MRI files: {missing_mri}")

    raise FileNotFoundError("\n".join(problems))

FIGURES_DIR = Path("reports/figures")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_output_path(output_dir: Path, ct_path: Path) -> Path:
    from kidney_abnormality_segmentation.utils import stem
    name = stem(str(ct_path))
    ext = ".mha" if str(ct_path).endswith(".mha") else ".nii.gz"
    return output_dir / f"{name}{ext}"


def _assert_valid_segmentation(output_path: Path, original_ct_path: Path):
    assert output_path.exists(), f"Output file not created: {output_path}"
    mask = sitk.ReadImage(str(output_path))
    arr  = sitk.GetArrayFromImage(mask)
    unique = set(int(v) for v in np.unique(arr))
    unexpected = unique - {0, 1, 2}
    assert not unexpected, f"Unexpected label(s) in mask: {unexpected}"
    assert unique == {0, 1, 2}, (
        f"Expected labels {{0, 1, 2}}, got {unique}. "
        "All three classes (background, kidney, tumor) must be present."
    )
    orig = sitk.ReadImage(str(original_ct_path))
    for got, want in zip(mask.GetSpacing(), orig.GetSpacing()):
        assert abs(got - want) < 0.01, (
            f"Output spacing {mask.GetSpacing()} does not match original {orig.GetSpacing()}"
        )


def _save_and_attach(extra, ct_path: Path, output_path: Path, img_name: str):
    """Generate a slice figure, save it to reports/figures/, attach to report."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_png = FIGURES_DIR / f"{img_name}.png"
    try:
        from pytest_html import extras
        from report_utils import save_slice_figure
        save_slice_figure(str(ct_path), str(output_path), img_name, str(out_png))
        extra.append(extras.image(str(out_png)))
    except Exception as exc:
        print(f"[report] Could not generate figure {img_name}: {exc}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

CT_IDS = [f"ct{i+1}" for i in range(len(TEST_CT_FILES))]
MRI_IDS = [f"mri{i+1}" for i in range(len(TEST_MRI_FILES))]


@pytest.mark.parametrize("ct_path,img_id", zip(TEST_CT_FILES, CT_IDS), ids=CT_IDS)
def test_ct_pipeline_no_crop(ct_path, img_id, tmp_path, extra):
    """Full CT pipeline without ROI cropping — one run per image."""
    from kidney_abnormality_segmentation import inference

    output_dir = tmp_path / "no_crop"
    output_dir.mkdir()
    inference.run(all_images=[ct_path], model_path=MODEL_PATH,
                  output_path=output_dir, crop_roi=False, run_fast=True)

    output_file = _get_output_path(output_dir, ct_path)
    _save_and_attach(extra, ct_path, output_file, f"{img_id}_no_crop")
    _assert_valid_segmentation(output_file, ct_path)    


@pytest.mark.parametrize("ct_path,img_id", zip(TEST_CT_FILES, CT_IDS), ids=CT_IDS)
def test_ct_pipeline_crop(ct_path, img_id, tmp_path, extra):
    """Full CT pipeline with ROI cropping — one run per image."""
    from kidney_abnormality_segmentation import inference

    output_dir = tmp_path / "crop"
    output_dir.mkdir()
    inference.run(all_images=[ct_path], model_path=MODEL_PATH,
                  output_path=output_dir, crop_roi=True, run_fast=True)

    output_file = _get_output_path(output_dir, ct_path)
    _save_and_attach(extra, ct_path, output_file, f"{img_id}_crop")
    _assert_valid_segmentation(output_file, ct_path)


@pytest.mark.parametrize("mri_path,img_id", zip(TEST_MRI_FILES, MRI_IDS), ids=MRI_IDS)
def test_mri_pipeline_no_crop(mri_path, img_id, tmp_path, extra):
    """Full MRI pipeline without ROI cropping — one run per image."""
    from kidney_abnormality_segmentation import inference

    output_dir = tmp_path / "no_crop"
    output_dir.mkdir()
    inference.run(all_images=[mri_path], model_path=MODEL_PATH,
                  output_path=output_dir, crop_roi=False, run_fast=True, mri=True)

    output_file = _get_output_path(output_dir, mri_path)
    _save_and_attach(extra, mri_path, output_file, f"MRI_{img_id}_no_crop")
    _assert_valid_segmentation(output_file, mri_path)
    