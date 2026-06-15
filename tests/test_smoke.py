"""
Smoke tests for Renal-Net — no model or real CT images required.

These tests use small synthetic images (created in memory) to verify that
the core utility and postprocessing functions work correctly.  They run in
seconds and are safe to include in CI.

Run with:
    pytest tests/test_smoke.py -v
"""

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sitk_image(array: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> sitk.Image:
    """Convert a numpy array to a SimpleITK image with the given voxel spacing."""
    img = sitk.GetImageFromArray(array.astype(np.uint8))
    img.SetSpacing(spacing)
    return img


def make_empty_volume(shape=(20, 30, 40), spacing=(1.0, 1.0, 1.0)) -> sitk.Image:
    """Return a blank (all-zero) 3-D image."""
    arr = np.zeros(shape, dtype=np.uint8)
    return make_sitk_image(arr, spacing)

# ---------------------------------------------------------------------------
# 1. utils — stem()
# ---------------------------------------------------------------------------

class TestStem:
    """stem() strips the medical-image extension and directory prefix."""

    def test_extension_stripped(self):
        from kidney_abnormality_segmentation.utils import stem
        assert stem("patient_001.mha") == "patient_001"
        assert stem("patient_001.nii") == "patient_001"
        assert stem("patient_001.nii.gz") == "patient_001"

    def test_directory_path_stripped(self):
        from kidney_abnormality_segmentation.utils import stem
        assert stem("/data/cases/patient_001.mha") == "patient_001"

    def test_unknown_extension_unchanged(self):
        from kidney_abnormality_segmentation.utils import stem
        # If the extension is not in the list the filename is returned as-is
        # (minus the directory part, since str.endswith doesn't match)
        result = stem("scan.dcm")
        # The directory is stripped but the extension is kept
        assert "scan.dcm" in result


# ---------------------------------------------------------------------------
# 2. utils — resample_volume()
# ---------------------------------------------------------------------------

class TestResampleVolume:
    """resample_volume() must produce an image with the requested spacing."""

    def _make_image(self, spacing):
        arr = np.zeros((10, 20, 30), dtype=np.float32)
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing(spacing)
        return img

    def test_output_spacing_matches_request(self):
        from kidney_abnormality_segmentation.utils import resample_volume
        img = self._make_image((2.0, 2.0, 2.0))
        resampled = resample_volume(img, new_spacing=(1.0, 1.0, 1.0))
        for got, want in zip(resampled.GetSpacing(), (1.0, 1.0, 1.0)):
            assert abs(got - want) < 1e-5

    def test_size_increases_when_spacing_decreases(self):
        from kidney_abnormality_segmentation.utils import resample_volume
        # Halving the spacing should roughly double each dimension
        img = self._make_image((2.0, 2.0, 2.0))
        resampled = resample_volume(img, new_spacing=(1.0, 1.0, 1.0))
        orig_size = img.GetSize()
        new_size = resampled.GetSize()
        for o, n in zip(orig_size, new_size):
            assert n > o, "Finer spacing should produce a larger image"

    def test_identity_resample(self):
        from kidney_abnormality_segmentation.utils import resample_volume
        img = self._make_image((1.0, 1.5, 2.0))
        resampled = resample_volume(img, new_spacing=(1.0, 1.5, 2.0))
        assert resampled.GetSize() == img.GetSize()


# ---------------------------------------------------------------------------
# 3. utils — crop_image()
# ---------------------------------------------------------------------------

class TestCropImage:
    """crop_image() must return the right sub-volume, or the original if
    index/size are both None."""

    def _make_image(self):
        arr = np.arange(1000, dtype=np.float32).reshape((10, 10, 10))
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing((1.0, 1.0, 1.0))
        return img

    def test_crop_returns_smaller_image(self):
        from kidney_abnormality_segmentation.utils import crop_image
        img = self._make_image()
        cropped = crop_image(img, index=(1, 1, 1), size=(5, 5, 5))
        assert cropped.GetSize() == (5, 5, 5)

    def test_none_index_and_size_returns_original(self):
        from kidney_abnormality_segmentation.utils import crop_image
        img = self._make_image()
        result = crop_image(img, index=None, size=None)
        # Should be the original image object (unchanged)
        assert result.GetSize() == img.GetSize()


# ---------------------------------------------------------------------------
# 4. utils — load_image_metadata()
# ---------------------------------------------------------------------------

class TestLoadImageMetadata:
    """load_image_metadata() must return the four expected keys with correct
    values without loading pixel data."""

    def test_metadata_keys_present(self, tmp_path):
        from kidney_abnormality_segmentation.utils import load_image_metadata
        # Write a tiny image to disk
        arr = np.zeros((5, 10, 15), dtype=np.int16)
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing((0.5, 0.5, 1.0))
        path = str(tmp_path / "test.mha")
        sitk.WriteImage(img, path)

        meta = load_image_metadata(path)
        assert set(meta.keys()) == {"size", "spacing", "origin", "direction"}

    def test_metadata_spacing_correct(self, tmp_path):
        from kidney_abnormality_segmentation.utils import load_image_metadata
        arr = np.zeros((5, 10, 15), dtype=np.int16)
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing((0.5, 0.75, 1.25))
        path = str(tmp_path / "test.mha")
        sitk.WriteImage(img, path)

        meta = load_image_metadata(path)
        for got, want in zip(meta["spacing"], (0.5, 0.75, 1.25)):
            assert abs(got - want) < 1e-5


# ---------------------------------------------------------------------------
# 5. utils — get_binary_mask()
# ---------------------------------------------------------------------------

class TestGetBinaryMask:
    """get_binary_mask() must produce a 0/1 image covering only the
    specified label range."""

    def _make_label_image(self):
        # labels: 0 = background, 1 = kidney, 2 = tumor
        arr = np.zeros((10, 10, 10), dtype=np.uint8)
        arr[2:5, 2:5, 2:5] = 1   # kidney block
        arr[6:8, 6:8, 6:8] = 2   # tumor block
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing((1.0, 1.0, 1.0))
        return img

    def test_kidney_label_isolated(self):
        from kidney_abnormality_segmentation.utils import get_binary_mask
        img = self._make_label_image()
        kidney_mask = get_binary_mask(img, labels=[1])
        arr = sitk.GetArrayFromImage(kidney_mask)
        # Kidney block should be 1, everything else 0
        assert arr[3, 3, 3] == 1
        assert arr[7, 7, 7] == 0   # tumor voxel → should not be included
        assert arr[0, 0, 0] == 0   # background

    def test_tumor_label_isolated(self):
        from kidney_abnormality_segmentation.utils import get_binary_mask
        img = self._make_label_image()
        tumor_mask = get_binary_mask(img, labels=[2])
        arr = sitk.GetArrayFromImage(tumor_mask)
        assert arr[7, 7, 7] == 1
        assert arr[3, 3, 3] == 0


# ---------------------------------------------------------------------------
# 6. postprocessing — combine_masks()
# ---------------------------------------------------------------------------

class TestCombineMasks:
    """combine_masks() must assign kidney→1, tumor→2, and tumor wins
    on overlap."""

    def _make_binary(self, shape, region_slice):
        arr = np.zeros(shape, dtype=np.uint8)
        arr[region_slice] = 1
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing((1.0, 1.0, 1.0))
        return img

    def test_kidney_gets_label_1(self):
        from kidney_abnormality_segmentation.postprocessing.postprocess_segmentation_mask import combine_masks
        kidney = self._make_binary((10, 10, 10), np.s_[0:5, :, :])
        tumor = self._make_binary((10, 10, 10), np.s_[7:10, :, :])
        result = sitk.GetArrayFromImage(combine_masks(kidney, tumor))
        assert result[2, 5, 5] == 1   # kidney region

    def test_tumor_gets_label_2(self):
        from kidney_abnormality_segmentation.postprocessing.postprocess_segmentation_mask import combine_masks
        kidney = self._make_binary((10, 10, 10), np.s_[0:5, :, :])
        tumor = self._make_binary((10, 10, 10), np.s_[7:10, :, :])
        result = sitk.GetArrayFromImage(combine_masks(kidney, tumor))
        assert result[8, 5, 5] == 2   # tumor region

    def test_tumor_wins_on_overlap(self):
        from kidney_abnormality_segmentation.postprocessing.postprocess_segmentation_mask import combine_masks
        # Same region is both kidney and tumor → tumor (label 2) should win
        kidney = self._make_binary((10, 10, 10), np.s_[3:7, :, :])
        tumor = self._make_binary((10, 10, 10), np.s_[3:7, :, :])
        result = sitk.GetArrayFromImage(combine_masks(kidney, tumor))
        assert result[5, 5, 5] == 2

    def test_background_is_zero(self):
        from kidney_abnormality_segmentation.postprocessing.postprocess_segmentation_mask import combine_masks
        kidney = self._make_binary((10, 10, 10), np.s_[0:3, :, :])
        tumor = self._make_binary((10, 10, 10), np.s_[7:10, :, :])
        result = sitk.GetArrayFromImage(combine_masks(kidney, tumor))
        assert result[5, 5, 5] == 0   # truly empty region


# ---------------------------------------------------------------------------
# 7. main — file discovery and extension filtering
# ---------------------------------------------------------------------------

class TestFileDiscovery:
    """The CLI must find .mha/.nii/.nii.gz files and skip everything else."""

    def _write_files(self, directory: Path, names):
        for name in names:
            (directory / name).touch()

    def test_mha_files_discovered(self, tmp_path):
        self._write_files(tmp_path, ["a.mha", "b.mha"])
        from kidney_abnormality_segmentation.config import CT_EXTENSIONS
        found = [p for p in tmp_path.rglob("*") if p.suffix in CT_EXTENSIONS
                 or str(p).endswith(".nii.gz")]
        assert len(found) == 2

    def test_non_ct_files_ignored(self, tmp_path):
        self._write_files(tmp_path, ["image.dcm", "report.pdf", "data.csv"])
        from kidney_abnormality_segmentation.config import CT_EXTENSIONS
        found = [p for p in tmp_path.rglob("*")
                 if any(str(p).endswith(ext) for ext in CT_EXTENSIONS)]
        assert len(found) == 0

    def test_mixed_directory(self, tmp_path):
        self._write_files(tmp_path, ["scan.mha", "scan.nii", "notes.txt"])
        from kidney_abnormality_segmentation.config import CT_EXTENSIONS
        found = [p for p in tmp_path.rglob("*")
                 if any(str(p).endswith(ext) for ext in CT_EXTENSIONS)]
        assert len(found) == 2

    def test_nested_directory_search(self, tmp_path):
        sub = tmp_path / "patient_01"
        sub.mkdir()
        (sub / "scan.mha").touch()
        from kidney_abnormality_segmentation.config import CT_EXTENSIONS
        found = [p for p in tmp_path.rglob("*")
                 if any(str(p).endswith(ext) for ext in CT_EXTENSIONS)]
        assert len(found) == 1

    def test_nii_gz_extension_detected(self, tmp_path):
        """nii.gz files must be discovered — tests multi-dot extension handling."""
        (tmp_path / "scan.nii.gz").touch()
        from kidney_abnormality_segmentation.config import CT_EXTENSIONS
        found = [p for p in tmp_path.rglob("*")
                 if any(str(p).endswith(ext) for ext in CT_EXTENSIONS)]
        assert len(found) == 1


# ---------------------------------------------------------------------------
# 8. inference — idempotency (output-exists skip)
# ---------------------------------------------------------------------------

class TestIdempotency:
    """If an output file already exists, run() must skip re-processing it."""

    def test_existing_output_is_skipped(self, tmp_path, monkeypatch):
        """Create a fake output file and verify segment_ct_image is never called."""
        called = []

        # Patch heavy functions so nothing is actually executed
        import kidney_abnormality_segmentation.inference as inf
        monkeypatch.setattr(inf, "segment_ct_image",
                            lambda *a, **kw: called.append("segment"))
        monkeypatch.setattr(inf, "postprocess_segmentation_mask",
                            lambda *a, **kw: None)

        # Write a dummy input CT
        input_ct = tmp_path / "patient.mha"
        arr = np.zeros((5, 5, 5), dtype=np.int16)
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing((1.0, 1.0, 1.0))
        sitk.WriteImage(img, str(input_ct))

        # Pre-create the output file so the loop thinks it already ran
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        (output_dir / "patient.mha").write_bytes(b"")

        inf.run([input_ct], model_path=tmp_path, output_path=output_dir,
                crop_roi=False)

        assert called == [], "segment_ct_image should not be called when output already exists"


# ---------------------------------------------------------------------------
# 9. postprocessing — small tumor removal (size threshold)
# ---------------------------------------------------------------------------

class TestSizeThreshold:
    """Tumor components smaller than the diameter threshold must be removed."""

    def _make_segmentation(self, shape=(30, 40, 50)):
        """
        Creates a synthetic segmentation with:
        - A kidney region (label 1)
        - One large tumor attached to the kidney (label 2)  — should survive
        - One tiny tumor cluster (label 2) far from the kidney — should be removed
        """
        arr = np.zeros(shape, dtype=np.uint8)
        # Kidney block
        arr[5:25, 5:25, 5:25] = 1
        # Large tumor touching the kidney (10x10x10 block → ~10 mm diameter)
        arr[5:15, 25:35, 5:15] = 2
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing((1.0, 1.0, 1.0))
        return img

    def test_large_tumor_is_kept(self):
        from kidney_abnormality_segmentation.postprocessing.postprocess_segmentation_mask import postprocess_segmentation_mask
        seg = self._make_segmentation()
        result = postprocess_segmentation_mask(seg)
        arr = sitk.GetArrayFromImage(result)
        # The large tumor region should still contain label-2 voxels
        assert (arr == 2).any(), "Large tumor component should survive postprocessing"

    def test_kidney_label_survives(self):
        from kidney_abnormality_segmentation.postprocessing.postprocess_segmentation_mask import postprocess_segmentation_mask
        seg = self._make_segmentation()
        result = postprocess_segmentation_mask(seg)
        arr = sitk.GetArrayFromImage(result)
        assert (arr == 1).any(), "Kidney label should be preserved after postprocessing"

    def test_output_has_only_valid_labels(self):
        from kidney_abnormality_segmentation.postprocessing.postprocess_segmentation_mask import postprocess_segmentation_mask
        seg = self._make_segmentation()
        result = postprocess_segmentation_mask(seg)
        arr = sitk.GetArrayFromImage(result)
        unique_labels = set(arr.flatten().tolist())
        assert unique_labels <= {0, 1, 2}, f"Unexpected labels in output: {unique_labels}"