# tests/report_utils.py
#
# Generates a three-panel CT + mask figure and saves it as a PNG.
# Called directly from integration tests.

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import SimpleITK as sitk

HU_MIN, HU_MAX = -160, 240   # soft-tissue window (W:400 L:40)

LABEL_COLORS = {
    1: (0.2, 0.6, 1.0, 0.35),  # kidney — blue
    2: (1.0, 0.2, 0.2, 0.50),  # tumor  — red
}


def _resample_to_isotropic(image: sitk.Image, interpolator) -> sitk.Image:
    """Resample image to 1mm isotropic spacing for display."""
    original_spacing = image.GetSpacing()
    original_size    = image.GetSize()
    new_spacing      = (1.0, 1.0, 1.0)
    new_size         = [
        int(round(original_size[i] * original_spacing[i] / new_spacing[i]))
        for i in range(3)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(interpolator)
    return resampler.Execute(image)


def save_slice_figure(ct_path: str, mask_path: str, title: str, out_png: str):
    """
    Save a three-panel (axial / coronal / sagittal) figure centered on the
    tumor center of mass to `out_png`.

    The CT is first resampled into the physical space of the mask (matching
    its origin, size, and spacing). This is essential for cropped masks which
    have a different origin than the original CT. Both are then resampled to
    1mm isotropic so coronal and sagittal views are not distorted.
    """
    ct   = sitk.ReadImage(ct_path)
    mask = sitk.ReadImage(mask_path)

    # Step 1: resample CT into mask space so they are pixel-perfectly aligned
    ct_in_mask_space = sitk.Resample(ct, mask, sitk.Transform(),
                                     sitk.sitkLinear, 0.0, ct.GetPixelID())

    # Step 2: resample both to 1mm isotropic for display
    ct   = _resample_to_isotropic(ct_in_mask_space, sitk.sitkLinear)
    mask = _resample_to_isotropic(mask,              sitk.sitkNearestNeighbor)

    ct_arr   = sitk.GetArrayFromImage(ct).astype(np.float32)
    mask_arr = sitk.GetArrayFromImage(mask).astype(np.uint8)

    # Center on tumor if present, fall back to kidney, then image center
    tumor_voxels  = np.argwhere(mask_arr == 2)
    kidney_voxels = np.argwhere(mask_arr == 1)
    if len(tumor_voxels) > 0:
        z, y, x = tumor_voxels.mean(axis=0).astype(int)
    elif len(kidney_voxels) > 0:
        z, y, x = kidney_voxels.mean(axis=0).astype(int)
    else:
        z, y, x = [s // 2 for s in mask_arr.shape]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(title, fontsize=11, fontweight="bold")

    for ax, (ct_slice, mask_slice, label) in zip(axes, [
        (ct_arr[z, :, :],  mask_arr[z, :, :],  f"Axial z={z}"),
        (ct_arr[:, y, :],  mask_arr[:, y, :],  f"Coronal y={y}"),
        (ct_arr[:, :, x],  mask_arr[:, :, x],  f"Sagittal x={x}"),
    ]):
        ct_display = np.clip(ct_slice, HU_MIN, HU_MAX)
        ct_display = (ct_display - HU_MIN) / (HU_MAX - HU_MIN)
        ax.imshow(ct_display, cmap="gray", origin="lower", interpolation="nearest")
        for lbl, color in LABEL_COLORS.items():
            overlay = np.zeros((*mask_slice.shape, 4), dtype=np.float32)
            overlay[mask_slice == lbl] = color
            ax.imshow(overlay, origin="lower", interpolation="nearest")
        ax.set_title(label, fontsize=9)
        ax.axis("off")

    patches = [mpatches.Patch(color=c[:3], label=n)
               for (lbl, c), n in zip(LABEL_COLORS.items(), ["Kidney (1)", "Tumor (2)"])]
    fig.legend(handles=patches, loc="lower center", ncol=2, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)