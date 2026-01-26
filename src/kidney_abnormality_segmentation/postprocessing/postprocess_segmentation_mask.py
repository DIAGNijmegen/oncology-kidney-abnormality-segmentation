#  Copyright 2022 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpy as np
import SimpleITK as sitk

from src.kidney_abnormality_segmentation.utils import get_binary_mask


def process_tumors(
    tumor_mask: sitk.Image,
    kidney_mask: sitk.Image,
    diameter_threshold: float = 3.0,
    physical_dilation_radius: float = 4.0,
    keep_large_tumors: bool = True,
) -> sitk.Image:
    """
    Retain only tumor regions connected to kidney regions.

    Parameters:
        tumor_mask (sitk.Image): Binary tumor mask.
        kidney_mask (sitk.Image): Binary kidney mask.
        diameter_threshold (float): Minimum size threshold for tumors.

    Returns:
        sitk.Image: Updated tumor mask with only connected tumors.
    """
    filter = sitk.ConnectedComponentImageFilter()
    # this returns a label in the mask for each connected component
    tumor_cc = filter.Execute(tumor_mask)
    num_components = filter.GetObjectCount()

    # Find tumor regions overlapping kidney regions
    connected_tumors = sitk.Image(tumor_mask.GetSize(), sitk.sitkUInt8)
    connected_tumors.CopyInformation(tumor_mask)

    # Dilate the kidney mask
    # Define the desired physical dilation radius in millimeters for each dimension
    physical_radius = [
        physical_dilation_radius,
        physical_dilation_radius,
        physical_dilation_radius,
    ]  # Adjust based on the physical spacing needs

    # Get the image spacing
    spacing = kidney_mask.GetSpacing()

    # Convert physical radius to voxel radius (unsigned integers)
    voxel_radius = [int(np.ceil(r / s)) for r, s in zip(physical_radius, spacing)]
    vector_radius = list(sitk.VectorUInt32(voxel_radius))

    # Dilate the kidney mask with anisotropic radius
    dilated_kidney_mask = sitk.BinaryDilate(kidney_mask, vector_radius)

    for tumor_label in range(1, num_components + 1):
        tumor_region = tumor_cc == tumor_label

        if keep_large_tumors:
            tumor_array = sitk.GetArrayFromImage(tumor_region)  # Convert to NumPy array

            # Compute voxel volume (spacing_x * spacing_y * spacing_z)
            voxel_volume = np.prod(spacing)  # Volume of a single voxel in mm³

            # Compute tumor volume in mm³
            tumor_voxel_count = tumor_array.sum()  # Number of tumor voxels
            tumor_volume_mm3 = tumor_voxel_count * voxel_volume

            # Keep large tumors (> 100,000 mm^3) even if they are not attached to a kidney region
            if tumor_volume_mm3 > 100000:
                connected_tumors = sitk.Or(connected_tumors, tumor_region)
                continue

        # Check overlap with any kidney region

        overlap = sitk.And(dilated_kidney_mask, tumor_region)

        # Use SimpleITK's StatisticsImageFilter to compute pixel statistics
        stats = sitk.StatisticsImageFilter()
        stats.Execute(overlap)

        # Check if the sum of pixel values is greater than zero
        if stats.GetSum() > 0:
            connected_tumors = sitk.Or(connected_tumors, tumor_region)

    # Apply size threshold if specified
    if diameter_threshold > 0:
        connected_tumors = apply_size_threshold(connected_tumors, diameter_threshold)

    return connected_tumors


# def apply_size_threshold(
#     tumor_mask: sitk.Image, diameter_threshold: float
# ) -> sitk.Image:
#     """
#     Removes tumor components below the specified size threshold.
#     """
#     relabel_filter = sitk.RelabelComponentImageFilter()
#     labeled_tumors = relabel_filter.Execute(tumor_mask)

#     stats_filter = sitk.LabelShapeStatisticsImageFilter()
#     stats_filter.Execute(labeled_tumors)

#     filtered_tumors = sitk.Image(tumor_mask.GetSize(), sitk.sitkUInt8)
#     filtered_tumors.CopyInformation(tumor_mask)

#     for label in stats_filter.GetLabels():
#         if (
#             stats_filter.GetEquivalentEllipsoidDiameter(label)[0] >= diameter_threshold
#             or stats_filter.GetEquivalentEllipsoidDiameter(label)[1]
#             >= diameter_threshold
#             or stats_filter.GetEquivalentEllipsoidDiameter(label)[2]
#             >= diameter_threshold
#         ):
#             filtered_tumors = sitk.Or(filtered_tumors, labeled_tumors == label)

#     return filtered_tumors


def apply_size_threshold(
    tumor_mask: sitk.Image, diameter_threshold: float
) -> sitk.Image:
    """
    Keeps connected components whose maximum axial-plane (XY) Feret diameter across slices
    is >= diameter_threshold (physical units, e.g. mm).

    Input: binary mask
    Output: binary mask
    """
    labeled = sitk.RelabelComponentImageFilter().Execute(tumor_mask)

    stats3d = sitk.LabelShapeStatisticsImageFilter()
    stats3d.Execute(labeled)

    out = sitk.Image(tumor_mask.GetSize(), sitk.sitkUInt8)
    out.CopyInformation(tumor_mask)

    size = list(labeled.GetSize())  # (x, y, z)

    stats2d = sitk.LabelShapeStatisticsImageFilter()
    stats2d.SetComputeFeretDiameter(True)

    for lab in stats3d.GetLabels():
        # Use bounding box to restrict which slices we check
        # BoundingBox is (x, y, z, size_x, size_y, size_z)
        bb = stats3d.GetBoundingBox(lab)
        z0 = int(bb[2])
        z1 = int(bb[2] + bb[5])  # exclusive

        best = 0.0
        for z in range(z0, z1):
            slice_lab = sitk.Extract(
                labeled, size=[size[0], size[1], 0], index=[0, 0, z]
            )
            slice_bin = sitk.Cast(slice_lab == lab, sitk.sitkUInt8)

            # If label exists on this slice, it will be the only foreground (value 1)
            stats2d.Execute(slice_bin)
            if stats2d.HasLabel(1):
                d = stats2d.GetFeretDiameter(1)
                if d > best:
                    best = d
                    if best >= diameter_threshold:
                        break

        if best >= diameter_threshold:
            out = sitk.Or(out, sitk.Cast(labeled == lab, sitk.sitkUInt8))

    return out


def combine_masks(kidney_mask: sitk.Image, tumor_mask: sitk.Image) -> sitk.Image:
    """
    Combine two masks with distinct labels.

    Parameters:
        kidney_mask (sitk.Image): Mask with kidney regions (binary or labeled).
        tumor_mask (sitk.Image): Mask with tumor regions (binary or labeled).

    Returns:
        sitk.Image: Combined mask where kidneys have label 1 and tumors have label 2.
    """
    # Ensure masks are binary and have non-overlapping regions
    kidney_mask = sitk.Cast(kidney_mask > 0, sitk.sitkUInt8) * 1  # Label kidneys as 1
    tumor_mask = sitk.Cast(tumor_mask > 0, sitk.sitkUInt8) * 2  # Label tumors as 2

    # Combine masks (assuming non-overlapping)
    combined_mask = sitk.Add(kidney_mask, tumor_mask)

    # Resolve any overlapping regions (if necessary, prioritize higher label)
    combined_mask = sitk.Maximum(combined_mask, tumor_mask)

    return combined_mask


def get_largest_cc(
    pred_mask: sitk.Image,
    diameter_threshold: float = 3.0,
    keep_large_tumors: bool = False,
) -> sitk.Image:
    """Does largest component analysis"""
    # Separate kidney and tumor regions
    binary_kidney_mask = get_binary_mask(pred_mask, [1])
    binary_tumor_mask = get_binary_mask(pred_mask, [2])

    kidney_mask = binary_kidney_mask

    tumor_mask = process_tumors(
        binary_tumor_mask,
        kidney_mask,
        diameter_threshold,
        keep_large_tumors,
    )

    # Combine the kidney and connected tumor masks
    combined_mask = combine_masks(kidney_mask, tumor_mask)

    return combined_mask


def postprocess_segmentation_mask(output_segmentation_mask):
    mask = get_largest_cc(
        output_segmentation_mask,
        diameter_threshold=3.0,
        keep_large_tumors=True,
    )
    return mask
