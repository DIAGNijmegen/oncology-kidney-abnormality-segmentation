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
import scipy.ndimage as ndi
import SimpleITK as sitk
from scipy.spatial import distance

connectivity_structure = ndi.generate_binary_structure(3, 3)


def stem(filename: str, extensions: list = [".nii", ".nii.gz", ".mha"]):
    """Remove the extension from a filename. And remove the leading path if present."""

    for ext in extensions:
        if filename.endswith(ext):
            filename = filename[: -len(ext)]
            break  # Stop once the matching extension is removed

    return filename.split("/")[-1]


def resample_volume(image, new_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkLinear):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputDirection(image.GetDirection())
    return resample.Execute(image)


def load_image_metadata(image_file: str):
    """
    Load only the metadata of a medical image using SimpleITK.

    Parameters:
        image_file (str): Path to the image file.

    Returns:
        dict: Dictionary containing metadata information (size, spacing, origin, etc.).
    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(image_file)

    # Read the image metadata without loading the full image
    reader.ReadImageInformation()

    # Extract metadata information
    metadata = {
        "size": tuple(reader.GetSize()),
        "spacing": tuple(reader.GetSpacing()),
        "origin": tuple(reader.GetOrigin()),
        "direction": tuple(reader.GetDirection()),
    }

    return metadata


def crop_image(
    input_image: sitk.Image,
    index: tuple | None,
    size: tuple | None,
) -> sitk.Image:
    """
    Crop an image using the given index and size of the bounding box around the region of interest

    Parameters:
    input_image (sitk.Image): Input image to be cropped
    index (tuple): Index of the bounding box
    size (tuple): Size of the bounding box
    output_file (str): Path to save the cropped image

    Returns:
    sitk.Image: Cropped image
    """
    if size is None and index is None:
        print(
            "No cropping will be performed because TotalSegmentator could not find the lungs and/or bladder. The algorithm will continue with the full scan. "
        )
        return input_image

    print("The algorithm will be performed on the cropped image.")
    cropped_image = sitk.RegionOfInterest(input_image, size=size, index=index)
    return cropped_image


connectivity_structure = ndi.generate_binary_structure(3, 3)


def convert_numpy_to_python(obj):
    """
    Convert numpy objects to native Python data types for JSON serialization.
    If the object is already a native Python type, return it unchanged.
    """
    if isinstance(obj, (np.integer, np.int_)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float_)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (int, float, str, list, dict, bool, type(None))):
        # Return the object as it is if it's already a Python native type
        return obj
    else:
        raise TypeError(
            f"Object of type {obj.__class__.__name__} is not JSON serializable"
        )


def find_longest_axes_diameters_axial(
    mask: np.ndarray,
    voxel_spacing: tuple[float, float, float],
    tumor_label: int = None,
    slice_step: int = 5,
    num_sample_points: int = 100,
) -> list[tuple[int, float]]:
    """
    Find the longest axes diameters (Feret diameters) of tumors in the axial plane.

    Parameters:
    mask (np.ndarray): The mask array with tumors labeled.
    voxel_spacing (tuple[float, float, float]): The voxel spacing in the (x, y, z) directions.
    tumor_label (int or None): The label value representing tumors. If None, compute for all labels > 0.
    slice_step (int): The step for skipping slices in the z-dimension to speed up computation.
    num_sample_points (int): Number of points to sample for distance calculation.

    Returns:
    list[tuple[int, float]]: A list of tuples where each tuple contains a tumor index and its
    corresponding longest axis diameter in millimeters.
    """
    if tumor_label is not None:
        # Use only the specified tumor_label
        labeled_tumors, num_tumors = ndi.label(
            mask == tumor_label, structure=connectivity_structure
        )
    else:
        # Use existing labels in the mask (labels should be > 0)
        labeled_tumors = mask
        num_tumors = int(labeled_tumors.max())

    tumor_diameters = []

    for tumor_index in range(1, num_tumors + 1):
        # Extract coordinates for a single tumor component
        tumor_voxel_indices = np.argwhere(labeled_tumors == tumor_index)

        # Skip if there are no tumor voxels
        if tumor_voxel_indices.shape[0] == 0:
            tumor_diameters.append((tumor_index, 0))
            continue

        # Skip slices by selecting every nth slice in the z-axis
        tumor_voxel_indices = tumor_voxel_indices[::slice_step]

        # Scale the coordinates based on voxel spacing
        scaled_voxel_coords = tumor_voxel_indices * np.array(voxel_spacing)

        # Project onto the axial plane (x, y)
        scaled_voxel_coords_axial = scaled_voxel_coords[:, :2]

        # Skip if there are not enough points
        if len(scaled_voxel_coords_axial) < 2:
            tumor_diameters.append((tumor_index, 0))
            continue

        # Sample points to reduce computation
        num_points = len(scaled_voxel_coords_axial)
        sample_indices = np.random.choice(
            num_points, min(num_sample_points, num_points), replace=False
        )
        sampled_coords = scaled_voxel_coords_axial[sample_indices]

        # Compute the pairwise distances in the axial plane (2D)
        pairwise_distances = distance.pdist(sampled_coords)
        max_distance = np.max(pairwise_distances)

        tumor_diameters.append((tumor_index, max_distance))

    return tumor_diameters


def get_binary_mask(mask: sitk.Image, labels: list) -> sitk.Image:
    """
    Get binary mask from a mask image with multiple labels

    Parameters:
    mask (sitk.Image) Mask image with multiple labels
    labels (list) List of labels to be included in the binary mask

    Returns:
    sitk.Image: Binary mask image
    bool: Found lung
    bool: Found bladder
    """
    return sitk.BinaryThreshold(
        mask,
        lowerThreshold=labels[0],
        upperThreshold=labels[-1],
        insideValue=1,
        outsideValue=0,
    )
