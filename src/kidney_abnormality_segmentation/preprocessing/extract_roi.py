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

import gc
import os
import sys
import tempfile

import SimpleITK as sitk
from totalsegmentator.python_api import totalsegmentator

from src.kidney_abnormality_segmentation.utils import crop_image, resample_volume


def extract_roi(ct_image: sitk.Image) -> sitk.Image:
    """
    1) Downsample the highres CT to ~3 mm isotropic and write it to /tmp.
    2) Run TotalSegmentator on that small volume (no max_res or resample_kwargs needed).
    3) Read back the lowres mask, find its bounding box, and map it
       back to the original CT index space via physical point transforms.
    4) Crop the original CT directly in highres, then clean up.
    """

    try:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["NNUNET_NUM_PROCESSORS"] = "2"

        fast_ct = resample_volume(ct_image, new_spacing=(3.0, 3.0, 3.0))

        with tempfile.NamedTemporaryFile(
            suffix=".nii.gz", delete=False, dir="/tmp"
        ) as tmp_fast_ct_file:
            sitk.WriteImage(fast_ct, tmp_fast_ct_file.name)
            fast_ct_path = tmp_fast_ct_file.name

        # Prepare a second temp filename (for TotalSegmentator’s output mask)
        with tempfile.NamedTemporaryFile(
            suffix=".nii.gz", delete=False, dir="/tmp"
        ) as tmp_fast_mask_file:
            fast_mask_path = tmp_fast_mask_file.name

        # Only keep the lobes we need (same as your original mask_id_mapping).
        ts_kwargs = {
            "input": fast_ct_path,
            "output": fast_mask_path,
            "fast": True,  # already downsampled, but “fast=True” ensures no extra postprocessing.
            "ml": True,
            "roi_subset": [
                "lung_lower_lobe_right",
                "lung_lower_lobe_left",
                "urinary_bladder",
            ],
            "verbose": False,
        }

        totalsegmentator(**ts_kwargs)

        # At this point, there is a small (~fast_size) mask on disk at fast_mask_path.
        # --------------------------------------------------------------------

        lowres_mask = sitk.ReadImage(fast_mask_path)

        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(lowres_mask)

        labels_of_interest = [11, 14, 21]  # same IDs as “lung_lower_lobe_left=11”, etc.

        # Initialize extreme indices in low-res index space:
        min_idx_low = [
            lowres_mask.GetSize()[0],
            lowres_mask.GetSize()[1],
            lowres_mask.GetSize()[2],
        ]
        max_idx_low = [0, 0, 0]

        for lbl in labels_of_interest:
            if not stats.HasLabel(lbl):
                continue

            # GetBoundingBox returns (startX, startY, startZ, sizeX, sizeY, sizeZ).
            bb = stats.GetBoundingBox(lbl)
            sx, sy, sz, dx, dy, dz = bb

            # Compute the min & max corners in low-res index space:
            lx0, ly0, lz0 = sx, sy, sz
            lx1 = sx + dx - 1  # inclusive
            ly1 = sy + dy - 1
            lz1 = sz + dz - 1

            min_idx_low[0] = min(min_idx_low[0], lx0)
            min_idx_low[1] = min(min_idx_low[1], ly0)
            min_idx_low[2] = min(min_idx_low[2], lz0)

            max_idx_low[0] = max(max_idx_low[0], lx1)
            max_idx_low[1] = max(max_idx_low[1], ly1)
            max_idx_low[2] = max(max_idx_low[2], lz1)

        # If we didn’t find any of the requested labels, bail out:
        if min_idx_low[0] > max_idx_low[0]:
            raise RuntimeError("No ROI labels found in the low-res mask.")

        # Get the physical coordinates of the min and max corners in the low-res grid:
        min_phys = lowres_mask.TransformIndexToPhysicalPoint(tuple(min_idx_low))
        max_phys = lowres_mask.TransformIndexToPhysicalPoint(tuple(max_idx_low))

        # Now map those back into the original high-res CT’s index space:
        orig_min_idx = ct_image.TransformPhysicalPointToIndex(min_phys)
        orig_max_idx = ct_image.TransformPhysicalPointToIndex(max_phys)

        # Clamp indices to be within original CT bounds:
        orig_size = ct_image.GetSize()  # (X, Y, Z)
        orig_min_idx = [max(0, orig_min_idx[i]) for i in range(3)]
        orig_max_idx = [min(orig_size[i] - 1, orig_max_idx[i]) for i in range(3)]

        # Compute the high-res “size” in each dim:
        size_x = orig_max_idx[0] - orig_min_idx[0] + 1
        size_y = orig_max_idx[1] - orig_min_idx[1] + 1
        size_z = orig_max_idx[2] - orig_min_idx[2] + 1

        cropped_image = crop_image(
            input_image=ct_image,
            index=(orig_min_idx[0], orig_min_idx[1], orig_min_idx[2]),
            size=(size_x, size_y, size_z),
        )

        del lowres_mask
        del fast_ct
        gc.collect()

        try:
            os.remove(fast_ct_path)
        except OSError:
            pass
        try:
            os.remove(fast_mask_path)
        except OSError:
            pass

        return cropped_image

    except Exception as e:
        print(
            "An error occurred during ROI extraction:",
            e,
            "\nAborting execution of the algorithm.",
        )
        sys.exit(1)
