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
import SimpleITK
from pathlib import Path

from kidney_abnormality_segmentation.config import ensure_totalsegmentator_env
ensure_totalsegmentator_env()

from kidney_abnormality_segmentation.config import EXTENSIONS, CT_MODEL_PATH, MRI_MODEL_PATH
from kidney_abnormality_segmentation.postprocessing.postprocess_segmentation_mask import postprocess_segmentation_mask
from kidney_abnormality_segmentation.preprocessing.extract_roi import extract_roi
from kidney_abnormality_segmentation.segmentation.segment_ct_image import build_predictor, segment_image
from kidney_abnormality_segmentation.utils import resample_volume, stem


def run(all_cts,  model_path: Path, output_path: Path, crop_roi: bool = False, run_fast:bool = False, mri: bool = False, presample: bool = False):

    weights_path = model_path / (MRI_MODEL_PATH if mri else CT_MODEL_PATH)
    supported_folds = ("all",) if mri else (0, 1, 2, 3, 4)  
    predictor = build_predictor(weights_path, run_fast=run_fast, supported_folds=supported_folds)

    for input_ct_image_path in all_cts:
        print(f"[run] Processing {input_ct_image_path.name}")
        if input_ct_image_path.name.startswith("."):
            print(f"[run] Skipping {input_ct_image_path.name} because not an image.")
            continue
        image_name = stem(str(input_ct_image_path))
        file_extension = (
            ".mha" if str(input_ct_image_path).endswith(".mha") else ".nii.gz"
        )
        out_folder = output_path

        # define output location
        if any(str(output_path).endswith(ext) for ext in EXTENSIONS):
            if len(all_cts) > 1:
                raise ValueError(
                    f"Output path {output_path} is a file, but multiple input images were provided. Please provide a directory for output."
                )
            out_path = out_folder
        else:
            out_path = out_folder / f"{image_name}{file_extension}"

        # skip already segmented images
        if out_path.is_file():
            print(
                f"[run] Skipping {input_ct_image_path.name} because output segmentation already exists for this image."
            )
            continue
        
        # 3) Decide what to hand to segment_ct_image:
        #    - If cropping: read into memory, crop, then pass the cropped SITK.Image.
        #    - If no cropping: NEVER read the full CT. Pass the filepath string instead.
        orig_spacing = SimpleITK.ReadImage(str(input_ct_image_path)).GetSpacing()
        if crop_roi:
            print("[run] Cropping ROI; will read full CT into memory.")
            full_ct = SimpleITK.ReadImage(str(input_ct_image_path))
            try:
                input_for_seg = extract_roi(full_ct)
            except RuntimeError as e:
                print(f"[run] ROI extraction failed ({e}), falling back to segmenting the full CT.")
                input_for_seg = str(input_ct_image_path)
            # free the full CT
            del full_ct
            gc.collect()
        else:
            print(
                "[run] No cropping requested; will segment from disk without reading full CT."
            )
            input_for_seg = str(input_ct_image_path)

        # 4) Segment (this now never loads the full on-disk CT into RAM)
        print("[run] Calling segment_ct_image() …")
        segmentation_sitk = segment_image(input_for_seg, predictor, presample=presample)

        # 5) Free any remaining cropped image if it was in RAM
        if isinstance(input_for_seg, SimpleITK.Image):
            del input_for_seg
            gc.collect()

        # 6) Postprocess & write out
        post_sitk = postprocess_segmentation_mask(segmentation_sitk)
        final_image = resample_volume(
            post_sitk,
            new_spacing=orig_spacing,
            interpolator=SimpleITK.sitkNearestNeighbor,
        )
        print(f"[run] Writing final mask to: {out_path}")

        SimpleITK.WriteImage(final_image, str(out_path))

    print("[run] Done.")
    return 0