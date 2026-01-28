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
import sys
from argparse import ArgumentParser
from pathlib import Path

import SimpleITK

from src.kidney_abnormality_segmentation.config import (
    default_input_path,
    default_model_path,
    default_output_path,
)
from src.kidney_abnormality_segmentation.postprocessing.postprocess_segmentation_mask import (
    postprocess_segmentation_mask,
)
from src.kidney_abnormality_segmentation.preprocessing.extract_roi import extract_roi
from src.kidney_abnormality_segmentation.segmentation.segment_ct_image import (
    segment_ct_image,
)
from src.kidney_abnormality_segmentation.utils import resample_volume, stem


def build_parser() -> ArgumentParser:
    """Create and return the CLI argument parser."""
    p = ArgumentParser(description="Process images and perform segmentation")
    p.add_argument(
        "--use-cropping",
        action="store_true",
        default=False,
        help="Enable ROI cropping based on TotalSegmentator (default: disabled).",
    )
    # Use Path typing and safer argument names (avoid shadowing builtins like `input`)
    p.add_argument(
        "--input-path",
        type=Path,
        default=default_input_path(),
        help="Input folder path (defaults to /input or $INPUT_PATH).",
    )
    p.add_argument(
        "--output-path",
        type=Path,
        default=default_output_path(),
        help="Output folder path (defaults to /output or $OUTPUT_PATH).",
    )
    p.add_argument(
        "--model-path",
        type=Path,
        default=default_model_path(),
        help="Model weights folder path (defaults to /opt/ml/model or $MODEL_PATH).",
    )
    return p


def run():
    parser = build_parser()
    args = parser.parse_args()

    # List all CT files under /input
    ct_folder = args.input_path
    if not ct_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {ct_folder}")
    if not ct_folder.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {ct_folder}")

    try:
        all_cts = list(ct_folder.rglob("*.mha")) + list(ct_folder.rglob("*.nii.gz"))
    except PermissionError as e:
        raise PermissionError(f"Cannot access {args.input_path}: {e}") from e

    if not all_cts:
        print(f"No CT files found under {ct_folder}")
        sys.exit(1)

    print(f"[run] Found {len(all_cts)} input CTs to process")
    crop_roi = args.use_cropping

    for input_ct_image_path in all_cts:
        print(f"[run] Processing {input_ct_image_path.name}")
        if input_ct_image_path.name.startswith("."):
            print(f"[run] Skipping {input_ct_image_path.name} because not an image.")
            continue
        image_name = stem(str(input_ct_image_path))
        file_extension = (
            ".mha" if str(input_ct_image_path).endswith(".mha") else ".nii.gz"
        )
        out_folder = args.output_path
        out_folder.mkdir(parents=True, exist_ok=True)

        # find output
        out_path = out_folder / f"{image_name}{file_extension}"
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
            input_for_seg = extract_roi(full_ct)
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
        segmentation_sitk = segment_ct_image(input_for_seg, str(args.model_path))

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


if __name__ == "__main__":
    sys.exit(run())
