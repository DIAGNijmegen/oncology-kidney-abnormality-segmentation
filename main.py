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

import argparse
import gc
import SimpleITK
import sys

from pathlib import Path

from kidney_abnormality_segmentation.config import CT_EXTENSIONS, ensure_totalsegmentator_env
ensure_totalsegmentator_env()

from kidney_abnormality_segmentation.postprocessing.postprocess_segmentation_mask import (
    postprocess_segmentation_mask,
)
from kidney_abnormality_segmentation.preprocessing.extract_roi import extract_roi
from kidney_abnormality_segmentation.segmentation.segment_ct_image import (
    segment_ct_image,
)
from kidney_abnormality_segmentation.utils import resample_volume, stem




class TermStyle:
    """
    Manages ANSI escape codes for terminal styling, disabling them
    if the output stream is not a TTY (e.g., output is redirected to a file).
    """

    # Check if the output stream is a terminal (TTY)
    # We check stderr since that is where argparse prints help messages.
    _is_tty = sys.stderr.isatty()

    # --- Color Definitions ---
    if _is_tty:
        RED = "\033[31m"
        GREEN = "\033[32m"   # darker green
        YELLOW = "\033[33m"
        CYAN = "\033[34m"    # dark blue
        BOLD = "\033[1m"
        RESET = "\033[0m"
    else:
        # Define empty strings if not running in a terminal
        RED = GREEN = YELLOW = CYAN = BOLD = RESET = ""

    @staticmethod
    def style(text: str, color: str = "", bold: bool = False) -> str:
        """Helper method to wrap text with color and optional bolding."""
        prefix = color
        if bold:
            prefix = TermStyle.BOLD + prefix

        # This will return "text" if TermStyle is using empty strings (non-TTY)
        return f"{prefix}{text}{TermStyle.RESET}"


def initialize_parser() -> argparse.Namespace:

    name = TermStyle.style("Oncology Kidney Segmentation", bold=True)
    desc = TermStyle.style(
        "Segment kidneys and kidney abnomalities in abdominal contrast-enhanced CT", TermStyle.GREEN
    )

    epilog = (
        f"{TermStyle.BOLD}{'-'*20} Radboudumc OncoAI – 2026  {'-'*20}{TermStyle.RESET}\n"
        f"Group website: {TermStyle.CYAN}https://www.diagnijmegen.nl/research/oncology/{TermStyle.RESET}\n"
        f"Published paper: {TermStyle.CYAN}https://www.melba-journal.org/papers/2026:012.html{TermStyle.RESET}\n"
    )

    parser = argparse.ArgumentParser(
        prog=name,
        description=desc,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input-path",
        type=Path,
        required=True,
        help="input image or directory with CT images",
    )

    parser.add_argument(
        "-m",
        "--model-path",
        type=Path,
        required=True,
        help="path to the model weights",
    )

    parser.add_argument("-o",
                        "--output-path", 
                        type=Path, 
                        default="onco_segmentations", 
                        help="output directory")

    parser.add_argument(
        "--use-cropping",
        action="store_true",
        help="Enable ROI cropping based on TotalSegmentator (default: disabled).",
        default=False,
    )
    
    parser.add_argument(
        "--no_postprocessing",
        action="store_true",
        help="Disable postprocessing",
        default=False,
    )

    # Print help if no arguments are provided at all
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(2)

    args = parser.parse_args()
    return args


def run():
    args = initialize_parser()

    # List all CT files under /input
    input_path = args.input_path
    if not input_path.exists():
        raise FileNotFoundError(f"Input does not exist: {input_path}")

    if input_path.is_dir():
        try:
            patterns = ["*"+ext for ext in CT_EXTENSIONS]
            all_cts = [
                file
                for pattern in patterns
                for file in input_path.rglob(pattern)
            ]
        except PermissionError as e:
            raise PermissionError(f"Cannot access {args.input_path}: {e}") from e
    else:
        if not any(str(input_path).endswith(ext) for ext in CT_EXTENSIONS):
            raise ValueError(f"File type not support. Supported files are: {CT_EXTENSIONS}")
        all_cts = [input_path]

    if not all_cts:
        print(f"No CT files found under {input_path}")
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
