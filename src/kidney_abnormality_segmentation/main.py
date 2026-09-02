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
import sys

from pathlib import Path
from kidney_abnormality_segmentation.config import EXTENSIONS, CT_MODEL_PATH, MRI_MODEL_PATH



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
        help="path to the base nnUnet directory containing model weights",
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
        "--fast",
        action="store_true",
        help="Enable faster segmentations. Useful for experimentation and debugging (default: disabled).",
        default=False,
    )

    parser.add_argument(
        "--mri",
        action="store_true",
        help="Enable MRI segmentation (default: disabled).",
        default=False,
    )

    parser.add_argument(
        "--presample",
        action="store_true",
        help="Resample input images to 0.75mm isotropic spacing before inference (default: disabled). This resolves memory issues specifically on the Grand Challenge platform.",
        default=False,
    )

    # Print help if no arguments are provided at all
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(2)

    args = parser.parse_args()
    return args


def main():
    args = initialize_parser()

    # List all CT files under /input
    input_path = args.input_path
    if not input_path.exists():
        raise FileNotFoundError(f"Input does not exist: {input_path}")

    # Check model path
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model base path does not exist: {input_path}")

    # Check weights path
    if args.mri:
        weights_path = args.model_path / MRI_MODEL_PATH
        if not weights_path.exists():
            raise FileNotFoundError(f"Model base path does not include the required MRI model folder: {MRI_MODEL_PATH}",
                                    "Weights will soon be made available.")
    else:
        weights_path = args.model_path / CT_MODEL_PATH
        if not weights_path.exists():
            raise FileNotFoundError(f"Model base path does not include the required CT model folder: {CT_MODEL_PATH}",
                                    "Weights are available at: https://doi.org/10.5281/zenodo.15315330")

    # Output path
    if not args.output_path.exists():
        args.output_path.mkdir(parents=True, exist_ok = True)

    # Read files
    if input_path.is_dir():
        try:
            patterns = ["*"+ext for ext in EXTENSIONS]
            all_cts = [
                file
                for pattern in patterns
                for file in input_path.rglob(pattern)
                if not file.name.startswith(".")
            ]
        except PermissionError as e:
            raise PermissionError(f"Cannot access {args.input_path}: {e}") from e
    else:
        if not any(str(input_path).endswith(ext) for ext in EXTENSIONS):
            raise ValueError(f"File type not support. Supported files are: {EXTENSIONS}")
        all_cts = [input_path]

    if not all_cts:
        print(f"No CT files found under {input_path}")
        sys.exit(1)

    print(f"[main] Found {len(all_cts)} input CTs to process")
    if args.fast:
        print("[main] Running segmentation in fast mode.")

    from kidney_abnormality_segmentation.inference import run  
    run(all_cts, weights_path, args.output_path, args.use_cropping, args.fast, args.mri, presample=args.presample)

    
if __name__ == "__main__":
    sys.exit(main())
