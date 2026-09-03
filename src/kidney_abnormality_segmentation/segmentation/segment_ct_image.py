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

import json
import os
import tempfile

import SimpleITK as sitk
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from kidney_abnormality_segmentation.config import EXTENSIONS
from kidney_abnormality_segmentation.utils import resample_volume


def segment_image(input_image, weights_path: str, run_fast: bool = False, presample: bool = False, supported_folds = (0, 1, 2, 3, 4)) -> sitk.Image:
    """
    input_image: either a SimpleITK.Image or a string path to a .mha file
    weights_path: path to the directory containing the trained model weights
    run_fast: use a faster but less accurate inference mode
    presample: resample the input image to 0.75mm isotropic spacing before inference 
        (this resolves memory issues specifically on the Grand Challenge platform)
    supported_folds: tuple of fold indices to use for inference (default is all folds 0-4)
        (for MRI this needs to be (all,))

    Behaviour:
      - path input, presample=False  -> file is used directly, untouched
      - path input, presample=True   -> resampled copy written to a temp dir
      - sitk.Image input             -> written to a temp dir (resampled only if presample)

    Everything this function creates lives inside a single TemporaryDirectory that is
    removed automatically on exit. 
    """
    # limit threads…
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NNUNET_NUM_PROCESSORS"] = "2"

    new_spacing = (0.75, 0.75, 0.75)
    CASE_STEM = "renalnet_case"
    CASE_NAME = CASE_STEM + "_0000" # add a suffic of size 5, will be removed by the nnUnet predictor later (nnU-Net quirk)

    try:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
            in_dir = os.path.join(tmp_dir, "RenalNet_in")
            out_dir = os.path.join(tmp_dir, "RenalNet_out")
            os.makedirs(in_dir)
            os.makedirs(out_dir)

            # --- Resolve the input to a single path on disk -------------------
            # nnU-Net truncates output filenames by len(file_ending) + 5 chars, so the temp
            # input must use the model's declared file_ending, not the input's own extension.
            with open(os.path.join(weights_path, "dataset.json")) as f:
                file_ending = json.load(f)["file_ending"]

            if isinstance(input_image, sitk.Image):
                image = resample_volume(input_image, new_spacing=new_spacing) if presample else input_image
                tmp_input = os.path.join(in_dir, CASE_NAME + file_ending)
                sitk.WriteImage(image, tmp_input)

            elif isinstance(input_image, str):
                tmp_input = os.path.join(in_dir, CASE_NAME + file_ending)
                if not presample and input_image.endswith(file_ending):
                    # symlink: required to counter the nnU-Net quirk of truncating names
                    # no copy, and rmtree removes only the link, never the target
                    # only safe when the real extension already matches file_ending, since
                    # SimpleITK picks its reader from the filename extension, not the content
                    os.symlink(os.path.abspath(input_image), tmp_input)
                else:
                    image = sitk.ReadImage(input_image)
                    if presample:
                        image = resample_volume(image, new_spacing=new_spacing)
                    sitk.WriteImage(image, tmp_input)
            else:
                raise TypeError("segment_image: input_image must be sitk.Image or filepath")

            # --- Predictor ----------------------------------------------------
            device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
            print(f"[nnUNet] Using device: {device}")

            if device.type == "mps":
                print("[nnUNet] MPS device detected. Setting environment variable 'PYTORCH_ENABLE_MPS_FALLBACK' to '1' to enable fallback for unsupported operations.")
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

            
            predictor = nnUNetPredictor(
                tile_step_size=0.5 if not run_fast else 0.8,
                use_gaussian=True,
                use_mirroring= not run_fast,
                perform_everything_on_device=True,
                device=device,
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=True,
            )

            print(f"[nnUNet] Looking for trained model weights in: {weights_path}")
            predictor.initialize_from_trained_model_folder(
                model_training_output_dir=weights_path,
                use_folds= supported_folds if not run_fast else (supported_folds[0],),
                checkpoint_name="checkpoint_best.pth",
            )
            print("[nnUNet] Model loaded successfully.")

            # --- Inference ----------------------------------------------------
            print(f"[nnUNet] Running inference on: {tmp_input}")
            result_list = predictor.predict_from_files(
                list_of_lists_or_source_folder=[[tmp_input]],
                output_folder_or_list_of_truncated_output_files=out_dir,
                num_processes_preprocessing=1,
                num_processes_segmentation_export=1,
            )

            # result_list should be something like ["/tmp/shdgi/RenalNet_out/renalnet_case.nii.gz"]
            if result_list and isinstance(result_list[0], str):
                seg_path = result_list[0]
            else:
                candidates = [
                    p for p in (os.path.join(out_dir, CASE_STEM + e) for e in EXTENSIONS)
                    if os.path.isfile(p)
                ]
                if len(candidates) != 1:
                    raise RuntimeError(
                        f"Expected exactly 1 segmentation named '{CASE_STEM}<ext>' in {out_dir}, "
                        f"found {len(candidates)}. Contents: {os.listdir(out_dir)}"
                    )
                seg_path = candidates[0]

            print(f"[nnUNet] Using segmentation file at: {seg_path}")
            # image at seg_path will be deleted when the TemporaryDirectory is cleaned up, so we read it into memory first
            return sitk.ReadImage(seg_path)
        
    except Exception as e:
        raise RuntimeError(f"Failed during segmentation: {e}") from e