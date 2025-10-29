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

import glob
import os
import shutil
import sys
import tempfile

import SimpleITK as sitk
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from src.kidney_abnormality_segmentation.utils import resample_volume


def segment_ct_image(input_ct, model_path: str) -> sitk.Image:
    """
    input_ct: either a SimpleITK.Image or a string path to a .mha file
    model_path: base folder containing nnUNet_results/...

    Runs nnU-Net on a CT by:
      1. Writing the CT to /tmp as a .mha
      2. Calling nnUNetPredictor.predict_from_files(...) with a temporary output folder
      3. If predict_from_files() returns [None], we scan that folder for a .nii/.nii.gz
      4. Read the resulting segmentation back into SimpleITK and clean up.
    """
    # limit threads…
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NNUNET_NUM_PROCESSORS"] = "2"

    try:
        if isinstance(input_ct, sitk.Image):
            with tempfile.NamedTemporaryFile(
                suffix=".mha", delete=False, dir="/tmp"
            ) as tmp_in:
                ct = resample_volume(input_ct, new_spacing=(0.75, 0.75, 0.75))
                sitk.WriteImage(ct, tmp_in.name)
                tmp_input = tmp_in.name
        elif isinstance(input_ct, str):
            # copy to /tmp to guarantee write-perms / uniform path
            ct = sitk.ReadImage(input_ct)
            ct = resample_volume(ct, new_spacing=(0.75, 0.75, 0.75))
            tmp_input = os.path.join("/tmp", os.path.basename(input_ct))
            sitk.WriteImage(ct, tmp_input)
        else:
            raise ValueError(
                "segment_ct_image: input_ct must be sitk.Image or filepath"
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Instantiate the predictor
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True,
        )

        # Load the trained model weights
        weights_path = os.path.join(
            model_path,
            "nnUNet_results",
            "Dataset102_KidneyCT",
            "nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres",
        )
        print(f"[nnUNet] Looking for trained model weights in: {weights_path}")
        predictor.initialize_from_trained_model_folder(
            model_training_output_dir=weights_path,
            use_folds=(0, 1, 2, 3, 4),
            checkpoint_name="checkpoint_best.pth",
        )
        print("[nnUNet] Model loaded successfully.")

        # Create a temporary directory under /tmp for nnU-Net’s outputs
        tmp_output_dir = tempfile.mkdtemp(dir="/tmp")

        # Run inference. Because we give an existing folder as output, nnU-Net writes a .nii.gz into it.
        print(f"[nnUNet] Running inference on: {tmp_input}")
        result_list = predictor.predict_from_files(
            list_of_lists_or_source_folder=[[tmp_input]],
            output_folder_or_list_of_truncated_output_files=tmp_output_dir,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
        )

        # result_list should be something like ["/tmp/tmpXYZ/CaseName_seg.nii.gz"]
        seg_path = None
        if result_list and isinstance(result_list[0], str):
            seg_path = result_list[0]
        else:
            files = glob.glob(os.path.join(tmp_output_dir, "*.nii*"))
            if not files:
                raise RuntimeError(f"No segmentation found in {tmp_output_dir}")
            seg_path = files[0]

        print(f"[nnUNet] Using segmentation file at: {seg_path}")

        # Read the segmentation back into SimpleITK
        segmentation_sitk = sitk.ReadImage(seg_path)

        # Clean up temporary files & directory
        try:
            os.remove(tmp_input)
        except OSError:
            pass

        try:
            shutil.rmtree(tmp_output_dir)
        except OSError:
            pass

        # Return the SimpleITK segmentation image
        return segmentation_sitk

    except Exception as e:
        print(f"Failed during segmentation and postprocessing due to: {e}")
        sys.exit(1)
