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
import threading
import time

import SimpleITK as sitk
import torch

from kidney_abnormality_segmentation.segmentation.batched_predictor import (
    BatchedNNUNetPredictor,
)
from kidney_abnormality_segmentation.utils import resample_volume

# ---- Load model and store it in cache ---
_lock = threading.Lock()  # guard for multi-threading
_cached_model = None
_cached_model_path = None


def get_predictor(model_path, run_fast: bool = False, run_one_fold: bool = False,
                   sw_batch_size: int = 4) -> BatchedNNUNetPredictor:
    """
    Return a cached nnUNet predictor, reloading it only if the weights path changes to avoid repeated loading time.
    """

    global _cached_model, _cached_model_path

    with _lock:
        if _cached_model is None or _cached_model_path != model_path:

            _cached_model_path = model_path
            _cached_model = BatchedNNUNetPredictor(
                sw_batch_size=sw_batch_size,
                tile_step_size=0.5 if not run_fast else 0.8,
                use_gaussian=True,
                use_mirroring=True if not run_fast else False,
                perform_everything_on_device=True,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
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
            _cached_model.initialize_from_trained_model_folder(
                model_training_output_dir=weights_path,
                use_folds=(0, 1, 2, 3, 4) if not (run_fast or run_one_fold) else (0,),
                checkpoint_name="checkpoint_best.pth",
            )
            print("[nnUNet] Model loaded successfully.")

        # sw_batch_size is cheap to update without reloading weights, so keep it in sync
        # even when the cached predictor is reused.
        _cached_model.sw_batch_size = sw_batch_size

    return _cached_model



def segment_ct_image(input_ct, model_path: str, run_fast: bool = False, run_one_fold: bool = False,
                      sw_batch_size: int = 4) -> sitk.Image:
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
        t_resample_start = time.perf_counter()
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

            # Make sure the tempname is longer than 5 characters (weird nnUNet quirk)
            temp_name =  "tempimage_" + os.path.basename(input_ct)
            tmp_input = os.path.join("/tmp", temp_name)
            sitk.WriteImage(ct, tmp_input)
        else:
            raise ValueError(
                "segment_ct_image: input_ct must be sitk.Image or filepath"
            )
        print(f"[timing] read + resample to 0.75mm + write tmp file: {time.perf_counter() - t_resample_start:.2f}s")

        # Load predictor
        predictor = get_predictor(model_path, run_fast=run_fast, run_one_fold=run_one_fold,
                                   sw_batch_size=sw_batch_size)

        # Create a temporary directory under /tmp for nnU-Net’s outputs
        tmp_output_dir = tempfile.mkdtemp(dir="/tmp")

        # Run inference. Because we give an existing folder as output, nnU-Net writes a .nii.gz into it.
        print(f"[nnUNet] Running inference on: {tmp_input}")
        t_predict_start = time.perf_counter()
        result_list = predictor.predict_from_files(
            list_of_lists_or_source_folder=[[tmp_input]],
            output_folder_or_list_of_truncated_output_files=tmp_output_dir,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
        )
        print(f"[timing] predict_from_files total (preprocess+infer+export): "
              f"{time.perf_counter() - t_predict_start:.2f}s")

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
        raise RuntimeError(f"Failed during segmentation: {e}") from e
