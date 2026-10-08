import os
from pathlib import Path


def ensure_totalsegmentator_env():
    default = str(Path.home() / ".totalsegmentator" / "nnunet" / "results")
    for var in ("nnUNet_preprocessed", "nnUNet_raw", "nnUNet_results"):
        os.environ.setdefault(var, default)


EXTENSIONS = [".nii.gz", ".nii", ".mha"]
CT_MODEL_PATH = Path("nnUNet_results/Dataset102_KidneyCT/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres")
MRI_MODEL_PATH = Path("nnUNet_results/Dataset103_KidneyMRI/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres")