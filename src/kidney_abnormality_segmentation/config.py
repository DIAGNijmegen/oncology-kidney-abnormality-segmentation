import os
from pathlib import Path


def ensure_totalsegmentator_env():
    default = str(Path.home() / ".totalsegmentator" / "nnunet" / "results")
    for var in ("nnUNet_preprocessed", "nnUNet_raw", "nnUNet_results"):
        os.environ.setdefault(var, default)


CT_EXTENSIONS = [".nii.gz", ".nii", ".mha"]