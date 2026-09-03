# tests/integration_config.py
#
# Configuration for the full integration test suite.
# Please copy it, remove the .example from the filename, and provide your own paths.
# `integration_config.py` is intentionally excluded from version control (.gitignore.). Do not commit it!
#
# Fill in the paths below before running:
#   pytest tests/test_integration.py -v

from pathlib import Path

# Path to the directory that contains the nnUNet_results/ subdirectory.
MODEL_PATH = Path("/path/to/model")

# A list of real CT files (.mha, .nii, or .nii.gz) to run through the pipeline.
# Three images is a good starting point: covers variety without taking too long.
TEST_CT_FILES = [
    Path("/path/to/ct_image_1.mha"),
    Path("/path/to/ct_image_2.nii.gz"),
    Path("/path/to/ct_image_3.mha"),
]

TEST_MRI_FILES = [
    Path("/path/to/mri_image_1.nii"),
    Path("/path/to/mri_image_2.mha"),
    Path("/path/to/mri_image_3.nii.gz"),
]