import os
from pathlib import Path

# Defaults match docker mounts
DEFAULT_INPUT_PATH = Path("/input")
DEFAULT_OUTPUT_PATH = Path("/output")
DEFAULT_MODEL_PATH = Path("/opt/ml/model")


def default_input_path() -> Path:
    """Return default input path, allowing override via env var."""
    return Path(os.getenv("INPUT_PATH", DEFAULT_INPUT_PATH))


def default_output_path() -> Path:
    """Return default output path, allowing override via env var."""
    return Path(os.getenv("OUTPUT_PATH", DEFAULT_OUTPUT_PATH))


def default_model_path() -> Path:
    """Return default model path, allowing override via env var."""
    return Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
