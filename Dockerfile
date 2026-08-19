FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir .
