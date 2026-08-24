FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y openssh-server \
    && mkdir /var/run/sshd

RUN apt-get install -y build-essential

COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir ".[dev]"
