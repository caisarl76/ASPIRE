# cuRobo V2 motion-planning service for ASPIRE (cap_harness.providers.curobo.service, port 8118).
#
# Build from the repo root with the pinned cuRobo submodule checked out
# (configs/dependency-lock.json names the commit):
#   git submodule update --init third_party/curobo
#   docker build -f docker/curobo.Dockerfile -t aspire-curobo .
#   docker run -d --gpus '"device=2"' -p 8118:8118 aspire-curobo
#
# Same install as scripts/bootstrap_curobo.sh. This cuRobo compiles its kernels at
# runtime through cuda.core (no nvcc at build time), so the first plan for a robot
# model takes longer; the CUDA base image supplies the runtime libraries.
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /usr/local/bin/uv

WORKDIR /app
COPY . /app
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy
RUN uv venv /opt/venv --python 3.12 \
    && uv pip install --python /opt/venv/bin/python \
        'torch==2.9.1' 'torchvision==0.24.1' fastapi pydantic requests uvicorn \
    && uv pip install --python /opt/venv/bin/python --editable "/app/third_party/curobo[cu12-torch]" \
    && uv pip install --python /opt/venv/bin/python --editable /app --no-deps \
    && /opt/venv/bin/python -c "import curobo, torch, cap_harness.providers.curobo.service; assert torch.__version__.startswith('2.9.1')"

ENV PATH=/opt/venv/bin:$PATH
EXPOSE 8118
CMD ["python", "-m", "cap_harness.providers.curobo.service", "--host", "0.0.0.0", "--port", "8118"]
