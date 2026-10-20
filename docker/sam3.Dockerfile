# SAM3 segmentation service for ASPIRE (cap_harness.providers.sam3.service, port 8114).
#
# Build from the repo root with the third_party/sam3 submodule checked out:
#   git submodule update --init third_party/sam3
#   docker build -f docker/sam3.Dockerfile -t aspire-sam3 .
#
# Weights are not baked in. Mount a Hugging Face cache holding facebook/sam3
# (config.json and sam3.pt; the repo is gated) at /hf. The service then loads
# offline and the container needs no token:
#   docker run -d --gpus '"device=3"' -p 8114:8114 -v <hf-cache>:/hf:ro aspire-sam3
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.11.14

WORKDIR /app
COPY . /app
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy
# Same resolution as scripts/bootstrap_provider.sh sam3: the frozen lock plus the extra.
RUN uv sync --frozen --extra providers-sam3 \
    && /opt/venv/bin/python -c "import sam3, cap_harness.providers.sam3.service"

ENV PATH=/opt/venv/bin:$PATH HF_HUB_CACHE=/hf HF_HUB_OFFLINE=1
EXPOSE 8114
CMD ["python", "-m", "cap_harness.providers.sam3.service", "--host", "0.0.0.0", "--port", "8114", "--device", "cuda"]
