FROM python:3.13-slim

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /bin/
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY . .
RUN uv sync --frozen --no-dev

# This tracked inherited scenario is selected by release/operate_v0_61_0/.
CMD ["uv", "run", "python", "run.py", "--scenario", "operate_v0_58_0/datacenter/gpu_cluster_queue_control/deep_planning/high/alibaba_gpu_native_500_dfc0551ac1_c9da905bb4_high", "--agent", "wait_only", "--seed", "42"]
