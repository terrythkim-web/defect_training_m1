# DefectLab development container (Linux). Made for Docker Desktop on Windows 11 (WSL2 + WSLg shows
# the app window on the Windows desktop) and works on any Linux host. See docs/docker_dev.md.
#
#   docker compose build dev        CUDA PyTorch (NVIDIA GPU at run time)
#   docker compose build dev-cpu    CPU PyTorch
#
# The image holds the Python environment (/opt/venv) with the package installed editable from
# /workspace; compose bind-mounts the source over /workspace, so code edits need no rebuild.
# Rebuild after changing dependencies in pyproject.toml.
FROM python:3.12-slim-bookworm

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG USER_UID=1000
ARG USER_GID=1000
ARG INSTALL_CLAUDE_CODE=1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Qt (PySide6) needs the xcb / EGL / font libraries; fonts-noto-cjk = Korean UI text and report charts.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential sudo \
        libgl1 libegl1 libglib2.0-0 libfontconfig1 libdbus-1-3 \
        libxkbcommon0 libxkbcommon-x11-0 libx11-xcb1 \
        libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
        libxcb-shape0 libxcb-xinerama0 libxcb-xfixes0 libxcb-xkb1 \
        libwayland-client0 libwayland-cursor0 libwayland-egl1 \
        fonts-noto-cjk \
        nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# Non-root user with the usual WSL user id, so files written into the bind mount stay yours.
RUN groupadd --gid "$USER_GID" dev \
    && useradd --uid "$USER_UID" --gid "$USER_GID" --create-home --shell /bin/bash dev \
    && echo "dev ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/dev \
    && python -m venv /opt/venv && chown -R dev:dev /opt/venv \
    && mkdir -p /workspace /projects /home/dev/.cache /home/dev/.claude \
    && chown dev:dev /workspace /projects /home/dev/.cache /home/dev/.claude

# Claude Code CLI (`claude`), to continue development inside the container.
RUN if [ "$INSTALL_CLAUDE_CODE" = "1" ]; then npm install -g @anthropic-ai/claude-code; fi

USER dev
WORKDIR /workspace
RUN pip install --upgrade pip \
    && pip install torch torchvision --index-url "$TORCH_INDEX"
COPY --chown=dev:dev pyproject.toml README.md ./
COPY --chown=dev:dev defect_inspection ./defect_inspection
RUN pip install -e ".[dev,package,bench]"

CMD ["bash"]
