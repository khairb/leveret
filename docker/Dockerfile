# syntax=docker/dockerfile:1

# =============================================================================
# Scout Production Container
#
# Multi-stage build for running Scout scrapers in production.
# Supports both headless and headful (via Xvfb) browser modes.
#
# Build:   docker compose build
# Run:     docker compose up
# Test:    docker compose --profile test up battle-test
# =============================================================================

# =============================================================================
# Stage 1: Builder — install Python dependencies
# =============================================================================
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Disable HTTP pipelining — Docker VM networking can corrupt pipelined responses
RUN echo 'Acquire::http::Pipeline-Depth "0";' > /etc/apt/apt.conf.d/99docker-fix

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    gcc libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy project files for pip install.
COPY pyproject.toml ./
COPY src/ ./src/

# Install Scout + psutil system-wide (builder stage only).
RUN pip3 install --no-cache-dir --break-system-packages . psutil

# =============================================================================
# Stage 2: Runtime — minimal image with Chrome + Xvfb
# =============================================================================
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DISPLAY=:99

# Install runtime dependencies:
#   - xvfb: virtual display for headful Chrome in containers
#   - x11-utils: xdpyinfo for display readiness checks
#   - tini: PID 1 init process — reaps zombie Chrome processes
#   - procps: ps command for process monitoring
#   - fonts: Chrome needs fonts to render pages correctly
# Disable HTTP pipelining — Docker VM networking can corrupt pipelined responses
RUN echo 'Acquire::http::Pipeline-Depth "0";' > /etc/apt/apt.conf.d/99docker-fix

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    xvfb \
    x11-utils \
    tini \
    procps \
    fonts-liberation \
    fonts-noto-color-emoji \
    && rm -rf /var/lib/apt/lists/* \
              /usr/share/doc \
              /usr/share/man

WORKDIR /app

# Copy installed Python packages from builder.
COPY --from=builder /usr/local/lib/python3.12/dist-packages /usr/local/lib/python3.12/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Install Patchright's Chrome browser + system dependencies.
# On amd64: installs Google Chrome. On arm64: installs Chromium (Chrome unavailable).
RUN python3 -m patchright install --with-deps chrome 2>/dev/null || \
    python3 -m patchright install --with-deps chromium
RUN chmod -R 755 /root/.cache/ms-playwright

# Copy entrypoint.
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Copy the Scout source.
COPY src/ /app/src/

# Copy non-Python data files that pip doesn't include in the wheel.
# These are needed by the agent at runtime.
COPY src/scout/agent/patchright_guide.md /usr/local/lib/python3.12/dist-packages/scout/agent/patchright_guide.md
COPY src/scout/agent/_model_db_cache.json /usr/local/lib/python3.12/dist-packages/scout/agent/_model_db_cache.json

# Create required directories.
RUN mkdir -p /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix \
    && mkdir -p /app/scripts

# Create non-root user for security.
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app /tmp/.X11-unix \
    && mkdir -p /home/appuser/.cache \
    && cp -r /root/.cache/ms-playwright /home/appuser/.cache/ms-playwright \
    && chown -R appuser:appuser /home/appuser/.cache
USER appuser

# Scripts volume mount point — persistent across container restarts.
VOLUME ["/app/scripts"]

# Use tini as PID 1 to properly reap zombie Chrome processes.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/entrypoint.sh"]
