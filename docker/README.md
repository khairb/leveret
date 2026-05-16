# Scout Docker

Run Scout scrapers in a Docker container with Chrome, Xvfb (headful support), and tini (zombie reaping).

## Quick Start

```bash
cd docker

# Build
docker compose build

# Run (mount your script as /app/run.py)
docker compose up
```

## What's Included

- **Ubuntu 24.04** with Chrome/Chromium via Patchright
- **Xvfb** for headful browser mode (no physical display needed)
- **tini** as PID 1 for zombie process reaping
- **Startup diagnostics** that validate Chrome, memory, shm, FDs before running
- **Stale resource cleanup** on container start

## Configuration

| Variable | Default | Description |
|---|---|---|
| `SCOUT_HEADLESS` | `false` | Set to `true` to skip Xvfb |
| `SCOUT_BROWSER_CHANNEL` | auto-detect | Force `chrome` or `chromium` |
| `ANTHROPIC_API_KEY` | - | API key for script generation |

## Container Requirements

| Resource | Minimum | Recommended |
|---|---|---|
| Memory | 1 GB | 4 GB |
| /dev/shm | 512 MB | 2 GB |

The `docker-compose.yml` sets `shm_size: 2gb`, `init: true`, and `memory: 4g` by default.

## Building on macOS (Apple Silicon)

If `apt-get install` fails with "Hash Sum mismatch", the Dockerfile includes a fix
(`Acquire::http::Pipeline-Depth "0"`) that resolves this Docker Desktop networking issue.

If it persists, reset Docker Desktop: Settings > Troubleshoot > Reset to factory defaults.
