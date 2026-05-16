#!/bin/bash

# =============================================================================
# Scout Production Entrypoint
#
# 1. Validates system prerequisites (Chrome, Xvfb, memory, shm, etc.)
# 2. Cleans up stale resources from previous runs
# 3. Starts Xvfb virtual display (for headful mode)
# 4. Launches the user's application
# =============================================================================

set -euo pipefail

WARNINGS=0
ERRORS=0

echo ""
echo "=============================================================================="
echo "  SCOUT CONTAINER — STARTUP DIAGNOSTICS"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=============================================================================="

# -----------------------------------------------------------------------------
# 1. INIT PROCESS CHECK (Critical for zombie reaping)
# -----------------------------------------------------------------------------
echo ""
echo "[1/8] INIT PROCESS (Zombie Reaper)"
PID1_NAME=$(cat /proc/1/comm 2>/dev/null || echo "unknown")
echo "      PID 1: ${PID1_NAME}"

if [ "$PID1_NAME" = "tini" ] || [ "$PID1_NAME" = "dumb-init" ]; then
    echo "      [OK] Proper init process for zombie reaping"
elif [ "$PID1_NAME" = "bash" ] || [ "$PID1_NAME" = "sh" ]; then
    echo "      [ERROR] Shell as PID 1 — zombies will accumulate!"
    echo "      FIX: Add 'init: true' to docker-compose.yml"
    ERRORS=$((ERRORS + 1))
else
    echo "      [WARN] Unknown init — zombie reaping may not work"
    WARNINGS=$((WARNINGS + 1))
fi

# -----------------------------------------------------------------------------
# 2. SHARED MEMORY CHECK (Critical for Chrome)
# -----------------------------------------------------------------------------
echo ""
echo "[2/8] SHARED MEMORY (/dev/shm)"
SHM_SIZE_KB=$(df /dev/shm 2>/dev/null | tail -1 | awk '{print $2}')
if [ -n "$SHM_SIZE_KB" ]; then
    SHM_SIZE_MB=$((SHM_SIZE_KB / 1024))
    echo "      Size: ${SHM_SIZE_MB}MB"

    if [ "$SHM_SIZE_MB" -lt 512 ]; then
        echo "      [ERROR] Too small! Chrome needs at least 512MB, recommended 2GB"
        echo "      FIX: Add 'shm_size: 2gb' to docker-compose.yml"
        ERRORS=$((ERRORS + 1))
    elif [ "$SHM_SIZE_MB" -lt 1024 ]; then
        echo "      [WARN] Acceptable but 2GB recommended for concurrent scrapers"
        WARNINGS=$((WARNINGS + 1))
    else
        echo "      [OK]"
    fi
else
    echo "      [WARN] Could not check /dev/shm"
    WARNINGS=$((WARNINGS + 1))
fi

# -----------------------------------------------------------------------------
# 3. SYSTEM MEMORY CHECK (Container-aware)
# -----------------------------------------------------------------------------
echo ""
echo "[3/8] SYSTEM MEMORY"

MEM_SOURCE="unknown"
TOTAL_MEM_MB=0
AVAIL_MEM_MB=0

# Try cgroup v2 first (modern containers)
if [ -f /sys/fs/cgroup/memory.max ] && [ -f /sys/fs/cgroup/memory.current ]; then
    CGROUP_MAX=$(cat /sys/fs/cgroup/memory.max 2>/dev/null)
    CGROUP_CURRENT=$(cat /sys/fs/cgroup/memory.current 2>/dev/null)

    if [ "$CGROUP_MAX" != "max" ] && [ -n "$CGROUP_MAX" ] && [ -n "$CGROUP_CURRENT" ]; then
        TOTAL_MEM_MB=$((CGROUP_MAX / 1024 / 1024))
        USED_MEM_MB=$((CGROUP_CURRENT / 1024 / 1024))
        AVAIL_MEM_MB=$((TOTAL_MEM_MB - USED_MEM_MB))
        MEM_SOURCE="cgroup_v2"
    fi
fi

# Try cgroup v1
if [ "$MEM_SOURCE" = "unknown" ] && [ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
    CGROUP_LIMIT=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null)
    CGROUP_USAGE=$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null)

    if [ -n "$CGROUP_LIMIT" ] && [ "$CGROUP_LIMIT" -lt 1099511627776 ]; then
        TOTAL_MEM_MB=$((CGROUP_LIMIT / 1024 / 1024))
        USED_MEM_MB=$((CGROUP_USAGE / 1024 / 1024))
        AVAIL_MEM_MB=$((TOTAL_MEM_MB - USED_MEM_MB))
        MEM_SOURCE="cgroup_v1"
    fi
fi

# Fall back to /proc/meminfo
if [ "$MEM_SOURCE" = "unknown" ] && [ -f /proc/meminfo ]; then
    TOTAL_MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    AVAIL_MEM_KB=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
    TOTAL_MEM_MB=$((TOTAL_MEM_KB / 1024))
    AVAIL_MEM_MB=$((AVAIL_MEM_KB / 1024))
    MEM_SOURCE="proc_meminfo"
fi

if [ "$TOTAL_MEM_MB" -gt 0 ]; then
    echo "      Source: $MEM_SOURCE"
    echo "      Total: ${TOTAL_MEM_MB}MB, Available: ${AVAIL_MEM_MB}MB"

    if [ "$AVAIL_MEM_MB" -lt 1024 ]; then
        echo "      [ERROR] Less than 1GB available! Need at least 1GB per concurrent scraper"
        ERRORS=$((ERRORS + 1))
    elif [ "$AVAIL_MEM_MB" -lt 2048 ]; then
        echo "      [WARN] Less than 2GB available"
        WARNINGS=$((WARNINGS + 1))
    else
        echo "      [OK]"
    fi
else
    echo "      [WARN] Could not determine memory limits"
    WARNINGS=$((WARNINGS + 1))
fi

# -----------------------------------------------------------------------------
# 4. DISK SPACE CHECK (/tmp)
# -----------------------------------------------------------------------------
echo ""
echo "[4/8] DISK SPACE (/tmp)"
TMP_AVAIL_KB=$(df /tmp 2>/dev/null | tail -1 | awk '{print $4}')
if [ -n "$TMP_AVAIL_KB" ]; then
    TMP_AVAIL_MB=$((TMP_AVAIL_KB / 1024))
    echo "      Available: ${TMP_AVAIL_MB}MB"

    if [ "$TMP_AVAIL_MB" -lt 500 ]; then
        echo "      [ERROR] Less than 500MB free on /tmp"
        ERRORS=$((ERRORS + 1))
    elif [ "$TMP_AVAIL_MB" -lt 1000 ]; then
        echo "      [WARN] Less than 1GB free on /tmp"
        WARNINGS=$((WARNINGS + 1))
    else
        echo "      [OK]"
    fi
else
    echo "      [WARN] Could not check /tmp"
    WARNINGS=$((WARNINGS + 1))
fi

# -----------------------------------------------------------------------------
# 5. FILE DESCRIPTOR LIMITS
# -----------------------------------------------------------------------------
echo ""
echo "[5/8] FILE DESCRIPTOR LIMITS"
FD_SOFT=$(ulimit -Sn 2>/dev/null)
FD_HARD=$(ulimit -Hn 2>/dev/null)
echo "      Soft: ${FD_SOFT:-unknown}, Hard: ${FD_HARD:-unknown}"

if [ -n "$FD_SOFT" ] && [ "$FD_SOFT" -lt 4096 ]; then
    echo "      [WARN] Soft limit below 4096, may cause 'Too many open files'"
    WARNINGS=$((WARNINGS + 1))
else
    echo "      [OK]"
fi

# -----------------------------------------------------------------------------
# 6. CHROME INSTALLATION
# -----------------------------------------------------------------------------
echo ""
echo "[6/8] CHROME (Patchright)"
PATCHRIGHT_OK=$(python3 -c "import patchright; print('OK')" 2>/dev/null)
if [ "$PATCHRIGHT_OK" = "OK" ]; then
    echo "      patchright: installed"
    # Check for actual browser binary
    PATCHRIGHT_CHROME=$(find /home -name "chrome" -type f -executable 2>/dev/null | head -1)
    if [ -n "$PATCHRIGHT_CHROME" ]; then
        echo "      Chrome binary: $PATCHRIGHT_CHROME"
        echo "      [OK]"
    else
        echo "      [ERROR] Chrome binary not found! Run: python -m patchright install chromium"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "      [ERROR] patchright not installed!"
    ERRORS=$((ERRORS + 1))
fi

# -----------------------------------------------------------------------------
# 7. XVFB INSTALLATION
# -----------------------------------------------------------------------------
echo ""
echo "[7/8] XVFB (Virtual Display)"
XVFB_PATH=$(which Xvfb 2>/dev/null)
if [ -n "$XVFB_PATH" ]; then
    echo "      Path: $XVFB_PATH"
    echo "      [OK]"
else
    echo "      [WARN] Xvfb not found — headful mode will not work"
    WARNINGS=$((WARNINGS + 1))
fi

# -----------------------------------------------------------------------------
# 8. PYTHON DEPENDENCIES
# -----------------------------------------------------------------------------
echo ""
echo "[8/8] PYTHON DEPENDENCIES"
SCOUT_OK=$(python3 -c "import scout; print('OK')" 2>/dev/null)
if [ "$SCOUT_OK" = "OK" ]; then
    echo "      scout: installed"
else
    echo "      [ERROR] scout not installed!"
    ERRORS=$((ERRORS + 1))
fi

PSUTIL_OK=$(python3 -c "import psutil; print('OK')" 2>/dev/null)
if [ "$PSUTIL_OK" = "OK" ]; then
    echo "      psutil: installed (enhanced monitoring)"
else
    echo "      [WARN] psutil not installed (reduced monitoring)"
    WARNINGS=$((WARNINGS + 1))
fi
echo "      [OK]"

# -----------------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------------
echo ""
echo "=============================================================================="
if [ $ERRORS -gt 0 ]; then
    echo "  RESULT: $ERRORS ERROR(S), $WARNINGS WARNING(S)"
    echo "  Some checks failed! The container may not work correctly."
    echo "=============================================================================="
    echo ""
    echo "  Continuing startup anyway..."
elif [ $WARNINGS -gt 0 ]; then
    echo "  RESULT: OK with $WARNINGS WARNING(S)"
    echo "=============================================================================="
else
    echo "  RESULT: ALL CHECKS PASSED"
    echo "=============================================================================="
fi
echo ""

# -----------------------------------------------------------------------------
# CLEANUP STALE RESOURCES FROM PREVIOUS RUNS
# -----------------------------------------------------------------------------
echo "Cleaning stale resources from previous runs..."

# Remove old browser profiles (older than 15 minutes)
find /tmp -maxdepth 1 -name "scraping_agent_profile_*" -mmin +15 -exec rm -rf {} + 2>/dev/null || true
find /tmp -maxdepth 1 -name "scraper_profile_*" -mmin +15 -exec rm -rf {} + 2>/dev/null || true
find /tmp -maxdepth 1 -name "scrape_run_*" -mmin +15 -exec rm -rf {} + 2>/dev/null || true
find /tmp -maxdepth 1 -name "scrape_cp_*" -mmin +15 -exec rm -rf {} + 2>/dev/null || true
find /tmp -maxdepth 1 -name "scrape_checkpoints_*" -mmin +15 -exec rm -rf {} + 2>/dev/null || true

# Remove stale X11 lock files
find /tmp -maxdepth 1 -name ".X*-lock" -mmin +15 -delete 2>/dev/null || true

echo "Cleanup complete."
echo ""

# -----------------------------------------------------------------------------
# START XVFB (for headful mode support)
# -----------------------------------------------------------------------------
if [ "${SCOUT_HEADLESS:-false}" != "true" ] && [ -n "$XVFB_PATH" ]; then
    echo "Starting Xvfb on display ${DISPLAY:-:99}..."
    Xvfb ${DISPLAY:-:99} -screen 0 1920x1080x24 -ac -noreset &
    XVFB_PID=$!

    # Wait for Xvfb to be ready
    for i in $(seq 1 20); do
        if xdpyinfo -display ${DISPLAY:-:99} >/dev/null 2>&1; then
            echo "Xvfb ready (PID: $XVFB_PID)"
            break
        fi
        sleep 0.3
    done

    if ! xdpyinfo -display ${DISPLAY:-:99} >/dev/null 2>&1; then
        echo "[WARN] Xvfb did not become ready — headful mode may fail"
    fi
    echo ""
fi

# -----------------------------------------------------------------------------
# START APPLICATION
# -----------------------------------------------------------------------------
echo "Starting Scout application..."
echo ""

# If a custom command is provided, execute it.
# Otherwise, start the default Python application.
if [ $# -gt 0 ]; then
    exec "$@"
else
    # Default: run the user's script or start an interactive shell
    if [ -f /app/run.py ]; then
        exec python3 /app/run.py
    else
        echo "No /app/run.py found. Container is ready."
        echo "Mount your script as /app/run.py or pass a command."
        echo ""
        # Keep container alive for debugging / manual use
        exec tail -f /dev/null
    fi
fi
