#!/usr/bin/env bash
# launch_local.sh — bring up the full reviewing stack on the local host.
#
#   1. Verify nvidia-smi finds a CUDA GPU.
#   2. Launch `paperlens serve` (paperlens-training-and-inference) in background
#      and wait for its /health.
#   3. Launch `paperlensreview serve` in foreground; print its URL.
#
# Stop:
#   pkill -f "paperlens serve"
#   pkill -f "paperlensreview serve"
#
# Env / overrides:
#   PAPERLENS_SERVE_CONFIG   path to paperlens-training-and-inference/configs/serve.yaml
#   PAPERLENSREVIEW_CONFIG   path to paperlens-reviewing/configs/server.yaml
#   PAPERLENS_SERVE_PORT     default 8002
#   PAPERLENSREVIEW_PORT     default 8003
#   PAPERLENS_TRAIN_AND_INFER_VENV  python venv that has the `paperlens` CLI
#   PAPERLENSREVIEW_VENV     python venv that has the `paperlensreview` CLI
#
# If both share a venv, set them to the same path.

set -euo pipefail

HERE="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
REVIEW_ROOT="$(cd "$HERE/.." && pwd)"

# ----- 0. Defaults --------------------------------------------------------
PAPERLENS_SERVE_PORT="${PAPERLENS_SERVE_PORT:-8002}"
PAPERLENSREVIEW_PORT="${PAPERLENSREVIEW_PORT:-8003}"

# Best-effort discovery of the sibling paperlens-training-and-inference clone.
if [[ -z "${PAPERLENS_SERVE_CONFIG:-}" ]]; then
    for candidate in \
        "$REVIEW_ROOT/../../paperlens-training-and-inference/configs/serve.yaml" \
        "$REVIEW_ROOT/../paperlens-training-and-inference/configs/serve.yaml" \
        "/scratch/gpfs/ZHUANGL/sk7524/paperlens-training-and-inference/configs/serve.yaml"
    do
        if [[ -f "$candidate" ]]; then
            PAPERLENS_SERVE_CONFIG="$candidate"
            break
        fi
    done
fi
if [[ -z "${PAPERLENS_SERVE_CONFIG:-}" || ! -f "$PAPERLENS_SERVE_CONFIG" ]]; then
    echo "ERROR: PAPERLENS_SERVE_CONFIG not set and no sibling config found." >&2
    echo "Point it at paperlens-training-and-inference/configs/serve.yaml" >&2
    exit 2
fi
PAPERLENSREVIEW_CONFIG="${PAPERLENSREVIEW_CONFIG:-$REVIEW_ROOT/configs/server.yaml}"

echo "[launcher] review root:           $REVIEW_ROOT"
echo "[launcher] paperlens serve cfg:   $PAPERLENS_SERVE_CONFIG  (port $PAPERLENS_SERVE_PORT)"
echo "[launcher] paperlensreview cfg:   $PAPERLENSREVIEW_CONFIG  (port $PAPERLENSREVIEW_PORT)"

# ----- 1. NVIDIA GPU sanity ----------------------------------------------
echo ""
echo "[launcher] step 1/3: NVIDIA GPU check"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "  ERROR: nvidia-smi not on PATH; paperlens serve needs a CUDA GPU." >&2
    exit 2
fi
GPU_LINE=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
if [[ -z "$GPU_LINE" ]]; then
    echo "  ERROR: nvidia-smi found no GPUs." >&2
    exit 2
fi
echo "  ✓ GPU: $GPU_LINE"

# ----- 2. Launch paperlens serve ------------------------------------------
echo ""
echo "[launcher] step 2/3: launch paperlens serve"
LOG_DIR="${LOG_DIR:-$REVIEW_ROOT/logs}"
mkdir -p "$LOG_DIR"
PAPERLENS_LOG="$LOG_DIR/paperlens_serve.log"

# Activate the upstream venv if specified
PAPERLENS_VENV="${PAPERLENS_TRAIN_AND_INFER_VENV:-}"
PAPERLENS_PY="paperlens"
if [[ -n "$PAPERLENS_VENV" && -f "$PAPERLENS_VENV/bin/activate" ]]; then
    # shellcheck disable=SC1090,SC1091
    source "$PAPERLENS_VENV/bin/activate"
    PAPERLENS_PY="$(command -v paperlens || true)"
    if [[ -z "$PAPERLENS_PY" ]]; then
        echo "  ERROR: 'paperlens' not found in venv $PAPERLENS_VENV" >&2
        exit 2
    fi
fi

# Already running? Skip.
if curl -sf "http://127.0.0.1:${PAPERLENS_SERVE_PORT}/health" >/dev/null 2>&1; then
    echo "  ✓ paperlens serve already up on :${PAPERLENS_SERVE_PORT}"
else
    echo "  starting paperlens serve, logs -> $PAPERLENS_LOG"
    nohup paperlens serve \
        --config "$PAPERLENS_SERVE_CONFIG" \
        --port "$PAPERLENS_SERVE_PORT" \
        >"$PAPERLENS_LOG" 2>&1 &
    PAPERLENS_PID=$!
    echo "  paperlens serve PID: $PAPERLENS_PID"
    echo "  waiting for /health (up to 10 min) ..."
    for _ in {1..120}; do
        if curl -sf "http://127.0.0.1:${PAPERLENS_SERVE_PORT}/health" >/dev/null 2>&1; then
            echo "  ✓ paperlens serve healthy"
            break
        fi
        sleep 5
    done
    if ! curl -sf "http://127.0.0.1:${PAPERLENS_SERVE_PORT}/health" >/dev/null 2>&1; then
        echo "  ERROR: paperlens serve never came up. Check $PAPERLENS_LOG" >&2
        exit 1
    fi
fi

# Deactivate paperlens venv before activating the reviewing one
if [[ -n "$PAPERLENS_VENV" ]]; then
    deactivate 2>/dev/null || true
fi

# ----- 3. Launch paperlensreview server ----------------------------------
echo ""
echo "[launcher] step 3/3: launch paperlensreview"
REVIEW_VENV="${PAPERLENSREVIEW_VENV:-}"
if [[ -n "$REVIEW_VENV" && -f "$REVIEW_VENV/bin/activate" ]]; then
    # shellcheck disable=SC1090,SC1091
    source "$REVIEW_VENV/bin/activate"
fi

if ! command -v paperlensreview >/dev/null 2>&1; then
    echo "  ERROR: paperlensreview CLI not on PATH (uv pip install -e .)" >&2
    exit 2
fi

URL="http://$(hostname):${PAPERLENSREVIEW_PORT}"
echo ""
echo "================================================================"
echo "  🌐 PaperLens reviewing UI ready at:"
echo "     $URL"
echo "================================================================"
echo ""

exec paperlensreview serve \
    --config "$PAPERLENSREVIEW_CONFIG" \
    --port "$PAPERLENSREVIEW_PORT"
