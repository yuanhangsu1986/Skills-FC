#!/usr/bin/env bash
# Serve the S2S FC Eval Scorecard via HTTP from /lustre, then print the SSH
# tunnel command so you can open it in a browser on your laptop.
#
# Usage: bash scripts/serve_scorecard.sh [--name <filename>] [--port <port>]
#   --name  HTML filename under asset/ (default: scorecard.html).
#           Extension .html is added automatically if omitted.
#   --port  Port to listen on (default: auto-select from 8780-8799).
#
# Examples:
#   bash scripts/serve_scorecard.sh
#   bash scripts/serve_scorecard.sh --name my_run
#   bash scripts/serve_scorecard.sh --name my_run --port 8790

set -euo pipefail

SERVE_ROOT="/lustre"
ASSET_REL="fsw/portfolios/llmservice/users/yuanhangs/codes/Skills-FC/asset"

# ── Parse args ────────────────────────────────────────────────────────────────
NAME="scorecard"
PORT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name) NAME="${2%.html}"; shift 2 ;;
        --port) PORT="$2";        shift 2 ;;
        *)      echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

SCORECARD_REL="${ASSET_REL}/${NAME}.html"

# ── Find a free port ──────────────────────────────────────────────────────────
find_free_port() {
    for port in $(seq 8780 8799); do
        if ! ss -ltn 2>/dev/null | grep -q ":${port} " && \
           ! lsof -iTCP:"${port}" -sTCP:LISTEN -t 2>/dev/null | grep -q .; then
            echo "$port"; return 0
        fi
    done
    echo ""
}

if [[ -z "$PORT" ]]; then
    PORT=$(find_free_port)
fi

if [[ -z "$PORT" ]]; then
    echo "ERROR: no free port in 8780-8799. Use --port to specify one." >&2
    exit 1
fi

# ── Verify scorecard exists ───────────────────────────────────────────────────
SCORECARD_ABS="${SERVE_ROOT}/${SCORECARD_REL}"
if [[ ! -f "$SCORECARD_ABS" ]]; then
    echo "WARNING: scorecard not found at ${SCORECARD_ABS}" >&2
fi

# ── Start server ──────────────────────────────────────────────────────────────
LOGIN_HOST="draco-oci-login-01.draco-oci-iad.nvidia.com"
LOG_FILE="/tmp/scorecard_server_${PORT}.log"

# ── List available scorecards ─────────────────────────────────────────────────
ASSET_ABS="${SERVE_ROOT}/${ASSET_REL}"
echo "Available scorecards in ${ASSET_ABS}:"
if compgen -G "${ASSET_ABS}/*.html" > /dev/null 2>&1; then
    for f in "${ASSET_ABS}"/*.html; do
        fname=$(basename "$f")
        echo "  http://localhost:${PORT}/${ASSET_REL}/${fname}"
    done
else
    echo "  (none found)"
fi
echo ""

echo "Starting HTTP server on ${LOGIN_HOST}:${PORT} (root: ${SERVE_ROOT})"
echo "Scorecard: ${SCORECARD_ABS}"
echo "Log: ${LOG_FILE}"
echo ""

cd "${SERVE_ROOT}"
python3 -m http.server "${PORT}" --bind 127.0.0.1 \
    > "${LOG_FILE}" 2>&1 &
SERVER_PID=$!

sleep 1
if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "ERROR: server failed to start. Check ${LOG_FILE}" >&2
    exit 1
fi

echo "Server PID: ${SERVER_PID}"
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Run this on your LAPTOP:"
echo ""
echo "    ssh -L ${PORT}:127.0.0.1:${PORT} ${USER}@${LOGIN_HOST} -N &"
echo ""
echo "  Then open in browser:"
echo "    http://localhost:${PORT}/${SCORECARD_REL}"
echo ""
echo "  To stop the server:"
echo "    kill ${SERVER_PID}"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Waiting (Ctrl-C to stop)..."

TAIL_PID=""
trap 'echo ""; echo "Stopping server (PID ${SERVER_PID})..."; kill "${SERVER_PID}" 2>/dev/null; [[ -n "${TAIL_PID}" ]] && kill "${TAIL_PID}" 2>/dev/null; exit 0' INT TERM

tail -f "${LOG_FILE}" &
TAIL_PID=$!

wait "${SERVER_PID}"
