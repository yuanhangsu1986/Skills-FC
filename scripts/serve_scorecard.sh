#!/usr/bin/env bash
# Serve the S2S FC Eval Scorecard via HTTP from /lustre, then print the SSH
# tunnel command so you can open it in a browser on your laptop.
#
# Usage: bash scripts/serve_scorecard.sh [--name <filename>] [--port <port>]
#   --name  Highlight a specific HTML file in the printed URL list.
#           Extension .html is added automatically if omitted.
#           If omitted, all files in asset/ are listed.
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
NAME=""
PORT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name) NAME="${2%.html}"; shift 2 ;;
        --port) PORT="$2";        shift 2 ;;
        *)      echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

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

# ── Start server ──────────────────────────────────────────────────────────────
LOGIN_HOST="draco-oci-login-01.draco-oci-iad.nvidia.com"
LOG_FILE="/tmp/scorecard_server_${PORT}.log"
ASSET_ABS="${SERVE_ROOT}/${ASSET_REL}"

cd "${SERVE_ROOT}"
python3 -c "
import http.server, socketserver, sys
class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write('%s - - [%s] %s\n' % (self.address_string(),
            self.log_date_time_string(), fmt % args))
socketserver.ThreadingTCPServer.allow_reuse_address = True
with socketserver.ThreadingTCPServer(('127.0.0.1', ${PORT}), Handler) as s:
    s.serve_forever()
" > "${LOG_FILE}" 2>&1 &
SERVER_PID=$!

sleep 1
if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "ERROR: server failed to start. Check ${LOG_FILE}" >&2
    exit 1
fi

# ── Build URL list ────────────────────────────────────────────────────────────
url_lines=""
if compgen -G "${ASSET_ABS}/*.html" > /dev/null 2>&1; then
    for f in "${ASSET_ABS}"/*.html; do
        fname=$(basename "$f" .html)
        url="http://localhost:${PORT}/${ASSET_REL}/${fname}.html"
        if [[ -n "$NAME" && "$fname" == "$NAME" ]]; then
            url_lines+="    ${url}  ← ${NAME}"$'\n'
        else
            url_lines+="    ${url}"$'\n'
        fi
    done
else
    url_lines="    (no HTML files found in ${ASSET_ABS})"$'\n'
fi

echo "════════════════════════════════════════════════════════════"
echo "  Run this on your LAPTOP:"
echo ""
echo "    ssh -L ${PORT}:127.0.0.1:${PORT} ${USER}@${LOGIN_HOST} -N &"
echo ""
echo "  Then open in browser:"
echo "${url_lines}"
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
