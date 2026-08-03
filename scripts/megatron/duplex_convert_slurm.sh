#!/usr/bin/env bash
# Run this file inside the conversion container.
set -euo pipefail

usage() {
    echo "Usage: $0 --checkpoint PATH --output-dir PATH --repo-root PATH --container-image PATH [--python PATH]" >&2
}

CHECKPOINT=""
OUTPUT_DIR=""
REPO_ROOT=""
PYTHON_BIN="python3"
CONTAINER_IMAGE=""
INSIDE_CONTAINER=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --repo-root) REPO_ROOT="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        --container-image) CONTAINER_IMAGE="$2"; shift 2 ;;
        --inside-container) INSIDE_CONTAINER=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$CHECKPOINT" || -z "$OUTPUT_DIR" || -z "$REPO_ROOT" || -z "$CONTAINER_IMAGE" ]]; then
    usage
    exit 1
fi

# Pyxis options are srun options on this cluster, not sbatch options. The
# allocation starts this host-side wrapper, which re-enters itself inside the
# conversion container with the same explicit arguments.
if [[ "$INSIDE_CONTAINER" != "true" ]]; then
    INNER_WRAPPER="$REPO_ROOT/scripts/megatron/duplex_convert_slurm.sh"
    if [[ ! -x "$INNER_WRAPPER" ]]; then
        echo "ERROR: Conversion wrapper is not executable: $INNER_WRAPPER" >&2
        exit 1
    fi
    exec srun \
        --nodes 1 \
        --ntasks 1 \
        --gpus-per-node 1 \
        --container-image "$CONTAINER_IMAGE" \
        --container-mounts "/lustre,/home" \
        "$INNER_WRAPPER" \
        --checkpoint "$CHECKPOINT" \
        --output-dir "$OUTPUT_DIR" \
        --repo-root "$REPO_ROOT" \
        --python "$PYTHON_BIN" \
        --container-image "$CONTAINER_IMAGE" \
        --inside-container
fi

MANAGER="$REPO_ROOT/scripts/megatron/duplex_checkpoint_manager.py"
EXPORTER="$REPO_ROOT/scripts/megatron/duplex_export_hybrid_checkpoint.py"
VERIFIER="$REPO_ROOT/scripts/megatron/duplex_verify_hybrid_checkpoint.py"
LOCK_DIR="$OUTPUT_DIR/conversion_logs"
LOCK_FILE="$LOCK_DIR/.conversion.lock"

mkdir -p "$LOCK_DIR"
exec 9>"$LOCK_FILE"
if ! flock -w 7200 9; then
    echo "ERROR: Timed out waiting for conversion lock: $LOCK_FILE" >&2
    exit 1
fi

if "$PYTHON_BIN" "$MANAGER" validate-export \
    --checkpoint "$CHECKPOINT" --export-dir "$OUTPUT_DIR" --quiet; then
    echo "Valid converted checkpoint already exists; skipping conversion: $OUTPUT_DIR"
    exit 0
fi

echo "Converting Megatron Duplex checkpoint"
echo "  source: $CHECKPOINT"
echo "  output: $OUTPUT_DIR"
"$PYTHON_BIN" "$EXPORTER" \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUTPUT_DIR" \
    --overwrite

"$PYTHON_BIN" "$VERIFIER" validate-export --export-dir "$OUTPUT_DIR"
"$PYTHON_BIN" "$MANAGER" validate-export \
    --checkpoint "$CHECKPOINT" --export-dir "$OUTPUT_DIR"
echo "Megatron Duplex conversion completed successfully: $OUTPUT_DIR"
