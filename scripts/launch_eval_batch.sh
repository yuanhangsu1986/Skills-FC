#!/bin/bash
# Launch S2S FC eval suite (run_all_benchmarks.sh) for every ckpt in a list.
#
# For each line in --list (one ckpt path per line), runs:
#   bash scripts/run_all_benchmarks.sh \
#       --benchmarks <BENCHMARKS> \
#       --top_p 1.0 --repetition_penalty 1.0 --temperature 0.0 \
#       --force_turn_taking false \
#       --model <ckpt> \
#       --output_dir <OUTPUT_ROOT>/<basename(ckpt)>
#
# Output for each ckpt lands at <OUTPUT_ROOT>/<basename(ckpt)>/<benchmark>_<commit>/...
# Launcher logs (stdout+stderr of each run_all_benchmarks invocation) land at
# <OUTPUT_ROOT>/_launcher_logs/<basename(ckpt)>.log and are also printed to the
# screen via tee.
#
# Sequential: one ckpt's 74 sbatch submissions all complete (the launcher exits)
# before the next ckpt starts. Slurm jobs themselves run async after submission.
#
# Auto-detach: the script re-execs itself under nohup on first invocation, so an
# ssh disconnect or terminal close cannot kill the loop. The foreground call
# prints the detach-log path + PID and exits; the background worker runs the
# main loop. Follow live with `tail -f <detach-log>`. Per-ckpt logs at
# $OUTPUT_ROOT/_launcher_logs/<ckpt>.log still work exactly as before.
#
# Usage:
#   bash launch_eval_batch.sh \
#     [--list PATH] \
#     [--output_root PATH] \
#     [--benchmarks LIST] \
#     [--repo PATH] \
#     [--start N] \
#     [--dry_run]
#
# Defaults reflect the S2S_hf workflow:
#   --list         /lustre/fsw/.../yuanhangs/models/S2S_hf/ckpts_list.txt
#   --output_root  /lustre/fsw/.../yuanhangs/workspace/voice_chat/sampling+greedy/greedy
#   --benchmarks   vb_mcq,vb_nonmcq,fdb_v1,fdb_v1_5,fdb_v3,conv_behav,bfcl,bba
#   --repo         /lustre/fs12/.../yuanhangs/codes/Skills-FC
#   --start        1  (line number to start at — use 2 to skip line 1, etc.)
#   --max_jobs     unset (run_all_benchmarks.sh auto-detects from SLURM limits)
#   --resume       off (pass --resume to run_all_benchmarks.sh — re-enters
#                       existing output dirs without forcing scoring redo;
#                       useful when a previous batch had partial/failed jobs)

# pipefail intentionally NOT set: we want the loop to continue to the next ckpt
# even if one invocation of run_all_benchmarks.sh fails or emits non-zero.
set -eu

# ---- defaults ----
LIST=/lustre/fsw/portfolios/llmservice/users/yuanhangs/models/S2S_hf/ckpts_list.txt
OUTPUT_ROOT=/lustre/fsw/portfolios/llmservice/users/yuanhangs/workspace/voice_chat/sampling+greedy/greedy
BENCHMARKS="vb_mcq,vb_nonmcq,fdb_v1,fdb_v1_5,fdb_v3,conv_behav,bfcl,bba"
REPO=/lustre/fs12/portfolios/llmservice/projects/llmservice_nemo_mlops/users/yuanhangs/codes/Skills-FC
START=1
DRY_RUN=""
MAX_JOBS=""
RESUME=""
USE_RNNT_TURN_TAKING=false   # opt-in

# Snapshot original args BEFORE the while loop below consumes them via `shift`,
# so the nohup re-exec in the auto-detach block can forward them verbatim.
ORIGINAL_ARGS=("$@")

# ---- args ----
while [ $# -gt 0 ]; do
  case "$1" in
    --list)         LIST="$2"; shift 2 ;;
    --output_root)  OUTPUT_ROOT="$2"; shift 2 ;;
    --benchmarks)   BENCHMARKS="$2"; shift 2 ;;
    --repo)         REPO="$2"; shift 2 ;;
    --start)        START="$2"; shift 2 ;;
    --max_jobs)     MAX_JOBS="$2"; shift 2 ;;
    --use_rnnt_turn_taking) USE_RNNT_TURN_TAKING="$2"; shift 2 ;;
    --resume)       RESUME="--resume"; shift ;;
    --dry_run)      DRY_RUN="--dry_run"; shift ;;
    -h|--help)      sed -n '1,50p' "$0"; exit 0 ;;
    *) echo "ERROR: unknown arg: $1" >&2; exit 1 ;;
  esac
done

[ -f "$LIST" ] || { echo "ERROR: --list not found: $LIST" >&2; exit 1; }
[ -d "$REPO" ] || { echo "ERROR: --repo not found: $REPO" >&2; exit 1; }
[ -f "$REPO/scripts/run_all_benchmarks.sh" ] \
    || { echo "ERROR: $REPO/scripts/run_all_benchmarks.sh missing" >&2; exit 1; }
case "$START" in *[!0-9]*|"") echo "ERROR: --start must be a positive integer" >&2; exit 1;; esac
[ "$START" -ge 1 ] || { echo "ERROR: --start must be >= 1" >&2; exit 1; }
if [ -n "$MAX_JOBS" ]; then
    case "$MAX_JOBS" in *[!0-9]*|"") echo "ERROR: --max_jobs must be a positive integer" >&2; exit 1;; esac
    [ "$MAX_JOBS" -ge 1 ] || { echo "ERROR: --max_jobs must be >= 1" >&2; exit 1; }
fi

LOG_DIR="$OUTPUT_ROOT/_launcher_logs"
mkdir -p "$LOG_DIR"

# Auto-detach via nohup so the loop survives ssh disconnect / terminal close.
# Re-execs self in the background; the child skips this block via the env flag.
if [ "${BATCH_LAUNCHER_DETACHED:-}" != "1" ]; then
    DETACH_LOG="$LOG_DIR/_batch_launcher_$(date +%Y%m%d_%H%M%S).log"
    export BATCH_LAUNCHER_DETACHED=1
    echo "Detaching launcher (nohup) — survives ssh disconnect."
    echo "  Log:    $DETACH_LOG"
    echo "  Follow: tail -f $DETACH_LOG"
    nohup bash "$0" "${ORIGINAL_ARGS[@]+"${ORIGINAL_ARGS[@]}"}" >"$DETACH_LOG" 2>&1 </dev/null &
    disown
    echo "  PID:    $!"
    exit 0
fi

cd "$REPO"

echo "=== launch_eval_batch ==="
echo "  list:        $LIST  ($(grep -c . "$LIST") non-empty lines)"
echo "  output:      $OUTPUT_ROOT"
echo "  benchmarks:  $BENCHMARKS"
echo "  repo (cwd):  $REPO"
echo "  start line:  $START"
echo "  max_jobs:    ${MAX_JOBS:-auto-detect by run_all_benchmarks.sh}"
echo "  resume:      ${RESUME:-no}"
echo "  log dir:     $LOG_DIR"
echo "  dry_run:     ${DRY_RUN:-no}"
echo

idx=0
tail -n "+$START" "$LIST" | while IFS= read -r ckpt; do
    idx=$((idx+1))
    [ -z "$ckpt" ] && continue
    [ "${ckpt#\#}" != "$ckpt" ] && continue
    name=$(basename "$ckpt")
    log="$LOG_DIR/${name}.log"
    echo "[$idx] $(date '+%Y-%m-%d %H:%M:%S')  launching: $name"
    echo "    log:   $log"
    # `</dev/null` is critical: without it, the ckpt-list file is the stdin of
    # the whole while loop, and any read() inside nemo-run/nemo-skills would
    # consume the rest of the ckpts -> the next read in the loop hits EOF and
    # the loop bails after ckpt #1.
    bash scripts/run_all_benchmarks.sh \
        --benchmarks "$BENCHMARKS" \
        --top_p 1.0 \
        --repetition_penalty 1.0 \
        --temperature 0.0 \
        --force_turn_taking false \
        --model "$ckpt" \
        --output_dir "$OUTPUT_ROOT/$name" \
        --use_rnnt_turn_taking "$USE_RNNT_TURN_TAKING" \
        ${MAX_JOBS:+--max_jobs $MAX_JOBS} \
        $RESUME \
        $DRY_RUN \
        </dev/null 2>&1 | tee "$log"
done

echo
echo "done."
