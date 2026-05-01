#!/usr/bin/env bash
# run_all_benchmarks.sh
#
# Submit all S2S FC eval benchmarks sequentially, throttling submission based
# on the SLURM job-count limit (queried from sacctmgr QOS settings and
# scontrol partition settings; the tighter of the two is used).
#
# Run from the repo root:
#   bash asset/run_all_benchmarks.sh [options]
#
# Examples:
#   # Run everything with defaults
#   bash asset/run_all_benchmarks.sh
#
#   # Run only BBA and BFCL, override the checkpoint
#   bash asset/run_all_benchmarks.sh --benchmarks bba,bfcl --model /path/to/ckpt
#
#   # Dry run — preview job submissions without actually submitting
#   bash asset/run_all_benchmarks.sh --dry_run

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMMIT=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo "unknown")

VB_SCRIPT="${REPO_ROOT}/nemo_skills/dataset/voicebench/scripts/generate_from_api_and_score_official.py"
FDB_SCRIPT="${REPO_ROOT}/nemo_skills/dataset/fdb/scripts/run_eval.py"
BBA_SCRIPT="${REPO_ROOT}/nemo_skills/dataset/bba/scripts/run_bba_eval.py"
BFCL_SCRIPT="${REPO_ROOT}/nemo_skills/dataset/bfcl_single_turn_function_channel/scripts/run_eval.py"
CONV_BEHAV_SCRIPT="${REPO_ROOT}/nemo_skills/dataset/conv_behav/scripts/run_eval.py"

# Config base directories — used by config_for() to derive default YAML paths.
VB_BASE="${REPO_ROOT}/nemo_skills/dataset/voicebench/scripts"
FDB_BASE="${REPO_ROOT}/nemo_skills/dataset/fdb/scripts"
BBA_BASE="${REPO_ROOT}/nemo_skills/dataset/bba/scripts"
BFCL_BASE="${REPO_ROOT}/nemo_skills/dataset/bfcl_single_turn_function_channel/scripts"
CB_BASE="${REPO_ROOT}/nemo_skills/dataset/conv_behav/scripts"

# ---------------------------------------------------------------------------
# Argument defaults
# ---------------------------------------------------------------------------
ALL_BENCHMARKS="conv_behav fdb bba bfcl vb_mcq vb_nonmcq"
SELECTED_BENCHMARKS=""

# Left empty until resolve_configs() fills them from --eval_mode.
# Individual --config_* flags override the resolved defaults.
CONFIG_VB_NONMCQ=""
CONFIG_VB_MCQ=""
CONFIG_FDB=""
CONFIG_BBA=""
CONFIG_BFCL=""
CONFIG_CONV_BEHAV=""

EVAL_MODE="greedy+sampling"
MODEL_OVERRIDE=""
CODE_PATH_OVERRIDE=""
OUTPUT_DIR_OVERRIDE=""
MAX_JOBS_OVERRIDE=""
POLL_INTERVAL=60
DRY_RUN=false
FORCE_RERUN=false

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
usage() {
    cat <<EOF
Usage: bash asset/run_all_benchmarks.sh [options]

Submit S2S FC eval benchmarks sequentially.  Before each benchmark the script
checks the current SLURM job count against the detected (or supplied) limit and
waits until a slot is free.

Benchmark names: vb_nonmcq  vb_mcq  fdb  bba  bfcl  conv_behav

Options:
  --eval_mode         MODE  Decoding mode (default: greedy+sampling).
                            Accepted values:
                              greedy           — greedy configs only
                              sampling         — sampling configs only
                              greedy+sampling  — greedy first, then sampling
                            Selects the matching *_greedy.yaml / *_sampling.yaml
                            config for each benchmark automatically.
  --benchmarks LIST         Comma-separated subset of the benchmark names above.
                            Default: all in the order listed above.
  --config_vb_nonmcq  PATH Config YAML for VoiceBench non-MCQ (overrides --eval_mode)
  --config_vb_mcq     PATH Config YAML for VoiceBench MCQ     (overrides --eval_mode)
  --config_fdb        PATH Config YAML for FDB                 (overrides --eval_mode)
  --config_bba        PATH Config YAML for BBA                 (overrides --eval_mode)
  --config_bfcl       PATH Config YAML for BFCL                (overrides --eval_mode)
  --config_conv_behav PATH Config YAML for conv_behav          (overrides --eval_mode)
  --output_dir        PATH  Redirect all benchmark outputs under PATH/{mode}
                            (e.g. PATH/greedy, PATH/sampling). The Python scripts
                            will further append the git commit hash. If unset,
                            each benchmark uses its own output_dir from its YAML.
  --model             PATH  Override the model checkpoint for every benchmark.
                            Must be paired with --code_path.
  --code_path         PATH  Override the NeMo source code directory for every
                            benchmark. Must be paired with --model.
  --max_jobs          N     Override SLURM job limit (auto-detected by default)
  --poll_interval     N     Seconds between SLURM queue checks (default: 60)
  --dry_run                 Pass --dry_run to every benchmark script
  --force_rerun             Re-run all benchmarks even if results already exist
  --help                    Show this message and exit
EOF
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --eval_mode)         EVAL_MODE="$2";            shift 2 ;;
        --benchmarks)        SELECTED_BENCHMARKS="$2";  shift 2 ;;
        --config_vb_nonmcq)  CONFIG_VB_NONMCQ="$2";    shift 2 ;;
        --config_vb_mcq)     CONFIG_VB_MCQ="$2";        shift 2 ;;
        --config_fdb)        CONFIG_FDB="$2";            shift 2 ;;
        --config_bba)        CONFIG_BBA="$2";            shift 2 ;;
        --config_bfcl)       CONFIG_BFCL="$2";           shift 2 ;;
        --config_conv_behav) CONFIG_CONV_BEHAV="$2";     shift 2 ;;
        --output_dir)        OUTPUT_DIR_OVERRIDE="$2";    shift 2 ;;
        --model)             MODEL_OVERRIDE="$2";         shift 2 ;;
        --code_path)         CODE_PATH_OVERRIDE="$2";    shift 2 ;;
        --max_jobs)          MAX_JOBS_OVERRIDE="$2";     shift 2 ;;
        --poll_interval)     POLL_INTERVAL="$2";          shift 2 ;;
        --dry_run)           DRY_RUN=true;                shift ;;
        --force_rerun)       FORCE_RERUN=true;            shift ;;
        --help|-h)           usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

# Validate eval mode and expand to an ordered list of individual modes.
case "$EVAL_MODE" in
    greedy)                          EVAL_MODES="greedy" ;;
    sampling)                        EVAL_MODES="sampling" ;;
    greedy+sampling|sampling+greedy) EVAL_MODES="greedy sampling" ;;
    *)
        echo "ERROR: --eval_mode must be 'greedy', 'sampling', or 'greedy+sampling', got: '$EVAL_MODE'" >&2
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Resolve benchmark list
# ---------------------------------------------------------------------------
if [[ -n "$SELECTED_BENCHMARKS" ]]; then
    BENCHMARKS="${SELECTED_BENCHMARKS//,/ }"
else
    BENCHMARKS="$ALL_BENCHMARKS"
fi

# Validate each name
for b in $BENCHMARKS; do
    if [[ ! " $ALL_BENCHMARKS " =~ " $b " ]]; then
        echo "Unknown benchmark: '$b'. Valid names: $ALL_BENCHMARKS" >&2
        exit 1
    fi
done

# Reorder to size order (ALL_BENCHMARKS is already ordered smallest → largest).
ordered=""
for b in $ALL_BENCHMARKS; do
    if [[ " $BENCHMARKS " =~ " $b " ]]; then
        ordered="$ordered $b"
    fi
done
BENCHMARKS="${ordered# }"

# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------
# Returns the YAML path for a given benchmark + decoding mode.
# If the user supplied an explicit --config_<name> override, that takes
# precedence over the mode-derived default for both greedy and sampling.
config_for() {
    local name="$1" mode="$2"
    case "$name" in
        vb_nonmcq)
            echo "${CONFIG_VB_NONMCQ:-${VB_BASE}/vb_matched_demo_v2_02mar_config_fc_${mode}.yaml}"
            ;;
        vb_mcq)
            echo "${CONFIG_VB_MCQ:-${VB_BASE}/vb_matched_demo_v2_02mar_mcq_config_fc_${mode}.yaml}"
            ;;
        fdb)
            echo "${CONFIG_FDB:-${FDB_BASE}/fdb_s2s_incremental_v2_02mar_config_fc_${mode}.yaml}"
            ;;
        bba)
            echo "${CONFIG_BBA:-${BBA_BASE}/bba_config_fc_${mode}.yaml}"
            ;;
        bfcl)
            echo "${CONFIG_BFCL:-${BFCL_BASE}/bfcl_fc_config_${mode}.yaml}"
            ;;
        conv_behav)
            echo "${CONFIG_CONV_BEHAV:-${CB_BASE}/conv_behav_config_${mode}.yaml}"
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Output dir resolution + completion detection
# ---------------------------------------------------------------------------

# Returns the fully-resolved output dir for a benchmark+mode, including the
# git commit hash suffix that Python scripts append.
resolve_output_dir() {
    local name="$1" mode="$2"
    if [[ -n "$OUTPUT_DIR_OVERRIDE" ]]; then
        echo "${OUTPUT_DIR_OVERRIDE}/${mode}_${COMMIT}"
    else
        local config yaml_dir
        config=$(config_for "$name" "$mode")
        yaml_dir=$(grep -m1 '^output_dir:' "$config" 2>/dev/null | sed 's/^output_dir: *//')
        echo "${yaml_dir}_${COMMIT}"
    fi
}

# Echoes the path of the first result file found for a benchmark+mode
# (a metrics.json under eval-results/ or a report.html in the output dir),
# or nothing if no results exist yet.
find_benchmark_result() {
    local name="$1" mode="$2"
    local outdir
    outdir=$(resolve_output_dir "$name" "$mode")
    [[ -d "$outdir" ]] || return 0
    local f
    f=$(find "${outdir}/eval-results" -name "metrics.json" -maxdepth 2 2>/dev/null | head -1)
    [[ -n "$f" ]] && { echo "$f"; return; }
    f=$(find "${outdir}" -name "report.html" -maxdepth 1 2>/dev/null | head -1)
    [[ -n "$f" ]] && echo "$f"
}

# Echoes the path of the aggregate scorecard if it exists:
# checks output_dir/scorecard.html first, then asset/scorecard.html (CWD-relative).
find_scorecard() {
    if [[ -n "$OUTPUT_DIR_OVERRIDE" && -f "${OUTPUT_DIR_OVERRIDE}/scorecard.html" ]]; then
        echo "${OUTPUT_DIR_OVERRIDE}/scorecard.html"
    elif [[ -f "${SCRIPT_DIR}/scorecard.html" ]]; then
        echo "${SCRIPT_DIR}/scorecard.html"
    fi
}

# ---------------------------------------------------------------------------
# SLURM helpers
# ---------------------------------------------------------------------------

# Returns the SLURM max-submit-jobs limit for the current user.
# Checks (in order): user association, QOS, partition.  Prints the minimum
# positive value found, or nothing if no limit is detected.
detect_max_jobs() {
    local -a limits=()

    # 1. Per-user association MaxSubmitJobs
    local assoc_max
    assoc_max=$(sacctmgr show association user="$USER" format=MaxSubmitJobs \
        -n -P 2>/dev/null | tr -d ' ' | grep -E '^[0-9]+$' | head -1 || true)
    if [[ -n "$assoc_max" && "$assoc_max" -gt 0 ]]; then
        limits+=("$assoc_max")
    fi

    # 2. QOS MaxSubmitJobsPerUser — iterate over every QOS assigned to the user
    local qos_list
    qos_list=$(sacctmgr show association user="$USER" format=QOS \
        -n 2>/dev/null \
        | tr ',' '\n' | tr -d ' ' | grep -v '^$' | sort -u | head -10 || true)
    while IFS= read -r qos; do
        [[ -z "$qos" ]] && continue
        local qos_max
        qos_max=$(sacctmgr show qos name="$qos" format=MaxSubmitJobsPerUser \
            -n -P 2>/dev/null | tr -d ' ' | grep -E '^[0-9]+$' | head -1 || true)
        if [[ -n "$qos_max" && "$qos_max" -gt 0 ]]; then
            limits+=("$qos_max")
        fi
    done <<< "$qos_list"

    # 3. Partition MaxSubmitJobsPerUser — check every comma-separated partition
    #    from our known cluster config; fall back to whatever scontrol reports.
    local partitions_to_check="batch_block1 batch_block3 batch_block4 cpu"
    for part in $partitions_to_check; do
        local part_max
        part_max=$(scontrol show partition "$part" 2>/dev/null \
            | grep -o 'MaxSubmitJobsPerUser=[^ ]*' \
            | cut -d= -f2 | grep -E '^[0-9]+$' | head -1 || true)
        if [[ -n "$part_max" && "$part_max" -gt 0 ]]; then
            limits+=("$part_max")
            break  # first partition with a concrete limit is enough
        fi
    done

    if [[ ${#limits[@]} -eq 0 ]]; then
        echo ""
        return
    fi

    # Return the minimum across all sources
    local min="${limits[0]}"
    for v in "${limits[@]}"; do
        if [[ "$v" -lt "$min" ]]; then min="$v"; fi
    done
    echo "$min"
}

count_user_jobs() {
    squeue -u "$USER" -h 2>/dev/null | wc -l | tr -d ' '
}

# Block until at least one job slot is free.
wait_for_slot() {
    local max_jobs="$1"
    local benchmark="$2"
    local first=true
    while true; do
        local current
        current=$(count_user_jobs)
        if [[ "$current" -lt "$max_jobs" ]]; then
            if [[ "$first" != "true" ]]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Slot opened ($current/$max_jobs). Proceeding with: $benchmark"
            fi
            return
        fi
        if [[ "$first" == "true" ]]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Queue full ($current/$max_jobs jobs). Waiting for a free slot before: $benchmark"
            first=false
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Still waiting ($current/$max_jobs jobs)..."
        fi
        sleep "$POLL_INTERVAL"
    done
}

# ---------------------------------------------------------------------------
# Model + code_path validation
# ---------------------------------------------------------------------------

# --model and --code_path must always be specified together because each
# checkpoint ships with its own NeMo source code.  When neither is given,
# each benchmark simply uses whatever is in its own config YAML.
check_model_code_path() {
    if [[ -n "$MODEL_OVERRIDE" && -z "$CODE_PATH_OVERRIDE" ]]; then
        echo "" >&2
        echo "ERROR: --model was specified without --code_path." >&2
        echo "Each checkpoint ships with its own NeMo source code." >&2
        echo "Pass --code_path PATH alongside --model PATH." >&2
        echo "" >&2
        exit 1
    fi
    if [[ -z "$MODEL_OVERRIDE" && -n "$CODE_PATH_OVERRIDE" ]]; then
        echo "" >&2
        echo "ERROR: --code_path was specified without --model." >&2
        echo "Pass --model PATH alongside --code_path PATH." >&2
        echo "" >&2
        exit 1
    fi
    if [[ -n "$MODEL_OVERRIDE" ]]; then
        echo " Model        : $MODEL_OVERRIDE"
        echo " Code path    : $CODE_PATH_OVERRIDE"
    fi
}

# ---------------------------------------------------------------------------
# Config patching
# ---------------------------------------------------------------------------

# Creates a temp YAML with MODEL_OVERRIDE and CODE_PATH_OVERRIDE applied.
# Handles --code_path inside server_args AND nemo_code_path top-level field.
# Caller is responsible for deleting the returned temp file.
make_patched_config() {
    local config="$1"
    local tmp
    tmp=$(mktemp /tmp/benchmark_config_XXXXXX.yaml)
    sed "s|^model:.*|model: ${MODEL_OVERRIDE}|;
         s|--code_path [^ ]*|--code_path ${CODE_PATH_OVERRIDE}|g;
         s|^nemo_code_path:.*|nemo_code_path: ${CODE_PATH_OVERRIDE}|" \
        "$config" > "$tmp"
    echo "$tmp"
}

# ---------------------------------------------------------------------------
# Per-benchmark run logic
# ---------------------------------------------------------------------------

# Build CLI args common to every benchmark call
# Outputs one argument per line so callers can read into an array safely.
extra_args() {
    if [[ -n "$MODEL_OVERRIDE" ]]; then
        echo "--model"
        echo "$MODEL_OVERRIDE"
    fi
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "--dry_run"
    fi
}

run_benchmark() {
    local name="$1"
    local mode="$2"
    local -a extra=()
    while IFS= read -r arg; do
        extra+=("$arg")
    done < <(extra_args)

    # --output_dir override: each mode gets its own subdirectory so greedy and
    # sampling results never collide.  The Python scripts append the git commit
    # hash, producing e.g. OUTPUT_DIR_OVERRIDE/greedy_a1b2c3d.
    if [[ -n "$OUTPUT_DIR_OVERRIDE" ]]; then
        extra+=("--output_dir" "${OUTPUT_DIR_OVERRIDE}/${mode}")
    fi

    local base_config
    base_config=$(config_for "$name" "$mode")

    local config="$base_config"
    local tmp_config=""
    if [[ -n "$MODEL_OVERRIDE" || -n "$CODE_PATH_OVERRIDE" ]]; then
        tmp_config=$(make_patched_config "$base_config")
        config="$tmp_config"
    fi

    cd "$REPO_ROOT"
    export NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1

    echo ""
    echo "======================================================================"
    printf ' %-30s  %s\n' "Benchmark:" "$name"
    printf ' %-30s  %s\n' "Mode:" "$mode"
    printf ' %-30s  %s\n' "Started:" "$(date '+%Y-%m-%d %H:%M:%S')"
    printf ' %-30s  %s\n' "Config:" "$base_config"
    [[ -n "$tmp_config" ]] && printf ' %-30s  %s\n' "Patched config:" "$tmp_config"
    echo "======================================================================"

    case "$name" in
        vb_nonmcq)
            python "$VB_SCRIPT" --config "$config" "${extra[@]}"
            ;;
        vb_mcq)
            python "$VB_SCRIPT" --config "$config" "${extra[@]}"
            ;;
        fdb)
            python "$FDB_SCRIPT" --config "$config" "${extra[@]}"
            ;;
        bba)
            python "$BBA_SCRIPT" --config "$config" "${extra[@]}"
            ;;
        bfcl)
            python "$BFCL_SCRIPT" --config "$config" "${extra[@]}"
            ;;
        conv_behav)
            python "$CONV_BEHAV_SCRIPT" --config "$config" "${extra[@]}"
            ;;
    esac

    [[ -n "$tmp_config" ]] && rm -f "$tmp_config"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done submitting: $name"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
check_model_code_path

echo ""
echo "======================================================================"
echo " S2S FC Benchmark Runner"
echo "======================================================================"
echo " Eval mode    : $EVAL_MODE"
echo " Benchmarks   : $BENCHMARKS"
[[ -n "$OUTPUT_DIR_OVERRIDE" ]] && echo " Output dir   : $OUTPUT_DIR_OVERRIDE/{mode}_{commit}"
echo " Poll interval: ${POLL_INTERVAL}s"
echo " Dry run      : $DRY_RUN"
echo " Force rerun  : $FORCE_RERUN"
echo " Commit       : $COMMIT"

# Resolve job limit
if [[ -n "$MAX_JOBS_OVERRIDE" ]]; then
    MAX_JOBS="$MAX_JOBS_OVERRIDE"
    echo " Max jobs     : $MAX_JOBS (manual override)"
else
    echo " Detecting SLURM job limit..."
    MAX_JOBS=$(detect_max_jobs)
    if [[ -n "$MAX_JOBS" ]]; then
        echo " Max jobs     : $MAX_JOBS (auto-detected)"
    else
        echo " Max jobs     : none detected — will submit without throttling"
    fi
fi
echo "======================================================================"

# If the aggregate scorecard already exists, all benchmarks are considered done.
if [[ "$FORCE_RERUN" != "true" ]]; then
    SCORECARD=$(find_scorecard)
    if [[ -n "$SCORECARD" ]]; then
        echo ""
        echo "Scorecard found: $SCORECARD"
        echo "All benchmarks appear complete. Use --force_rerun to re-run anyway."
        exit 0
    fi
fi

# Submit all benchmarks for each mode in order (greedy first, then sampling).
for mode in $EVAL_MODES; do
    echo ""
    echo "--- Mode: $mode ---"
    for benchmark in $BENCHMARKS; do
        if [[ "$FORCE_RERUN" != "true" ]]; then
            result=$(find_benchmark_result "$benchmark" "$mode")
            if [[ -n "$result" ]]; then
                echo "  Skipping $benchmark ($mode): results found at $result"
                continue
            fi
        fi
        if [[ -n "$MAX_JOBS" && "$DRY_RUN" != "true" ]]; then
            wait_for_slot "$MAX_JOBS" "$benchmark ($mode)"
        fi
        run_benchmark "$benchmark" "$mode"
    done
done

echo ""
echo "======================================================================"
echo " All benchmarks submitted successfully."
echo "======================================================================"
