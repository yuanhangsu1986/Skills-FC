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

VB_SCRIPT="${REPO_ROOT}/nemo_skills/dataset/voicebench/scripts/generate_from_api_and_score_official.py"
FDB_SCRIPT="${REPO_ROOT}/nemo_skills/dataset/fdb/scripts/run_eval.py"
BBA_SCRIPT="${REPO_ROOT}/nemo_skills/dataset/bba/scripts/run_bba_eval.py"
BFCL_SCRIPT="${REPO_ROOT}/nemo_skills/dataset/bfcl_single_turn_function_channel/scripts/run_eval.py"
CONV_BEHAV_SCRIPT="${REPO_ROOT}/nemo_skills/dataset/conv_behav/scripts/run_eval.py"

# Default configs — point at the greedy variants; override with --config_* flags.
DEFAULT_CONFIG_VB_NONMCQ="${REPO_ROOT}/nemo_skills/dataset/voicebench/scripts/vb_matched_demo_v2_02mar_config_fc_greedy.yaml"
DEFAULT_CONFIG_VB_MCQ="${REPO_ROOT}/nemo_skills/dataset/voicebench/scripts/vb_matched_demo_v2_02mar_mcq_config_fc_greedy.yaml"
DEFAULT_CONFIG_FDB="${REPO_ROOT}/nemo_skills/dataset/fdb/scripts/fdb_s2s_incremental_v2_02mar_config_fc_greedy.yaml"
DEFAULT_CONFIG_BBA="${REPO_ROOT}/nemo_skills/dataset/bba/scripts/bba_config_fc_greedy.yaml"
DEFAULT_CONFIG_BFCL="${REPO_ROOT}/nemo_skills/dataset/bfcl_single_turn_function_channel/scripts/bfcl_fc_config_greedy.yaml"
DEFAULT_CONFIG_CONV_BEHAV="${REPO_ROOT}/nemo_skills/dataset/conv_behav/scripts/conv_behav_config_greedy.yaml"

# ---------------------------------------------------------------------------
# Argument defaults
# ---------------------------------------------------------------------------
ALL_BENCHMARKS="vb_nonmcq vb_mcq fdb bba bfcl conv_behav"
SELECTED_BENCHMARKS=""

CONFIG_VB_NONMCQ="$DEFAULT_CONFIG_VB_NONMCQ"
CONFIG_VB_MCQ="$DEFAULT_CONFIG_VB_MCQ"
CONFIG_FDB="$DEFAULT_CONFIG_FDB"
CONFIG_BBA="$DEFAULT_CONFIG_BBA"
CONFIG_BFCL="$DEFAULT_CONFIG_BFCL"
CONFIG_CONV_BEHAV="$DEFAULT_CONFIG_CONV_BEHAV"

MODEL_OVERRIDE=""
CODE_PATH_OVERRIDE=""
MAX_JOBS_OVERRIDE=""
POLL_INTERVAL=60
DRY_RUN=false

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
  --benchmarks LIST          Comma-separated subset of the benchmark names above.
                             Default: all in the order listed above.
  --config_vb_nonmcq  PATH  Config YAML for VoiceBench non-MCQ
  --config_vb_mcq     PATH  Config YAML for VoiceBench MCQ
  --config_fdb        PATH  Config YAML for FDB
  --config_bba        PATH  Config YAML for BBA
  --config_bfcl       PATH  Config YAML for BFCL
  --config_conv_behav PATH  Config YAML for conv_behav
  --model             PATH  Override the model checkpoint for every benchmark
  --code_path         PATH  Override the NeMo source code path for every benchmark
                            (must be set whenever --model is set; each checkpoint
                            ships with its own NeMo code)
  --max_jobs          N     Override SLURM job limit (auto-detected by default)
  --poll_interval     N     Seconds between SLURM queue checks (default: 60)
  --dry_run                 Pass --dry_run to every benchmark script
  --help                    Show this message and exit
EOF
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --benchmarks)        SELECTED_BENCHMARKS="$2"; shift 2 ;;
        --config_vb_nonmcq)  CONFIG_VB_NONMCQ="$2";   shift 2 ;;
        --config_vb_mcq)     CONFIG_VB_MCQ="$2";       shift 2 ;;
        --config_fdb)        CONFIG_FDB="$2";           shift 2 ;;
        --config_bba)        CONFIG_BBA="$2";           shift 2 ;;
        --config_bfcl)       CONFIG_BFCL="$2";          shift 2 ;;
        --config_conv_behav) CONFIG_CONV_BEHAV="$2";    shift 2 ;;
        --model)             MODEL_OVERRIDE="$2";        shift 2 ;;
        --code_path)         CODE_PATH_OVERRIDE="$2";   shift 2 ;;
        --max_jobs)          MAX_JOBS_OVERRIDE="$2";    shift 2 ;;
        --poll_interval)     POLL_INTERVAL="$2";         shift 2 ;;
        --dry_run)           DRY_RUN=true;               shift ;;
        --help|-h)           usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

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
        -n -P 2>/dev/null \
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
# Model + code_path consistency checks
# ---------------------------------------------------------------------------

# Returns the code_path embedded in a config YAML.
# Handles two conventions:
#   1. --code_path <path>  inside the server_args multi-line string
#   2. nemo_code_path: <path>  as a top-level YAML field (conv_behav)
_extract_code_path() {
    local config="$1"
    local cp
    cp=$(grep -oP '(?<=--code_path )\S+' "$config" | head -1)
    if [[ -z "$cp" ]]; then
        cp=$(grep -E '^nemo_code_path:[[:space:]]' "$config" | head -1 \
             | sed 's/^nemo_code_path:[[:space:]]*//')
    fi
    echo "$cp"
}

# Validates that all selected benchmarks agree on model AND code_path.
# When both --model and --code_path are supplied the overrides are applied
# uniformly, so no cross-config check is needed.
# If only one of the two is supplied, we error: they must travel together.
check_consistency() {
    # Paired-override validation
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

    # Both overrides supplied — no cross-config check needed.
    if [[ -n "$MODEL_OVERRIDE" && -n "$CODE_PATH_OVERRIDE" ]]; then
        echo " Model        : $MODEL_OVERRIDE"
        echo " Code path    : $CODE_PATH_OVERRIDE"
        return
    fi

    # Neither override: read values from each config and verify they agree.
    local -a models=()
    local -a code_paths=()
    local -a bnames=()
    local config model cp bname

    for bname in $BENCHMARKS; do
        case "$bname" in
            vb_nonmcq)  config="$CONFIG_VB_NONMCQ" ;;
            vb_mcq)     config="$CONFIG_VB_MCQ" ;;
            fdb)        config="$CONFIG_FDB" ;;
            bba)        config="$CONFIG_BBA" ;;
            bfcl)       config="$CONFIG_BFCL" ;;
            conv_behav) config="$CONFIG_CONV_BEHAV" ;;
        esac

        if [[ ! -f "$config" ]]; then
            echo "WARNING: Config not found, skipping consistency check: $config" >&2
            continue
        fi

        model=$(grep -E '^model:[[:space:]]' "$config" | head -1 \
                | sed 's/^model:[[:space:]]*//')
        cp=$(_extract_code_path "$config")

        if [[ -z "$model" ]]; then
            echo "WARNING: No 'model:' in $config — skipping for $bname" >&2
            continue
        fi
        if [[ -z "$cp" ]]; then
            echo "WARNING: No code_path in $config — skipping for $bname" >&2
            continue
        fi

        models+=("$model")
        code_paths+=("$cp")
        bnames+=("$bname")
    done

    [[ ${#models[@]} -eq 0 ]] && return

    local first_model="${models[0]}"
    local first_cp="${code_paths[0]}"
    local model_mismatch=false
    local cp_mismatch=false

    for i in "${!models[@]}"; do
        [[ "${models[$i]}"      != "$first_model" ]] && model_mismatch=true
        [[ "${code_paths[$i]}"  != "$first_cp"    ]] && cp_mismatch=true
    done

    if [[ "$model_mismatch" == "true" || "$cp_mismatch" == "true" ]]; then
        echo "" >&2
        echo "ERROR: Benchmarks reference different model/code_path combinations." >&2
        echo "Pass --model PATH --code_path PATH to use the same pair for all, or" >&2
        echo "align the fields in the config YAMLs." >&2
        echo "" >&2
        printf '  %-15s  %-70s  %s\n' "benchmark" "model" "code_path" >&2
        printf '  %-15s  %-70s  %s\n' "---------" "-----" "---------" >&2
        for i in "${!bnames[@]}"; do
            printf '  %-15s  %-70s  %s\n' \
                "${bnames[$i]}" "${models[$i]}" "${code_paths[$i]}" >&2
        done
        echo "" >&2
        exit 1
    fi

    echo " Model        : $first_model"
    echo " Code path    : $first_cp"
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
    local -a extra=()
    while IFS= read -r arg; do
        extra+=("$arg")
    done < <(extra_args)

    # Resolve per-benchmark config; patch into a temp file when overrides are active.
    local base_config
    case "$name" in
        vb_nonmcq)  base_config="$CONFIG_VB_NONMCQ" ;;
        vb_mcq)     base_config="$CONFIG_VB_MCQ" ;;
        fdb)        base_config="$CONFIG_FDB" ;;
        bba)        base_config="$CONFIG_BBA" ;;
        bfcl)       base_config="$CONFIG_BFCL" ;;
        conv_behav) base_config="$CONFIG_CONV_BEHAV" ;;
    esac

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
echo ""
echo "======================================================================"
echo " S2S FC Benchmark Runner"
echo "======================================================================"
echo " Benchmarks  : $BENCHMARKS"
echo " Poll interval: ${POLL_INTERVAL}s"
echo " Dry run      : $DRY_RUN"
check_consistency

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

# Submit benchmarks one at a time
for benchmark in $BENCHMARKS; do
    if [[ -n "$MAX_JOBS" && "$DRY_RUN" != "true" ]]; then
        wait_for_slot "$MAX_JOBS" "$benchmark"
    fi
    run_benchmark "$benchmark"
done

echo ""
echo "======================================================================"
echo " All benchmarks submitted successfully."
echo "======================================================================"
