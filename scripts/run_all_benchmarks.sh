#!/usr/bin/env bash
# run_all_benchmarks.sh
#
# Submit all S2S FC eval benchmarks sequentially, throttling submission based
# on the SLURM job-count limit (queried from sacctmgr QOS settings and
# scontrol partition settings; the tighter of the two is used).
#
# Run from the repo root:
#   bash scripts/run_all_benchmarks.sh [options]
#
# Examples:
#   # Run everything with defaults
#   bash scripts/run_all_benchmarks.sh
#
#   # Run only BBA and BFCL, override the checkpoint
#   bash scripts/run_all_benchmarks.sh --benchmarks bba,bfcl --model /path/to/ckpt
#
#   # Dry run — preview job submissions without actually submitting
#   bash scripts/run_all_benchmarks.sh --dry_run

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMMIT=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo "unknown")
PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="$(command -v python3 || command -v python)"
fi

# Make the repo root findable by Python (for nemo_skills editable install).
# Do NOT add the venv site-packages here: that path would propagate into
# SLURM container jobs where a different Python version would try to load
# incompatible compiled extensions (.so files) and fail.
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

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
EVAL_MODE_EXPLICITLY_SET=false
MODEL_OVERRIDE=""
CODE_PATH_OVERRIDE=""
OUTPUT_DIR_OVERRIDE=""
HTML_NAME="scorecard"
MAX_JOBS_OVERRIDE=""
POLL_INTERVAL=60
DRY_RUN=false
FORCE_RERUN=false

# Customized-mode decoding params (all four must be set together, with --output_dir)
CUSTOM_FORCE_TURN_TAKING=""   # "true" or "false"
CUSTOM_TOP_P=""
CUSTOM_REPETITION_PENALTY=""
CUSTOM_TEMPERATURE=""

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
usage() {
    cat <<EOF
Usage: bash scripts/run_all_benchmarks.sh [options]

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
                              customized       — use base configs + explicit params
                            Selects the matching *_greedy.yaml / *_sampling.yaml
                            config for each benchmark automatically.
                            Set automatically to 'customized' when any of the four
                            decoding param flags below are provided.
  --force_turn_taking BOOL  Customized mode: "true" or "false". Adds --force_turn_taking
                            to server_args (or sets the top-level bool for conv_behav).
                            Requires all four decoding params + --output_dir.
  --top_p             VAL   Customized mode: LLM top-p sampling value.
  --repetition_penalty VAL  Customized mode: repetition penalty value.
  --temperature       VAL   Customized mode: sampling temperature value.
  --benchmarks LIST         Comma-separated subset of the benchmark names above.
                            Default: all in the order listed above.
  --config_vb_nonmcq  PATH Config YAML for VoiceBench non-MCQ (overrides --eval_mode)
  --config_vb_mcq     PATH Config YAML for VoiceBench MCQ     (overrides --eval_mode)
  --config_fdb        PATH Config YAML for FDB                 (overrides --eval_mode)
  --config_bba        PATH Config YAML for BBA                 (overrides --eval_mode)
  --config_bfcl       PATH Config YAML for BFCL                (overrides --eval_mode)
  --config_conv_behav PATH Config YAML for conv_behav          (overrides --eval_mode)
  --html_name         NAME  Base name for the scorecard HTML (default: scorecard).
                            Used to locate an existing scorecard for the skip-all
                            check. Pass the same value you use with /make-scorecard
                            name=NAME so the two commands stay in sync.
  --output_dir        PATH  Redirect all benchmark outputs under PATH/{mode}/{name}
                            (e.g. PATH/greedy/bba_a1b2c3d, PATH/sampling/fdb_a1b2c3d).
                            Each benchmark gets its own subdirectory so result-detection
                            across benchmarks cannot cross-contaminate. The Python
                            scripts append the git commit hash. If unset, each
                            benchmark uses its own output_dir from its YAML.
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
        --eval_mode)         EVAL_MODE="$2"; EVAL_MODE_EXPLICITLY_SET=true; shift 2 ;;
        --benchmarks)        SELECTED_BENCHMARKS="$2";  shift 2 ;;
        --config_vb_nonmcq)  CONFIG_VB_NONMCQ="$2";    shift 2 ;;
        --config_vb_mcq)     CONFIG_VB_MCQ="$2";        shift 2 ;;
        --config_fdb)        CONFIG_FDB="$2";            shift 2 ;;
        --config_bba)        CONFIG_BBA="$2";            shift 2 ;;
        --config_bfcl)       CONFIG_BFCL="$2";           shift 2 ;;
        --config_conv_behav) CONFIG_CONV_BEHAV="$2";     shift 2 ;;
        --html_name)         HTML_NAME="$2";               shift 2 ;;
        --output_dir)        OUTPUT_DIR_OVERRIDE="$2";    shift 2 ;;
        --model)             MODEL_OVERRIDE="$2";         shift 2 ;;
        --code_path)         CODE_PATH_OVERRIDE="$2";    shift 2 ;;
        --max_jobs)          MAX_JOBS_OVERRIDE="$2";     shift 2 ;;
        --poll_interval)     POLL_INTERVAL="$2";          shift 2 ;;
        --force_turn_taking) CUSTOM_FORCE_TURN_TAKING="$2"; shift 2 ;;
        --top_p)             CUSTOM_TOP_P="$2";            shift 2 ;;
        --repetition_penalty) CUSTOM_REPETITION_PENALTY="$2"; shift 2 ;;
        --temperature)       CUSTOM_TEMPERATURE="$2";     shift 2 ;;
        --dry_run)           DRY_RUN=true;                shift ;;
        --force_rerun)       FORCE_RERUN=true;            shift ;;
        --help|-h)           usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

# Validate customized decoding params and resolve eval mode.
# Use if/fi to safely count under set -e (&&-chain with failing [[ ]] would exit).
_n_custom=0
if [[ -n "$CUSTOM_FORCE_TURN_TAKING" ]];  then (( ++_n_custom )); fi
if [[ -n "$CUSTOM_TOP_P" ]];              then (( ++_n_custom )); fi
if [[ -n "$CUSTOM_REPETITION_PENALTY" ]]; then (( ++_n_custom )); fi
if [[ -n "$CUSTOM_TEMPERATURE" ]];        then (( ++_n_custom )); fi

if [[ "$_n_custom" -gt 0 || "$EVAL_MODE" == "customized" ]]; then
    if [[ "$EVAL_MODE_EXPLICITLY_SET" == "true" && "$EVAL_MODE" != "customized" ]]; then
        echo "ERROR: --eval_mode cannot be combined with custom decoding param flags (--force_turn_taking, --top_p, etc.)." >&2
        exit 1
    fi

    # Prompt for any missing required values; error if stdin is not a terminal.
    _missing=false
    if [[ -z "$CUSTOM_FORCE_TURN_TAKING" || -z "$CUSTOM_TOP_P" || -z "$CUSTOM_REPETITION_PENALTY" \
       || -z "$CUSTOM_TEMPERATURE" || -z "$OUTPUT_DIR_OVERRIDE" ]]; then
        _missing=true
    fi
    if [[ "$_missing" == "true" ]]; then
        if [[ ! -t 0 ]]; then
            echo "ERROR: Customized mode requires all of: --force_turn_taking, --top_p," >&2
            echo "       --repetition_penalty, --temperature, --output_dir." >&2
            exit 1
        fi
        echo ""
        echo "Customized mode — provide missing values (leave blank to abort):"
        if [[ -z "$CUSTOM_FORCE_TURN_TAKING" ]]; then
            read -r -p "  --force_turn_taking (true/false): " CUSTOM_FORCE_TURN_TAKING
        fi
        if [[ -z "$CUSTOM_TOP_P" ]]; then
            read -r -p "  --top_p: " CUSTOM_TOP_P
        fi
        if [[ -z "$CUSTOM_REPETITION_PENALTY" ]]; then
            read -r -p "  --repetition_penalty: " CUSTOM_REPETITION_PENALTY
        fi
        if [[ -z "$CUSTOM_TEMPERATURE" ]]; then
            read -r -p "  --temperature: " CUSTOM_TEMPERATURE
        fi
        if [[ -z "$OUTPUT_DIR_OVERRIDE" ]]; then
            read -r -p "  --output_dir: " OUTPUT_DIR_OVERRIDE
        fi
        echo ""
    fi

    # Validate that all values were supplied (either via CLI or prompts).
    if [[ -z "$CUSTOM_FORCE_TURN_TAKING" || -z "$CUSTOM_TOP_P" || -z "$CUSTOM_REPETITION_PENALTY" \
       || -z "$CUSTOM_TEMPERATURE" || -z "$OUTPUT_DIR_OVERRIDE" ]]; then
        echo "ERROR: Customized mode requires all of: --force_turn_taking, --top_p," >&2
        echo "       --repetition_penalty, --temperature, --output_dir." >&2
        exit 1
    fi
    if [[ "$CUSTOM_FORCE_TURN_TAKING" != "true" && "$CUSTOM_FORCE_TURN_TAKING" != "false" ]]; then
        echo "ERROR: --force_turn_taking must be 'true' or 'false', got: '$CUSTOM_FORCE_TURN_TAKING'" >&2
        exit 1
    fi
    EVAL_MODE="customized"
fi

# Validate eval mode and expand to an ordered list of individual modes.
case "$EVAL_MODE" in
    greedy)                          EVAL_MODES="greedy" ;;
    sampling)                        EVAL_MODES="sampling" ;;
    greedy+sampling|sampling+greedy) EVAL_MODES="greedy sampling" ;;
    customized)                      EVAL_MODES="customized" ;;
    *)
        echo "ERROR: --eval_mode must be 'greedy', 'sampling', 'greedy+sampling', or 'customized', got: '$EVAL_MODE'" >&2
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
            local default
            [[ "$mode" == "customized" ]] \
                && default="${VB_BASE}/vb_matched_demo_v2_02mar_config_fc.yaml" \
                || default="${VB_BASE}/vb_matched_demo_v2_02mar_config_fc_${mode}.yaml"
            echo "${CONFIG_VB_NONMCQ:-$default}"
            ;;
        vb_mcq)
            local default
            [[ "$mode" == "customized" ]] \
                && default="${VB_BASE}/vb_matched_demo_v2_02mar_mcq_config_fc.yaml" \
                || default="${VB_BASE}/vb_matched_demo_v2_02mar_mcq_config_fc_${mode}.yaml"
            echo "${CONFIG_VB_MCQ:-$default}"
            ;;
        fdb)
            local default
            [[ "$mode" == "customized" ]] \
                && default="${FDB_BASE}/fdb_s2s_incremental_v2_02mar_config_fc.yaml" \
                || default="${FDB_BASE}/fdb_s2s_incremental_v2_02mar_config_fc_${mode}.yaml"
            echo "${CONFIG_FDB:-$default}"
            ;;
        bba)
            local default
            [[ "$mode" == "customized" ]] \
                && default="${BBA_BASE}/bba_config_fc.yaml" \
                || default="${BBA_BASE}/bba_config_fc_${mode}.yaml"
            echo "${CONFIG_BBA:-$default}"
            ;;
        bfcl)
            local default
            [[ "$mode" == "customized" ]] \
                && default="${BFCL_BASE}/bfcl_fc_config.yaml" \
                || default="${BFCL_BASE}/bfcl_fc_config_${mode}.yaml"
            echo "${CONFIG_BFCL:-$default}"
            ;;
        conv_behav)
            local default
            [[ "$mode" == "customized" ]] \
                && default="${CB_BASE}/conv_behav_config.yaml" \
                || default="${CB_BASE}/conv_behav_config_${mode}.yaml"
            echo "${CONFIG_CONV_BEHAV:-$default}"
            ;;
    esac
}

# Returns the expname base (without mode suffix) for a benchmark.
# Used by customized mode and sibling-exclusion logic in cancel_stale_jobs.
expname_base_for() {
    local name="$1"
    case "$name" in
        vb_nonmcq)  echo "vb_matched_demo_v2_02mar_fc" ;;
        vb_mcq)     echo "vb_matched_demo_v2_02mar_mcq_fc" ;;
        fdb)        echo "fdb_v1_0_s2s_incremental_v2_02mar_fc" ;;
        bba)        echo "bba_fc" ;;
        bfcl)       echo "bfcl_fc" ;;
        conv_behav) echo "conv_behav" ;;
    esac
}

# Returns the expname for a given benchmark+mode.
# For customized mode, derives expname from expname_base_for() since base YAMLs
# have no expname key.
expname_for() {
    local name="$1" mode="$2"
    if [[ "$mode" == "customized" ]]; then
        echo "$(expname_base_for "$name")_customized"
        return
    fi
    local config
    config=$(config_for "$name" "$mode")
    grep -m1 '^expname:' "$config" 2>/dev/null | sed 's/^expname: *//' || true
}

# ---------------------------------------------------------------------------
# Output dir resolution + completion detection
# ---------------------------------------------------------------------------

# Returns the fully-resolved output dir for a benchmark+mode, including the
# git commit hash suffix that Python scripts append.
# When --output_dir is set each benchmark gets its own subdirectory so
# find_benchmark_result can't confuse one benchmark's metrics.json for another's.
resolve_output_dir() {
    local name="$1" mode="$2"
    if [[ -n "$OUTPUT_DIR_OVERRIDE" ]]; then
        echo "${OUTPUT_DIR_OVERRIDE}/${mode}/${name}_${COMMIT}"
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
    f=$(find "${outdir}/eval-results" -name "metrics.json" -maxdepth 2 2>/dev/null | head -1 || true)
    [[ -n "$f" ]] && { echo "$f"; return; }
    f=$(find "${outdir}" -name "report.html" -maxdepth 1 2>/dev/null | head -1 || true)
    [[ -n "$f" ]] && echo "$f"
    return 0
}

# Echoes the path of the aggregate scorecard if it exists:
# checks output_dir/{html_name}.html first, then asset/{html_name}.html.
# When --output_dir is set we only check that location; the asset/ fallback
# would otherwise match a scorecard from a previous run and incorrectly
# short-circuit a new run with a different mode or checkpoint.
find_scorecard() {
    if [[ -n "$OUTPUT_DIR_OVERRIDE" && -f "${OUTPUT_DIR_OVERRIDE}/${HTML_NAME}.html" ]]; then
        echo "${OUTPUT_DIR_OVERRIDE}/${HTML_NAME}.html"
    elif [[ -z "$OUTPUT_DIR_OVERRIDE" && -f "${REPO_ROOT}/asset/${HTML_NAME}.html" ]]; then
        echo "${REPO_ROOT}/asset/${HTML_NAME}.html"
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

# Deletes a directory if it exists and contains no complete or in-progress results.
# Preserves the directory if any of these are found:
#   - metrics.json / report.html  (scoring complete)
#   - output.jsonl.done or output_chunk_*.jsonl.done  (generation complete, scoring pending)
# This avoids throwing away valid generation output that scoring can still use.
_delete_incomplete_dir() {
    local dir="$1"
    [[ -d "$dir" ]] || return 0
    local f
    # Scoring complete
    f=$(find "${dir}/eval-results" -name "metrics.json" -maxdepth 2 2>/dev/null | head -1 || true)
    [[ -z "$f" ]] && f=$(find "${dir}" -name "report.html" -maxdepth 1 2>/dev/null | head -1 || true)
    if [[ -n "$f" ]]; then
        printf '  [cleanup] Skipping (scoring complete): %s\n' "$dir"
        return 0
    fi
    # Generation complete but scoring not yet done — preserve so scoring can reuse it
    f=$(find "${dir}/eval-results" -name "output.jsonl.done" -o -name "output_chunk_*.jsonl.done" \
        2>/dev/null | head -1 || true)
    if [[ -n "$f" ]]; then
        printf '  [cleanup] Skipping (generation complete, scoring pending): %s\n' "$dir"
        return 0
    fi
    if [[ "$DRY_RUN" == "true" ]]; then
        printf '  [cleanup] DRY RUN — would delete: %s\n' "$dir"
    else
        printf '  [cleanup] Deleting: %s\n' "$dir"
        rm -rf "$dir"
    fi
}

# Finds and cancels any running/pending SLURM jobs whose name matches the
# expname prefix for the given benchmark+mode.  Jobs belonging to a sibling
# mode of the same benchmark (e.g. sampling when running greedy) are excluded
# so we never accidentally cancel an unrelated concurrent run.
#
# After cancellation, cleans up the output directories of the cancelled jobs.
# Because job names now embed the git commit (appended by the Python pipeline),
# we extract the commit from each job name and delete
# {base_output_dir}_{extracted_commit}, which is the exact directory that job
# was writing to — regardless of whether it matches the current commit.
#
# If no stale jobs are found we still clean up the current commit's output dir
# if it exists with only partial data (e.g. left behind by a prior Python crash).
cancel_stale_jobs() {
    local name="$1" mode="$2"

    local expname
    expname=$(expname_for "$name" "$mode") || true
    if [[ -z "$expname" ]]; then
        echo "  [dedup] WARNING: cannot read expname for $name/$mode — skipping duplicate check" >&2
        return 0
    fi

    # Collect expnames for every other mode of this benchmark so we can
    # exclude their jobs from cancellation (e.g. don't cancel bba_fc_sampling_*
    # when we're about to run greedy whose expname is the shorter bba_fc).
    local -a exclude_prefixes=()
    for m in greedy sampling customized; do
        [[ "$m" == "$mode" ]] && continue
        local other_exp
        other_exp=$(expname_for "$name" "$m") || true
        [[ -n "$other_exp" && "$other_exp" != "$expname" ]] && exclude_prefixes+=("$other_exp")
    done

    # awk filter: include if job name equals expname or starts with expname_,
    # then subtract any job that matches a sibling-mode expname by the same rule.
    local awk_include
    awk_include="(\$2 == \"$expname\" || index(\$2, \"${expname}_\") == 1)"
    local awk_exclude=""
    for excl in "${exclude_prefixes[@]}"; do
        awk_exclude+=" && !(\$2 == \"$excl\" || index(\$2, \"${excl}_\") == 1)"
    done

    # Base output dir (without commit suffix).
    # When --output_dir is set, every benchmark gets its own subdirectory
    # (matching resolve_output_dir / run_benchmark).  Otherwise read it from
    # the config YAML (greedy/sampling mode-specific YAMLs have output_dir:).
    local base_outdir
    if [[ -n "$OUTPUT_DIR_OVERRIDE" ]]; then
        base_outdir="${OUTPUT_DIR_OVERRIDE}/${mode}/${name}"
    else
        base_outdir=$(grep -m1 '^output_dir:' "$(config_for "$name" "$mode")" 2>/dev/null \
            | sed 's/^output_dir: *//' || true)
    fi

    local matching
    matching=$(squeue -u "$USER" -h -o "%i %j %T %r" 2>/dev/null \
        | awk "{ if (${awk_include}${awk_exclude}) print }" || true)

    if [[ -z "$matching" ]]; then
        # No stale jobs — still clean up the current commit's dir if it has
        # only partial data (e.g. left behind by an earlier Python crash).
        [[ -n "$base_outdir" ]] && _delete_incomplete_dir "${base_outdir}_${COMMIT}"
        return 0
    fi

    local count job_ids
    count=$(printf '%s\n' "$matching" | wc -l | tr -d ' ')
    job_ids=$(printf '%s\n' "$matching" | awk '{printf "%s ", $1}')

    echo ""
    printf '  [dedup] %s (%s): %d existing job(s) match prefix "%s"\n' \
        "$name" "$mode" "$count" "$expname"
    printf '%s\n' "$matching" | \
        awk '{printf "    %-12s %-50s %-12s %s\n", $1, $2, $3, $4}'

    if [[ "$DRY_RUN" == "true" ]]; then
        printf '  [dedup] DRY RUN — would cancel job IDs: %s\n' "$job_ids"
    else
        printf '  [dedup] Cancelling job IDs: %s\n' "$job_ids"
        # shellcheck disable=SC2086
        scancel $job_ids 2>&1 || true
        printf '  [dedup] Cancelled.\n'
    fi

    # Clean up the output dir of each cancelled job.  The Python pipeline
    # appends _{commit} to every SLURM job name, so we extract it from the
    # job name to find the exact directory that job was writing to.
    if [[ -n "$base_outdir" ]]; then
        local -a seen_commits=()
        while IFS= read -r line; do
            local job_name job_commit
            job_name=$(printf '%s\n' "$line" | awk '{print $2}')
            # The commit is the trailing _[0-9a-f]{7,8} component.
            job_commit=$(printf '%s\n' "$job_name" | grep -oE '_[0-9a-f]{7,8}$' | tr -d '_' || true)
            [[ -z "$job_commit" ]] && continue
            # Process each unique commit only once.
            local dup=false
            for c in "${seen_commits[@]+"${seen_commits[@]}"}"; do
                [[ "$c" == "$job_commit" ]] && { dup=true; break; }
            done
            $dup && continue
            seen_commits+=("$job_commit")
            _delete_incomplete_dir "${base_outdir}_${job_commit}"
        done <<< "$matching"

        # Also clean the current commit's dir if not already covered above.
        local covered=false
        for c in "${seen_commits[@]+"${seen_commits[@]}"}"; do
            [[ "$c" == "$COMMIT" ]] && { covered=true; break; }
        done
        $covered || _delete_incomplete_dir "${base_outdir}_${COMMIT}"
    fi

    echo ""
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

# Creates a temp YAML for customized mode: injects decoding params into server_args,
# sets output_dir/expname/decoding_mode, and applies any model/code_path overrides.
# Usage: make_patched_config_customized <benchmark_name> <output_dir_no_commit>
# Prints path to the temp file.
make_patched_config_customized() {
    local name="$1"
    local output_dir="$2"   # top-level output_dir to embed (without commit suffix)

    local base_config expname artifacts_dir tmp
    base_config=$(config_for "$name" "customized")
    expname="$(expname_base_for "$name")_customized"
    # Artifacts dir mirrors the output_dir with _artifacts suffix (server audio output).
    artifacts_dir="${output_dir}_artifacts"
    tmp=$(mktemp /tmp/benchmark_config_XXXXXX.yaml)

    PATCH_SRC="$base_config" \
    PATCH_DST="$tmp" \
    PATCH_FORCE_TT="$CUSTOM_FORCE_TURN_TAKING" \
    PATCH_TOP_P="$CUSTOM_TOP_P" \
    PATCH_REP_PENALTY="$CUSTOM_REPETITION_PENALTY" \
    PATCH_TEMP="$CUSTOM_TEMPERATURE" \
    PATCH_ARTIFACTS_DIR="$artifacts_dir" \
    PATCH_OUTPUT_DIR="$output_dir" \
    PATCH_EXPNAME="$expname" \
    PATCH_MODEL="${MODEL_OVERRIDE:-}" \
    PATCH_CODE_PATH="${CODE_PATH_OVERRIDE:-}" \
    PATCH_BENCHMARK="$name" \
    "$PYTHON" - <<'PYEOF'
import os, yaml, re, sys

src           = os.environ['PATCH_SRC']
dst           = os.environ['PATCH_DST']
force_tt      = os.environ['PATCH_FORCE_TT']
top_p         = os.environ['PATCH_TOP_P']
rep_pen       = os.environ['PATCH_REP_PENALTY']
temp          = os.environ['PATCH_TEMP']
artifacts_dir = os.environ['PATCH_ARTIFACTS_DIR']
output_dir    = os.environ['PATCH_OUTPUT_DIR']
expname       = os.environ['PATCH_EXPNAME']
model         = os.environ.get('PATCH_MODEL', '')
code_path     = os.environ.get('PATCH_CODE_PATH', '')
benchmark     = os.environ['PATCH_BENCHMARK']

with open(src) as f:
    cfg = yaml.safe_load(f)

# Model override
if model:
    cfg['model'] = model

# Patch server_args — inject decoding params and artifacts output dir
if 'server_args' in cfg:
    sa = cfg['server_args']
    if code_path:
        sa = re.sub(r'--code_path \S+', f'--code_path {code_path}', sa)
    additions = []
    if force_tt == 'true':
        additions.append('--force_turn_taking')
    additions += [
        f'--top_p {top_p}',
        f'--repetition_penalty {rep_pen}',
        f'--temperature {temp}',
        f'--output_dir {artifacts_dir}',
    ]
    cfg['server_args'] = sa.rstrip() + ' ' + ' '.join(additions)

# nemo_code_path (conv_behav style)
if code_path and 'nemo_code_path' in cfg:
    cfg['nemo_code_path'] = code_path

# Set mode-specific top-level keys
cfg['output_dir']    = output_dir
cfg['expname']       = expname
cfg['decoding_mode'] = 'customized'

# conv_behav: top-level force_turn_taking bool (no server_args)
if benchmark == 'conv_behav':
    cfg['force_turn_taking'] = (force_tt == 'true')

with open(dst, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True,
              sort_keys=False, width=2147483647)
PYEOF
    echo "$tmp"
}

# Writes the patched YAML config as a JSON file into the benchmark's output folder.
# Called once per benchmark in customized mode before job submission.
dump_config_json() {
    local config_yaml="$1"
    local benchmark_name="$2"
    local resolved_outdir="$3"   # output dir with commit suffix

    [[ "$DRY_RUN" == "true" ]] && return 0

    mkdir -p "$resolved_outdir"
    local json_path="${resolved_outdir}/${benchmark_name}_config.json"
    "$PYTHON" - "$config_yaml" "$json_path" <<'PYEOF'
import sys, yaml, json
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
with open(sys.argv[2], 'w') as f:
    json.dump(cfg, f, indent=2, default=str)
print(f"  Config JSON: {sys.argv[2]}")
PYEOF
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

    local base_config
    base_config=$(config_for "$name" "$mode")

    local config="$base_config"
    local tmp_config=""

    if [[ "$mode" == "customized" ]]; then
        # Customized mode: patch base config with decoding params + output_dir/expname.
        # output_dir passed to patcher is without commit; Python scripts append it.
        local cust_output_dir="${OUTPUT_DIR_OVERRIDE}/${mode}/${name}"
        tmp_config=$(make_patched_config_customized "$name" "$cust_output_dir")
        config="$tmp_config"
        # Dump config.json into the resolved output dir before submission.
        dump_config_json "$config" "$name" "${OUTPUT_DIR_OVERRIDE}/${mode}/${name}_${COMMIT}"
    else
        # --output_dir override: each benchmark gets its own subdirectory so
        # result files from different benchmarks never collide.  Python scripts
        # append the git commit hash, producing e.g. OUTPUT_DIR_OVERRIDE/greedy/bba_a1b2c3d.
        if [[ -n "$OUTPUT_DIR_OVERRIDE" ]]; then
            extra+=("--output_dir" "${OUTPUT_DIR_OVERRIDE}/${mode}/${name}")
        fi
        if [[ -n "$MODEL_OVERRIDE" || -n "$CODE_PATH_OVERRIDE" ]]; then
            tmp_config=$(make_patched_config "$base_config")
            config="$tmp_config"
        fi
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
            "$PYTHON" "$VB_SCRIPT" --config "$config" "${extra[@]}"
            ;;
        vb_mcq)
            "$PYTHON" "$VB_SCRIPT" --config "$config" "${extra[@]}"
            ;;
        fdb)
            "$PYTHON" "$FDB_SCRIPT" --config "$config" "${extra[@]}"
            ;;
        bba)
            "$PYTHON" "$BBA_SCRIPT" --config "$config" "${extra[@]}"
            ;;
        bfcl)
            "$PYTHON" "$BFCL_SCRIPT" --config "$config" "${extra[@]}"
            ;;
        conv_behav)
            "$PYTHON" "$CONV_BEHAV_SCRIPT" --config "$config" "${extra[@]}"
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
[[ -n "$OUTPUT_DIR_OVERRIDE" ]] && echo " Output dir   : $OUTPUT_DIR_OVERRIDE/{mode}/{name}_{commit}"
if [[ "$EVAL_MODE" == "customized" ]]; then
    echo " force_turn_taking  : $CUSTOM_FORCE_TURN_TAKING"
    echo " top_p              : $CUSTOM_TOP_P"
    echo " repetition_penalty : $CUSTOM_REPETITION_PENALTY"
    echo " temperature        : $CUSTOM_TEMPERATURE"
fi
echo " HTML name    : ${HTML_NAME}.html"
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
        cancel_stale_jobs "$benchmark" "$mode"
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
