#!/usr/bin/env bash
# run_all_benchmarks.sh
#
# Submit all S2S FC eval benchmarks sequentially, throttling submission based
# on the SLURM job-count limit (queried from sacctmgr QOS settings and
# scontrol partition settings; the tighter of the two is used).
#
# Decoding defaults to greedy via each benchmark's _greedy YAML.  Pass
# --temperature/--top_p/--repetition_penalty/--force_turn_taking to override
# (all four + --output_dir required together).
#
# Run from the repo root:
#   bash scripts/run_all_benchmarks.sh [options]
#
# Examples:
#   # Run the default suite with greedy, incremental decoding
#   bash scripts/run_all_benchmarks.sh
#
#   # Run offline-decoding configs for all benchmarks
#   bash scripts/run_all_benchmarks.sh --decoding_mode offline
#
#   # Run only BBA and BFCL, override the checkpoint
#   bash scripts/run_all_benchmarks.sh --benchmarks bba,bfcl --model /path/to/ckpt
#
#   # Customized configs (per benchmark)
#   bash scripts/run_all_benchmarks.sh --decoding_mode customized \
#        --benchmarks bba,bfcl --config_bba /my/bba.yaml --config_bfcl /my/bfcl.yaml
#
#   # Sampling-style override (requires all four + --output_dir)
#   bash scripts/run_all_benchmarks.sh --temperature 0.8 --top_p 0.8 \
#        --repetition_penalty 1.0 --force_turn_taking false --output_dir /path/out

set -euo pipefail

# Some submission paths allocate a pseudo-terminal.  If one exits abruptly, the
# parent terminal can be left with echo disabled, making typed input invisible.
_TTY_STATE=""
if [[ -t 0 ]]; then
    _TTY_STATE="$(stty -g 2>/dev/null || true)"
fi

restore_terminal() {
    if [[ -n "$_TTY_STATE" && -t 0 ]]; then
        stty "$_TTY_STATE" 2>/dev/null || stty sane 2>/dev/null || true
    fi
}
trap restore_terminal EXIT HUP INT TERM

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
FDB_V3_SCRIPT="${REPO_ROOT}/nemo_skills/dataset/fdb/scripts/fdb_v3/run_eval.py"
FDB_V3_CHEN_CHEN_SCRIPT="${REPO_ROOT}/nemo_skills/dataset/fdb/scripts/fdb_v3_chen_chen/run_eval.py"
# fdb_v3_official reuses the shared fdb_v3 run_eval.py; only its config differs.
FDB_V3_OFFICIAL_SCRIPT="${FDB_V3_SCRIPT}"
BBA_SCRIPT="${REPO_ROOT}/nemo_skills/dataset/bba/scripts/run_bba_eval.py"
BFCL_SCRIPT="${REPO_ROOT}/nemo_skills/dataset/bfcl_single_turn_function_channel/scripts/run_eval.py"
CONV_BEHAV_SCRIPT="${REPO_ROOT}/nemo_skills/dataset/conv_behav/scripts/run_eval.py"
MEGATRON_CHECKPOINT_MANAGER="${REPO_ROOT}/scripts/megatron/duplex_checkpoint_manager.py"

# Config base directories — used by config_for() to derive default YAML paths.
VB_BASE="${REPO_ROOT}/nemo_skills/dataset/voicebench/scripts"
FDB_BASE="${REPO_ROOT}/nemo_skills/dataset/fdb/scripts"
FDB_V3_BASE="${REPO_ROOT}/nemo_skills/dataset/fdb/scripts/fdb_v3"
FDB_V3_CHEN_CHEN_BASE="${REPO_ROOT}/nemo_skills/dataset/fdb/scripts/fdb_v3_chen_chen"
FDB_V3_OFFICIAL_BASE="${REPO_ROOT}/nemo_skills/dataset/fdb/scripts/fdb_v3_official"
BBA_BASE="${REPO_ROOT}/nemo_skills/dataset/bba/scripts"
BFCL_BASE="${REPO_ROOT}/nemo_skills/dataset/bfcl_single_turn_function_channel/scripts"
CB_BASE="${REPO_ROOT}/nemo_skills/dataset/conv_behav/scripts"

# ---------------------------------------------------------------------------
# Argument defaults
# ---------------------------------------------------------------------------
KNOWN_BENCHMARKS="conv_behav fdb_v1 fdb_v1_5 fdb_v3 fdb_v3_chen_chen fdb_v3_official bba bfcl vb_mcq vb_nonmcq"
# ChenChen remains explicitly selectable for conventional checkpoints, but is
# not part of the default suite. Its custom native inference stack cannot load
# the split hybrid-vLLM Megatron export.
DEFAULT_BENCHMARKS="conv_behav fdb_v1 fdb_v1_5 fdb_v3 fdb_v3_official bba bfcl vb_mcq vb_nonmcq"
# Aliases that expand to multiple benchmark names in --benchmarks.
declare -A BENCHMARK_ALIASES=(
    [fdb]="fdb_v1 fdb_v1_5 fdb_v3"
)
SELECTED_BENCHMARKS=""

CONFIG_VB_NONMCQ=""
CONFIG_VB_MCQ=""
CONFIG_FDB_V1=""
CONFIG_FDB_V1_5=""
CONFIG_FDB_V3=""
CONFIG_FDB_V3_CHEN_CHEN=""
CONFIG_FDB_V3_OFFICIAL=""
CONFIG_BBA=""
CONFIG_BFCL=""
CONFIG_CONV_BEHAV=""

MODEL_OVERRIDE=""
OUTPUT_DIR_OVERRIDE=""
PROCESSED_CKPT_DIR=""
MEGATRON_CONVERSION_JOB_ID=""
MEGATRON_MODEL_ACTIVE=false
DECODING_MODE="incremental"   # incremental | offline | customized
HTML_NAME="scorecard"
MAX_JOBS_OVERRIDE=""
POLL_INTERVAL=60
DRY_RUN=false
FORCE_RERUN=false
# RESUME bypasses the "config.json exists -> error out" guard so partial/failed
# runs can resume in place. Unlike FORCE_RERUN, it does NOT pass --scoring_force
# to benchmark scripts — they'll skip generation if output.jsonl is present and
# skip scoring on the compute node if metrics.json is complete.
RESUME=false

# Decoding-param overrides (all four required together when any is set, plus --output_dir)
CUSTOM_FORCE_TURN_TAKING=""   # "true" or "false"
CUSTOM_TOP_P=""
CUSTOM_REPETITION_PENALTY=""
CUSTOM_TEMPERATURE=""

# RNNT turn-taking toggle. Standalone (NOT part of the all-four decoding group)
# because it selects a turn-taking mechanism rather than a sampling parameter,
# and needs to be flippable on its own for A/B runs. Empty = leave the YAML
# alone, which means the backend default (RNNT on) applies.
CUSTOM_USE_RNNT_TT=""   # "true" or "false"

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
usage() {
    cat <<EOF
Usage: bash scripts/run_all_benchmarks.sh [options]

Submit S2S FC eval benchmarks sequentially.  Before each benchmark the script
checks the current SLURM job count against the detected (or supplied) limit and
waits until a slot is free.

Decoding defaults to greedy via each benchmark's *_greedy.yaml config.

Benchmark names: vb_nonmcq  vb_mcq  fdb_v1  fdb_v1_5  fdb_v3  fdb_v3_chen_chen  fdb_v3_official  bba  bfcl  conv_behav
Aliases:         fdb -> fdb_v1,fdb_v1_5,fdb_v3   (fdb_v3_chen_chen is opt-in: name it explicitly)

Options:
  --benchmarks LIST         Comma-separated subset of the benchmark names above.
                            Default: all except fdb_v3_chen_chen, in the order listed above.
                            "fdb" expands to all three FDB versions.
  --config_vb_nonmcq  PATH Config YAML for VoiceBench non-MCQ (overrides default greedy YAML)
  --config_vb_mcq     PATH Config YAML for VoiceBench MCQ
  --config_fdb_v1     PATH Config YAML for FDB v1.0
  --config_fdb_v1_5   PATH Config YAML for FDB v1.5
  --config_fdb_v3     PATH Config YAML for FDB v3
  --config_fdb_v3_chen_chen PATH Config YAML for FDB v3 ChenChen (upstream end-to-end variant)
  --config_fdb_v3_official PATH Config YAML for FDB v3 Official (faithful to public Full-Duplex-Bench v3)
  --config_bba        PATH Config YAML for BBA
  --config_bfcl       PATH Config YAML for BFCL
  --config_conv_behav PATH Config YAML for conv_behav

  --output_dir        PATH  Redirect all benchmark outputs under PATH/{name}_{commit}.
                            Each benchmark gets its own subdirectory so result-detection
                            across benchmarks cannot cross-contaminate. The Python
                            scripts append the git commit hash. If unset, each
                            benchmark uses its own output_dir from its YAML.
  --model             PATH  Override the model checkpoint for every benchmark.
  --processed_ckpt_dir PATH Exact destination for a converted Megatron checkpoint.
                            If unset, the preferred destination is
                            <model>/nemo_skills_converted. When that location
                            cannot be selected automatically, an interactive
                            prompt offers it or <output_dir>/nemo_skills_converted.
  --decoding_mode     MODE  incremental (default) | offline | customized.
                            incremental: use each benchmark's *_incremental_v2_greedy YAML
                                         (DRIRF codebase, triton sqsh).
                            offline:     use each benchmark's *_s2s_offline_greedy YAML
                                         (DSFTS codebase, nemo_duplex sqsh).
                            customized:  every selected benchmark must be given an
                                         explicit --config_<name> PATH.
                            conv_behav uses its incremental/offline greedy YAMLs
                            according to the selected mode, like the other benchmarks.

  Decoding-param overrides (all four required together when any is set; also
  requires --output_dir).  Without these, defaults from the greedy YAML are used.
  --force_turn_taking BOOL  "true" or "false"
  --top_p             VAL   LLM top-p sampling value
  --repetition_penalty VAL  Repetition penalty value
  --temperature       VAL   Sampling temperature value

  RNNT turn-taking (standalone; may be used on its own, no --output_dir needed).
  --use_rnnt_turn_taking BOOL
                            "true"  -> RNNT blank/non-blank drives agent BOS/EOS
                                       (this is the backend default)
                            "false" -> revert to the legacy ASR-text-channel
                                       heuristic
                            Omitted -> leave the YAML untouched.

  --html_name         NAME  Base name for the scorecard HTML (default: scorecard).
  --max_jobs          N     Override SLURM job limit (auto-detected by default)
  --poll_interval     N     Seconds between SLURM queue checks (default: 60)
  --dry_run                 Pass --dry_run to every benchmark script
  --force_rerun             Re-run all benchmarks even if results already exist,
                            and pass --scoring_force to each benchmark (forces
                            re-scoring even when metrics.json is already correct).
                            If the existing config.json doesn't match the current
                            settings, the per-benchmark output dir is wiped so
                            generation also redoes (otherwise the benchmark python
                            script would skip generation when output.jsonl + .done
                            markers are present).
  --resume                  Allow re-entering output dirs that already have a
                            config.json (e.g. partial/failed prior run), WITHOUT
                            forcing scoring to redo. Fully-complete benchmarks
                            are still skipped via find_benchmark_result; partial
                            ones re-invoke the benchmark python script, which
                            does per-stage skipping (generation skipped if
                            output.jsonl + .done markers present; scoring jobs
                            no-op on the compute node if metrics.json is good).
                            If the existing config.json doesn't match the current
                            settings, the benchmark is SKIPPED with the diff
                            printed — pass --force_rerun to override and wipe.
  --help                    Show this message and exit
EOF
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --benchmarks)        SELECTED_BENCHMARKS="$2";  shift 2 ;;
        --config_vb_nonmcq)  CONFIG_VB_NONMCQ="$2";    shift 2 ;;
        --config_vb_mcq)     CONFIG_VB_MCQ="$2";        shift 2 ;;
        --config_fdb_v1)     CONFIG_FDB_V1="$2";         shift 2 ;;
        --config_fdb_v1_5)   CONFIG_FDB_V1_5="$2";       shift 2 ;;
        --config_fdb_v3)     CONFIG_FDB_V3="$2";         shift 2 ;;
        --config_fdb_v3_chen_chen) CONFIG_FDB_V3_CHEN_CHEN="$2"; shift 2 ;;
        --config_fdb_v3_official) CONFIG_FDB_V3_OFFICIAL="$2"; shift 2 ;;
        --config_bba)        CONFIG_BBA="$2";            shift 2 ;;
        --config_bfcl)       CONFIG_BFCL="$2";           shift 2 ;;
        --config_conv_behav) CONFIG_CONV_BEHAV="$2";     shift 2 ;;
        --html_name)         HTML_NAME="$2";               shift 2 ;;
        --output_dir)        OUTPUT_DIR_OVERRIDE="$2";    shift 2 ;;
        --model)             MODEL_OVERRIDE="$2";         shift 2 ;;
        --processed_ckpt_dir) PROCESSED_CKPT_DIR="$2";    shift 2 ;;
        --decoding_mode)     DECODING_MODE="$2";         shift 2 ;;
        --max_jobs)          MAX_JOBS_OVERRIDE="$2";     shift 2 ;;
        --poll_interval)     POLL_INTERVAL="$2";          shift 2 ;;
        --force_turn_taking) CUSTOM_FORCE_TURN_TAKING="$2"; shift 2 ;;
        --use_rnnt_turn_taking) CUSTOM_USE_RNNT_TT="$2";  shift 2 ;;
        --top_p)             CUSTOM_TOP_P="$2";            shift 2 ;;
        --repetition_penalty) CUSTOM_REPETITION_PENALTY="$2"; shift 2 ;;
        --temperature)       CUSTOM_TEMPERATURE="$2";     shift 2 ;;
        --dry_run)           DRY_RUN=true;                shift ;;
        --force_rerun)       FORCE_RERUN=true;            shift ;;
        --resume)            RESUME=true;                 shift ;;
        --help|-h)           usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

# Validate decoding-param overrides: if any set, require all four + --output_dir.
_n_custom=0
if [[ -n "$CUSTOM_FORCE_TURN_TAKING" ]];  then (( ++_n_custom )); fi
if [[ -n "$CUSTOM_TOP_P" ]];              then (( ++_n_custom )); fi
if [[ -n "$CUSTOM_REPETITION_PENALTY" ]]; then (( ++_n_custom )); fi
if [[ -n "$CUSTOM_TEMPERATURE" ]];        then (( ++_n_custom )); fi

if [[ "$_n_custom" -gt 0 ]]; then
    # Prompt for any missing required values; error if stdin is not a terminal.
    _missing=false
    if [[ -z "$CUSTOM_FORCE_TURN_TAKING" || -z "$CUSTOM_TOP_P" || -z "$CUSTOM_REPETITION_PENALTY" \
       || -z "$CUSTOM_TEMPERATURE" || -z "$OUTPUT_DIR_OVERRIDE" ]]; then
        _missing=true
    fi
    if [[ "$_missing" == "true" ]]; then
        if [[ ! -t 0 ]]; then
            echo "ERROR: When any of --top_p/--temperature/--repetition_penalty/--force_turn_taking" >&2
            echo "       is set, all four are required, plus --output_dir." >&2
            exit 1
        fi
        echo ""
        echo "Decoding-param overrides — provide missing values (leave blank to abort):"
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

    if [[ -z "$CUSTOM_FORCE_TURN_TAKING" || -z "$CUSTOM_TOP_P" || -z "$CUSTOM_REPETITION_PENALTY" \
       || -z "$CUSTOM_TEMPERATURE" || -z "$OUTPUT_DIR_OVERRIDE" ]]; then
        echo "ERROR: When any of --top_p/--temperature/--repetition_penalty/--force_turn_taking" >&2
        echo "       is set, all four are required, plus --output_dir." >&2
        exit 1
    fi
    if [[ "$CUSTOM_FORCE_TURN_TAKING" != "true" && "$CUSTOM_FORCE_TURN_TAKING" != "false" ]]; then
        echo "ERROR: --force_turn_taking must be 'true' or 'false', got: '$CUSTOM_FORCE_TURN_TAKING'" >&2
        exit 1
    fi
fi
HAS_DECODING_OVERRIDES=$([[ "$_n_custom" -gt 0 ]] && echo "true" || echo "false")

# Validated separately from the decoding group: it is independently optional.
if [[ -n "$CUSTOM_USE_RNNT_TT" \
      && "$CUSTOM_USE_RNNT_TT" != "true" && "$CUSTOM_USE_RNNT_TT" != "false" ]]; then
    echo "ERROR: --use_rnnt_turn_taking must be 'true' or 'false', got: '$CUSTOM_USE_RNNT_TT'" >&2
    exit 1
fi

# Validate --decoding_mode value
case "$DECODING_MODE" in
    incremental|offline|customized) ;;
    *)
        echo "ERROR: --decoding_mode must be one of {incremental, offline, customized}, got: '$DECODING_MODE'" >&2
        exit 1
        ;;
esac

# Derive a short suffix from OUTPUT_DIR_OVERRIDE so that concurrent runs with
# different output directories produce distinct SLURM job names and
# cancel_stale_jobs cannot accidentally cancel an unrelated parallel run.
_EXPNAME_SUFFIX=""
if [[ -n "$OUTPUT_DIR_OVERRIDE" ]]; then
    _EXPNAME_SUFFIX="_$(printf '%s' "${OUTPUT_DIR_OVERRIDE%/}" | md5sum | cut -c1-8)"
fi

# ---------------------------------------------------------------------------
# Resolve benchmark list
# ---------------------------------------------------------------------------
if [[ -n "$SELECTED_BENCHMARKS" ]]; then
    BENCHMARKS="${SELECTED_BENCHMARKS//,/ }"
else
    BENCHMARKS="$DEFAULT_BENCHMARKS"
fi

# Expand aliases (e.g. "fdb" -> "fdb_v1 fdb_v1_5 fdb_v3"), then de-duplicate
# while preserving first-seen order.
expanded=""
for b in $BENCHMARKS; do
    if [[ -n "${BENCHMARK_ALIASES[$b]+x}" ]]; then
        expanded="$expanded ${BENCHMARK_ALIASES[$b]}"
    else
        expanded="$expanded $b"
    fi
done
declare -A _seen=()
deduped=""
for b in $expanded; do
    if [[ -z "${_seen[$b]+x}" ]]; then
        _seen[$b]=1
        deduped="$deduped $b"
    fi
done
BENCHMARKS="${deduped# }"

# Validate each name
for b in $BENCHMARKS; do
    if [[ ! " $KNOWN_BENCHMARKS " =~ " $b " ]]; then
        echo "Unknown benchmark: '$b'. Valid names: $KNOWN_BENCHMARKS" >&2
        echo "Aliases: ${!BENCHMARK_ALIASES[*]}" >&2
        exit 1
    fi
done

# Reorder to size order (KNOWN_BENCHMARKS is ordered smallest → largest).
ordered=""
for b in $KNOWN_BENCHMARKS; do
    if [[ " $BENCHMARKS " =~ " $b " ]]; then
        ordered="$ordered $b"
    fi
done
BENCHMARKS="${ordered# }"

# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------
# Returns the YAML path for a given benchmark.  Selects the incremental or
# offline greedy YAML based on $DECODING_MODE; a user-supplied --config_<name>
# always wins.
config_for() {
    local name="$1"
    local inc off
    case "$name" in
        vb_nonmcq)
            inc="${VB_BASE}/vb_matched_demo_v2_02mar_config_fc_s2s_incremental_v2_greedy.yaml"
            off="${VB_BASE}/vb_matched_demo_v2_02mar_config_fc_s2s_offline_greedy.yaml"
            _emit_config "$CONFIG_VB_NONMCQ" "$inc" "$off"
            ;;
        vb_mcq)
            inc="${VB_BASE}/vb_matched_demo_v2_02mar_mcq_config_fc_s2s_incremental_v2_greedy.yaml"
            off="${VB_BASE}/vb_matched_demo_v2_02mar_mcq_config_fc_s2s_offline_greedy.yaml"
            _emit_config "$CONFIG_VB_MCQ" "$inc" "$off"
            ;;
        fdb_v1)
            inc="${FDB_BASE}/fdb_s2s_incremental_v2_02mar_config_fc_greedy.yaml"
            off="${FDB_BASE}/fdb_s2s_offline_02mar_config_fc_greedy.yaml"
            _emit_config "$CONFIG_FDB_V1" "$inc" "$off"
            ;;
        fdb_v1_5)
            inc="${FDB_BASE}/fdb_s2s_incremental_v2_v1.5_02mar_config_fc_greedy.yaml"
            off="${FDB_BASE}/fdb_s2s_offline_v1.5_02mar_config_fc_greedy.yaml"
            _emit_config "$CONFIG_FDB_V1_5" "$inc" "$off"
            ;;
        fdb_v3)
            inc="${FDB_V3_BASE}/fdb_v3_s2s_incremental_v2_config_fc_greedy.yaml"
            off="${FDB_V3_BASE}/fdb_v3_s2s_offline_config_fc_greedy.yaml"
            _emit_config "$CONFIG_FDB_V3" "$inc" "$off"
            ;;
        fdb_v3_chen_chen)
            # Upstream-end-to-end variant: single YAML drives the FD3 orchestrator
            # (Backend Agent + Qwen3 vLLM + DRIRF-wrapped S2S), so decoding_mode is
            # irrelevant — same config in both incremental and offline slots.
            inc="${FDB_V3_CHEN_CHEN_BASE}/fdb_v3_chen_chen_config.yaml"
            off="${FDB_V3_CHEN_CHEN_BASE}/fdb_v3_chen_chen_config.yaml"
            _emit_config "$CONFIG_FDB_V3_CHEN_CHEN" "$inc" "$off"
            ;;
        fdb_v3_official)
            # Faithful-to-public-benchmark variant: single incremental_v2 YAML
            # (official tool spec/prompt/mock/scoring/data). No separate offline
            # config -> same YAML in both incremental and offline slots.
            inc="${FDB_V3_OFFICIAL_BASE}/fdb_v3_official_s2s_incremental_v2_config_fc_greedy.yaml"
            off="${FDB_V3_OFFICIAL_BASE}/fdb_v3_official_s2s_incremental_v2_config_fc_greedy.yaml"
            _emit_config "$CONFIG_FDB_V3_OFFICIAL" "$inc" "$off"
            ;;
        bba)
            inc="${BBA_BASE}/bba_config_fc_s2s_incremental_v2_greedy.yaml"
            off="${BBA_BASE}/bba_config_fc_s2s_offline_greedy.yaml"
            _emit_config "$CONFIG_BBA" "$inc" "$off"
            ;;
        bfcl)
            inc="${BFCL_BASE}/bfcl_fc_config_s2s_incremental_v2_greedy.yaml"
            off="${BFCL_BASE}/bfcl_fc_config_s2s_offline_greedy.yaml"
            _emit_config "$CONFIG_BFCL" "$inc" "$off"
            ;;
        conv_behav)
            inc="${CB_BASE}/conv_behav_incremental_config_greedy.yaml"
            # Default offline mode uses DSFTS (matches the pre-split conv_behav_config_greedy.yaml behavior).
            # To run drirf_offline instead, pass --config_conv_behav <path to conv_behav_offline_config_greedy.yaml>.
            off="${CB_BASE}/conv_behav_dsfts_offline_config_greedy.yaml"
            _emit_config "$CONFIG_CONV_BEHAV" "$inc" "$off"
            ;;
    esac
}

# Helper: returns the user override if non-empty, otherwise picks between
# incremental and offline YAMLs based on $DECODING_MODE.  In customized mode
# the user override is mandatory — emits empty if missing (caller must check).
_emit_config() {
    local user="$1" inc="$2" off="$3"
    if [[ -n "$user" ]]; then
        echo "$user"
        return
    fi
    case "$DECODING_MODE" in
        incremental) echo "$inc" ;;
        offline)     echo "$off" ;;
        customized)  echo "" ;;
    esac
}

# Validate that customized mode has --config_<name> for every selected
# benchmark.
validate_customized_configs() {
    [[ "$DECODING_MODE" == "customized" ]] || return 0
    local missing=()
    for b in $BENCHMARKS; do
        local got
        got=$(config_for "$b")
        if [[ -z "$got" ]]; then
            missing+=("$b")
        fi
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "" >&2
        echo "ERROR: --decoding_mode=customized requires --config_<name> for every selected benchmark." >&2
        echo "Missing configs for: ${missing[*]}" >&2
        echo "Pass --config_${missing[0]} PATH (and similarly for any other missing benchmarks)." >&2
        exit 1
    fi
}

# Returns the expname for a given benchmark, including the _EXPNAME_SUFFIX
# that differentiates concurrent runs writing to different output directories.
expname_for() {
    local name="$1"
    local config base_exp
    config=$(config_for "$name")
    base_exp=$(grep -m1 '^expname:' "$config" 2>/dev/null | sed 's/^expname: *//' || true)
    if [[ -z "$base_exp" ]]; then
        echo ""
        return
    fi
    echo "${base_exp}${_EXPNAME_SUFFIX}"
}

# ---------------------------------------------------------------------------
# Output dir resolution + completion detection
# ---------------------------------------------------------------------------

# Returns the fully-resolved output dir for a benchmark, including the
# git commit hash suffix that Python scripts append.
# When --output_dir is set each benchmark gets its own subdirectory so
# find_benchmark_result can't confuse one benchmark's metrics.json for another's.
resolve_output_dir() {
    local name="$1"
    if [[ -n "$OUTPUT_DIR_OVERRIDE" ]]; then
        echo "${OUTPUT_DIR_OVERRIDE}/${name}_${COMMIT}"
    else
        local config yaml_dir
        config=$(config_for "$name")
        yaml_dir=$(grep -m1 '^output_dir:' "$config" 2>/dev/null | sed 's/^output_dir: *//')
        echo "${yaml_dir}_${COMMIT}"
    fi
}

# Returns the base output dir (without commit suffix) for a benchmark.
base_output_dir_for() {
    local name="$1"
    if [[ -n "$OUTPUT_DIR_OVERRIDE" ]]; then
        echo "${OUTPUT_DIR_OVERRIDE}/${name}"
    else
        local config
        config=$(config_for "$name")
        grep -m1 '^output_dir:' "$config" 2>/dev/null | sed 's/^output_dir: *//' || true
    fi
}

# Echoes the path of the first result file found for a benchmark
# (a metrics.json under eval-results/ or a report.html in the output dir),
# or nothing if no results exist yet.
find_benchmark_result() {
    local name="$1"
    local outdir config
    outdir=$(resolve_output_dir "$name")
    [[ -d "$outdir" ]] || return 0
    config=$(config_for "$name")

    # Completion checks are benchmark-aware so a partial multi-stage run cannot
    # suppress resubmission.  In particular, VoiceBench incremental runs need
    # both generated and ASR metrics, and BBA needs its aggregate metrics.
    local result_path
    result_path=$("$PYTHON" - "$config" "$outdir" "$name" <<'PYEOF'
import json
import sys
from pathlib import Path

import yaml

config_path, outdir, name = sys.argv[1], sys.argv[2], sys.argv[3]
with open(config_path) as f:
    cfg = yaml.safe_load(f) or {}

eval_results = Path(outdir) / "eval-results"

VOICEBENCH_ALL = [
    "advbench",
    "alpacaeval",
    "alpacaeval_full",
    "alpacaeval_speaker",
    "bbh",
    "commoneval",
    "ifeval",
    "mmsu",
    "mtbench",
    "openbookqa",
    "sd_qa",
    "sd_qa_usa",
    "wildvoice",
]
FDB_V1_ALL = ["pause_candor", "pause_synthetic", "backchannel", "turn_taking", "interruption"]
FDB_V1_5_ALL = ["background_speech", "talking_to_other", "backchannel", "interruption"]
BBA_ALL = ["formal_fallacies", "navigate", "object_counting", "web_of_lies"]
BFCL_ALL = ["simple", "parallel", "multiple", "parallel_multiple", "irrelevance"]


def as_items(value, default):
    if not value or value == "all":
        return list(default)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


def as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def load_json(path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def metric_file_has(path, key):
    data = load_json(path)
    return isinstance(data, dict) and bool(data.get(key))


def voicebench_complete(path, subtest, require_asr):
    data = load_json(path)
    if not isinstance(data, dict):
        return False
    metrics = data.get(f"voicebench.{subtest}")
    if not isinstance(metrics, dict):
        return False
    has_generated = any((not k.startswith("agent_")) and (not k.endswith("_asr")) for k in metrics)
    if not has_generated:
        return False
    if require_asr and not any(k.endswith("_asr") for k in metrics):
        return False
    return True


def complete_all(paths_and_checks):
    first = None
    for path, check in paths_and_checks:
        if first is None:
            first = path
        if not path.exists() or not check(path):
            return None
    return first


if name in {"vb_mcq", "vb_nonmcq"}:
    require_asr = cfg.get("agent_audio_stage_enabled")
    if require_asr is None:
        require_asr = "--decode_audio" in (cfg.get("server_args") or "")
    subtests = as_items(cfg.get("subtests", "all"), VOICEBENCH_ALL)
    checks = [
        (
            eval_results / f"voicebench.{subtest}" / "metrics.json",
            lambda path, subtest=subtest: voicebench_complete(path, subtest, as_bool(require_asr)),
        )
        for subtest in subtests
    ]
    result = complete_all(checks)
    if result:
        print(result)
elif name in {"fdb_v1", "fdb_v1_5"}:
    fdb_version = cfg.get("fdb_version", "v1.0")
    prefix = "fdb_v1_5" if fdb_version == "v1.5" else "fdb_v1"
    subtests = as_items(cfg.get("subtests", "all"), FDB_V1_5_ALL if fdb_version == "v1.5" else FDB_V1_ALL)
    checks = [
        (
            eval_results / f"{prefix}.{subtest}" / "metrics.json",
            lambda path, prefix=prefix, subtest=subtest: metric_file_has(path, f"{prefix}.{subtest}"),
        )
        for subtest in subtests
    ]
    result = complete_all(checks)
    if result:
        print(result)
elif name == "fdb_v3":
    path = eval_results / "fdb_v3.tool_call" / "metrics.json"
    if path.exists() and metric_file_has(path, "fdb_v3.tool_call"):
        print(path)
elif name == "fdb_v3_chen_chen":
    path = eval_results / "fdb_v3_chen_chen.tool_call" / "metrics.json"
    if path.exists() and metric_file_has(path, "fdb_v3_chen_chen.tool_call"):
        print(path)
elif name == "fdb_v3_official":
    # Generation reuses benchmark fdb_v3.tool_call, so results land in the
    # fdb_v3.tool_call/ dir, but metrics are keyed fdb_v3_official.tool_call.
    path = eval_results / "fdb_v3.tool_call" / "metrics.json"
    if path.exists() and metric_file_has(path, "fdb_v3_official.tool_call"):
        print(path)
elif name == "bba":
    categories = as_items(cfg.get("categories", "all"), BBA_ALL)
    checks = [
        (
            eval_results / category / "metrics.json",
            lambda path, category=category: metric_file_has(path, f"bba.{category}"),
        )
        for category in categories
    ]
    checks.append((eval_results / "bba_aggregate" / "metrics.json", lambda path: metric_file_has(path, "bba.aggregate")))
    result = complete_all(checks)
    if result:
        print(result)
elif name == "bfcl":
    categories = as_items(cfg.get("categories", "all"), BFCL_ALL)
    checks = [
        (
            eval_results / category / "metrics.json",
            lambda path, category=category: metric_file_has(path, f"bfcl_fc.{category}"),
        )
        for category in categories
    ]
    result = complete_all(checks)
    if result:
        print(result)
elif name == "conv_behav":
    path = eval_results / "metrics.json"
    if path.exists() and metric_file_has(path, "conv_behav"):
        print(path)
else:
    first_metric = next(eval_results.glob("*/metrics.json"), None)
    if first_metric:
        print(first_metric)
    else:
        report = Path(outdir) / "report.html"
        if report.exists():
            print(report)
PYEOF
    2>/dev/null || true)

    if [[ -n "$result_path" ]]; then
        echo "$result_path"
        return 0
    fi
    return 0
}

# Echoes the path of the aggregate scorecard if it exists.
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

# Echoes the space-separated, deduped list of SLURM partitions referenced by
# the selected benchmarks' cluster configs (first token of `partition:` plus
# `cpu_partition:`). Empty if no cluster config is resolvable.
partitions_from_configs() {
    local -a parts=()
    local seen_clusters=" "
    for b in $BENCHMARKS; do
        local bench_cfg cluster cluster_cfg
        bench_cfg=$(config_for "$b" 2>/dev/null || true)
        [[ -z "$bench_cfg" || ! -f "$bench_cfg" ]] && continue
        cluster=$(grep -m1 '^cluster:' "$bench_cfg" 2>/dev/null \
            | sed 's/^cluster:[[:space:]]*//;s/[[:space:]]*$//' || true)
        [[ -z "$cluster" ]] && continue
        [[ "$seen_clusters" == *" $cluster "* ]] && continue
        seen_clusters+="$cluster "
        cluster_cfg="${REPO_ROOT}/cluster_configs/${cluster}.yaml"
        [[ ! -f "$cluster_cfg" ]] && continue
        local p
        for key in partition cpu_partition; do
            p=$(grep -m1 "^${key}:" "$cluster_cfg" 2>/dev/null \
                | sed "s/^${key}:[[:space:]]*//;s/[[:space:]]*\$//;s/,.*//" || true)
            [[ -n "$p" ]] && parts+=("$p")
        done
    done
    # Dedupe while preserving order.
    local seen=" " out=""
    for p in "${parts[@]}"; do
        if [[ "$seen" != *" $p "* ]]; then
            seen+="$p "
            out+="$p "
        fi
    done
    echo "${out% }"
}

# Returns the SLURM max-submit-jobs limit for the current user.
detect_max_jobs() {
    local -a limits=()

    local assoc_max
    assoc_max=$(sacctmgr show association user="$USER" format=MaxSubmitJobs \
        -n -P 2>/dev/null | tr -d ' ' | grep -E '^[0-9]+$' | head -1 || true)
    if [[ -n "$assoc_max" && "$assoc_max" -gt 0 ]]; then
        limits+=("$assoc_max")
    fi

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

    local partitions_to_check
    partitions_to_check=$(partitions_from_configs)
    for part in $partitions_to_check; do
        local part_max
        part_max=$(scontrol show partition "$part" 2>/dev/null \
            | grep -o 'MaxSubmitJobsPerUser=[^ ]*' \
            | cut -d= -f2 | grep -E '^[0-9]+$' | head -1 || true)
        if [[ -n "$part_max" && "$part_max" -gt 0 ]]; then
            limits+=("$part_max")
            break
        fi
    done

    if [[ ${#limits[@]} -eq 0 ]]; then
        echo ""
        return
    fi

    local min="${limits[0]}"
    for v in "${limits[@]}"; do
        if [[ "$v" -lt "$min" ]]; then min="$v"; fi
    done
    echo "$min"
}

count_user_jobs() {
    squeue -u "$USER" -h 2>/dev/null | wc -l | tr -d ' '
}

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
_delete_incomplete_dir() {
    local dir="$1"
    [[ -d "$dir" ]] || return 0
    local f
    f=$(find "${dir}/eval-results" -name "metrics.json" -maxdepth 2 2>/dev/null | head -1 || true)
    [[ -z "$f" ]] && f=$(find "${dir}" -name "report.html" -maxdepth 1 2>/dev/null | head -1 || true)
    if [[ -n "$f" ]]; then
        printf '  [cleanup] Skipping (scoring complete): %s\n' "$dir"
        return 0
    fi
    f=$(find "${dir}/eval-results" -name "output.jsonl.done" -o -name "output_chunk_*.jsonl.done" \
        2>/dev/null | head -1 || true)
    [[ -z "$f" ]] && f=$(find "${dir}/eval-results/validation_logs/metadatas" -name "*.json.done" \
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
# expname prefix for the given benchmark.  After cancellation, cleans up the
# output directories of the cancelled jobs.
cancel_stale_jobs() {
    local name="$1"

    local expname
    expname=$(expname_for "$name") || true
    if [[ -z "$expname" ]]; then
        echo "  [dedup] WARNING: cannot read expname for $name — skipping duplicate check" >&2
        return 0
    fi

    local cluster cluster_cfg job_name_prefix=""
    cluster=$(grep -m1 '^cluster:' "$(config_for "$name")" 2>/dev/null \
        | sed 's/^cluster:[[:space:]]*//' || true)
    if [[ -n "$cluster" ]]; then
        cluster_cfg="${REPO_ROOT}/cluster_configs/${cluster}.yaml"
        if [[ -f "$cluster_cfg" ]]; then
            job_name_prefix=$("$PYTHON" -c "
import yaml, sys
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
print(cfg.get('job_name_prefix', ''), end='')
" "$cluster_cfg" 2>/dev/null || true)
        fi
    fi

    local full_expname="${job_name_prefix}${expname}"
    local awk_include
    awk_include="(\$2 == \"$full_expname\" || index(\$2, \"${full_expname}_\") == 1)"

    local base_outdir
    base_outdir=$(base_output_dir_for "$name")

    local matching
    matching=$(squeue -u "$USER" -h -o "%i %j %T %r" 2>/dev/null \
        | awk "{ if (${awk_include}) print }" || true)

    if [[ -z "$matching" ]]; then
        [[ -n "$base_outdir" ]] && _delete_incomplete_dir "${base_outdir}_${COMMIT}"
        return 0
    fi

    local count job_ids
    count=$(printf '%s\n' "$matching" | wc -l | tr -d ' ')
    job_ids=$(printf '%s\n' "$matching" | awk '{printf "%s ", $1}')

    echo ""
    printf '  [dedup] %s: %d existing job(s) match prefix "%s"\n' \
        "$name" "$count" "$full_expname"
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

    if [[ -n "$base_outdir" ]]; then
        local -a seen_commits=()
        while IFS= read -r line; do
            local job_name job_commit
            job_name=$(printf '%s\n' "$line" | awk '{print $2}')
            job_commit=$(printf '%s\n' "$job_name" | grep -oE '_[0-9a-f]{7,8}$' | tr -d '_' || true)
            [[ -z "$job_commit" ]] && continue
            local dup=false
            for c in "${seen_commits[@]+"${seen_commits[@]}"}"; do
                [[ "$c" == "$job_commit" ]] && { dup=true; break; }
            done
            $dup && continue
            seen_commits+=("$job_commit")
            _delete_incomplete_dir "${base_outdir}_${job_commit}"
        done <<< "$matching"

        local covered=false
        for c in "${seen_commits[@]+"${seen_commits[@]}"}"; do
            [[ "$c" == "$COMMIT" ]] && { covered=true; break; }
        done
        $covered || _delete_incomplete_dir "${base_outdir}_${COMMIT}"
    fi

    echo ""
}

# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------

check_model() {
    if [[ -n "$MODEL_OVERRIDE" ]]; then
        echo " Model        : $MODEL_OVERRIDE"
    fi
    echo " Decoding mode: $DECODING_MODE"
}

is_megatron_dcp() {
    "$PYTHON" "$MEGATRON_CHECKPOINT_MANAGER" probe-source --checkpoint "$1" --quiet
}

is_megatron_export() {
    "$PYTHON" "$MEGATRON_CHECKPOINT_MANAGER" probe-export --export-dir "$1" --quiet
}

detect_megatron_model() {
    [[ -n "$MODEL_OVERRIDE" ]] || return 0
    if is_megatron_export "$MODEL_OVERRIDE" || is_megatron_dcp "$MODEL_OVERRIDE"; then
        MEGATRON_MODEL_ACTIVE=true
    fi
}

filter_unsupported_megatron_benchmarks() {
    [[ "$MEGATRON_MODEL_ACTIVE" == "true" ]] || return 0
    local benchmark filtered=""
    if [[ -z "$SELECTED_BENCHMARKS" ]]; then
        echo "Megatron compatibility: fdb_v3_chen_chen is excluded from the default suite because its native inference engine does not support the split converted checkpoint."
    fi
    for benchmark in $BENCHMARKS; do
        if [[ "$benchmark" == "fdb_v3_chen_chen" ]]; then
            echo "Skipping fdb_v3_chen_chen: its native inference engine does not support the split Megatron converted checkpoint."
            echo "  Native-engine Megatron conversion is not implemented; use fdb_v3 or fdb_v3_official instead."
            continue
        fi
        filtered="${filtered:+$filtered }$benchmark"
    done
    BENCHMARKS="$filtered"
}

prepare_megatron_model() {
    [[ -n "$MODEL_OVERRIDE" ]] || return 0
    if [[ "$MEGATRON_MODEL_ACTIVE" == "true" && "$DECODING_MODE" == "offline" ]]; then
        echo "ERROR: Megatron Duplex checkpoints currently require incremental (DRIRF hybrid-vLLM) decoding." >&2
        exit 1
    fi
    is_megatron_export "$MODEL_OVERRIDE" && return 0
    if [[ "$MODEL_OVERRIDE" == /* && ! -e "$MODEL_OVERRIDE" ]]; then
        echo "ERROR: --model does not exist: $MODEL_OVERRIDE" >&2
        exit 1
    fi
    [[ "$MEGATRON_MODEL_ACTIVE" == "true" ]] || return 0
    local source_model="$MODEL_OVERRIDE"
    local export_dir preferred_dir default_dir model_writable=false
    local conversion_script="${REPO_ROOT}/scripts/megatron/duplex_convert_slurm.sh"
    echo "Detected Megatron Duplex Torch-DCP checkpoint: $source_model"

    preferred_dir="${source_model%/}/nemo_skills_converted"
    if "$PYTHON" "$MEGATRON_CHECKPOINT_MANAGER" probe-writable --directory "$source_model" --quiet; then
        model_writable=true
    fi

    if [[ -n "$PROCESSED_CKPT_DIR" ]]; then
        export_dir="$PROCESSED_CKPT_DIR"
    elif [[ "$model_writable" == "true" && ! -e "$preferred_dir" ]]; then
        export_dir="$preferred_dir"
    else
        if [[ "$model_writable" == "true" ]]; then
            default_dir="$preferred_dir"
        elif [[ -n "$OUTPUT_DIR_OVERRIDE" ]]; then
            default_dir="${OUTPUT_DIR_OVERRIDE%/}/nemo_skills_converted"
        else
            default_dir=""
        fi

        if [[ ! -t 0 ]]; then
            echo "ERROR: The converted-checkpoint destination requires a choice, but stdin is not interactive." >&2
            [[ -e "$preferred_dir" ]] \
                && echo "       The preferred destination already exists: $preferred_dir" >&2
            [[ "$model_writable" != "true" ]] \
                && echo "       The model directory is not writable: $source_model" >&2
            echo "       Pass --processed_ckpt_dir PATH and retry." >&2
            exit 1
        fi

        echo ""
        echo "Choose the converted-checkpoint directory:"
        if [[ -n "$default_dir" ]]; then
            echo "  Enter  Use default: $default_dir"
        else
            echo "  Enter  No default is available (choose 2 or 3)"
        fi
        echo "  2      Enter another path"
        echo "  3      Exit"
        local selection custom_dir
        read -r -p "Selection: " selection
        case "$selection" in
            "")
                if [[ -z "$default_dir" ]]; then
                    echo "ERROR: No default destination is available; pass --processed_ckpt_dir PATH." >&2
                    exit 1
                fi
                export_dir="$default_dir"
                ;;
            2)
                read -r -p "Converted-checkpoint directory: " custom_dir
                if [[ -z "$custom_dir" ]]; then
                    echo "ERROR: Converted-checkpoint directory cannot be empty." >&2
                    exit 1
                fi
                export_dir="$custom_dir"
                ;;
            3)
                echo "Exiting without submitting conversion or evaluation jobs."
                exit 0
                ;;
            *)
                echo "ERROR: Invalid selection: $selection" >&2
                exit 1
                ;;
        esac
    fi

    if [[ -e "$export_dir" && ! -d "$export_dir" ]]; then
        echo "ERROR: Converted-checkpoint destination exists and is not a directory: $export_dir" >&2
        exit 1
    fi
    mkdir -p "$export_dir"
    export_dir=$(readlink -f "$export_dir")
    PROCESSED_CKPT_DIR="$export_dir"

    if "$PYTHON" "$MEGATRON_CHECKPOINT_MANAGER" validate-export \
        --checkpoint "$source_model" --export-dir "$export_dir" --quiet; then
        echo "Using valid converted checkpoint: $export_dir"
        MODEL_OVERRIDE="$export_dir"
        return 0
    fi

    local first_benchmark base_config cluster cluster_config account partition container
    first_benchmark=${BENCHMARKS%% *}
    base_config=$(config_for "$first_benchmark")
    cluster=$(grep -m1 '^cluster:' "$base_config" | sed 's/^cluster:[[:space:]]*//')
    cluster_config="${REPO_ROOT}/cluster_configs/${cluster}.yaml"
    if [[ ! -f "$cluster_config" ]]; then
        echo "ERROR: Cannot resolve conversion Slurm settings; cluster config not found: $cluster_config" >&2
        exit 1
    fi
    account=$("$PYTHON" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1])).get("account", ""))' "$cluster_config")
    partition=$("$PYTHON" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1])).get("partition", ""))' "$cluster_config")
    container="${NEMO_SKILLS_MEGATRON_CONVERSION_CONTAINER:-/lustre/fsw/portfolios/llmservice/users/nsrihari/full_duplex/avlm/containers/megatron_voicechat_0626.sqsh}"
    mkdir -p "$export_dir/conversion_logs"

    local -a sbatch_args=(
        --parsable
        --job-name "megatron-duplex-convert-${COMMIT}"
        --nodes 1
        --gpus-per-node 1
        --ntasks-per-node 1
        --mem "${NEMO_SKILLS_MEGATRON_CONVERSION_MEMORY:-220G}"
        --time "${NEMO_SKILLS_MEGATRON_CONVERSION_TIME:-02:00:00}"
        --output "$export_dir/conversion_logs/%x_%j.log"
    )
    [[ -n "$account" ]] && sbatch_args+=(--account "$account")
    [[ -n "$partition" ]] && sbatch_args+=(--partition "$partition")
    sbatch_args+=(
        "$conversion_script"
        --checkpoint "$source_model"
        --output-dir "$export_dir"
        --repo-root "$REPO_ROOT"
        --container-image "$container"
    )

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "DRY RUN — would submit Megatron conversion job:"
        printf '  sbatch'
        printf ' %q' "${sbatch_args[@]}"
        printf '\n'
        echo "DRY RUN — evaluation jobs would use dependency: afterok:<conversion-job-id>"
    else
        if ! command -v sbatch >/dev/null 2>&1; then
            echo "ERROR: sbatch is required to convert a Megatron checkpoint." >&2
            exit 1
        fi
        MEGATRON_CONVERSION_JOB_ID=$(env -u SBATCH_DEPENDENCY sbatch "${sbatch_args[@]}")
        MEGATRON_CONVERSION_JOB_ID=${MEGATRON_CONVERSION_JOB_ID%%;*}
        if [[ ! "$MEGATRON_CONVERSION_JOB_ID" =~ ^[0-9]+$ ]]; then
            echo "ERROR: Could not parse conversion job ID: $MEGATRON_CONVERSION_JOB_ID" >&2
            exit 1
        fi
        export NEMO_SKILLS_SLURM_AFTEROK_JOB_ID="$MEGATRON_CONVERSION_JOB_ID"
        echo "Conversion job submitted: $MEGATRON_CONVERSION_JOB_ID"
        echo "Evaluation dependency: afterok:$MEGATRON_CONVERSION_JOB_ID (explicit NeMo Run dependency)"
    fi
    MODEL_OVERRIDE="$export_dir"
}

# ---------------------------------------------------------------------------
# Config patching (single unified path)
# ---------------------------------------------------------------------------

# Build a patched temp YAML for a benchmark.
# Applies (in order, when set):
#   - MODEL_OVERRIDE                     -> top-level model:
#   - _EXPNAME_SUFFIX                    -> appended to expname
#   - decoding-param overrides           -> for each config, patches whichever
#                                            knob is present:
#                                              * server_args CLI flags
#                                                (incremental: --temperature X
#                                                 --top_p Y --repetition_penalty Z,
#                                                 --force_turn_taking)
#                                              * inference_overrides Hydra string
#                                                (offline: ++inference.temperature=X
#                                                 ++inference.top_p=Y
#                                                 ++inference.repetition_penalty=Z)
#                                              * top-level YAML keys
#                                                (conv_behav: temperature, top_p,
#                                                 repetition_penalty, force_turn_taking)
#   - OUTPUT_DIR_OVERRIDE                -> rewrites top-level output_dir (without commit suffix)
# Caller is responsible for deleting the returned temp file.
make_patched_config() {
    local name="$1"
    local base_config="$2"
    local tmp output_dir_no_commit=""
    tmp=$(mktemp /tmp/benchmark_config_XXXXXX.yaml)

    if [[ -n "$OUTPUT_DIR_OVERRIDE" ]]; then
        output_dir_no_commit="${OUTPUT_DIR_OVERRIDE}/${name}"
    fi

    PATCH_SRC="$base_config" \
    PATCH_DST="$tmp" \
    PATCH_BENCHMARK="$name" \
    PATCH_MODEL="${MODEL_OVERRIDE:-}" \
    PATCH_EXPNAME_SUFFIX="${_EXPNAME_SUFFIX:-}" \
    PATCH_HAS_DECODING_OVERRIDES="$HAS_DECODING_OVERRIDES" \
    PATCH_FORCE_TT="${CUSTOM_FORCE_TURN_TAKING:-}" \
    PATCH_USE_RNNT_TT="${CUSTOM_USE_RNNT_TT:-}" \
    PATCH_TOP_P="${CUSTOM_TOP_P:-}" \
    PATCH_REP_PENALTY="${CUSTOM_REPETITION_PENALTY:-}" \
    PATCH_TEMP="${CUSTOM_TEMPERATURE:-}" \
    PATCH_OUTPUT_DIR="$output_dir_no_commit" \
    "$PYTHON" - <<'PYEOF'
import os, re, yaml

src           = os.environ['PATCH_SRC']
dst           = os.environ['PATCH_DST']
benchmark     = os.environ['PATCH_BENCHMARK']
model         = os.environ.get('PATCH_MODEL', '')
expname_suffix = os.environ.get('PATCH_EXPNAME_SUFFIX', '')
has_overrides = os.environ.get('PATCH_HAS_DECODING_OVERRIDES', 'false') == 'true'
force_tt      = os.environ.get('PATCH_FORCE_TT', '')
use_rnnt_tt   = os.environ.get('PATCH_USE_RNNT_TT', '')
top_p         = os.environ.get('PATCH_TOP_P', '')
rep_pen       = os.environ.get('PATCH_REP_PENALTY', '')
temp          = os.environ.get('PATCH_TEMP', '')
output_dir    = os.environ.get('PATCH_OUTPUT_DIR', '')

with open(src) as f:
    cfg = yaml.safe_load(f)

# Model override
if model:
    cfg['model'] = model

if has_overrides:
    # Strategy 1: server_args carries direct CLI flags (incremental + offline).
    # --top_p / --repetition_penalty / --temperature: always append; argparse takes the
    # last value for repeated flags, so the override wins regardless of whether the
    # original YAML had the flag.
    # --force_turn_taking is presence-only (argparse store_true). Want_ftt=true: ensure
    # the flag is present (append if missing). Want_ftt=false: ensure the flag is absent
    # (strip if present) so the override actually turns it off.
    if 'server_args' in cfg and cfg.get('server_args'):
        sa = cfg['server_args']
        want_ftt = (force_tt == 'true')
        if want_ftt:
            if '--force_turn_taking' not in sa:
                sa = sa.rstrip() + ' --force_turn_taking'
        else:
            # Remove all occurrences of --force_turn_taking, including any leading whitespace.
            sa = re.sub(r'\s*--force_turn_taking\b', '', sa)
        sa = sa.rstrip() + (
            f' --top_p {top_p}'
            f' --repetition_penalty {rep_pen}'
            f' --temperature {temp}'
        )
        cfg['server_args'] = sa

    # Strategy 2: inference_overrides carries Hydra ++inference.<key>=<val>
    # (offline configs). Substitute the values in place; do not inject new
    # keys, so a config that intentionally omits a knob stays omitted.
    if 'inference_overrides' in cfg and cfg.get('inference_overrides'):
        io = cfg['inference_overrides']
        io = re.sub(r'\+\+inference\.temperature=\S+', f'++inference.temperature={temp}', io)
        io = re.sub(r'\+\+inference\.top_p=\S+', f'++inference.top_p={top_p}', io)
        io = re.sub(r'\+\+inference\.repetition_penalty=\S+', f'++inference.repetition_penalty={rep_pen}', io)
        cfg['inference_overrides'] = io

    # Strategy 3: top-level YAML keys (conv_behav and the offline configs that
    # also expose them for the runner). Only update keys that already exist.
    if 'temperature' in cfg:
        cfg['temperature'] = float(temp)
    if 'top_p' in cfg:
        cfg['top_p'] = float(top_p)
    if 'repetition_penalty' in cfg:
        cfg['repetition_penalty'] = float(rep_pen)
    if 'force_turn_taking' in cfg:
        cfg['force_turn_taking'] = (force_tt == 'true')

# RNNT turn-taking override. Deliberately outside the `has_overrides` block:
# it is independently settable and does not require the decoding-param group.
#
# serve_unified exposes this as argparse.BooleanOptionalAction with default
# True, so the pair --use_rnnt_turn_taking / --no_use_rnnt_turn_taking are both
# valid and the LAST one wins. Strip both before appending so repeated patching
# of an already-patched config stays idempotent.
if use_rnnt_tt in ('true', 'false'):
    if 'server_args' in cfg and cfg.get('server_args'):
        sa = cfg['server_args']
        sa = re.sub(r'\s*--(?:no_)?use_rnnt_turn_taking\b', '', sa)
        flag = '--use_rnnt_turn_taking' if use_rnnt_tt == 'true' else '--no_use_rnnt_turn_taking'
        cfg['server_args'] = sa.rstrip() + ' ' + flag

    # Top-level key form, for configs that surface it to the runner directly.
    # Only rewrite when already present, matching Strategy 3's convention of
    # never injecting a knob a config intentionally omits.
    if 'turn_taking_source' in cfg:
        cfg['turn_taking_source'] = 'rnnt' if use_rnnt_tt == 'true' else 'asr_head'

# Expname suffix
if expname_suffix and 'expname' in cfg:
    cfg['expname'] = f"{cfg['expname']}{expname_suffix}"

# Output dir override (without commit suffix; Python scripts append the commit)
if output_dir:
    cfg['output_dir'] = output_dir

with open(dst, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True,
              sort_keys=False, width=2147483647)
PYEOF
    echo "$tmp"
}

# Dump the resolved (patched) YAML config + run-level metadata as JSON into
# the benchmark's output folder.  Called once per benchmark before submission.
dump_config_json() {
    local config_yaml="$1"
    local benchmark_name="$2"
    local resolved_outdir="$3"

    [[ "$DRY_RUN" == "true" ]] && return 0

    mkdir -p "$resolved_outdir"
    local json_path="${resolved_outdir}/config.json"
    PATCH_OVERRIDES_JSON="$(printf '%s' "$_OVERRIDES_JSON")" \
    "$PYTHON" - "$config_yaml" "$json_path" "$benchmark_name" "$COMMIT" <<'PYEOF'
import json, os, sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
overrides = json.loads(os.environ.get('PATCH_OVERRIDES_JSON') or '{}')
record = {
    'benchmark': sys.argv[3],
    'commit': sys.argv[4],
    'cli_overrides': overrides,
    'resolved_config': cfg,
}
with open(sys.argv[2], 'w') as f:
    json.dump(record, f, indent=2, default=str)
print(f"  Config JSON: {sys.argv[2]}")
PYEOF
}

# Build a JSON blob of the CLI-level overrides for inclusion in config.json.
# Stored once per script run, reused across all benchmarks.
_build_overrides_json() {
    "$PYTHON" - <<PYEOF
import json
print(json.dumps({
    'model': "${MODEL_OVERRIDE}" or None,
    'output_dir': "${OUTPUT_DIR_OVERRIDE}" or None,
    'decoding_mode': "${DECODING_MODE}",
    'force_turn_taking': "${CUSTOM_FORCE_TURN_TAKING}" or None,
    'use_rnnt_turn_taking': "${CUSTOM_USE_RNNT_TT}" or None,
    'top_p': "${CUSTOM_TOP_P}" or None,
    'repetition_penalty': "${CUSTOM_REPETITION_PENALTY}" or None,
    'temperature': "${CUSTOM_TEMPERATURE}" or None,
}, indent=None))
PYEOF
}

# Deep-diff the to-be-written config (current patched YAML + current CLI
# overrides) against an existing config.json. Used in --resume / --force_rerun
# modes to detect when a partial output dir was produced under different
# settings than the current invocation.
#
# Args:
#   $1 new_yaml_path     patched temp YAML produced by make_patched_config
#   $2 existing_json     path to the existing <outdir>/config.json
#   $3 benchmark_name    benchmark label (for diff header)
#
# Stdout: nothing on match; "CONFIG MISMATCH ..." + per-key diff lines on mismatch.
# Exit:   0 on match (or if the JSON can't be parsed — silent), 1 on mismatch.
compare_resolved_config() {
    local new_yaml="$1"
    local existing_json="$2"
    local benchmark="$3"
    PATCH_OVERRIDES_JSON="$_OVERRIDES_JSON" \
    "$PYTHON" - "$new_yaml" "$existing_json" "$benchmark" <<'PYEOF'
import json, os, sys, yaml

new_yaml_path = sys.argv[1]
existing_json_path = sys.argv[2]
benchmark = sys.argv[3]

try:
    with open(existing_json_path) as f:
        existing = json.load(f)
except Exception as e:
    # Unreadable existing config -> can't compare; treat as mismatch.
    print(f"CONFIG MISMATCH ({benchmark}): existing config.json unreadable ({e})", file=sys.stderr)
    sys.exit(1)

existing_overrides = existing.get('cli_overrides') or {}
existing_resolved  = existing.get('resolved_config') or {}

try:
    new_overrides = json.loads(os.environ.get('PATCH_OVERRIDES_JSON') or '{}')
except Exception:
    new_overrides = {}
with open(new_yaml_path) as f:
    new_resolved = yaml.safe_load(f) or {}

def deep_diff(a, b, path=""):
    """List of human-readable lines describing where a differs from b."""
    out = []
    if type(a) != type(b):
        out.append(f"  ~ {path or '<root>'}: {type(a).__name__}({a!r}) -> {type(b).__name__}({b!r})")
        return out
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            sub = f"{path}.{k}" if path else k
            if k not in a:
                out.append(f"  + {sub}: {b[k]!r}")
            elif k not in b:
                out.append(f"  - {sub}: {a[k]!r}")
            else:
                out.extend(deep_diff(a[k], b[k], sub))
    elif isinstance(a, list):
        if a != b:
            out.append(f"  ~ {path}: {a!r} -> {b!r}")
    else:
        if a != b:
            out.append(f"  ~ {path}: {a!r} -> {b!r}")
    return out

diffs  = deep_diff(existing_overrides, new_overrides, "cli_overrides")
diffs += deep_diff(existing_resolved,  new_resolved,  "resolved_config")

if diffs:
    print(f"CONFIG MISMATCH ({benchmark}):", file=sys.stderr)
    for d in diffs[:50]:
        print(d, file=sys.stderr)
    if len(diffs) > 50:
        print(f"  ... ({len(diffs)-50} more differences truncated)", file=sys.stderr)
    sys.exit(1)
sys.exit(0)
PYEOF
}

# ---------------------------------------------------------------------------
# Pre-flight: existing-config conflict detection
# ---------------------------------------------------------------------------

# If a config.json already exists at any benchmark's resolved output dir,
# repeatedly prompt the user for a new --output_dir until either there is no
# conflict or the user aborts (blank input).  Updates OUTPUT_DIR_OVERRIDE,
# _EXPNAME_SUFFIX in place.
check_existing_configs_and_maybe_reprompt() {
    [[ "$FORCE_RERUN" == "true" ]] && return 0
    [[ "$RESUME" == "true" ]] && return 0
    [[ "$DRY_RUN" == "true" ]] && return 0
    while true; do
        local -a conflicts=()
        for b in $BENCHMARKS; do
            local outdir
            outdir=$(resolve_output_dir "$b")
            if [[ -f "${outdir}/config.json" ]]; then
                conflicts+=("${outdir}/config.json")
            fi
        done
        if [[ ${#conflicts[@]} -eq 0 ]]; then
            return 0
        fi
        echo ""
        echo "Existing config.json found in:"
        for c in "${conflicts[@]+"${conflicts[@]}"}"; do
            echo "  $c"
        done
        if [[ ! -t 0 ]]; then
            echo "ERROR: Output dirs are already claimed by previous runs. Pass a new --output_dir." >&2
            exit 1
        fi
        local new_dir
        read -r -p "Provide a new --output_dir (leave blank to abort): " new_dir
        if [[ -z "$new_dir" ]]; then
            echo "Aborted."
            exit 1
        fi
        OUTPUT_DIR_OVERRIDE="$new_dir"
        _EXPNAME_SUFFIX="_$(printf '%s' "${OUTPUT_DIR_OVERRIDE%/}" | md5sum | cut -c1-8)"
    done
}

# ---------------------------------------------------------------------------
# Per-benchmark run logic
# ---------------------------------------------------------------------------

extra_args() {
    if [[ -n "$MODEL_OVERRIDE" ]]; then
        echo "--model"
        echo "$MODEL_OVERRIDE"
    fi
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "--dry_run"
    fi
    if [[ "$FORCE_RERUN" == "true" ]]; then
        echo "--scoring_force"
    fi
}

run_benchmark() {
    local name="$1"
    local -a extra=()
    while IFS= read -r arg; do
        extra+=("$arg")
    done < <(extra_args)

    local base_config
    base_config=$(config_for "$name")

    # Always patch (if any of model/code_path/server_backend/expname_suffix/decoding/output_dir is set,
    # patcher applies them; otherwise the patcher just copies the file through).
    local tmp_config
    tmp_config=$(make_patched_config "$name" "$base_config")
    local config="$tmp_config"

    # --output_dir override: each benchmark gets its own subdirectory so
    # result files from different benchmarks never collide.  Python scripts
    # append the git commit hash, producing e.g. OUTPUT_DIR_OVERRIDE/bba_a1b2c3d.
    if [[ -n "$OUTPUT_DIR_OVERRIDE" ]]; then
        extra+=("--output_dir" "${OUTPUT_DIR_OVERRIDE}/${name}")
    fi

    # Dump config.json into the resolved output dir before submission.
    dump_config_json "$config" "$name" "$(resolve_output_dir "$name")"

    cd "$REPO_ROOT"
    export NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1

    echo ""
    echo "======================================================================"
    printf ' %-30s  %s\n' "Benchmark:" "$name"
    printf ' %-30s  %s\n' "Started:" "$(date '+%Y-%m-%d %H:%M:%S')"
    printf ' %-30s  %s\n' "Config:" "$base_config"
    printf ' %-30s  %s\n' "Patched config:" "$tmp_config"
    echo "======================================================================"

    local rc=0
    case "$name" in
        vb_nonmcq)
            "$PYTHON" "$VB_SCRIPT" --config "$config" "${extra[@]}"
            ;;
        vb_mcq)
            "$PYTHON" "$VB_SCRIPT" --config "$config" "${extra[@]}"
            ;;
        fdb_v1|fdb_v1_5)
            "$PYTHON" "$FDB_SCRIPT" --config "$config" "${extra[@]}"
            ;;
        fdb_v3)
            "$PYTHON" "$FDB_V3_SCRIPT" --config "$config" "${extra[@]}"
            ;;
        fdb_v3_chen_chen)
            "$PYTHON" "$FDB_V3_CHEN_CHEN_SCRIPT" --config "$config" "${extra[@]}"
            ;;
        fdb_v3_official)
            "$PYTHON" "$FDB_V3_OFFICIAL_SCRIPT" --config "$config" "${extra[@]}"
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
    esac || rc=$?

    rm -f "$tmp_config"

    if [[ $rc -ne 0 ]]; then
        # Some Python scripts crash during interpreter shutdown after successfully
        # submitting their SLURM jobs.  If the expected job already appears in
        # squeue, treat it as a success.
        local exp found_job=""
        exp=$(expname_for "$name") || true
        if [[ -n "$exp" ]]; then
            found_job=$(squeue -u "$USER" -h -o "%j" 2>/dev/null \
                | awk -v p="$exp" '($0 == p || index($0, p "_") == 1)' \
                | head -1 || true)
        fi
        if [[ -n "$found_job" ]]; then
            echo "" >&2
            printf 'WARNING: %s exited with code %d but SLURM job "%s" is queued — likely a Python shutdown crash. Continuing.\n' \
                "$name" "$rc" "$found_job" >&2
        else
            printf 'ERROR: %s failed with exit code %d and no matching SLURM job found.\n' \
                "$name" "$rc" >&2
            exit $rc
        fi
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done submitting: $name"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
detect_megatron_model
filter_unsupported_megatron_benchmarks
if [[ -z "$BENCHMARKS" ]]; then
    echo "No compatible benchmarks remain; no conversion or evaluation jobs were submitted."
    exit 0
fi
prepare_megatron_model
check_model
validate_customized_configs

echo ""
echo "======================================================================"
echo " S2S FC Benchmark Runner"
echo "======================================================================"
echo " Benchmarks   : $BENCHMARKS"
[[ -n "$OUTPUT_DIR_OVERRIDE" ]] && echo " Output dir   : $OUTPUT_DIR_OVERRIDE/{name}_{commit}"
[[ -n "$PROCESSED_CKPT_DIR" ]] && echo " Processed ckpt: $PROCESSED_CKPT_DIR"
[[ -n "$MEGATRON_CONVERSION_JOB_ID" ]] && echo " Conversion job: $MEGATRON_CONVERSION_JOB_ID (afterok)"
if [[ "$HAS_DECODING_OVERRIDES" == "true" ]]; then
    echo " Decoding overrides:"
    echo "   force_turn_taking  : $CUSTOM_FORCE_TURN_TAKING"
    echo "   top_p              : $CUSTOM_TOP_P"
    echo "   repetition_penalty : $CUSTOM_REPETITION_PENALTY"
    echo "   temperature        : $CUSTOM_TEMPERATURE"
fi
# Separate from the decoding-overrides block: settable on its own.
[[ -n "$CUSTOM_USE_RNNT_TT" ]] && echo " RNNT turn-taking: $CUSTOM_USE_RNNT_TT"
echo " HTML name    : ${HTML_NAME}.html"
echo " Poll interval: ${POLL_INTERVAL}s"
echo " Dry run      : $DRY_RUN"
echo " Force rerun  : $FORCE_RERUN"
echo " Resume       : $RESUME"
echo " Commit       : $COMMIT"

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

# An existing scorecard is informational only.  It may be stale or generated
# from a different subset, so per-benchmark metrics remain the source of truth.
if [[ "$FORCE_RERUN" != "true" ]]; then
    SCORECARD=$(find_scorecard)
    if [[ -n "$SCORECARD" ]]; then
        echo ""
        echo "Scorecard found: $SCORECARD"
        echo "Continuing with per-benchmark result checks."
    fi
fi

# Conflict check: if any selected benchmark already has a config.json under the
# resolved output dir, prompt for a new --output_dir or abort.
check_existing_configs_and_maybe_reprompt

# Build CLI-level overrides JSON once for re-use in dump_config_json.
_OVERRIDES_JSON=$(_build_overrides_json)

for benchmark in $BENCHMARKS; do
    # Config-mismatch handling (--resume or --force_rerun only). Runs BEFORE
    # find_benchmark_result so that stale-config results don't get used
    # silently in --resume mode.
    #   --resume + mismatch      -> skip this benchmark with diff printed;
    #                               require --force_rerun to override
    #   --force_rerun + mismatch -> wipe the output dir so the run starts truly
    #                               clean (otherwise the benchmark python
    #                               script would skip generation when
    #                               output.jsonl + .done markers exist)
    #   match (either mode) or no existing config -> proceed normally
    if [[ "$RESUME" == "true" || "$FORCE_RERUN" == "true" ]]; then
        _outdir=$(resolve_output_dir "$benchmark")
        if [ -f "$_outdir/config.json" ]; then
            _new_patched=$(make_patched_config "$benchmark" "$(config_for "$benchmark")")
            if ! compare_resolved_config "$_new_patched" "$_outdir/config.json" "$benchmark"; then
                rm -f "$_new_patched"
                if [[ "$RESUME" == "true" ]]; then
                    echo "  Skipping $benchmark: --resume + config mismatch (see diff above). Use --force_rerun to override and rerun from scratch."
                    continue
                else
                    echo "  $benchmark: config mismatch under --force_rerun; clearing $_outdir for full rerun."
                    if [[ "$DRY_RUN" == "true" ]]; then
                        echo "    (dry_run — would rm -rf $_outdir)"
                    else
                        rm -rf "$_outdir"
                    fi
                fi
            else
                rm -f "$_new_patched"
            fi
        fi
    fi

    if [[ "$FORCE_RERUN" != "true" ]]; then
        result=$(find_benchmark_result "$benchmark")
        if [[ -n "$result" ]]; then
            echo "  Skipping $benchmark: results found at $result"
            continue
        fi
    fi

    cancel_stale_jobs "$benchmark"
    if [[ -n "$MAX_JOBS" && "$DRY_RUN" != "true" ]]; then
        wait_for_slot "$MAX_JOBS" "$benchmark"
    fi
    run_benchmark "$benchmark"
done

echo ""
echo "======================================================================"
echo " All benchmarks submitted successfully."
echo "======================================================================"
