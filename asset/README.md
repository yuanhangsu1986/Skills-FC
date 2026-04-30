# S2S FC Benchmark Runner

`asset/run_all_benchmarks.sh` submits all S2S function-calling eval benchmarks sequentially,
throttling submission so the SLURM queue never exceeds the cluster's max-jobs limit.

## Prerequisites

- Run from the **repo root** (one level above `asset/`).
- The default per-benchmark config YAMLs must exist (or supply your own via `--config_*` flags).
- SLURM tools (`sacctmgr`, `scontrol`, `squeue`) must be on `$PATH`.

## Quick start

```bash
# Run all six benchmarks, both greedy and sampling (default)
bash asset/run_all_benchmarks.sh

# Greedy only
bash asset/run_all_benchmarks.sh --eval_mode greedy

# Sampling only
bash asset/run_all_benchmarks.sh --eval_mode sampling

# Run a subset in sampling mode
bash asset/run_all_benchmarks.sh --eval_mode sampling --benchmarks bba,bfcl

# Override model + code for a greedy-only run (must always be paired)
bash asset/run_all_benchmarks.sh \
  --eval_mode greedy \
  --model /lustre/path/to/checkpoint \
  --code_path /lustre/path/to/NeMo_code

# Preview what would be submitted without actually submitting
bash asset/run_all_benchmarks.sh --dry_run
```

## Benchmark names

| Name | Dataset |
|------|---------|
| `vb_nonmcq` | VoiceBench non-MCQ subtests (sd_qa, alpacaeval, ifeval, …) |
| `vb_mcq` | VoiceBench MCQ subtests (bbh, openbookqa, mmsu) |
| `fdb` | Full-Duplex Bench (pause, backchannel, turn-taking, interruption) |
| `bba` | BigBench Audio (formal_fallacies, navigate, object_counting, web_of_lies) |
| `bfcl` | BFCL single-turn function-channel (simple, parallel, multiple, …) |
| `conv_behav` | Conversational Behavior (turn-taking, barge-in, back-channeling) |

Default order when `--benchmarks` is omitted: `vb_nonmcq vb_mcq fdb bba bfcl conv_behav`.

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `--eval_mode MODE` | `greedy`, `sampling`, or `greedy+sampling`. Selects the matching `*_greedy.yaml` / `*_sampling.yaml` configs. `greedy+sampling` runs all benchmarks greedy-first, then repeats for sampling. | `greedy+sampling` |
| `--benchmarks LIST` | Comma-separated subset of benchmark names | all six, in listed order |
| `--config_vb_nonmcq PATH` | Config YAML for VoiceBench non-MCQ | `nemo_skills/dataset/voicebench/scripts/vb_matched_demo_v2_02mar_config_fc_greedy.yaml` |
| `--config_vb_mcq PATH` | Config YAML for VoiceBench MCQ | `nemo_skills/dataset/voicebench/scripts/vb_matched_demo_v2_02mar_mcq_config_fc_greedy.yaml` |
| `--config_fdb PATH` | Config YAML for FDB | `nemo_skills/dataset/fdb/scripts/fdb_s2s_incremental_v2_02mar_config_fc_greedy.yaml` |
| `--config_bba PATH` | Config YAML for BBA | `nemo_skills/dataset/bba/scripts/bba_config_fc_greedy.yaml` |
| `--config_bfcl PATH` | Config YAML for BFCL | `nemo_skills/dataset/bfcl_single_turn_function_channel/scripts/bfcl_fc_config_greedy.yaml` |
| `--config_conv_behav PATH` | Config YAML for conv_behav | `nemo_skills/dataset/conv_behav/scripts/conv_behav_config_greedy.yaml` |
| `--model PATH` | Override the model checkpoint for every benchmark | (from each config YAML) |
| `--code_path PATH` | Override the NeMo source code directory for every benchmark | (from each config YAML) |
| `--max_jobs N` | Hard-code the SLURM job limit instead of auto-detecting | auto-detected |
| `--poll_interval N` | Seconds between SLURM queue checks while waiting | `60` |
| `--dry_run` | Pass `--dry_run` to every benchmark script; no jobs submitted | `false` |
| `--help` | Print usage and exit | |

## How job throttling works

Before each benchmark is submitted the script queries the SLURM job limit from three sources
and takes the minimum:

1. Per-user association `MaxSubmitJobs` (via `sacctmgr show association`)
2. `MaxSubmitJobsPerUser` for every QOS assigned to the user (via `sacctmgr show qos`)
3. `MaxSubmitJobsPerUser` on partitions `batch_block1`, `batch_block3`, `batch_block4`, `cpu`
   (via `scontrol show partition`)

It then polls `squeue -u $USER` every `--poll_interval` seconds and waits until the current
job count drops below the limit before submitting the next benchmark.  If no limit is detected
the benchmarks are submitted without throttling.

Pass `--max_jobs N` to skip auto-detection and use a fixed limit.

## Model and code_path

Each model checkpoint ships with a matched NeMo source directory (`--code_path`).
The script enforces this pairing:

- If `--model` is set, `--code_path` **must** also be set (and vice versa).
- If neither is set, the script reads `model:` and the code_path from each selected
  config YAML and verifies they all agree before submitting anything.  If they differ
  it prints a table of the mismatches and exits without submitting.

The code_path override patches two YAML conventions transparently:
- `--code_path <path>` embedded inside `server_args` (all benchmarks except conv_behav)
- `nemo_code_path: <path>` top-level field (conv_behav)

## Examples

```bash
# Run only BBA and BFCL with a custom checkpoint + matching NeMo code
bash asset/run_all_benchmarks.sh \
  --benchmarks bba,bfcl \
  --model /lustre/path/to/my_checkpoint \
  --code_path /lustre/path/to/NeMo_code \
  --poll_interval 30

# Run everything but use a custom FDB config
bash asset/run_all_benchmarks.sh \
  --config_fdb nemo_skills/dataset/fdb/scripts/my_fdb_config.yaml

# Hard-code the job limit (useful if sacctmgr is unavailable)
bash asset/run_all_benchmarks.sh --max_jobs 50

# Dry run — inspect commands without submitting
bash asset/run_all_benchmarks.sh --dry_run
```

## Analyzing results

After benchmarks complete, use the Claude Code skills in this directory to generate HTML reports:

```
/analyze-bba       <OUTPUT_DIR>
/analyze-bfcl      <OUTPUT_DIR>
/analyze-conv-behav <OUTPUT_DIR>
```

To produce a single aggregate scorecard across all benchmarks:

```
/make-scorecard vb_nonmcq=<PATH> vb_mcq=<PATH> fdb=<PATH> bba=<PATH> bfcl=<PATH> conv_behav=<PATH>
```

Any subset of benchmark paths may be supplied; missing benchmarks are shown as "Not run".
The scorecard is written to **`asset/scorecard.html`** in the repo root by default.
Override with `output=<absolute_path>`.
