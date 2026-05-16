# silero-vad setup for FDB v1.5 scoring

FDB v1.5 scoring runs `Full-Duplex-Bench/evaluation/get_timing.py`, which calls
`torch.hub.load('snakers4/silero-vad', ...)`. That call downloads the repo from
GitHub on every fresh container/node. On clusters where compute nodes have no
outbound internet (or GitHub is flaky), the download fails with
`zipfile.BadZipFile: File is not a zip file` and the timing metrics
(`stop_latency`, `response_latency`) come out empty.

`run_fdb_scoring.py` accepts `--silero_vad_dir <path>` and, when set, pre-populates
`TORCH_HOME` so `torch.hub.load` finds the cached repo and skips the GitHub
download entirely. This page is the one-time setup for that path.

## Where it lives

All v1.5 yaml configs under this directory already point at:

```
silero_vad_dir: /lustre/fsw/portfolios/llmservice/users/yuanhangs/silero-vad
```

If you want a different location, override `silero_vad_dir` in the yaml and pass
the same path to the setup script below.

## One-time install

From a node with outbound internet (typically the login/head node):

```bash
bash nemo_skills/dataset/fdb/scripts/setup_silero_vad.sh
```

That:
1. Tries `git clone --depth 1` first.
2. Falls back to downloading the tagged master zip with retries.
3. Verifies the resulting tree (`hubconf.py`, `src/silero_vad/utils_vad.py`,
   `src/silero_vad/data/`).
4. Atomically moves the verified tree into place.

The script is idempotent — a re-run on an already-good install exits 0 without
re-downloading. Pass `--force` to redownload.

### Custom path

```bash
bash nemo_skills/dataset/fdb/scripts/setup_silero_vad.sh /path/you/own/silero-vad
# or:
SILERO_VAD_DIR=/path/you/own/silero-vad \
  bash nemo_skills/dataset/fdb/scripts/setup_silero_vad.sh
```

Don't forget to update `silero_vad_dir:` in the yaml configs to match.

## Manual install (if both git and curl fail)

On a machine with good connectivity:

```bash
git clone https://github.com/snakers4/silero-vad.git
```

Then rsync the resulting `silero-vad/` directory to the target path on lustre.
The minimum required tree is:

```
silero-vad/
├── hubconf.py
└── src/silero_vad/
    ├── __init__.py
    ├── utils_vad.py
    └── data/
        ├── silero_vad.jit
        ├── silero_vad.onnx
        └── ... (other model files)
```

## What the runtime does with it

In `run_fdb_scoring.py`, before invoking `get_timing.py`:

1. A fresh temp dir is created and `TORCH_HOME` is set to it.
2. `<temp>/hub/snakers4_silero-vad_master` is symlinked to `silero_vad_dir`.
3. The subprocess sees `TORCH_HOME` pointing at a cache that already contains
   the expected repo, so `torch.hub.load` returns immediately without hitting
   GitHub.

If `silero_vad_dir` is unset, missing, or lacks `hubconf.py`, the script logs a
warning and falls back to the original `torch.hub.load` behavior (which will
attempt the GitHub download).

## Verifying the install

```bash
ls /lustre/fsw/portfolios/llmservice/users/yuanhangs/silero-vad/hubconf.py \
   /lustre/fsw/portfolios/llmservice/users/yuanhangs/silero-vad/src/silero_vad/utils_vad.py
```

Both paths should exist. If they do, your next FDB v1.5 scoring run will use the
local checkout — look for the absence of the `Downloading: "https://github.com/snakers4/silero-vad/zipball/master"`
line in the score log.
