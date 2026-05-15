#!/usr/bin/env bash
# Download snakers4/silero-vad into the path referenced by the v1.5 yaml configs
# (`silero_vad_dir`). FDB v1.5 scoring (get_timing.py) uses this checkout to
# pre-populate TORCH_HOME so torch.hub.load skips the GitHub download at scoring
# time — useful when compute nodes have no internet or GitHub is flaky.
#
# Usage:
#   bash nemo_skills/dataset/fdb/scripts/setup_silero_vad.sh [TARGET_DIR]
#
# Defaults to /lustre/fsw/portfolios/llmservice/users/yuanhangs/silero-vad
# (the path baked into the v1.5 yamls). Override by passing a different
# TARGET_DIR or by setting SILERO_VAD_DIR.
#
# Idempotent: if TARGET_DIR already has a valid hubconf.py + src/silero_vad/,
# the script exits 0 without redownloading. Pass --force to redownload.

set -euo pipefail

DEFAULT_TARGET="/lustre/fsw/portfolios/llmservice/users/yuanhangs/silero-vad"
TARGET="${1:-${SILERO_VAD_DIR:-$DEFAULT_TARGET}}"
FORCE=0
if [[ "${1:-}" == "--force" || "${2:-}" == "--force" ]]; then
    FORCE=1
    [[ "${1:-}" == "--force" ]] && TARGET="${SILERO_VAD_DIR:-$DEFAULT_TARGET}"
fi

REPO_URL="https://github.com/snakers4/silero-vad.git"
ZIP_URL="https://codeload.github.com/snakers4/silero-vad/zip/refs/heads/master"

is_valid_checkout() {
    local d="$1"
    [[ -f "$d/hubconf.py" && -d "$d/src/silero_vad" && -f "$d/src/silero_vad/utils_vad.py" ]]
}

echo "Target: $TARGET"

if [[ $FORCE -eq 0 ]] && is_valid_checkout "$TARGET"; then
    echo "OK: existing checkout at $TARGET looks valid (hubconf.py + src/silero_vad/utils_vad.py present)."
    echo "    Pass --force to redownload."
    exit 0
fi

# Make sure parent exists and is writable.
parent="$(dirname "$TARGET")"
if [[ ! -d "$parent" ]]; then
    echo "ERROR: parent dir $parent does not exist." >&2
    exit 1
fi
if [[ ! -w "$parent" ]]; then
    echo "ERROR: parent dir $parent is not writable by $(whoami)." >&2
    exit 1
fi

# Stage into a tmp dir alongside TARGET so we can atomically swap at the end.
stage="$(mktemp -d -p "$parent" .silero_vad_setup.XXXXXX)"
trap 'rm -rf "$stage"' EXIT

try_git_clone() {
    echo "Attempting git clone (full repo)..."
    if timeout 600 git clone --depth 1 "$REPO_URL" "$stage/silero-vad"; then
        if is_valid_checkout "$stage/silero-vad"; then
            return 0
        fi
        echo "git clone finished but checkout looks incomplete." >&2
    fi
    rm -rf "$stage/silero-vad"
    return 1
}

try_zip_download() {
    # Retry the zip download a few times — GitHub sometimes hangs up partway.
    local zip="$stage/silero-vad-master.zip"
    local attempt
    for attempt in 1 2 3 4 5; do
        echo "Attempting zip download (attempt $attempt)..."
        if timeout 600 curl -sSfL --connect-timeout 30 -o "$zip" "$ZIP_URL"; then
            # Verify EOCD signature (PK\x05\x06) in last 22 bytes.
            if tail -c 22 "$zip" | grep -q $'PK\x05\x06'; then
                echo "Zip download OK. Extracting..."
                ( cd "$stage" && unzip -q "$zip" )
                rm -f "$zip"
                # GitHub names the extracted dir `silero-vad-master/`.
                if [[ -d "$stage/silero-vad-master" ]]; then
                    mv "$stage/silero-vad-master" "$stage/silero-vad"
                fi
                if is_valid_checkout "$stage/silero-vad"; then
                    return 0
                fi
                echo "Extracted tree is incomplete; retrying..." >&2
                rm -rf "$stage/silero-vad"
            else
                echo "Downloaded zip is truncated (no EOCD); retrying..." >&2
                rm -f "$zip"
            fi
        else
            echo "curl failed (exit $?); retrying..." >&2
        fi
        sleep 3
    done
    return 1
}

if ! try_git_clone && ! try_zip_download; then
    echo "" >&2
    echo "ERROR: failed to download silero-vad after all attempts." >&2
    echo "Possible causes:" >&2
    echo "  - This node has no outbound internet (try the login/head node)." >&2
    echo "  - GitHub / codeload.github.com is rate-limiting or slow." >&2
    echo "" >&2
    echo "Workaround: clone manually from a machine with good connectivity and" >&2
    echo "rsync the result to $TARGET. The checkout must contain:" >&2
    echo "  hubconf.py  src/silero_vad/__init__.py  src/silero_vad/utils_vad.py" >&2
    echo "  src/silero_vad/data/  (the .jit/.onnx model files)" >&2
    exit 1
fi

# Atomic-ish swap. If TARGET exists, replace it.
if [[ -e "$TARGET" ]]; then
    backup="$TARGET.bak.$$"
    echo "Existing $TARGET found; moving aside to $backup"
    mv "$TARGET" "$backup"
fi
mv "$stage/silero-vad" "$TARGET"

if is_valid_checkout "$TARGET"; then
    echo ""
    echo "Done. silero-vad installed at:"
    echo "  $TARGET"
    echo ""
    echo "Verify the v1.5 yaml configs have:"
    echo "  silero_vad_dir: $TARGET"
    exit 0
fi

echo "ERROR: post-install validation failed at $TARGET" >&2
exit 1
