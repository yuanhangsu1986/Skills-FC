# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob as _glob
import logging
import os
import subprocess
import tarfile as _tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import nemo_run as run
from nemo_run.core.packaging.base import Packager as _NemoPackager

from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


@dataclass
class RepoMetadata:
    """Metadata for a repo that is used in the experiment."""

    name: str
    path: Path

    def __post_init__(self):
        if isinstance(self.path, str):
            self.path = Path(self.path)

        if not self.path.exists():
            raise ValueError(f"Repository path `{self.path}` does not exist.")


# Registry of external repos that should be packaged with the code in the experiment
EXTERNAL_REPOS = {
    "nemo_skills": RepoMetadata(
        name="nemo_skills", path=Path(__file__).absolute().parents[2]
    ),  # path to nemo_skills repo
}


def register_external_repo(metadata: RepoMetadata):
    """Register an external repo to be packaged with the code in the experiment.

    Args:
        metadata (RepoMetadata): Metadata for the external repo.
    """
    if metadata.name in EXTERNAL_REPOS:
        raise ValueError(f"External repo {metadata.name} is already registered.")

    EXTERNAL_REPOS[metadata.name] = metadata


def get_registered_external_repo(name: str) -> Optional[RepoMetadata]:
    """Get the path to the registered external repo.

    Args:
        name (str): Name of the external repo.

    Returns:
        A path to the external repo if it is registered, otherwise None.
    """
    if name not in EXTERNAL_REPOS:
        return None

    return EXTERNAL_REPOS[name]


def get_git_repo_path(path: str | Path = None):
    """Check if the path is a git repo.

    Args:
        path: Path to the directory to check. If None, will check the current directory.

    Returns:
        Path to the repo if it is a git repo, otherwise None.
    """
    original_path = os.getcwd()
    try:
        if path:
            os.chdir(path)

        repo_path = (
            subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True,
                check=True,
            )
            .stdout.decode()
            .strip()
        )
        return Path(repo_path)

    except subprocess.CalledProcessError:
        return None

    finally:
        os.chdir(original_path)


@dataclass(kw_only=True)
class GitWorkingTreePackager(_NemoPackager):
    """Packages the git working tree instead of committed HEAD.

    Uses `git ls-files` (tracked files, including locally modified ones) plus
    `git ls-files --others --exclude-standard` (new untracked non-ignored files).
    This respects .gitignore so __pycache__, large binaries, etc. are excluded,
    but local uncommitted edits ARE included — unlike GitArchivePackager.
    """

    basepath: str = ""
    include_pattern: str | list[str] = field(default_factory=list)
    include_pattern_relative_path: str | list[str] = field(default_factory=list)

    def package(self, path: Path, job_dir: str, name: str) -> str:
        output_file = os.path.join(job_dir, f"{name}.tar.gz")
        if os.path.exists(output_file):
            return output_file

        base = Path(self.basepath) if self.basepath else path
        git_base = Path(
            subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=str(base), check=True, capture_output=True,
            ).stdout.decode().strip()
        )

        tracked = subprocess.run(
            ["git", "ls-files"], cwd=str(git_base), check=True, capture_output=True,
        ).stdout.decode().splitlines()

        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=str(git_base), check=True, capture_output=True,
        ).stdout.decode().splitlines()

        patterns = [self.include_pattern] if isinstance(self.include_pattern, str) else list(self.include_pattern)
        rel_bases = (
            [self.include_pattern_relative_path]
            if isinstance(self.include_pattern_relative_path, str)
            else list(self.include_pattern_relative_path)
        )

        added: set[str] = set()
        with _tarfile.open(output_file, "w:gz") as tar:
            for rel in sorted(set(f.strip() for f in tracked + untracked if f.strip())):
                full = git_base / rel
                if full.exists():
                    tar.add(str(full), arcname=rel)
                    added.add(rel)

            for pattern, rel_base in zip(patterns, rel_bases):
                if not pattern:
                    continue
                for match in _glob.glob(pattern, recursive=True):
                    full = Path(match)
                    if not full.is_file():
                        continue
                    try:
                        arcname = str(full.relative_to(rel_base)) if rel_base else full.name
                    except ValueError:
                        arcname = full.name
                    if arcname not in added:
                        tar.add(str(full), arcname=arcname)
                        added.add(arcname)

        return output_file


def _check_and_warn_uncommitted(repo_root: Path) -> None:
    """Warn (and abort by default) when the working tree has uncommitted changes."""
    modified = subprocess.run(
        ["git", "diff", "--name-only", "HEAD"],
        cwd=str(repo_root), capture_output=True, text=True,
    ).stdout.strip()

    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=str(repo_root), capture_output=True, text=True,
    ).stdout.strip()

    if not modified and not untracked:
        return

    lines = ["Uncommitted local changes detected — these WILL be packaged into the job:"]
    if modified:
        lines.append("  Modified files:")
        for f in modified.splitlines():
            lines.append(f"    M {f}")
    if untracked:
        lines.append("  New untracked files:")
        for f in untracked.splitlines():
            lines.append(f"    ? {f}")
    lines.append("")
    lines.append("Set NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1 to suppress this check and proceed.")

    msg = "\n".join(lines)
    allow = os.getenv("NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK", "0").strip().lower()
    if allow in ("1", "true", "yes", "on"):
        LOG.warning(msg)
    else:
        raise RuntimeError(msg + "\n\nAborted.")


def get_packager(extra_package_dirs: tuple[str] | None = None):
    """Will check if we are running from a git repo and use git packager or default packager otherwise."""
    nemo_skills_dir = get_registered_external_repo("nemo_skills").path

    if extra_package_dirs:
        include_patterns = [str(Path(d) / "*") for d in extra_package_dirs]
        include_pattern_relative_paths = [str(Path(d).parent) for d in extra_package_dirs]
    else:
        include_patterns = []
        include_pattern_relative_paths = []

    # are we in a git repo? If yes, we are uploading the current code
    repo_path = get_git_repo_path(path=None)  # check if we are in a git repo in pwd

    if repo_path:
        _check_and_warn_uncommitted(repo_path)

        # Do we have nemo_skills package in this repo? If no, we need to pick it up from installed location
        if not (Path(repo_path) / "nemo_skills").is_dir():
            LOG.info(
                "Not running from Nemo-Skills repo, trying to upload installed package. "
                "Make sure there are no extra files in %s",
                str(nemo_skills_dir / "*"),
            )
            include_patterns.append(str(nemo_skills_dir / "*"))
        else:
            # picking up local dataset files if we are in the right repo
            include_patterns.append(str(nemo_skills_dir / "dataset/**/*.jsonl"))
            subfolder_datasets = ["ruler", "bfcl_v3"]  # TODO: read this from init.py in a dataset folder
            # special logic for any dataset that creates subfolders
            for subfolder_dataset in subfolder_datasets:
                include_pattern_relative_paths.append(str(nemo_skills_dir.parent))
                include_patterns.append(str(nemo_skills_dir / f"dataset/{subfolder_dataset}/*"))
        include_pattern_relative_paths.append(str(nemo_skills_dir.parent))

        root_package = GitWorkingTreePackager(
            include_pattern=include_patterns,
            include_pattern_relative_path=include_pattern_relative_paths,
        )
    else:
        LOG.info(
            "Not running from a git repo, trying to upload installed package. Make sure there are no extra files in %s",
            str(nemo_skills_dir / "*"),
        )
        include_patterns.append(str(nemo_skills_dir / "*"))
        include_pattern_relative_paths.append(str(nemo_skills_dir.parent))

        root_package = run.PatternPackager(
            include_pattern=include_patterns,
            relative_path=include_pattern_relative_paths,
        )

    extra_repos = {}
    if len(EXTERNAL_REPOS) > 1:
        # Insert root package as the first package
        extra_repos["nemo_run"] = root_package

        for repo_name, repo_meta in EXTERNAL_REPOS.items():
            if repo_name == "nemo_skills":
                continue

            repo_path = repo_meta.path
            if get_git_repo_path(repo_path):
                extra_repos[repo_name] = GitWorkingTreePackager(basepath=str(repo_path))
            else:
                # Extra repos is not a git repo, so we need to package all files in the directory
                repo_include_pattern = [str(Path(repo_path) / "*")]
                repo_include_pattern_relative_path = [str(Path(repo_path).parent)]
                extra_repos[repo_name] = run.PatternPackager(
                    include_pattern=repo_include_pattern,
                    relative_path=repo_include_pattern_relative_path,
                )

        # Return hybrid packager
        return run.HybridPackager(sub_packagers=extra_repos, extract_at_root=True)

    return root_package
