import subprocess
from pathlib import Path


def test_explicit_chenchen_is_skipped_for_megatron_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoints"
    iteration = checkpoint / "iter_0000001"
    iteration.mkdir(parents=True)
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("1\n")
    (iteration / ".metadata").write_bytes(
        b"model.audio_encoder.weight\nmodel.backbone.mamba_model.mamba_model.weight\n"
    )

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts/run_all_benchmarks.sh"),
            "--benchmarks",
            "fdb_v3_chen_chen",
            "--model",
            str(checkpoint),
            "--output_dir",
            str(tmp_path / "output"),
            "--dry_run",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "native inference engine does not support the split Megatron converted checkpoint" in result.stdout
    assert "no conversion or evaluation jobs were submitted" in result.stdout
