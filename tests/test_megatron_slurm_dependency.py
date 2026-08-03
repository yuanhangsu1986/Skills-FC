import pytest

from nemo_skills.pipeline.utils.exp import _add_external_slurm_afterok_dependency


def test_external_conversion_job_is_added_as_explicit_slurm_handle(monkeypatch):
    monkeypatch.setenv("NEMO_SKILLS_SLURM_AFTEROK_JOB_ID", "11445689")

    dependencies, enabled = _add_external_slurm_afterok_dependency(["slurm://existing/123/master/0"])

    assert enabled is True
    assert dependencies == [
        "slurm://existing/123/master/0",
        "slurm://nemo-skills-external/11445689/master/0",
    ]


def test_external_conversion_job_must_be_numeric(monkeypatch):
    monkeypatch.setenv("NEMO_SKILLS_SLURM_AFTEROK_JOB_ID", "afterok:123")

    with pytest.raises(ValueError, match="numeric Slurm job ID"):
        _add_external_slurm_afterok_dependency(None)
