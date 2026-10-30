import pytest

from ehds_hdab import audit, config, spe
from ehds_hdab.pipeline import reset_query_history


@pytest.fixture(autouse=True)
def isolate_runtime_artifacts(monkeypatch, tmp_path):
    reset_query_history()
    spe_root = tmp_path / "spe_workspaces"
    audit_log = tmp_path / "audit_log.jsonl"
    monkeypatch.setattr(config, "SPE_ROOT", spe_root)
    monkeypatch.setattr(config, "AUDIT_LOG", audit_log)
    monkeypatch.setattr(spe, "SPE_ROOT", spe_root)
    monkeypatch.setattr(audit, "AUDIT_LOG", audit_log)
