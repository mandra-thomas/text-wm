"""Most tests call one run in isolation, so the dependency gate is switched on only where tested."""
import pytest


@pytest.fixture(autouse=True)
def _runs_may_be_called_alone(monkeypatch):
    monkeypatch.setenv("WF_MOCK_CONTINUE", "1")
