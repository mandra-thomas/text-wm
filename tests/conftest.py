"""Most tests call one run in isolation, so the dependency gate is switched on only where tested."""
import pytest


@pytest.fixture(autouse=True)
def _runs_may_be_called_alone(monkeypatch):
    monkeypatch.setenv("WF_MOCK_CONTINUE", "1")


# TextCraft's recipe database (wf/textcraft/data/recipes_frozen.json) is Mojang game data and is not committed.
# In a clone without it, a TextCraft test that errors is skipped with the rebuild instruction instead of failing.
# With the file present nothing is skipped, and tests outside tests/test_textcraft_*.py are never affected.
import subprocess
from pathlib import Path

_RECIPES = Path(__file__).resolve().parents[1] / "wf" / "textcraft" / "data" / "recipes_frozen.json"


def _skip_if_recipes_missing(item, exc):
    if (not _RECIPES.exists() and Path(str(item.fspath)).name.startswith("test_textcraft_")
            and isinstance(exc, (FileNotFoundError, subprocess.CalledProcessError))):
        pytest.skip("TextCraft recipes not present: rebuild with `python -m wf.textcraft.freeze --fetch`")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item):
    try:
        return (yield)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        _skip_if_recipes_missing(item, exc)
        raise


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    try:
        return (yield)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        _skip_if_recipes_missing(item, exc)
        raise
