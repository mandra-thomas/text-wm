"""The TextCraft and τ-bench mutation runners (wf/textcraft/mutation_check.py, scripts/tau_mutation_check.py).

Both run pytest without bytecode (PYTHONDONTWRITEBYTECODE=1) on a copy made without __pycache__, as
scripts/mutation_check.py does: a .pyc is trusted when its source's mtime, in whole seconds, and its size match, so a
file restored within the second its same-length mutant was compiled would be read back from the mutant's bytecode,
and every later mutant would "die" against code that is still broken (notes/deviations.md, 2026-11-24). Both apply
the strict kill rule of scripts/mutation_check.py (notes/testing.md). The end-to-end tests run each runner on a tiny
repository of its own, so each takes a few seconds."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from wf.textcraft import mutation_check as tc

TAU_RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "tau_mutation_check.py"


@pytest.fixture(scope="module")
def tau():
    spec = importlib.util.spec_from_file_location("_tau_mutation_check_under_test", TAU_RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeRun:
    """Stands in for subprocess.run: records each call and returns a pytest run that failed one named test."""

    def __init__(self):
        self.calls = []

    def __call__(self, cmd, **kw):
        self.calls.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 1, "FAILED tests/test_x.py::test_y - assert 0\n", "")


def test_textcraft_runner_runs_pytest_without_bytecode(monkeypatch, tmp_path):
    fake = _FakeRun()
    monkeypatch.setattr(tc.subprocess, "run", fake)
    monkeypatch.setenv("SOME_OTHER_SETTING", "kept")
    monkeypatch.delenv("PYTHONDONTWRITEBYTECODE", raising=False)  # the runner must set it, not inherit it
    res = tc._run_tests(tmp_path, 10)
    cmd, kw = fake.calls[0]
    assert cmd[1:3] == ["-m", "pytest"]
    assert kw["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert kw["env"]["SOME_OTHER_SETTING"] == "kept"  # the rest of the environment is passed on
    assert res == {"outcome": "killed", "by": "tests/test_x.py::test_y", "returncode": 1}


def test_tau_runner_runs_pytest_without_bytecode(tau, monkeypatch, tmp_path):
    fake = _FakeRun()
    monkeypatch.setattr(tau.subprocess, "run", fake)
    monkeypatch.setenv("SOME_OTHER_SETTING", "kept")
    monkeypatch.delenv("PYTHONDONTWRITEBYTECODE", raising=False)  # the runner must set it, not inherit it
    outcome, by, rc = tau._run(sys.executable, tmp_path, 10)
    cmd, kw = fake.calls[0]
    assert cmd[1:3] == ["-m", "pytest"]
    assert kw["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert kw["env"]["SOME_OTHER_SETTING"] == "kept"
    assert (outcome, rc) == ("killed", 1) and by.startswith("tests/test_x.py::test_y | ")


def _root_with_bytecode(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    for d in ("pkg/__pycache__", "tests/__pycache__", ".git", "runs"):
        (root / d).mkdir(parents=True)
    (root / "pkg" / "mod.py").write_text("X = 1\n", encoding="utf-8")
    (root / "pkg" / "__pycache__" / "mod.cpython-314.pyc").write_bytes(b"stale")
    (root / "tests" / "__pycache__" / "test_mod.cpython-314-pytest-9.0.pyc").write_bytes(b"stale")
    (root / "pkg" / "legacy.pyc").write_bytes(b"stale")
    return root


@pytest.mark.parametrize("runner", ["textcraft", "tau"])
def test_working_copy_starts_without_bytecode(runner, tau, tmp_path):
    copy = {"textcraft": tc._fresh_copy, "tau": tau._fresh_copy}[runner]
    root = _root_with_bytecode(tmp_path)
    work = tmp_path / "work"
    copy(root, work)
    assert (work / "pkg" / "mod.py").read_text(encoding="utf-8") == "X = 1\n"
    assert not list(work.rglob("__pycache__")) and not list(work.rglob("*.pyc"))
    assert not (work / ".git").exists() and not (work / "runs").exists()


MOD = "def double(x):\n    return 2 * x\n\n\ndef untested(x):\n    return x + 1\n"
TEST = "from pkg import mod\n\n\ndef test_double():\n    assert mod.double(3) == 6\n"
MUTANTS = [
    ("breaks the import", "pkg/mod.py", "def double(x):\n", "import no_such_module_for_mutation_tests\ndef double(x):\n"),
    ("caught by a named test", "pkg/mod.py", "return 2 * x", "return 3 * x"),
    ("untested change survives", "pkg/mod.py", "return x + 1", "return x + 2"),
    ("correct text missing", "pkg/mod.py", "this text is not in the file", "x"),
]


def _repo(tmp_path: Path, test_name: str) -> Path:
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "pkg" / "mod.py").write_text(MOD, encoding="utf-8")
    (repo / "tests" / test_name).write_text(TEST, encoding="utf-8")
    (repo / "pytest.ini").write_text("[pytest]\npythonpath = .\ntestpaths = tests\n", encoding="utf-8")
    return repo


def _records(path: Path) -> dict[str, dict]:
    return {r["name"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def _check_strict(out: dict[str, dict], by_prefix: str) -> None:
    # a mutant that breaks the import is error-only, not killed: pytest exits 2 with a collection error
    assert out["breaks the import"]["outcome"] == "error" and out["breaks the import"]["returncode"] == 2
    assert out["caught by a named test"]["outcome"] == "killed"
    assert out["caught by a named test"]["by"].startswith(by_prefix)
    assert out["untested change survives"]["outcome"] == "survived"
    assert out["correct text missing"]["outcome"] == "not applied"
    assert all(r["rule"] == "strict" for r in out.values())


def test_textcraft_runner_end_to_end(monkeypatch, tmp_path, capsys):
    repo = _repo(tmp_path, "test_textcraft_mod.py")
    monkeypatch.setattr(tc, "ROOT", repo)
    monkeypatch.setattr(tc, "MUTANTS", MUTANTS)
    monkeypatch.delenv("MUTATION_TIMEOUT", raising=False)
    res, log = tmp_path / "results.jsonl", tmp_path / "log.json"
    before = (repo / "pkg" / "mod.py").read_bytes()
    assert tc.main(["1", str(log), "--results", str(res)]) == 1
    _check_strict(_records(res), "tests/test_textcraft_mod.py::test_double")
    assert "rule strict: 1/4 killed by a named failing test" in capsys.readouterr().out
    assert [r["outcome"] for r in json.loads(log.read_text(encoding="utf-8"))] == [
        "ERROR-ONLY", "killed", "SURVIVED", "NOT APPLIED"]
    assert (repo / "pkg" / "mod.py").read_bytes() == before  # the checkout itself is never mutated
    n = len(res.read_text(encoding="utf-8").splitlines())
    tc.main(["1", "--results", str(res)])
    assert len(res.read_text(encoding="utf-8").splitlines()) == n  # resumed: nothing rerun
    tc.main(["1", "--results", str(res), "--legacy-any-exit"])
    legacy = {r["name"]: r for r in map(json.loads, res.read_text(encoding="utf-8").splitlines()[n:])}
    assert legacy["breaks the import"]["outcome"] == "killed" and legacy["breaks the import"]["rule"] == "any-exit"


def test_tau_runner_end_to_end(tau, monkeypatch, tmp_path, capsys):
    repo = _repo(tmp_path, "test_mod.py")
    monkeypatch.setattr(tau, "ROOT", repo)
    monkeypatch.setattr(tau, "MUTANTS", MUTANTS)
    monkeypatch.setattr(tau, "TESTS", ["tests/test_mod.py"])
    monkeypatch.delenv("MUTANTS_MATCHING", raising=False)
    monkeypatch.delenv("MUTATION_TIMEOUT", raising=False)
    res, out = tmp_path / "results.jsonl", tmp_path / "out.json"
    assert tau.main([f"--results={res}", f"--out={out}"]) == 1
    _check_strict(_records(res), "tests/test_mod.py::test_double | ")
    assert "rule strict: 1/4 killed by a named failing test" in capsys.readouterr().out
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["rule"] == "strict" and [n for n, _ in summary["error"]] == ["breaks the import"]
    n = len(res.read_text(encoding="utf-8").splitlines())
    tau.main([f"--results={res}"])
    assert len(res.read_text(encoding="utf-8").splitlines()) == n
    tau.main([f"--results={res}", "--legacy-any-exit"])
    legacy = {r["name"]: r for r in map(json.loads, res.read_text(encoding="utf-8").splitlines()[n:])}
    assert legacy["breaks the import"]["outcome"] == "killed" and legacy["breaks the import"]["rule"] == "any-exit"


def test_tau_failing_baseline_judges_no_mutant(tau, monkeypatch, tmp_path, capsys):
    repo = _repo(tmp_path, "test_mod.py")
    (repo / "tests" / "test_mod.py").write_text(TEST.replace("== 6", "== 7"), encoding="utf-8")
    monkeypatch.setattr(tau, "ROOT", repo)
    monkeypatch.setattr(tau, "MUTANTS", MUTANTS)
    monkeypatch.setattr(tau, "TESTS", ["tests/test_mod.py"])
    monkeypatch.delenv("MUTANTS_MATCHING", raising=False)
    res = tmp_path / "results.jsonl"
    assert tau.main([f"--results={res}"]) == 2
    assert "BASELINE FAILS" in capsys.readouterr().out and not res.exists()


@pytest.mark.parametrize("rc, stdout, outcome", [
    (1, "FAILED tests/test_x.py::test_y - AssertionError\n", "killed"),
    (0, "1 passed\n", "survived"),
    (None, "", "timeout"),
    (2, "ERROR tests/test_x.py - ImportError: boom\n", "error"),  # collection error: names a file, not a test
    (1, "ERROR tests/test_x.py::test_y - RuntimeError\n", "error"),  # an error in setup is not a failing test
    (2, "FAILED tests/test_x.py::test_y\n", "error"),  # interrupted run
    (5, "no tests ran\n", "error"),
])
def test_strict_rule_matches_the_main_runner(tau, rc, stdout, outcome):
    assert tc.classify(rc, stdout)[0] == outcome
    assert tau._classify(rc, stdout, "")[0] == outcome
    if rc:  # the legacy rule counts any non-zero exit
        assert tc.classify(rc, stdout, tc.ANY_EXIT)[0] == "killed"
        assert tau._classify(rc, stdout, "", tau.ANY_EXIT)[0] == "killed"
