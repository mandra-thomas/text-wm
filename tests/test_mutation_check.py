"""The mutation runner's kill rule (scripts/mutation_check.py; notes/testing.md, "Mutation runner: strict kill rule").

A mutant counts as killed only when pytest exits 1 and a named test fails; a mutant that breaks the import is
error-only. Each end-to-end test runs the runner on a tiny repository of its own, so it takes a few seconds."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import types
import sys
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "mutation_check.py"


@pytest.fixture(scope="module")
def mc():
    sys.path.insert(0, str(RUNNER.parent))
    try:
        spec = importlib.util.spec_from_file_location("_mutation_check_under_test", RUNNER)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(RUNNER.parent))
    return mod


MOD = "def double(x):\n    return 2 * x\n\n\ndef untested(x):\n    return x + 1\n"
TEST = "from pkg import mod\n\n\ndef test_double():\n    assert mod.double(3) == 6\n"
MUTANTS = [
    ("breaks the import", "pkg/mod.py", "def double(x):\n", "import no_such_module_for_mutation_tests\ndef double(x):\n"),
    ("caught by a named test", "pkg/mod.py", "return 2 * x", "return 3 * x"),
    ("untested change survives", "pkg/mod.py", "return x + 1", "return x + 2"),
    ("correct text missing", "pkg/mod.py", "this text is not in the file", "x"),
]


def _repo(tmp_path: Path, test: str = TEST, mutants=MUTANTS) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "pkg" / "mod.py").write_text(MOD, encoding="utf-8")
    (repo / "tests" / "test_mod.py").write_text(test, encoding="utf-8")
    (repo / "pytest.ini").write_text("[pytest]\npythonpath = .\ntestpaths = tests\n", encoding="utf-8")
    mfile = tmp_path / "mutants.py"
    mfile.write_text(f"MUTANTS = {mutants!r}\n", encoding="utf-8")
    return repo, mfile


def _run(repo: Path, mfile: Path, results: Path, *extra: str) -> subprocess.CompletedProcess:
    # The runner's own settings are dropped: when this file runs inside a mutation run, MUTANTS_MATCHING is set.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("MUTANTS_", "MUTATION_"))}
    return subprocess.run([sys.executable, str(RUNNER), "--root", str(repo), "--mutants-from", str(mfile),
                           "--results", str(results), *extra], capture_output=True, text=True, timeout=300, env=env)


def _outcomes(results: Path) -> dict[str, dict]:
    return {r["name"]: r for r in map(json.loads, results.read_text(encoding="utf-8").splitlines())}


def test_strict_rule_end_to_end(tmp_path):
    repo, mfile = _repo(tmp_path)
    before = (repo / "pkg" / "mod.py").read_bytes()
    res = tmp_path / "results.jsonl"
    r = _run(repo, mfile, res)
    out = _outcomes(res)
    # a mutant that breaks the import is error-only, not killed: pytest exits 2 with a collection error
    assert out["breaks the import"]["outcome"] == "error"
    assert out["breaks the import"]["returncode"] == 2
    assert "ERROR-ONLY   breaks the import" in r.stdout
    # a mutant caught by a named failing test is killed, and the test is recorded
    assert out["caught by a named test"]["outcome"] == "killed"
    assert out["caught by a named test"]["by"] == "tests/test_mod.py::test_double"
    # a mutant no test notices survives the covering tests and then the whole suite
    assert out["untested change survives"]["outcome"] == "survived"
    assert out["untested change survives"]["targets"] == ["tests"]
    assert out["correct text missing"]["outcome"] == "not applied"
    assert all(rec["rule"] == "strict" for rec in out.values())
    assert r.returncode == 1
    assert "1/4 killed by a named failing test" in r.stdout
    assert (repo / "pkg" / "mod.py").read_bytes() == before  # the checkout itself is never mutated


def test_legacy_flag_counts_any_non_zero_exit(tmp_path):
    repo, mfile = _repo(tmp_path)
    res = tmp_path / "legacy.jsonl"
    _run(repo, mfile, res, "--legacy-any-exit")
    out = _outcomes(res)
    assert out["breaks the import"]["outcome"] == "killed"
    assert out["breaks the import"]["rule"] == "any-exit"
    assert out["caught by a named test"]["outcome"] == "killed"
    assert out["untested change survives"]["outcome"] == "survived"


def test_resume_skips_recorded_mutants_and_tolerates_a_cut_off_line(tmp_path):
    repo, mfile = _repo(tmp_path)
    res = tmp_path / "results.jsonl"
    _run(repo, mfile, res)
    lines = res.read_text(encoding="utf-8").splitlines()
    res.write_text("\n".join(lines[:2]) + "\n" + lines[2][:20], encoding="utf-8")  # cut off in the third record
    r = _run(repo, mfile, res)
    out = _outcomes_lenient(res)
    assert len(out) == 4 and [k for k in out] == [m[0] for m in MUTANTS]
    assert "1/4 killed" in r.stdout
    n = len(res.read_text(encoding="utf-8").splitlines())
    _run(repo, mfile, res)
    assert len(res.read_text(encoding="utf-8").splitlines()) == n  # nothing rerun
    # records of another rule are not reused
    _run(repo, mfile, res, "--legacy-any-exit")
    assert len(res.read_text(encoding="utf-8").splitlines()) == n + 4


def _outcomes_lenient(results: Path) -> dict[str, dict]:
    out = {}
    for line in results.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[rec["name"]] = rec
    return out


def test_failing_baseline_judges_no_mutant(tmp_path):
    repo, mfile = _repo(tmp_path, test=TEST.replace("== 6", "== 7"))
    res = tmp_path / "results.jsonl"
    r = _run(repo, mfile, res)
    assert r.returncode != 0
    assert "BASELINE FAILS" in r.stderr
    assert not res.exists() or not res.read_text(encoding="utf-8").strip()


def test_setup_error_then_named_failure_is_killed(tmp_path):
    test = ("import pytest\nfrom pkg import mod\n\n\n@pytest.fixture\ndef broken():\n"
            "    if mod.double(1) != 2:\n        raise RuntimeError('fixture')\n\n\n"
            "def test_a(broken):\n    pass\n\n\ndef test_b():\n    assert mod.double(3) == 6\n")
    repo, mfile = _repo(tmp_path, test=test, mutants=[MUTANTS[1]])
    res = tmp_path / "results.jsonl"
    _run(repo, mfile, res)
    rec = _outcomes(res)["caught by a named test"]
    assert rec["outcome"] == "killed" and rec["by"].startswith("tests/test_mod.py::test_b")


def test_setup_error_alone_is_error_only(tmp_path):
    test = ("import pytest\nfrom pkg import mod\n\n\n@pytest.fixture\ndef broken():\n"
            "    if mod.double(1) != 2:\n        raise RuntimeError('fixture')\n\n\ndef test_a(broken):\n    pass\n")
    repo, mfile = _repo(tmp_path, test=test, mutants=[MUTANTS[1]])
    res = tmp_path / "results.jsonl"
    _run(repo, mfile, res)
    rec = _outcomes(res)["caught by a named test"]
    assert rec["outcome"] == "error" and "tests/test_mod.py::test_a" in rec["by"]


def test_classify(mc):
    assert mc.classify(1, "FAILED tests/test_x.py::test_y - AssertionError\n") == ("killed", "tests/test_x.py::test_y")
    assert mc.classify(1, "FAILED tests/test_x.py::test_y[case one] - assert 0\n")[0] == "killed"
    assert mc.classify(0, "1 passed")[0] == "survived"
    assert mc.classify(None, "")[0] == "timeout"
    # collection and import errors: exit 2, the error line names a file, not a test
    assert mc.classify(2, "ERROR tests/test_x.py - ImportError: boom\n")[0] == "error"
    # a named error in setup with exit 1 is not a failing test
    assert mc.classify(1, "ERROR tests/test_x.py::test_y - RuntimeError\n")[0] == "error"
    # exit 1 without any named test, and other exit codes
    for rc in (1, 3, 4, 5):
        assert mc.classify(rc, "something went wrong\n")[0] == "error"
    # a FAILED line with a non-1 exit (interrupted) is not a kill under the strict rule
    assert mc.classify(2, "FAILED tests/test_x.py::test_y\n")[0] == "error"
    # the legacy rule counts any non-zero exit
    assert mc.classify(2, "ERROR tests/test_x.py - ImportError\n", mc.ANY_EXIT)[0] == "killed"
    assert mc.classify(None, "", mc.ANY_EXIT)[0] == "timeout"


def test_load_mutants_uses_the_files_own_sibling_imports(mc, tmp_path, monkeypatch):
    # As with a frozen tag's mutation_check.py importing its own copy of a mutant file this runner also imports.
    other = tmp_path / "other"
    other.mkdir()
    (other / "sibling_mutants.py").write_text('MUTANTS = [("theirs", "a.py", "x", "y")]\n', encoding="utf-8")
    (other / "mutation_check.py").write_text(
        "from sibling_mutants import MUTANTS as _M\nMUTANTS = [('own', 'b.py', 'p', 'q')] + _M\n", encoding="utf-8")
    ours = types.ModuleType("sibling_mutants")
    ours.MUTANTS = [("ours", "c.py", "s", "t")]
    monkeypatch.setitem(sys.modules, "sibling_mutants", ours)
    assert mc.load_mutants(other / "mutation_check.py") == [("own", "b.py", "p", "q"), ("theirs", "a.py", "x", "y")]
    assert sys.modules["sibling_mutants"] is ours
