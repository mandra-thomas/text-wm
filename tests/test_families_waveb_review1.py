"""Model families, wave B: the fixes of review 1 (notes/families_waveb_review1.md; 2026-11-22, before any wave B score).

The forward passes with the real tokenizers and classes are in tests/test_families_waveb_serve2.py (serve2 only).
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import sys
import types
from pathlib import Path

import pytest

from wf import constants as C, families, sizes
from wf.scoring.logodds import LogoddsScorer, template_input_ids
from tests.test_rq1 import mock_root, mock_score  # noqa: F401
from tests.test_families_waveb import (ChatTok, WordTok, _cfg, _chat_scorer, _chat_spec, _items, _plain_scorer,
                                       _report)

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = json.loads((ROOT / "tests" / "golden" / "default_prompt.json").read_text())
B = families.WAVE_B


# ---------------- M1: apply_chat_template returns a BatchEncoding in transformers 5 ----------------


class FT:
    """A minimal stand-in for a torch tensor (nested lists), with the operations the scorer uses."""

    def __init__(self, data):
        self.data = data

    @property
    def shape(self):
        s, d = [], self.data
        while isinstance(d, list):
            s.append(len(d))
            d = d[0] if d else None
        return tuple(s)

    def __getitem__(self, k):
        if isinstance(k, tuple):
            out = self
            for i in k:
                out = out[i]
            return out
        return FT(self.data[k])

    def tolist(self):
        return self.data

    def to(self, device):
        return self

    def float(self):
        return self

    def item(self):
        assert not isinstance(self.data, list)
        return self.data


class _NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _cat(ts, dim):
    if not all(isinstance(t, FT) for t in ts):  # as torch.cat: "expected Tensor as element 0, but got BatchEncoding"
        raise TypeError(f"expected Tensor, got {[type(t).__name__ for t in ts]}")
    assert dim == 1
    return FT([sum((t.data[0] for t in ts), [])])


def _log_softmax(x, dim):
    m = max(x.data)
    lse = m + math.log(sum(math.exp(v - m) for v in x.data))
    return FT([v - lse for v in x.data])


FAKE_TORCH = types.SimpleNamespace(cat=_cat, no_grad=_NoGrad, log_softmax=_log_softmax)
V = 512


class FakeModel:
    """Logits at each position depend only on the token there (deterministic, causal)."""

    def parameters(self):
        yield types.SimpleNamespace(device="cpu")

    def __call__(self, input_ids):
        if not isinstance(input_ids, FT):  # a model reads input_ids.shape; a BatchEncoding has none
            raise AttributeError(f"input_ids is a {type(input_ids).__name__}")
        rows = [[float((t * 31 + v * 17) % 13) for v in range(V)] for t in input_ids.data[0]]
        return types.SimpleNamespace(logits=FT([rows]))


class FakeBatchEncoding(collections.UserDict):
    """transformers' BatchEncoding: a mapping with attribute access to its keys."""

    def __getattr__(self, name):
        try:
            return self.data[name]
        except KeyError:
            raise AttributeError(name) from None


class BatchTok(ChatTok):
    """ChatTok whose tokenize=True template call returns what transformers 5 returns: a BatchEncoding."""

    def _id(self, w):
        return self.vocab.setdefault(w, len(self.vocab) % V)

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = super().__call__(text).input_ids
        return types.SimpleNamespace(input_ids=FT([ids]) if return_tensors else ids)

    def apply_chat_template(self, messages, tokenize=True, return_tensors=None, **kw):
        out = super().apply_chat_template(messages, tokenize=tokenize, **kw)
        if not tokenize:
            return out
        ids = out if isinstance(out, list) else out.input_ids
        return FakeBatchEncoding({"input_ids": FT([ids]), "attention_mask": FT([[1] * len(ids)])})

    def decode(self, ids):
        return super().decode(ids.tolist() if isinstance(ids, FT) else ids)


def test_scoring_reads_the_ids_out_of_a_batch_encoding(monkeypatch, mock_score):
    """Review 1, M1: with transformers 5's BatchEncoding, torch.cat and the model refused the prompt; every chat score
    crashed. The unfixed scorer fails this test."""
    monkeypatch.setitem(sys.modules, "torch", FAKE_TORCH)
    spec = _chat_spec()
    sc = _chat_scorer(BatchTok(), spec)
    item = GOLDEN["item"]
    msgs = families.family_messages(spec, item, "aligned")
    sc.mock, sc._model = False, FakeModel()
    assert isinstance(sc._encode(msgs), FT)
    r = sc.score_pair(item["id"], "aligned", msgs, item["candidates"]["o_plus"], item["candidates"]["o_minus"])
    assert math.isfinite(r.L) and r.n_tokens_plus == len(item["candidates"]["o_plus"].split())
    yn = families.family_messages(spec, item, "yn_aligned")
    one = sc.score_one_token_pair(item["id"], "yn_aligned", yn, "Yes", "No")
    full = sc.score_pair(item["id"] + "-full", "yn_aligned", yn, "Yes", "No")
    assert math.isfinite(one.L) and abs(one.L - full.L) < 1e-9 and one.first_token_plus == "Yes"
    assert len(sc._tok.vocab) < V  # no id wrapped around, so every decoded token is the word it stands for


def test_template_ids_are_unchanged_when_the_template_returns_the_tensor():
    """transformers 4 returns the ids themselves: they pass through untouched (every earlier run)."""
    ids = FT([[1, 2, 3]])
    assert template_input_ids(ids) is ids
    assert template_input_ids(FakeBatchEncoding({"input_ids": ids, "attention_mask": FT([[1, 1, 1]])})) is ids
    assert template_input_ids([[4, 5]]) == [[4, 5]]


# ---------------- M3, S1, S2, S3: node-local weights against the committed manifest ----------------


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_file_digest_matches_git_and_lfs():
    from wf.tools import weights_manifest as wm

    p = _write(Path(__import__("tempfile").mkdtemp()) / "x", b"hello\n")
    assert wm.file_digest(p, "git_sha1") == "ce013625030ba8dba906f756967f9e9ca394464a"  # git hash-object
    assert wm.file_digest(p, "sha256") == hashlib.sha256(b"hello\n").hexdigest()
    with pytest.raises(ValueError):
        wm.file_digest(p, "md5")


def _manifest(files: dict[str, bytes], lfs=("model.safetensors",)) -> dict:
    out = []
    for name, data in files.items():
        e = {"name": name, "size": len(data)}
        if name in lfs:
            e["sha256"] = hashlib.sha256(data).hexdigest()
        else:
            e["git_sha1"] = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        out.append(e)
    return {"hf_id": "org/m", "revision": "r" * 40, "files": out, "total_bytes": sum(len(d) for d in files.values())}


FILES = {"config.json": b'{"a": 1}', "model.safetensors": b"\x00weights\x01", "sub/tokenizer.json": b"{}"}


def _snapshot(tmp_path: Path, files=FILES) -> Path:
    snap = tmp_path / "models--org--m" / "snapshots" / ("r" * 40)
    blobs = tmp_path / "models--org--m" / "blobs"
    for name, data in files.items():  # the Hub cache: snapshot entries are symlinks to blobs
        blob = _write(blobs / hashlib.sha256(name.encode()).hexdigest(), data)
        link = snap / name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(blob)
    return snap


def test_verify_snapshot_passes_the_exact_files(tmp_path):
    from wf.tools import weights_manifest as wm

    rec = wm.verify_snapshot(_snapshot(tmp_path), _manifest(FILES))
    assert rec["ok"], rec["problems"]
    assert rec["n_files"] == 3 and rec["total_bytes"] == sum(map(len, FILES.values()))
    assert rec["files"]["model.safetensors"]["sha256"] == hashlib.sha256(FILES["model.safetensors"]).hexdigest()
    assert rec["files"]["config.json"]["git_sha1"] and rec["manifest"]["n_files"] == 3


@pytest.mark.parametrize("change, words", [
    ({"model.safetensors": b"\x00weightz\x01"}, "hash: model.safetensors"),  # same size, other content (LFS)
    ({"config.json": b'{"a": 2}'}, "hash: config.json"),  # same size, other content (Git blob)
    ({"config.json": b'{"a": 10}'}, "size: config.json"),
    ({"extra.bin": b"x"}, "extra: extra.bin"),
    ({"sub/tokenizer.json": None}, "missing: sub/tokenizer.json"),
])
def test_verify_snapshot_fails_on_any_difference(tmp_path, change, words):
    from wf.tools import weights_manifest as wm

    files = {k: v for k, v in {**FILES, **change}.items() if v is not None}
    rec = wm.verify_snapshot(_snapshot(tmp_path, files), _manifest(FILES))
    assert not rec["ok"] and any(p.startswith(words) for p in rec["problems"]), rec["problems"]


def test_check_disk_needs_the_manifest_total_plus_ten_percent():
    from wf.tools import weights_manifest as wm

    m = {"total_bytes": 1000}
    usage = lambda free: (lambda path: types.SimpleNamespace(free=free))
    assert wm.check_disk("/x", m, usage(1100))["ok"] and wm.check_disk("/x", m, usage(1100))["needed_bytes"] == 1100
    assert not wm.check_disk("/x", m, usage(1099))["ok"]
    assert not wm.check_disk("/x", m, usage(1000))["ok"]


@pytest.mark.parametrize("path, refused", [("/storage/ice1/2/1/gmat3/hfcache", True), ("/storage", True),
                                           ("/tmp/wf_hf_1", False), ("/storagex/a", False), ("/scratch/x", False)])
def test_node_local_cache_is_never_on_the_shared_file_system(path, refused):
    from wf.tools import weights_manifest as wm

    if refused:
        with pytest.raises(SystemExit, match="shared file system"):
            wm.refuse_shared_storage(path)
    else:
        assert wm.refuse_shared_storage(path).endswith(path.rsplit("/", 1)[-1])


def test_node_local_cache_behind_a_link_to_storage_is_refused(tmp_path, monkeypatch):
    from wf.tools import weights_manifest as wm

    monkeypatch.setattr(wm.os.path, "realpath", lambda p: "/storage/ice1/x" if "link" in str(p) else str(p))
    with pytest.raises(SystemExit):
        wm.refuse_shared_storage(tmp_path / "link")


def test_prefetch_checks_disk_before_and_hashes_after(tmp_path):
    from wf.tools import prefetch_models as pm

    calls, snap = [], _snapshot(tmp_path)

    def download(**kw):
        calls.append(kw)
        return str(snap)

    m = _manifest(FILES)
    plenty = lambda p: types.SimpleNamespace(free=10 ** 9)
    r = pm.prefetch("org/m", "r" * 40, str(tmp_path), download=download, manifest=m, usage=plenty)
    assert r["status"] == "OK" and r["verify"]["ok"] and r["disk"]["ok"] and calls[0]["cache_dir"] == str(tmp_path)
    r = pm.prefetch("org/m", "r" * 40, str(tmp_path), download=download, manifest=m,
                    usage=lambda p: types.SimpleNamespace(free=10))
    assert r["status"] == "ERROR not enough disk space" and len(calls) == 1  # nothing downloaded
    bad = _manifest({**FILES, "config.json": b'{"b": 1}'})
    r = pm.prefetch("org/m", "r" * 40, str(tmp_path), download=download, manifest=bad, usage=plenty)
    assert r["status"] == "ERROR files differ from the manifest" and not r["verify"]["ok"]
    r = pm.prefetch("org/m", "r" * 40, str(tmp_path), download=download)  # shared cache: no manifest, no hashing
    assert r["status"] == "OK" and "verify" not in r and "disk" not in r


def test_prefetch_node_local_downloads_into_the_named_cache(tmp_path, monkeypatch):
    """Review 1, S1: the folder is passed by name, refused under /storage, and every model needs a manifest."""
    from wf.tools import prefetch_models as pm, weights_manifest as wm

    spec = B["gemma4-e4b-it"]
    rev = C.REVISIONS[spec.hf_id]
    files = {"config.json": b"{}", "model.safetensors": b"w"}
    man = {**_manifest(files), "hf_id": spec.hf_id, "revision": rev}
    mdir = tmp_path / "manifests"
    _write(mdir / (spec.hf_id.replace("/", "__") + ".json"), json.dumps(man).encode())
    monkeypatch.setattr(wm, "MANIFEST_DIR", mdir)
    seen = {}

    def download(**kw):
        seen.update(kw)
        snap = Path(kw["cache_dir"]) / "snap" / rev
        for n, d in files.items():
            _write(snap / n, d)
        return str(snap)

    monkeypatch.setattr(pm, "_snapshot_download", download)
    cache, out = tmp_path / "local", tmp_path / "rec.json"
    assert pm.main(["gemma4-e4b-it", "--cache", str(cache), "--node-local", "--out", str(out)]) == 0
    assert seen["cache_dir"] == str(cache) and seen["revision"] == rev
    rec = json.loads(out.read_text())["gemma4-e4b-it"]
    assert rec["verify"]["ok"] and rec["hf_id"] == spec.hf_id and rec["revision"] == rev
    assert pm.main(["gemma4-e4b-it", "--node-local"]) == 2  # no --cache
    with pytest.raises(SystemExit, match="shared file system"):
        pm.main(["gemma4-e4b-it", "--cache", "/storage/ice1/2/1/gmat3/hfcache/x", "--node-local"])
    assert pm.main(["qwen3.5-9b", "--cache", str(cache), "--node-local"]) == 2  # no manifest in this folder
    files["model.safetensors"] = b"v"
    assert pm.main(["gemma4-e4b-it", "--cache", str(tmp_path / "l2"), "--node-local", "--out", str(out)]) == 1


def test_committed_manifests_cover_every_wave_b_model_at_its_pinned_revision():
    import fnmatch

    from wf.tools import prefetch_models as pm, weights_manifest as wm

    sizes_gb = {"mistralai/Ministral-3-8B-Instruct-2512-BF16": 17.9, "Qwen/Qwen3.5-9B": 19.3, "Qwen/Qwen3.5-9B-Base": 19.3,
                "google/gemma-4-E4B-it": 16.0, "Qwen/Qwen3.8-27B": 55.6, "google/gemma-4-31B-it": 62.6}
    assert {s.hf_id for s in B.values()} == set(sizes_gb)
    for hf_id, gb in sizes_gb.items():
        m = wm.load_manifest(hf_id, C.REVISIONS[hf_id])
        assert abs(m["total_bytes"] / 1e9 - gb) < 0.06 and m["total_bytes"] == sum(f["size"] for f in m["files"])
        assert m["dry_run_cross_check"]["same_files_and_sizes"] is True
        assert m["allow_patterns"] == pm.ALLOW and m["ignore_patterns"] == pm.IGNORE
        for f in m["files"]:
            assert any(fnmatch.fnmatch(f["name"], a) for a in pm.ALLOW)
            assert not any(fnmatch.fnmatch(f["name"], i) for i in pm.IGNORE)
            assert len(f.get("sha256") or "") == 64 or len(f.get("git_sha1") or "") == 40, f
            if f["name"].endswith(".safetensors"):
                assert len(f["sha256"]) == 64
        names = {f["name"] for f in m["files"]}
        assert "config.json" in names and any(n.endswith(".safetensors") for n in names)
    with pytest.raises(ValueError, match="not"):
        wm.load_manifest("Qwen/Qwen3.5-9B", "0" * 40)
    with pytest.raises(FileNotFoundError):
        wm.load_manifest("org/none", "0" * 40)


def _weights_record(tmp_path, spec, ok=True, status="OK", revision=None):
    rec = {"x": {"hf_id": spec.hf_id, "revision": revision or spec.revision, "status": status,
                 "verify": {"ok": ok, "problems": [] if ok else ["hash: x"], "n_files": 2, "total_bytes": 9,
                            "files": {"a": {"size": 1, "sha256": "0" * 64}}}}}
    f = tmp_path / "rec.json"
    f.write_text(json.dumps(rec))
    return f


def test_weights_check_requires_a_passed_record_for_this_model_and_revision(tmp_path):
    spec = B["gemma4-31b-it"]
    assert families.weights_check(spec, {}) is None  # shared cache
    with pytest.raises(families.WeightsCheckFailed, match="no verification record"):
        families.weights_check(spec, {"WF_WEIGHTS": "local"})
    env = lambda f: {"WF_WEIGHTS": "local", families.WEIGHTS_RECORD_ENV: str(f)}
    got = families.weights_check(spec, env(_weights_record(tmp_path, spec)))
    assert got["ok"] and got["revision"] == spec.revision and got["files"]["a"]["sha256"] == "0" * 64
    for kw, words in (({"ok": False}, "do not match"), ({"status": "ERROR files differ"}, "do not match"),
                      ({"revision": "1" * 40}, "no verified download")):
        with pytest.raises(families.WeightsCheckFailed, match=words):
            families.weights_check(spec, env(_weights_record(tmp_path, spec, **kw)))
    with pytest.raises(families.WeightsCheckFailed):
        families.weights_check(B["gemma4-e4b-it"], env(_weights_record(tmp_path, spec)))


class _FakeLoaded:
    def __init__(self):
        self.config = types.SimpleNamespace(_attn_implementation="sdpa")
        self.dtype = "bf16"
        self.hf_device_map = {"": 0}


def test_the_gate_saves_the_weights_hashes_in_the_shard(tmp_path, monkeypatch, mock_score):
    spec = _chat_spec(attn_agreement_check=False)
    sc = _chat_scorer(ChatTok(), spec)
    sc.mock, sc._model = False, _FakeLoaded()
    monkeypatch.setenv("WF_WEIGHTS", "local")
    monkeypatch.setenv(families.WEIGHTS_RECORD_ENV, str(_weights_record(tmp_path, spec)))
    d = tmp_path / "shard"
    d.mkdir()
    families._gate(spec, {"x": 1})(d, sc, *_items(2, 1))
    saved = json.loads((d / "weights_check.json").read_text())
    assert saved["ok"] and saved["files"]["a"]["sha256"] == "0" * 64
    assert json.loads((d / "tokenization_checks.json").read_text())["weights_check"]["ok"] is True
    monkeypatch.setenv(families.WEIGHTS_RECORD_ENV, str(_weights_record(tmp_path, spec, ok=False)))
    with pytest.raises(families.WeightsCheckFailed):
        families._gate(spec, {"x": 1})(d, sc, *_items(2, 1))


def test_serve2_job_downloads_node_local_and_checks_hashes_before_scoring():
    text = (ROOT / "pace" / "family_score_serve2.sbatch").read_text()
    local = text[text.index('if [[ "${WF_WEIGHTS:-shared}" == "local" ]]'):text.index("export HF_HUB_OFFLINE=1")]
    for line in ('export HF_HUB_CACHE="${LOCAL_CACHE}"', 'export HF_XET_CACHE="${LOCAL_CACHE}/xet"',
                 "export WF_WEIGHTS=local", 'export WF_WEIGHTS_RECORD="logs/familyB_prefetch_',
                 'python -m wf.tools.prefetch_models "${TAG}" --cache "${LOCAL_CACHE}" --node-local --out "${WF_WEIGHTS_RECORD}"'):
        assert line in local, line
    assert local.index('export HF_HUB_CACHE="${LOCAL_CACHE}"') < local.index("python -m wf.tools.prefetch_models")
    assert 'LOCAL_CACHE="${TMPDIR:-/tmp}/wf_hf_' in local
    assert "family_nopreserve_*) TAG=" in text


# ---------------- S4: the environment check's GPU type ----------------


def _shard_with_jobs(tmp_path, name, gpus_per_job):
    d = tmp_path / name
    d.mkdir()
    (d / "jobs.jsonl").write_text("".join(json.dumps({"gpu": g}) + "\n" for g in gpus_per_job))
    return d


@pytest.mark.parametrize("jobs, ok", [([["NVIDIA L40S"]], True), ([["NVIDIA L40S"], ["NVIDIA L40S"]], True),
                                      ([["NVIDIA A40"]], False), ([None], False), ([["NVIDIA L40S"], ["NVIDIA A100-SXM4-80GB"]], False),
                                      ([["NVIDIA L40S", "NVIDIA A40"]], False)])
def test_envcheck_gpu_must_be_the_pinned_type(tmp_path, jobs, ok):
    assert families.ENVCHECK_GPU == {"mistral-7b-v0.3": "NVIDIA A40", "qwen3-8b": "NVIDIA L40S"}
    rec = families.envcheck_gpu_check("qwen3-8b", [_shard_with_jobs(tmp_path, "shard_00_of_01", jobs)])
    assert rec["ok"] is ok and rec["expected"] == "NVIDIA L40S"
    assert families.envcheck_gpu_check("qwen3-8b", [])["ok"] is False


def test_envcheck_fails_on_the_wrong_gpu_type(mock_root, monkeypatch):
    from tests.test_families_waveb import _envcheck_runs
    from wf.runs import load_config

    _envcheck_runs(monkeypatch, 4, gpus={"mistral-7b-v0.3": "NVIDIA A40", "qwen3-8b": "NVIDIA A40"})
    assert families.run_family_score(_cfg("family_score_mistral-7b-v0.3", "mistral-7b-v0.3")) == 0
    assert sizes.run_size_score(_cfg("size_score_qwen3-8b", "qwen3-8b")) == 0
    (mock_root / "envcheck_tokens").mkdir()
    for tag in families.ENVCHECK_MODELS:
        (mock_root / "envcheck_tokens" / f"compare_{tag}.json").write_text(json.dumps({"ok": True}))
    cfg = {**load_config("families_envcheck"), "sources": ["rq12_transitions_smoke"], "n_items": 4}
    assert families.run_families_envcheck(cfg) == 0
    rep = json.loads((mock_root / "families_envcheck" / "metrics.json").read_text())
    assert rep["pass"] is False and rep["models"]["mistral-7b-v0.3"]["pass"] is True
    q = rep["models"]["qwen3-8b"]
    assert q["pass"] is False and q["checks"]["gpu_type"] is False and q["gpu"]["recorded"] == {"shard_00_of_01": ["NVIDIA A40"]}
    assert [k for k, v in q["checks"].items() if not v] == ["gpu_type"]


def test_serve2_job_stops_an_envcheck_on_another_gpu_type():
    text = (ROOT / "pace" / "family_score_serve2.sbatch").read_text()
    assert 'family_envcheck_*) TAG=""; ENVTAG="${RUN#family_envcheck_}"' in text
    check = text[text.index('if [[ -n "${ENVTAG}" ]]'):text.index('LOCAL_CACHE=""')]
    assert "from wf.families import ENVCHECK_GPU" in check and "if have != [want]:" in check and "sys.exit" in check
    assert text.index('if [[ -n "${ENVTAG}" ]]') < text.index('run_wf "${RUN}"')
    assert "--constraint=gpu-a40" in text and "--constraint=gpu-l40s" in text


# ---------------- S6: GPU memory and job memory at the start of a job ----------------


@pytest.mark.parametrize("tag, gpu_gb, mem_mb, ok", [
    ("gemma4-31b-it", 85.1, "131072", True), ("gemma4-31b-it", 95.4, "131072", True),
    ("gemma4-31b-it", 47.7, "131072", False), ("gemma4-31b-it", 85.1, "65536", False),
    ("gemma4-31b-it", 85.1, "", False), ("qwen3.8-27b", 85.1, "131071", False), ("qwen3.8-27b", 85.1, "131072", True),
    ("qwen3.5-9b", 47.7, "", True), ("qwen3.5-9b", 23.6, "", False), ("gemma4-e4b-it", 42.4, "65536", True),
])
def test_resources_are_checked_before_loading(tag, gpu_gb, mem_mb, ok):
    env = {"SLURM_MEM_PER_NODE": mem_mb} if mem_mb else {}
    if ok:
        rec = families.require_resources(B[tag], env, lambda: gpu_gb)
        assert rec["checked"] is True and rec["gpu_total_gb"] == round(gpu_gb, 1)
    else:
        with pytest.raises(SystemExit):
            families.require_resources(B[tag], env, lambda: gpu_gb)


def test_resource_settings_per_model_and_mock_runs():
    assert {t for t, s in B.items() if s.host_mem_min_gb == 128} == {"qwen3.8-27b", "gemma4-31b-it"}
    assert {t for t, s in B.items() if s.host_mem_min_gb} == {"qwen3.8-27b", "gemma4-31b-it"}
    never = lambda: pytest.fail("no GPU check for a model without gpu_min_gb")
    assert families.require_resources(families.MODELS["mistral-7b-v0.3"], {}, never)["checked"] is True
    assert families.require_resources(B["gemma4-31b-it"], {"WF_MOCK_SCORE": "1"}, never) == {"checked": "mock"}


# ---------------- S7, S10: the CPU gate tool ----------------


def test_gate_tool_resolves_check_variants_and_gates_their_shards_items():
    from wf.tools import family_gate_cpu as g

    ft = families.variant_spec("gemma4-e4b-it-firstturn")
    assert ft.tag == "gemma4-e4b-it-firstturn" and ft.system == "fold" and ft.hf_id == B["gemma4-e4b-it"].hf_id
    npz = families.variant_spec("qwen3.8-27b-nopreserve")
    assert npz.template_kwargs == {"preserve_thinking": False} and npz.forbidden_before_ending == ("<think>",)
    assert families.variant_spec("qwen3.5-9b") is B["qwen3.5-9b"]
    primary = [{"id": f"t{i:04d}"} for i in range(400)]
    for tag in ("gemma4-e4b-it-firstturn", "qwen3.8-27b-nopreserve"):
        p, s = g.shard_items(tag, primary, [{"id": "s"}])
        assert p == families.placement_items(primary) and len(p) == 300 and s == []
    assert g.shard_items("gemma4-e4b-it", primary, [{"id": "s"}]) == (primary, [{"id": "s"}])


@pytest.mark.parametrize("oks, code", [((True, True), 0), ((True, False), 1), ((False, True), 1)])
def test_gate_tool_exits_non_zero_when_a_gate_fails(mock_root, monkeypatch, tmp_path, oks, code):
    from wf.tools import family_gate_cpu as g

    seen = []
    monkeypatch.setattr(g, "load_tokenizer", lambda spec, cache: object())
    monkeypatch.setattr(g, "scorer_for", lambda spec, tok: object())

    def gate(spec, scorer, primary, successes, cache):
        seen.append((spec.tag, len(primary), len(successes)))
        return {"ok": oks[len(seen) - 1], "n_candidates_checked": 1, "n_yes_no_prompts_checked": 1, "n_problems": 0,
                "max_prompt_tokens": 1, "seconds": 0, "problems": []}

    monkeypatch.setattr(g, "gate_all_items", gate)
    out = tmp_path / "gate.json"
    assert g.main(["gemma4-e4b-it", "gemma4-e4b-it-firstturn", "--out", str(out), "--cache", str(tmp_path),
                   "--sources", "rq12_transitions_smoke"]) == code
    assert [t for t, _, _ in seen] == ["gemma4-e4b-it", "gemma4-e4b-it-firstturn"]
    assert seen[1][2] == 0 and seen[1][1] <= seen[0][1]  # the variant gates the placement items, no successes
    assert set(json.loads(out.read_text())["models"]) == {"gemma4-e4b-it", "gemma4-e4b-it-firstturn"}


def test_plain_gate_fails_when_the_candidates_share_their_first_token(mock_score):
    """Review 1, S10: the mutant removing the distinct-first-token check in plain_checks survived."""
    sc, spec = _plain_scorer(WordTok())
    primary, successes = _items(2, 0, t1={"candidates": {"o_plus": "You put it", "o_minus": "You fail"}})
    rec = families.tokenization_checks(sc, spec, primary, successes)
    assert not rec["ok"] and rec["problems"] == ["t1: the candidates share their first token"], rec["problems"]


def test_first_turn_note_is_right_for_each_template():
    """Review 1, S10: "not the last" describes Mistral's template only; wave A's saved note is kept word for word."""
    a = families.first_turn_spec(families.MODELS["mistral-7b-v0.3"])
    assert a.template_notes == "Placement check: the system text opens the first user turn, not the last."
    b = families.first_turn_spec(B["gemma4-e4b-it"])
    assert "not the last" not in b.template_notes and "instead of the template's own system turn" in b.template_notes
    assert b.attn_agreement_check and b.system == "fold"


# ---------------- S8: Qwen3.8 with preserve_thinking=False ----------------


def test_nopreserve_spec_and_run_are_registered():
    from wf.runs import RUNNERS, load_config

    s = families.nopreserve_spec(B["qwen3.8-27b"])
    assert s.tag == "qwen3.8-27b-nopreserve" and s.template_kwargs == {"preserve_thinking": False}
    assert s.thinking_off_ending == B["qwen3.8-27b"].thinking_off_ending and s.forbidden_before_ending == ("<think>",)
    with pytest.raises(ValueError):
        families.nopreserve_spec(B["qwen3.5-9b"])
    cfg = load_config("family_nopreserve_qwen3.8-27b")
    assert cfg["model"] == "qwen3.8-27b" and cfg["n_items"] == 300 and "family_nopreserve_qwen3.8-27b" in RUNNERS
    assert load_config("families_waveb_analysis")["template_checks"] == {"qwen3.8-27b": "family_nopreserve_qwen3.8-27b"}
    with pytest.raises(SystemExit):
        families.run_family_nopreserve({"model": "qwen3.5-9b", "run_id": "x"})


def test_the_gate_finds_think_blocks_in_the_examples(mock_score):
    spec = _chat_spec(forbidden_before_ending=("OFF",))
    primary, successes = _items(2, 1)
    rec = families.tokenization_checks(_chat_scorer(ChatTok(), spec), spec, primary, successes)
    assert rec["ok"], rec["problems"]
    rec = families.tokenization_checks(_chat_scorer(ChatTok(examples_off=True), spec), spec, primary, successes)
    assert not rec["ok"] and any("'OFF' before the thinking-off ending" in p for p in rec["problems"]), rec["problems"]
    plain = _chat_spec()  # no forbidden text declared: the same template passes
    assert families.tokenization_checks(_chat_scorer(ChatTok(examples_off=True), plain), plain, primary, successes)["ok"]


def _block(diff, ci, native):
    return {fmt: {"first_turn_minus_native": {"delta_L_nats": diff, "ci": list(ci)}, "native": {"delta_L_nats": native}}
            for fmt in ("yes_no", "two_candidate")}


@pytest.mark.parametrize("diff, ci, native, dependent", [
    (0.5, (0.2, 0.8), 2.0, True), (-0.5, (-0.8, -0.2), 2.0, True), (0.3, (0.1, 0.5), 2.0, False),
    (0.5, (-0.1, 1.1), 2.0, False), (0.41, (0.01, 0.8), 2.0, True), (0.4, (0.01, 0.8), 2.0, False),
    (float("nan"), (float("nan"), float("nan")), 2.0, False),
])
def test_one_fifth_rule(diff, ci, native, dependent):
    r = families.one_fifth_rule(_block(diff, ci, native), "yes_no")
    assert r["format_dependent"] is dependent and r["native_nats"] == native


# ---------------- S9: eager against SDPA attention for Gemma 4 ----------------


class _AttnModel:
    """A model whose text sub-config's attention can be switched (or not), as transformers 5's set_attn_implementation."""

    def __init__(self, can_switch=True, loaded="sdpa"):
        text = types.SimpleNamespace(_attn_implementation=loaded)
        self.config = types.SimpleNamespace(text_config=text, sub_configs={"text_config": None},
                                            get_text_config=lambda: text, _attn_implementation=loaded)
        self.can_switch, self.calls = can_switch, []

    def set_attn_implementation(self, impl):
        self.calls.append(impl)
        if self.can_switch:
            self.config.text_config._attn_implementation = impl["text_config"] if isinstance(impl, dict) else impl


class _AttnScorer:
    def __init__(self, model, eager_shift=0.0, one_shift=None):
        self._model, self._cache, self.eager_shift, self.one_shift = model, {"stale": 1}, eager_shift, one_shift
        self.n = 0

    def _L(self, tid):
        eager = self._model.config.text_config._attn_implementation == "eager"
        self.n += 1
        shift = self.eager_shift if eager else 0.0
        if eager and self.one_shift and tid == "t0":
            shift = self.one_shift
        return types.SimpleNamespace(L=1.0 + 0.1 * int(tid[1:]) + shift)

    def score_pair(self, tid, cond, msgs, a, b):
        assert not self._cache, "scored from the cache"
        return self._L(tid)

    def score_one_token_pair(self, tid, cond, msgs, a, b):
        assert not self._cache
        return self._L(tid)


def _attn(model, **kw):
    sc = _AttnScorer(model, **kw)
    return families.attention_agreement(sc, B["gemma4-e4b-it"], _items(7, 0)[0]), sc


def test_attention_agreement_passes_within_the_tolerance(mock_score):
    model = _AttnModel()
    rec, sc = _attn(model, eager_shift=0.05)
    assert rec["ok"], rec["problems"]
    assert rec["n_items"] == 5 and len(rec["scores"]) == 10 and sc.n == 20 and not sc._cache
    assert rec["switched"] == ["eager"] and rec["restored"] == ["sdpa"] and model.calls == [{"text_config": "eager"},
                                                                                              {"text_config": "sdpa"}]
    assert families.ATTN_AGREEMENT == {"n_items": 5, "arms": ("aligned", "yn_aligned"), "yes_no_mean_abs_max": 0.10,
                                       "two_candidate_mean_abs_max": 0.50, "any_abs_max": 1.0}


@pytest.mark.parametrize("model_kw, scorer_kw, words", [
    ({}, {"eager_shift": 0.2}, "(yes_no)"),
    ({}, {"eager_shift": 0.6}, "(two_candidate)"),
    ({}, {"one_shift": 1.5}, "on one score"),
    ({"can_switch": False}, {}, "could not switch to eager"),
    ({"loaded": "eager"}, {}, "not sdpa"),
])
def test_attention_agreement_fails_on_disagreement_or_no_switch(mock_score, model_kw, scorer_kw, words):
    rec, _ = _attn(_AttnModel(**model_kw), **scorer_kw)
    assert not rec["ok"] and any(words in p for p in rec["problems"]), rec["problems"]


def test_attention_agreement_fails_if_the_switch_back_fails(mock_score):
    model = _AttnModel()
    orig = model.set_attn_implementation

    def once(impl):
        if model.calls:  # the switch back does nothing
            model.calls.append(impl)
            return
        orig(impl)

    model.set_attn_implementation = once
    rec, _ = _attn(model)
    assert not rec["ok"] and any("could not switch back" in p for p in rec["problems"])


def test_the_gate_runs_the_attention_check_for_gemma4_and_stops_on_disagreement(tmp_path, monkeypatch, mock_score):
    spec = _chat_spec(attn_agreement_check=True)
    sc = _chat_scorer(ChatTok(), spec)
    sc.mock, sc._model = False, _FakeLoaded()
    calls = []
    monkeypatch.setattr(families, "attention_agreement",
                        lambda s, sp, p: calls.append(len(p)) or {"ok": len(calls) == 1, "problems": ["eager and sdpa differ"]})
    d = tmp_path / "shard"
    d.mkdir()
    families._gate(spec, {})(d, sc, *_items(2, 1))
    assert calls == [2] and json.loads((d / "tokenization_checks.json").read_text())["attention_agreement"]["ok"]
    with pytest.raises(families.TokenizationMismatch):
        families._gate(spec, {})(d, sc, *_items(2, 1))
    rec = json.loads((d / "tokenization_checks.json").read_text())
    assert rec["ok"] is False and "eager and sdpa differ" in rec["problems"]
    plain = _chat_spec(attn_agreement_check=False)
    families._gate(plain, {})(d, sc, *_items(2, 1))
    assert calls == [2, 2]  # not run without attn_agreement_check
    assert {t for t, s in B.items() if s.attn_agreement_check} == {"gemma4-e4b-it", "gemma4-31b-it"}


# ---------------- run order and amendments ----------------


def test_run_order_is_preregistered():
    from wf.runs import RUNNERS

    order = families.RUN_ORDER_B
    assert all(r in RUNNERS for r in order) and len(set(order)) == len(order)
    assert order[:3] == ("family_envcheck_mistral-7b-v0.3", "family_envcheck_qwen3-8b", "families_envcheck")
    assert order[-1] == "family_score_gemma4-31b-it"
    assert order.index("family_nopreserve_qwen3.8-27b") == order.index("family_score_qwen3.8-27b") + 1
    small = {f"family_score_{t}" for t in ("ministral3-8b-it", "qwen3.5-9b", "qwen3.5-9b-plain", "qwen3.5-9b-base-plain",
                                            "gemma4-e4b-it")}
    assert max(order.index(r) for r in small) < order.index("family_score_qwen3.8-27b")
    assert {f"family_score_{t}" for t in B} <= set(order)
