"""Reviewer (TextCraft round 3, review 1), no scores read: the two-L40S provenance path and the H3-P one-competent-model label."""
import json
import tempfile
from pathlib import Path

from wf import constants as C
from wf.provenance import run_record
from wf.textcraft import r2_analysis as A
from wf.textcraft import r3 as R3
from wf.textcraft import r3_analysis as Z

# 1. A shard that ran on two L40S: the job records both names; run_record collapses them to one.
with tempfile.TemporaryDirectory() as tmp:
    d = Path(tmp) / "shard_00_of_04"
    d.mkdir()
    (d / "git_sha").write_text("abc\n")
    (d / "jobs.jsonl").write_text(json.dumps({"gpu": ["NVIDIA L40S", "NVIDIA L40S"], "slurm_job_id": "1"}) + "\n")
    rec = run_record(d)
    print("run_record gpu:", rec["gpu"])
    print("gpu_class of the recorded list:", R3.gpu_class(rec["gpu"]))
    print("job-start check would see:", R3.gpu_class(["NVIDIA L40S", "NVIDIA L40S"]),
          R3.gpu_type_ok("qwen3.8-27b", ["NVIDIA L40S", "NVIDIA L40S"]))
    pin = C.REVISIONS["Qwen/Qwen3.8-27B"]
    probs = Z.provenance_problems({"qwen3.8-27b": [d]}, {"qwen3.8-27b": [{"model_revision": pin}]})
    print("analysis provenance problems:", probs)

# 2. H3-P with one competent core model, supported; the other two not competent for P.
V = lambda **kw: {c: kw.get(c, "inconclusive") for c in A.CONTRASTS}  # noqa: E731
v = {"a": V(parent_penalty="confirmed below zero"), "b": V(), "c": V()}
comp = {"a": True, "b": False, "c": False}
r2 = Z.r2_contemporary(v, comp)
print("H3-P, one competent model supported:", Z.h3p_reading(v, comp, {t: False for t in v}, r2))
