"""D-PT3 step 1: read only the competence gate of llama-base_plain7 with the frozen analysis's own functions."""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from wf import posttrain_analysis as A  # noqa: E402

x = A.load_arm(Path("runs/posttrain"), "llama-base_plain7", None)
g = A.arm_gates(x)
out = {"arm": "llama-base_plain7", "status": x["status"], "commit": x["commit"], "env": x["env"],
       "n_failures": len(x["fail"]), "n_successes": len(x["succ"]), "arm_ok": g["arm_ok"], "read": g["read"],
       "gates": {ro: g["by_readout"][ro] for ro in ("first", "full")}}
Path("logs/gate_llama_base_plain7.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
print(json.dumps(out, indent=1, default=float))
