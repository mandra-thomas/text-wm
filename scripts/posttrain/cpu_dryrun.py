"""CPU dry run of the post-training scorer (testing only; no checkpoint weights). The real tokenizer of a checkpoint
(from the access check's tokenizer cache) is paired with a tiny random model of the same vocabulary, and the scorer's
main() runs unchanged on a few items into the given RUNS_DIR. Checks prompts, token checks, cells, run folders and
resume; the numbers mean nothing.

    RUNS_DIR=/tmp/x python scripts/posttrain/cpu_dryrun.py --tokcache tokcache --ckpt tulu-sft \
        --arms tulu-sft_plain,tulu-sft_native,tulu-sft_w2w
"""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokcache", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--limit", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from scripts.lexsem import score as ls
    from wf import posttrain as P
    from wf import posttrain_score as S

    ck = P.CHECKPOINTS[a.ckpt]
    path = snapshot_download(ck["repo"], revision=ck["revision"], cache_dir=a.tokcache, local_files_only=True)

    def fake_load(model_id):
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        cfg = AutoConfig.from_pretrained(path, local_files_only=True)
        for k, v in dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=16, max_position_embeddings=8192).items():
            setattr(cfg, k, v)
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).eval()
        return tok, model, ck["revision"]

    ls.load = fake_load
    S.strict_load_check = lambda *x, **k: {"ok": True, "dryrun": True}
    torch.cuda.synchronize = lambda *x, **k: None
    sys.argv = ["posttrain_score", "--ckpt", a.ckpt, "--arms", a.arms, "--limit", str(a.limit)]
    if a.smoke:
        sys.argv += ["--smoke", str(a.smoke)]
    if a.resume:
        sys.argv += ["--resume"]
    S.main()
    print("dry run ok", flush=True)


if __name__ == "__main__":
    main()
