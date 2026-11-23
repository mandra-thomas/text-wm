"""Round 3, thinking arm: is the start check's padded-vs-alone difference padding contamination or bf16 batch noise?

Run from a checkout at the freeze tag (not the frozen run checkout), with node-local Qwen3.5-9B weights. Writes one
JSON to argv[1] and prints a summary. Nothing here reads or writes a round-3 run folder, and no record is scored for
any study quantity: only next-token distributions of the first batch's prompts, generated text, and timings.
"""
import json
import sys
import time

import torch

from wf.textcraft import prompts as P
from wf.textcraft import r3 as R3
from wf.textcraft import r3_think as T
from wf.textcraft.r2_score import load_records, verified_files

OUT = sys.argv[1]
res: dict = {}


def save():
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1, default=str)


rpath, spath, _ = verified_files({"manifest": "wf/textcraft/data/r2_manifest.json"})
records = load_records(rpath)
shots = P.fewshot_examples(spath)
all_jobs = T.think_jobs(records)
jobs = all_jobs[: T.BATCH]
eng = T._Engine()
tok, model = eng.tok, eng.model
dev = model.device
texts = sorted((eng.prompt_text(R3.model_messages(R3.THINK_SPEC, r, a, shots)) for r, a in jobs), key=len)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
PAD = tok.pad_token_id
ids = [tok(t, add_special_tokens=False).input_ids for t in texts]
res["model_class"] = type(model).__name__
res["pad_token"] = [tok.pad_token, PAD]
res["prompt_lengths"] = [len(x) for x in ids]
print("model", res["model_class"], "pad", res["pad_token"], "lengths", res["prompt_lengths"], flush=True)


def last_lp(rows, masks=None, pos=None, keep=None):
    x = torch.tensor(rows, device=dev)
    m = torch.tensor(masks if masks is not None else [[1] * len(r) for r in rows], device=dev)
    kw = {"input_ids": x, "attention_mask": m}
    if pos is not None:
        kw["position_ids"] = pos
    if keep:
        kw["logits_to_keep"] = 1
    with torch.no_grad():
        lg = model(**kw).logits[:, -1].float()
    return torch.log_softmax(lg, dim=-1)


def cmp(ref, other):
    top = torch.topk(ref, 20).indices
    d = (ref[top] - other[top]).abs()
    p = ref.exp()
    kl = float((p * (ref - other)).sum())
    return {"mean_abs_top20": round(float(d.mean()), 4), "max_abs_top20": round(float(d.max()), 4),
            "kl": round(kl, 6), "top1_same": bool(ref.argmax() == other.argmax())}


def left_pad(seq, k):
    return [PAD] * k + seq, [0] * k + [1] * len(seq)


def mask_positions(mask):
    m = torch.tensor(mask, device=dev)
    return (m.long().cumsum(-1) - 1).clamp(min=0)


def suite(prefix, keep=False):
    out = {}
    for j in range(len(ids)):  # every prompt of the first batch, the shortest first
        x = ids[j]
        a = last_lp([x], keep=keep)[0]
        r = {"n_tokens": len(x), "b_alone_again": cmp(a, last_lp([x], keep=keep)[0])}
        if j == 0 or not keep:
            for bs in (2, 8):
                lb = last_lp([x] * bs, keep=keep)
                r[f"c_same_length_batch{bs}"] = cmp(a, lb[0])
            for k in (8, 32, 105, 400):
                row, m = left_pad(x, k)
                r[f"d_pad{k}_batch1"] = cmp(a, last_lp([row], [m], keep=keep)[0])
                try:
                    r[f"d_pad{k}_batch1_maskpos"] = cmp(a, last_lp([row], [m], pos=mask_positions([m]), keep=keep)[0])
                except Exception as e:  # noqa: BLE001
                    r[f"d_pad{k}_batch1_maskpos"] = f"error: {str(e)[:200]}"
        # the real check's batch: every first-batch prompt, left-padded to the longest
        L = max(len(y) for y in ids)
        rows, masks = zip(*[left_pad(y, L - len(y)) for y in ids])
        lb = last_lp(list(rows), list(masks), keep=keep)
        r["e_real_batch"] = {**cmp(a, lb[j]), "n_padded": L - len(x)}
        out[j] = r
        print(prefix, j, json.dumps(r), flush=True)
        if j >= 3 and keep:
            break
    return out


t0 = time.time()
res["bf16"] = suite("bf16")
res["bf16_seconds"] = time.time() - t0
save()

# the start check itself, as the job runs it
res["start_check_as_run"] = eng.padded_check(texts)
print("start check as run:", {k: res["start_check_as_run"][k] for k in ("n_padded", "problems")}, flush=True)
save()


# greedy continuations: does padding make the text diverge early?
def greedy(rows, masks, n=256):
    x = torch.tensor(rows, device=dev)
    m = torch.tensor(masks, device=dev)
    with torch.no_grad():
        o = model.generate(input_ids=x, attention_mask=m, do_sample=False, max_new_tokens=n, temperature=None,
                           top_p=None, top_k=None)
    return [r[x.shape[1]:].tolist() for r in o]


def first_diff(u, v):
    for i, (p, q) in enumerate(zip(u, v)):
        if p != q:
            return i
    return None if len(u) == len(v) else min(len(u), len(v))


x0 = ids[0]
base = greedy([x0], [[1] * len(x0)])[0]
gen = {"alone_text_start": tok.decode(base[:60])}
gen["alone_again"] = first_diff(base, greedy([x0], [[1] * len(x0)])[0])
gen["same_length_batch8"] = first_diff(base, greedy([x0] * 8, [[1] * len(x0)] * 8)[0])
for k in (8, 105, 400):
    row, m = left_pad(x0, k)
    gen[f"pad{k}_batch1"] = first_diff(base, greedy([row], [m])[0])
L = max(len(y) for y in ids)
rows, masks = zip(*[left_pad(y, L - len(y)) for y in ids])
real = greedy(list(rows), list(masks))
gen["real_batch_shortest"] = first_diff(base, real[0])
# the longest prompt is unpadded in the real batch: its divergence is batch composition alone
xl = ids[-1]
base_l = greedy([xl], [[1] * len(xl)])[0]
gen["real_batch_longest_unpadded"] = first_diff(base_l, real[-1])
res["greedy_first_divergence_of_256"] = gen
print("greedy first divergence (token index, None = identical over 256):", json.dumps(gen), flush=True)
save()

# timing and length for the cost of each fix: the real procedure (sampling, cap 2048, stop at </think>)
t = time.time()
outs = eng.generate(texts, 12345)
res["batch8_sampled"] = {"seconds": time.time() - t, "n_new": [n for _g, n in outs]}
print("batch of 8:", res["batch8_sampled"], flush=True)
save()
one = []
for k, tx in enumerate(texts):
    t = time.time()
    (g, n), = eng.generate([tx], 12345 + k)
    one.append({"seconds": time.time() - t, "n_new": n})
res["batch1_sampled"] = one
print("batch of 1:", one, flush=True)
save()

# length bucketing: how many exact-length groups the 4,530 thinking prompts make
lens = [len(tok(eng.prompt_text(R3.model_messages(R3.THINK_SPEC, r, a, shots)), add_special_tokens=False).input_ids)
        for r, a in all_jobs]
from collections import Counter  # noqa: E402

c = Counter(lens)
res["bucketing"] = {"n_prompts": len(lens), "n_distinct_lengths": len(c),
                    "n_batches_exact_length_of_8": sum(-(-v // T.BATCH) for v in c.values()),
                    "n_batches_as_planned": -(-len(lens) // T.BATCH), "min": min(lens), "max": max(lens)}
print("bucketing:", res["bucketing"], flush=True)
save()

# float32: if padding contaminated, the difference would stay; if it is bf16 noise, it shrinks toward zero
del outs
torch.cuda.empty_cache()
model.float()
t0 = time.time()
res["fp32"] = suite("fp32", keep=True)
res["fp32_seconds"] = time.time() - t0
save()
print("done", flush=True)
