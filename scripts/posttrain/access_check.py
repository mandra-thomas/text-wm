"""Post-training pilot, step 1 (notes/posttraining_design.md): can the cluster's Hugging Face token reach every
checkpoint at its pinned revision, and what do the official tokenizer files say? CPU only; no weights are downloaded.

For every checkpoint of wf.posttrain.CHECKPOINTS:
- the Hub's model info at the pinned revision (the sha must equal the pin);
- a HEAD request on every weight file (proves access to the weights of a gated repo without downloading them; records
  the size);
- the tokenizer and config files at the pinned revision, downloaded into --cache (a few MB each), then loaded
  offline: tokenizer class, vocabulary size, special tokens, whether a BOS token is added by default, the added
  tokens above the base vocabulary, the chat template's sha256, and config facts (architecture, vocab_size,
  tie_word_embeddings, rope scaling, torch dtype).

Then, within each lineage, the token ids of a fixed set of texts (the golden item's plain-text prompt, the readout
words with and without a leading space, the success and failure texts, every ALFWorld object name with a leading
space, the pilot's carrier sentences) are compared with the lineage's reference tokenizer (the base model's), with and
without the tokenizer's default special tokens.

The token is HF_TOKEN from the environment (pace/pace.env via pace/_common.sh); it is never printed. Output: one JSON
file (--out) and a short summary on stdout.

    python scripts/posttrain/access_check.py --cache tokcache --out logs/posttrain_access.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

SMALL = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json", "*.jinja", "chat_template.json", "vocab.json", "merges.txt"]


def _status(e: BaseException) -> str:
    code = getattr(getattr(e, "response", None), "status_code", None)
    return f"HTTP {code}" if code else f"{type(e).__name__}"


OBJECT_NAMES = ("alarmclock", "apple", "baseballbat", "basketball", "book", "bowl", "box", "bread", "butterknife",
                "candle", "cd", "cellphone", "cloth", "creditcard", "cup", "dishsponge", "egg", "fork",
                "glassbottle", "handtowel", "kettle", "keychain", "knife", "ladle", "laptop", "lettuce", "mug",
                "newspaper", "pan", "pen", "pencil", "peppershaker", "pillow", "plate", "plunger", "pot", "potato",
                "remotecontrol", "saltshaker", "scrubbrush", "soapbar", "soapbottle", "spatula", "spoon",
                "spraybottle", "statue", "teddybear", "tennisracket", "tissuebox", "toiletpaper", "tomato", "vase",
                "watch", "wateringcan", "winebottle")
CARRIERS = {"assert": "There is a {obj} somewhere in the room.", "deny": "There is no {obj} in this room.",
            "use": "The following word appears in this note: {obj}.", "ignore": "Ignore the {obj}."}


def texts() -> dict[str, str]:
    """The fixed texts whose ids must agree across a lineage."""
    from scripts.qwen_audit.followup_score import plain_text

    golden = json.loads((Path(__file__).resolve().parents[2] / "tests" / "golden" / "default_prompt.json")
                        .read_text(encoding="utf-8"))
    out = {"golden_plain_aligned": plain_text(golden["aligned"]), "golden_plain_swapped": plain_text(golden["swapped"])}
    for w in ("You", "Nothing", "Yes", "No"):
        out[f"word:{w}"], out[f"word: {w}"] = w, " " + w
    out["success_text"] = " You pick up the saltshaker 1 from the shelf 2."
    out["failure_text"] = " Nothing happens."
    for o in OBJECT_NAMES:
        out[f"name: {o}"] = " " + o
    for k, c in CARRIERS.items():
        out[f"carrier:{k}"] = c.format(obj="saltshaker")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url, snapshot_download
    from transformers import AutoConfig, AutoTokenizer

    from wf.posttrain import CHECKPOINTS, LINEAGES

    api = HfApi()
    print(f"token from HF_TOKEN: {'set' if os.environ.get('HF_TOKEN') else 'not set'}", flush=True)
    res: dict = {}
    toks: dict = {}
    for tag, ck in CHECKPOINTS.items():
        r: dict = {"repo": ck["repo"], "revision": ck["revision"]}
        try:
            info = api.model_info(ck["repo"], revision=ck["revision"], files_metadata=False)
            r["sha"] = info.sha
            r["gated"] = getattr(info, "gated", None)
            r["sha_matches_pin"] = info.sha == ck["revision"]
            weights = sorted(s.rfilename for s in info.siblings if s.rfilename.endswith(".safetensors")
                             and not s.rfilename.startswith("original/"))
        except Exception as e:  # noqa: BLE001
            r["info"] = _status(e)
            res[tag] = r
            print(f"{tag:16s} {ck['repo']:44s} INFO {r['info']}", flush=True)
            continue
        sizes, wstat = {}, "OK"
        for f in weights:
            try:
                m = get_hf_file_metadata(hf_hub_url(ck["repo"], f, revision=ck["revision"]))
                sizes[f] = m.size
            except Exception as e:  # noqa: BLE001
                wstat = f"{f}: {_status(e)}"
                break
        r["weights"] = {"status": wstat, "n_files": len(weights), "bytes": sum(sizes.values())}
        try:
            path = snapshot_download(ck["repo"], revision=ck["revision"], cache_dir=args.cache, allow_patterns=SMALL)
            r["snapshot_dir_name"] = Path(path).name
            tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
            cfg = AutoConfig.from_pretrained(path, local_files_only=True)
            toks[tag] = tok
            tmpl = getattr(tok, "chat_template", None)
            base_ids = tok("Hello world", add_special_tokens=True)["input_ids"]
            plain_ids = tok("Hello world", add_special_tokens=False)["input_ids"]
            r["tokenizer"] = {
                "class": type(tok).__name__, "len": len(tok), "vocab_size_attr": tok.vocab_size,
                "bos": [tok.bos_token, tok.bos_token_id], "eos": [tok.eos_token, tok.eos_token_id],
                "pad": [tok.pad_token, tok.pad_token_id],
                "default_special_ids": [i for i in base_ids if i not in plain_ids],
                "adds_bos_by_default": len(base_ids) > len(plain_ids) and base_ids[0] == tok.bos_token_id,
                "added_tokens_above_128255": sorted(
                    [i, t.content] for i, t in tok.added_tokens_decoder.items() if i > 128255)[:40],
                "n_added_tokens": len(tok.added_tokens_decoder),
                "added_tokens_sha256": hashlib.sha256(json.dumps(sorted(
                    [i, t.content] for i, t in tok.added_tokens_decoder.items())).encode()).hexdigest()[:16],
                "chat_template_sha256": hashlib.sha256(tmpl.encode()).hexdigest()[:16] if isinstance(tmpl, str) else None,
                "chat_template_head": tmpl[:160] if isinstance(tmpl, str) else None,
            }
            r["config"] = {k: getattr(cfg, k, None) for k in ("architectures", "vocab_size", "tie_word_embeddings",
                                                                 "max_position_embeddings", "torch_dtype", "dtype",
                                                                 "bos_token_id", "eos_token_id", "pad_token_id")}
            r["config"]["rope_scaling"] = getattr(cfg, "rope_scaling", None)
            r["config"] = json.loads(json.dumps(r["config"], default=str))
        except Exception as e:  # noqa: BLE001
            r["tokenizer_error"] = f"{type(e).__name__}: {str(e)[:300]}"
        res[tag] = r
        t = r.get("tokenizer", {})
        print(f"{tag:16s} {ck['repo']:44s} sha_ok={r.get('sha_matches_pin')} gated={r.get('gated')} "
              f"weights={r['weights']['status']} ({r['weights']['n_files']} files, {r['weights']['bytes'] / 1e9:.2f} GB) "
              f"tok={t.get('class')} len={t.get('len')} bos_default={t.get('adds_bos_by_default')} "
              f"template={t.get('chat_template_sha256')}", flush=True)

    # cross-checkpoint token identity within each lineage, against the lineage's reference tokenizer
    tx = texts()
    ident: dict = {}
    for lin, members in LINEAGES.items():
        ref = members[0]
        if ref not in toks:
            ident[lin] = {"error": f"reference {ref} not loaded"}
            continue
        ident[lin] = {}
        for m in members[1:]:
            if m not in toks:
                ident[lin][m] = "not loaded"
                continue
            bad = []
            for name, s in tx.items():
                for special in (False, True):
                    a = toks[ref](s, add_special_tokens=special)["input_ids"]
                    b = toks[m](s, add_special_tokens=special)["input_ids"]
                    if a != b:
                        bad.append(f"{name} (special={special})")
            ident[lin][m] = {"n_texts": len(tx), "n_mismatch": len(bad), "mismatches": bad[:20]}
            print(f"identity {lin}: {ref} vs {m}: {len(bad)} mismatches of {2 * len(tx)}", flush=True)
    out = {"checkpoints": res, "token_identity": ident, "n_texts": len(tx),
           "golden_prompt_chars": len(tx["golden_plain_aligned"])}
    Path(args.out).write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
    ok = all(r.get("sha_matches_pin") and r.get("weights", {}).get("status") == "OK" and "tokenizer" in r
             for r in res.values())
    print("ALL OK" if ok else "SOME CHECKS FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
