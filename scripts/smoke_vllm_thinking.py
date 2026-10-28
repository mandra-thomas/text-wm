#!/usr/bin/env python3
"""Confirm Qwen3 thinking is disabled on a vLLM OpenAI-compatible server."""
from __future__ import annotations

import argparse
import sys

from openai import OpenAI


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--n", type=int, default=20)
    args = p.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    bad = 0
    for i in range(args.n):
        r = client.chat.completions.create(
            model=args.model,
            temperature=0,
            max_tokens=64,
            messages=[{"role": "user", "content": "Say hello in one short sentence."}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        text = r.choices[0].message.content or ""
        if "<think>" in text or "</think>" in text:
            bad += 1
            print(f"[{i}] THINK BLOCK:\n{text[:200]}")
    print(f"thinking_blocks={bad}/{args.n}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
