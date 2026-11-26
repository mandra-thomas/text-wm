"""Render one sample's scored prompt through Qwen3-8B's own chat template (thinking off), as text.

The template is read from the tokenizer_config.json of the pinned snapshot (wf/constants.REVISIONS,
b968826d9c46dd6066d109eabc6255188de91218), passed as the first argument, and rendered with jinja2 the way
transformers' apply_chat_template does (add_generation_prompt=True, enable_thinking=False), so the output shows
exactly where the empty think block sits. CPU only; no model is loaded.

    python scripts/walkthroughs/alfworld_render_qwen.py /path/to/tokenizer_config.json [sample id]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import jinja2

DATA = Path(__file__).resolve().parents[2] / "notes" / "walkthroughs" / "data" / "alfworld_samples.json"


def render(template: str, messages: list[dict]) -> str:
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])

    def raise_exception(msg: str) -> None:
        raise jinja2.TemplateError(msg)

    env.globals["raise_exception"] = raise_exception
    return env.from_string(template).render(messages=messages, add_generation_prompt=True, enable_thinking=False)


def main() -> None:
    cfg = json.loads(Path(sys.argv[1]).read_text())
    sid = sys.argv[2] if len(sys.argv) > 2 else "t_002295"
    sample = next(s for s in json.loads(DATA.read_text())["samples"] if s["id"] == sid)
    print(render(cfg["chat_template"], sample["messages_two_candidate_aligned"]), end="")


if __name__ == "__main__":
    main()
