"""Scoring: prompts (prompts.py), the teacher-forced scorer (logodds.py) and the two parsers (parser.py).

parser.py holds two parsers. `parse_observation` reads the environment's own replies, which follow ALFWorld's
fixed templates (its one-line header describes only this one). `parse_prediction` reads a model's written
prediction, which can paraphrase; it went through six blind audits (notes/deviations.md, 2026-11-17 to
2026-11-19), and its last version, 3c25d40cc71529e2, is the one every re-read uses. That version is the hash of
parser.py's bytes (`wf.tools.parser_audit.parser_version`), so the file, header included, is left byte-for-byte
as audited; a comment edit there would change the recorded version.
"""
from .logodds import score_transition
from .parser import parse_observation
from .prompts import agent_react_messages, world_model_prompt

__all__ = [
    "agent_react_messages",
    "parse_observation",
    "score_transition",
    "world_model_prompt",
]
