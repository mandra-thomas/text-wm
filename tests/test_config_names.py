"""A run's folder is its config file's name (`load_config`), so names written inside configs must agree (2026-11-18).

The RQ2 smoke run found scoring configs whose `run_id` differed from the file name: scoring wrote to the
file name, and the analysis looked for the `run_id` and found nothing.
"""
from __future__ import annotations

from pathlib import Path

import yaml

CONFIGS = Path(__file__).resolve().parents[1] / "wf" / "config"


def _all():
    return {p.stem: yaml.safe_load(p.read_text(encoding="utf-8")) or {} for p in CONFIGS.glob("*.yaml")}


def test_run_id_in_a_config_is_its_file_name():
    wrong = {name: c["run_id"] for name, c in _all().items() if "run_id" in c and c["run_id"] != name}
    assert not wrong


def test_every_source_and_sources_entry_names_a_config():
    configs = _all()
    missing = []
    for name, c in configs.items():
        refs = ([c["source"]] if isinstance(c.get("source"), str) else []) + list(c.get("sources") or [])
        missing += [(name, r) for r in refs if r not in configs]
    assert not missing
