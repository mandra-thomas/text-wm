"""Freeze TextCraft's recipe database into one committed file (notes/textcraft_design.md, "Environment").

Upstream builds its database from `os.listdir(recipes/)`, so the result depends on the order the filesystem lists
files: three pairs of recipes form two-way loops (ingot <-> nuggets, honey block <-> bottles), and the loader keeps
whichever direction it reads first. The frozen file lists every upstream recipe file in sorted file-name order,
minus the losing side of each loop. The rule for a loop: keep the direction that AgentGym's own released TextCraft
data shows (its training trajectories and evaluation set name only that direction), so the frozen database is the
one AgentGym's authors ran. With the losers removed, no loop remains, and the database no longer depends on order.

    python -m wf.textcraft.freeze <AgentGym checkout>     # rewrites wf/textcraft/data/recipes_frozen.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

FROZEN = Path(__file__).resolve().parent / "data" / "recipes_frozen.json"
UPSTREAM_COMMIT = "3ef9235d23e68e7c2920c5422ad957dc8ced5c6c"
RECIPES_SUBDIR = "agentenv-textcraft/agentenv_textcraft/recipes"

# (file dropped, the recipe it would add, the direction kept and the file that gives it). The released data
# (AgentTraj-L textcraft_train.json and the textcraft_test.json evaluation set) names only the kept direction:
# "craft 9 iron nugget using 1 iron ingot" 15 times and its reverse 0; "craft 1 gold ingot using 9 gold nugget"
# 72 times and its reverse 0; "craft 4 honey bottle using 1 honey block, 4 glass bottle" 4 times and its reverse 0.
LOOP_DROPS = (
    ("iron_ingot_from_nuggets.json", "craft 1 iron ingot using 9 iron nugget", "iron_nugget.json"),
    ("gold_nugget.json", "craft 9 gold nugget using 1 gold ingot", "gold_ingot_from_nuggets.json"),
    ("honey_block.json", "craft 1 honey block using 4 honey bottle", "honey_bottle.json"),
)


def freeze(agentgym_root: Path) -> dict:
    folder = agentgym_root / RECIPES_SUBDIR
    names = sorted(p.name for p in folder.iterdir() if p.suffix == ".json")
    dropped = {d for d, _, _ in LOOP_DROPS}
    missing = dropped - set(names)
    if missing:
        raise SystemExit(f"loop files not found upstream: {sorted(missing)}")
    files = [[n, json.loads((folder / n).read_text(encoding="utf-8"))] for n in names if n not in dropped]
    return {
        "source": f"AgentGym {UPSTREAM_COMMIT}, {RECIPES_SUBDIR}",
        "licence": "AgentGym: MIT. The recipe JSON is Minecraft game data (Mojang), redistributed by AgentGym and ADaPT.",
        "order": "sorted by file name",
        "loop_rule": "the direction AgentGym's released TextCraft data uses is kept; the other file is dropped",
        "dropped": [{"file": d, "recipe": r, "kept_file": k} for d, r, k in LOOP_DROPS],
        "n_upstream_files": len(names),
        "files": files,
    }


def load_frozen(path: Path = FROZEN) -> list[tuple[str, dict]]:
    """(file name, recipe JSON) in the frozen order, the input of the vendored CraftingTree."""
    data = json.loads(path.read_text(encoding="utf-8"))
    files = [(n, d) for n, d in data["files"]]
    if [n for n, _ in files] != sorted(n for n, _ in files):
        raise ValueError("frozen recipe files are not in sorted order")
    return files


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    data = freeze(Path(argv[0]))  # key order inside each recipe is kept: it fixes the order of a line's inputs
    FROZEN.parent.mkdir(parents=True, exist_ok=True)
    FROZEN.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(f"{FROZEN}: {len(data['files'])} files ({data['n_upstream_files']} upstream, {len(LOOP_DROPS)} dropped)")


if __name__ == "__main__":
    main()
