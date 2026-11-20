"""Rebuild TextCraft's frozen recipe database from AgentGym (notes/textcraft_design.md, "Environment", and §11 A6).

The recipe JSON is Minecraft game data (Mojang), so the frozen file is not committed. This script rebuilds it,
byte for byte, from AgentGym's copy at the pinned commit, and refuses to write anything whose sha256 differs from
the pinned hash (also recorded in wf/textcraft/data/manifest.json):

    python -m wf.textcraft.freeze --fetch                 # shallow, sparse git fetch of AgentGym at 3ef9235
    python -m wf.textcraft.freeze --from <AgentGym checkout at 3ef9235>

Upstream builds its database from `os.listdir(recipes/)`, so the result depends on the order the filesystem lists
files: three pairs of recipes form two-way loops (ingot <-> nuggets, honey block <-> bottles), and the loader keeps
whichever direction it reads first. The frozen file lists every upstream recipe file in sorted file-name order
(Python's sort of the names), minus the losing side of each loop. The rule for a loop: keep the direction that
AgentGym's own released TextCraft data shows (its training trajectories and evaluation set name only that
direction). With the losers removed, no loop remains, and the database no longer depends on order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

FROZEN = Path(__file__).resolve().parent / "data" / "recipes_frozen.json"
MANIFEST = FROZEN.parent / "manifest.json"
UPSTREAM_REPO = "https://github.com/WooooDyy/AgentGym.git"
UPSTREAM_COMMIT = "3ef9235d23e68e7c2920c5422ad957dc8ced5c6c"
RECIPES_SUBDIR = "agentenv-textcraft/agentenv_textcraft/recipes"
# sha256 of the file this script writes from the pinned commit; anything else is refused.
FROZEN_SHA256 = "5216d3a58a708a6181d3b948cc45e81c8f3ea6197246b4657fffd302df26d65f"
N_UPSTREAM_FILES = 860

# (file dropped, the recipe it would add, the file that gives the kept direction). The released data
# (AgentTraj-L textcraft_train.json and the textcraft_test.json evaluation set) names only the kept direction:
# "craft 9 iron nugget using 1 iron ingot" 15 times and its reverse 0; "craft 1 gold ingot using 9 gold nugget"
# 72 times and its reverse 0; "craft 4 honey bottle using 1 honey block, 4 glass bottle" 4 times and its reverse 0.
LOOP_DROPS = (
    ("iron_ingot_from_nuggets.json", "craft 1 iron ingot using 9 iron nugget", "iron_nugget.json"),
    ("gold_nugget.json", "craft 9 gold nugget using 1 gold ingot", "gold_ingot_from_nuggets.json"),
    ("honey_block.json", "craft 1 honey block using 4 honey bottle", "honey_bottle.json"),
)

MISSING = (f"{FROZEN} is missing. It is not committed (Mojang game data); rebuild it with "
           f"`python -m wf.textcraft.freeze --fetch` or `--from <AgentGym checkout at {UPSTREAM_COMMIT[:7]}>`.")


def freeze(agentgym_root: Path) -> dict:
    """The frozen database as a dict: every recipe file but the three loop losers, in sorted file-name order."""
    folder = agentgym_root / RECIPES_SUBDIR
    names = sorted(p.name for p in folder.iterdir() if p.suffix == ".json")
    dropped = {d for d, _, _ in LOOP_DROPS}
    missing = dropped - set(names)
    if missing:
        raise SystemExit(f"loop files not found upstream: {sorted(missing)}")
    # key order inside each recipe is kept (json.loads keeps it): it fixes the order of a line's inputs
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


def dumps(data: dict) -> str:
    return json.dumps(data, indent=1) + "\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def expected_sha() -> str:
    """The pinned hash; the manifest, when present, must record the same one."""
    if MANIFEST.exists():
        recorded = json.loads(MANIFEST.read_text(encoding="utf-8")).get("frozen_db_sha256")
        if recorded and recorded != FROZEN_SHA256:
            raise SystemExit(f"manifest records frozen_db_sha256 {recorded}, the script pins {FROZEN_SHA256}")
    return FROZEN_SHA256


def checkout_commit(root: Path) -> str | None:
    r = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def fetch(dest: Path) -> Path:
    """A shallow, sparse checkout of the recipe folder at the pinned commit (needs git and network)."""
    run = lambda *a: subprocess.run(["git", "-C", str(dest), *a], check=True, capture_output=True, text=True)
    dest.mkdir(parents=True, exist_ok=True)
    run("init", "-q")
    run("remote", "add", "origin", UPSTREAM_REPO)
    run("sparse-checkout", "set", "--no-cone", RECIPES_SUBDIR)
    run("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", UPSTREAM_COMMIT)
    run("checkout", "-q", "FETCH_HEAD")
    return dest


def rebuild(root: Path, out: Path = FROZEN) -> str:
    """Write the frozen file from an AgentGym checkout; refuse a wrong commit or a hash other than the pinned one."""
    commit = checkout_commit(root)
    if commit is not None and commit != UPSTREAM_COMMIT:
        raise SystemExit(f"{root} is at {commit}, not {UPSTREAM_COMMIT}")
    data = freeze(root)
    if data["n_upstream_files"] != N_UPSTREAM_FILES:
        raise SystemExit(f"{data['n_upstream_files']} recipe files upstream, expected {N_UPSTREAM_FILES}")
    text = dumps(data)
    got, want = sha256_text(text), expected_sha()
    if got != want:
        raise SystemExit(f"rebuilt database has sha256 {got}, expected {want}; nothing written")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return got


def load_frozen(path: Path = FROZEN) -> list[tuple[str, dict]]:
    """(file name, recipe JSON) in the frozen order, the input of the vendored CraftingTree."""
    if not path.exists():
        raise FileNotFoundError(MISSING)
    data = json.loads(path.read_text(encoding="utf-8"))
    files = [(n, d) for n, d in data["files"]]
    if [n for n, _ in files] != sorted(n for n, _ in files):
        raise ValueError("frozen recipe files are not in sorted order")
    return files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from", dest="root", type=Path, help="an AgentGym checkout at the pinned commit")
    src.add_argument("--fetch", action="store_true", help="fetch the recipe folder from GitHub at the pinned commit")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if args.fetch:
        with tempfile.TemporaryDirectory() as tmp:
            got = rebuild(fetch(Path(tmp) / "AgentGym"))
    else:
        got = rebuild(args.root)
    print(f"{FROZEN}: sha256 {got} (matches the pinned hash)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
