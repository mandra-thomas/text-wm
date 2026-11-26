"""Post-training pilot (notes/posttraining_design.md): checkpoints, lineages and arms.

Revisions are the Hub's `main` sha read on 2026-11-26 from https://huggingface.co/api/models/<id>, before any
download, except those already pinned in wf.constants.REVISIONS, which are reused.
"""
from __future__ import annotations

# tag -> repo, pinned revision, lineage role
CHECKPOINTS: dict[str, dict] = {
    "llama-base": {"repo": "meta-llama/Llama-3.1-8B", "revision": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
                   "stage": "base"},
    "tulu-sft": {"repo": "allenai/Llama-3.1-Tulu-3-8B-SFT", "revision": "f2a0b46b0cfda21003c6141b1ff837b7e165524d",
                 "stage": "sft"},
    "tulu-dpo": {"repo": "allenai/Llama-3.1-Tulu-3-8B-DPO", "revision": "a7beb67e33ffd01cc87ac3b46cadc1000985b8db",
                 "stage": "dpo"},
    "tulu-rl": {"repo": "allenai/Llama-3.1-Tulu-3-8B", "revision": "666943798adbde0b1aff34626007e26986a3c107",
                "stage": "rl-ppo"},
    "tulu31-grpo": {"repo": "allenai/Llama-3.1-Tulu-3.1-8B", "revision": "46239c2d07db76b412e1f1b0b4542f65b81fe01f",
                    "stage": "rl-grpo"},
    "llama-instruct": {"repo": "meta-llama/Llama-3.1-8B-Instruct",
                       "revision": "0e9e39f249a16976918f6564b8830bc894c89659", "stage": "meta-instruct"},
    "w2w-llama": {"repo": "X1AOX1A/WorldModel-Alfworld-Llama3.1-8B",
                  "revision": "8f089d2126f0406a296747041ebf33b902837cd3", "stage": "domain-sft"},
    "qwen-base": {"repo": "Qwen/Qwen3-8B-Base", "revision": "49e3418fbbbca6ecbdf9608b4d22e5a407081db4",
                  "stage": "base"},
    "qwen": {"repo": "Qwen/Qwen3-8B", "revision": "b968826d9c46dd6066d109eabc6255188de91218", "stage": "post"},
}

# lineage -> members; the first is the reference tokenizer for the token-identity gate
LINEAGES: dict[str, tuple[str, ...]] = {
    "llama": ("llama-base", "tulu-sft", "tulu-dpo", "tulu-rl", "tulu31-grpo", "llama-instruct", "w2w-llama"),
    "qwen": ("qwen-base", "qwen"),
}

OBJECT_NAMES = ("alarmclock", "apple", "baseballbat", "basketball", "book", "bowl", "box", "bread", "butterknife",
                "candle", "cd", "cellphone", "cloth", "creditcard", "cup", "dishsponge", "egg", "fork",
                "glassbottle", "handtowel", "kettle", "keychain", "knife", "ladle", "laptop", "lettuce", "mug",
                "newspaper", "pan", "pen", "pencil", "peppershaker", "pillow", "plate", "plunger", "pot", "potato",
                "remotecontrol", "saltshaker", "scrubbrush", "soapbar", "soapbottle", "spatula", "spoon",
                "spraybottle", "statue", "teddybear", "tennisracket", "tissuebox", "toiletpaper", "tomato", "vase",
                "watch", "wateringcan", "winebottle")
CARRIERS = {"assert": "There is a {obj} somewhere in the room.", "deny": "There is no {obj} in this room.",
            "use": "The following word appears in this note: {obj}.", "ignore": "Ignore the {obj}."}
