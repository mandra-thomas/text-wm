# Vendored unchanged from AgentGym (github.com/WooooDyy/AgentGym) at commit 3ef9235d23e68e7c2920c5422ad957dc8ced5c6c,
# agentenv-textcraft/agentenv_textcraft/utils.py
# (sha256 ee8528d1935c32737b1ab7ea1276bb41340ccc83a39b83887145313d614d14e4), MIT licence (LICENSE-AgentGym).
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ItemTag:
    tag: str = None
    item_id: str = None

    @property
    def name(self):
        return self.item_id or self.tag


@dataclass
class ItemTagWithCount:
    item_tag: ItemTag
    count: int


@dataclass(frozen=True)
class Recipe:
    input_items: List[ItemTagWithCount]
    output_item: ItemTagWithCount

    @property
    def recipe_str(self):
        output_str = "craft {} {} using ".format(
            self.output_item.count, item_id_to_str(self.output_item.item_tag.name)
        )
        for input_itemtag_count in self.input_items:
            output_str += "{} {}, ".format(
                input_itemtag_count.count,
                item_id_to_str(input_itemtag_count.item_tag.name),
            )
        output_str = output_str[:-2]
        return output_str


class ActionFailed(Exception):
    pass


def item_id_to_str(item_id: str):
    return item_id.replace("minecraft:", "").replace("_", " ")
