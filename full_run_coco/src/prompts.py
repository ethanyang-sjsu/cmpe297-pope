"""The three predeclared prompt formats (single source of truth for every run).

    ordinary  standard POPE question, yes/no only
    neutral   offers presence and absence symmetrically, yes/no only
    evidence  asks for an answer grounded in what is visible, and explicitly permits
              "not visible" (= no) and "not sure" (abstention)

All prompts are built from the frozen set's `object` field (not its `question` text) and
share one generation budget, MAX_NEW_TOKENS, so token budgets are matched across formats.
The wording is fixed before the full runs; changing it afterwards invalidates them.
"""
from pope_sampling import article

MAX_NEW_TOKENS = 16

PROMPTS = {
    "ordinary": "Is there {a} {obj} in the image? Answer with yes or no.",
    "neutral": "Does the image contain {a} {obj}, or does it not? Answer with yes or no.",
    "evidence": ("Look at the image carefully and answer based only on what is visibly present. "
                 "Is there {a} {obj} in the image? Answer \"yes\" only if you can clearly see it. "
                 "If it is not visible, answer \"no\". If you cannot tell, answer \"not sure\"."),
}


def build_prompt(kind: str, obj: str) -> str:
    return PROMPTS[kind].format(a=article(obj), obj=obj)
