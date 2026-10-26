"""Text prompts used for conditioning.

Training prompt (one per specimen), identical to the ones written by ``scripts/make_text_prompts.py``::

    "<name> exhibits a <PxTyCz> topology and a porosity of <phi>"

At generation time the model is queried with *partial* prompts that differ from the training
format::

    "with a <PxTyCz> topology"            (topology only)
    "with a porosity of <phi>"            (porosity only)
"""


def training_prompt(name: str, topology: str, porosity) -> str:
    return f"{name} exhibits a {topology} topology and a porosity of {porosity}"


def topology_prompt(topology: str) -> str:
    return f"with a {topology} topology"


def porosity_prompt(porosity) -> str:
    return f"with a porosity of {porosity}"
