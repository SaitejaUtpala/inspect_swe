"""GAIA-specific post-hoc answer repair.

Claude Code often returns the correct answer wrapped in extra explanation or
filler text. GAIA scores strictly against an answer-only format, so this repair
runs one final reformatting turn: it asks the same model to restate its final
answer in the strict GAIA format and sets ``state.output`` to that cleaned
answer for scoring.

This is intentionally scoped to GAIA + Claude Code via ``GAIA_ANSWER_REPAIR``.
Other datasets/agents should register their own repairs in ``registry.py``.
"""

from __future__ import annotations

from inspect_ai.model import ChatMessageUser, get_model
from inspect_ai.solver import TaskState

from .base import PostHocRepair

GAIA_ANSWER_FORMAT_INSTRUCTIONS = (
    "Return only your answer, which should be a number, or a short phrase with "
    "as few words as possible, or a comma separated list of numbers and/or "
    "strings.\n"
    "If the answer is a number, return only the number without any units unless "
    "specified otherwise.\n"
    "If the answer is a string, don't include articles, and don't use "
    "abbreviations (e.g. for states).\n"
    "If the answer is a comma separated list, apply the above rules to each "
    "element in the list."
)


def _build_repair_prompt(original_completion: str) -> ChatMessageUser:
    return ChatMessageUser(
        content=(
            "Your previous response may include extra explanation or filler "
            "text. Restate only your final answer, strictly following the "
            "format below. Do not add any explanation, units, or extra "
            "words.\n\n"
            f"{GAIA_ANSWER_FORMAT_INSTRUCTIONS}\n\n"
            "Previous response:\n"
            f"{original_completion}\n\n"
            "Final answer:"
        )
    )


async def repair_gaia_answer(state: TaskState) -> TaskState:
    """Run one reformatting turn so the scorer sees an answer-only completion."""
    original_completion = getattr(getattr(state, "output", None), "completion", None)
    if not isinstance(original_completion, str) or not original_completion.strip():
        return state

    messages = list(getattr(state, "messages", []) or [])
    prompt = _build_repair_prompt(original_completion)
    output = await get_model().generate(messages + [prompt])
    repaired = getattr(output, "completion", None)
    if not isinstance(repaired, str) or not repaired.strip():
        return state

    state.messages = messages + [prompt, output.message]
    state.output = output

    metadata = dict(getattr(state, "metadata", {}) or {})
    metadata["reliability_posthoc_repair"] = {
        "name": "gaia_answer_repair",
        "applied": True,
        "original_completion": original_completion,
        "repaired_completion": repaired,
    }
    state.metadata = metadata
    return state


GAIA_ANSWER_REPAIR = PostHocRepair(
    name="gaia_answer_repair",
    repair=repair_gaia_answer,
    agents=frozenset({"claude_code"}),
)
