"""Prompt perturbation adapter scaffolding."""

from __future__ import annotations

from copy import copy
from hashlib import sha256
from typing import Any, Literal

from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    GenerateFilter,
    GenerateInput,
    Model,
    get_model,
)
from pydantic import BaseModel, Field


class PromptPerturbationSpec(BaseModel):
    """Prompt perturbation policy for reliability prompt phase."""

    enabled: bool = True
    mode: Literal["noop", "rewrite_v1", "rewrite_llm_v1", "disabled"] = "rewrite_llm_v1"
    variant_count: int = Field(default=1, ge=1)
    seed: int | None = None
    policy_name: str = "prompt_rewrite_llm_v1"
    rewrite_model: str | None = None
    rewrite_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    rewrite_max_tokens: int = Field(default=512, ge=64)
    rewrite_strength: Literal["mild", "medium", "strong", "naturalistic"] = "medium"

    def effective_enabled(self) -> bool:
        return self.enabled and self.mode != "disabled"

    def metadata_tags(self, *, default_seed: int) -> dict[str, str | int | bool]:
        tags: dict[str, str | int | bool] = {
            "reliability_perturbation_type": "prompt",
            "reliability_perturbation_policy_name": self.policy_name,
            "reliability_perturbation_prompt_enabled": self.effective_enabled(),
            "reliability_perturbation_prompt_mode": self.mode,
            "reliability_perturbation_prompt_variant_count": self.variant_count,
            "reliability_perturbation_prompt_seed": self.seed
            if self.seed is not None
            else default_seed,
        }
        if self.mode == "rewrite_llm_v1":
            tags["reliability_perturbation_prompt_rewrite_model"] = (
                self.rewrite_model or "active_model"
            )
            tags["reliability_perturbation_prompt_rewrite_strength"] = (
                self.rewrite_strength
            )
        return tags

    def build_generate_filter(self, *, default_seed: int) -> GenerateFilter | None:
        """Build model generate filter for prompt rewrite perturbation."""
        if not self.effective_enabled():
            return None
        if self.mode == "noop":
            return None

        seed = self.seed if self.seed is not None else default_seed
        rewrite_cache: dict[str, str] = {}

        async def _prompt_rewrite_filter(model, input, tools, tool_choice, config):
            first_user_idx = _first_user_index(input)
            if first_user_idx is None:
                return None

            message = input[first_user_idx]
            content = getattr(message, "content", None)
            if not isinstance(content, str) or not content.strip():
                return None

            variant_index = _variant_index(
                seed=seed,
                sample_key=_sample_key(input),
                variant_count=self.variant_count,
            )
            if self.mode == "rewrite_v1":
                rewritten = _rewrite_prompt_text(content, variant_index=variant_index)
            elif self.mode == "rewrite_llm_v1":
                cache_key = f"{_sample_key(input)}:{variant_index}"
                rewritten = rewrite_cache.get(cache_key)
                if rewritten is None:
                    rewritten = await _rewrite_prompt_text_llm(
                        content,
                        variant_index=variant_index,
                        model=model,
                        rewrite_model=self.rewrite_model,
                        rewrite_temperature=self.rewrite_temperature,
                        rewrite_max_tokens=self.rewrite_max_tokens,
                        rewrite_strength=self.rewrite_strength,
                    )
                    rewrite_cache[cache_key] = rewritten
            else:
                return None
            if rewritten == content:
                return None

            updated_messages = list(input)
            updated = _copy_message_with_content(message, rewritten)
            updated_messages[first_user_idx] = updated
            return GenerateInput(updated_messages, tools, tool_choice, config)

        return _prompt_rewrite_filter


def _first_user_index(messages: list[Any]) -> int | None:
    for idx, message in enumerate(messages):
        if getattr(message, "role", None) == "user":
            return idx
    return None


def _sample_key(messages: list[object]) -> str:
    chunks: list[str] = []
    for message in messages:
        role = getattr(message, "role", None) or type(message).__name__
        content = getattr(message, "content", "")
        chunks.append(f"{role}:{content!r}")
    joined = "|".join(chunks)
    return sha256(joined.encode("utf-8")).hexdigest()


def _variant_index(*, seed: int, sample_key: str, variant_count: int) -> int:
    digest = sha256(f"{seed}:{sample_key}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value % max(1, variant_count)


async def _rewrite_prompt_text_llm(
    text: str,
    *,
    variant_index: int,
    model: Model | str,
    rewrite_model: str | None,
    rewrite_temperature: float,
    rewrite_max_tokens: int,
    rewrite_strength: str,
) -> str:
    rewriter = _resolve_rewriter_model(model=model, rewrite_model=rewrite_model)
    rewrite_prompt = _build_rewrite_prompt(
        text=text,
        variant_index=variant_index,
        rewrite_strength=rewrite_strength,
    )
    output = await rewriter.generate(
        input=[
            ChatMessageSystem(content=_rewrite_system_message()),
            ChatMessageUser(content=rewrite_prompt),
        ],
        config=GenerateConfig(
            temperature=rewrite_temperature,
            max_tokens=rewrite_max_tokens,
        ),
    )
    candidate = _sanitize_rewrite(output.completion)
    return candidate if candidate else text


def _resolve_rewriter_model(*, model: Model | str, rewrite_model: str | None) -> Model:
    if rewrite_model:
        return get_model(rewrite_model)
    if isinstance(model, Model):
        # Use a fresh model handle to avoid recursively re-entering this filter.
        return get_model(model.name)
    return get_model(model)


def _rewrite_system_message() -> str:
    return (
        "You rewrite benchmark task instructions for robustness evaluation.\n"
        "Preserve exact meaning, constraints, entities, and required output format.\n"
        "Do not solve the task, do not add extra requirements, and output only the rewritten instruction."
    )


def _build_rewrite_prompt(
    *,
    text: str,
    variant_index: int,
    rewrite_strength: str,
) -> str:
    return (
        f"Rewrite strength: {rewrite_strength}\n"
        f"Variant id: {variant_index}\n\n"
        "Rewrite the instruction below with different phrasing while preserving all semantics.\n"
        "Return only the rewritten instruction.\n\n"
        "Original instruction:\n"
        f"{text}"
    )


def _sanitize_rewrite(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = [line for line in cleaned.splitlines() if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    if len(cleaned) < 8:
        return ""
    return cleaned


def _rewrite_prompt_text(text: str, *, variant_index: int) -> str:
    marker = f"[INSPECT_SWE_PROMPT_VARIANT_{variant_index}]"
    if text.startswith(marker):
        return text
    templates = [
        (
            f"{marker}\n"
            "Task (rewritten):\n"
            f"{text}\n\n"
            "Please solve carefully and provide the final answer explicitly."
        ),
        (
            f"{marker}\n"
            "Restated task for clarity:\n"
            f"{text}\n\n"
            "Keep the response concise, but verify key assumptions."
        ),
        (
            f"{marker}\n"
            "You are given this task statement:\n"
            f"{text}\n\n"
            "Reason deliberately and include only the final answer."
        ),
    ]
    return templates[variant_index % len(templates)]


def _copy_message_with_content(message: Any, content: str) -> Any:
    if hasattr(message, "model_copy"):
        updated = message.model_copy(deep=True)
    else:
        updated = copy(message)
    updated.content = content
    return updated
