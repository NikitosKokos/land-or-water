"""Model registry: which Claude models can be tested and how each one is called.

Thinking is OFF for every model. Every model gets the same one-line prompt; the only
per-model difference is the request param needed to keep thinking off.

Not included: Claude Fable 5, Fable 5.1 and Opus 5.5. Their thinking can't be turned
off (the API rejects `thinking: {type: "disabled"}` on them).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Models that think by default but accept an explicit off switch.
_THINKING_OFF: dict[str, Any] = {"thinking": {"type": "disabled"}}
# Sonnet 5.5 rejects "disabled"; "between_tools" only thinks between tool calls,
# and this benchmark sends no tools, so it never thinks.
_SONNET_55_OFF: dict[str, Any] = {"thinking": {"type": "between_tools"}}

MAX_TOKENS = 64  # one-word answers; billing is on tokens actually generated


@dataclass(frozen=True)
class ModelSpec:
    key: str  # short CLI name, also the results folder name
    model_id: str  # exact API model ID
    label: str  # name shown on the plot
    price_in: float  # USD per 1M input tokens (standard rate; batch is half)
    price_out: float  # USD per 1M output tokens
    # Extra request params. Empty = thinking is already off when `thinking` is omitted.
    params: dict[str, Any] = field(default_factory=dict)


# Registry order = plot order (roughly release order).
_ALL = [
    ModelSpec("sonnet-4.5", "claude-sonnet-4-5", "Sonnet 4.5", 3, 15),  # retires 2026-11-30
    ModelSpec("haiku-4.5", "claude-haiku-4-5", "Haiku 4.5", 1, 5),
    ModelSpec("opus-4.5", "claude-opus-4-5", "Opus 4.5", 5, 25),
    ModelSpec("opus-4.6", "claude-opus-4-6", "Opus 4.6", 5, 25),
    ModelSpec("sonnet-4.6", "claude-sonnet-4-6", "Sonnet 4.6", 3, 15),
    ModelSpec("opus-4.7", "claude-opus-4-7", "Opus 4.7", 5, 25),
    ModelSpec("opus-4.8", "claude-opus-4-8", "Opus 4.8", 5, 25),
    ModelSpec("sonnet-5", "claude-sonnet-5", "Sonnet 5", 2, 10, _THINKING_OFF),
    ModelSpec("opus-5", "claude-opus-5", "Opus 5", 5, 25, _THINKING_OFF),
    ModelSpec("sonnet-5.5", "claude-sonnet-5-5", "Sonnet 5.5", 2, 10, _SONNET_55_OFF),
    ModelSpec("haiku-5.5", "claude-haiku-5-5", "Haiku 5.5", 0.10, 0.50, _THINKING_OFF),
]

MODELS: dict[str, ModelSpec] = {m.key: m for m in _ALL}

# Models `landwater run` uses when -m is not given. Edit this list to change the default.
DEFAULT_MODELS = ["haiku-4.5", "sonnet-4.6", "opus-4.8"]

PRESETS: dict[str, list[str]] = {
    "default": DEFAULT_MODELS,
    "all": [m.key for m in _ALL],
}


def resolve(names: list[str]) -> list[ModelSpec]:
    """Expand model keys and preset names into specs, keeping order, no duplicates."""
    keys: list[str] = []
    for name in names:
        if name in PRESETS:
            keys += PRESETS[name]
        elif name in MODELS:
            keys.append(name)
        else:
            raise SystemExit(f"Unknown model or preset {name!r}. Run `landwater models` to list them.")
    return [MODELS[k] for k in dict.fromkeys(keys)]
