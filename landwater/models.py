"""Model registry: which Claude models can be tested and how each one is called.

Every model gets the same one-line prompt. The only per-model difference is `params`,
the extra request fields that control thinking.

To experiment, edit this file:
  - DEFAULT_MODELS: what `landwater run` uses when no -m is given.
  - PRESETS: named groups for -m (e.g. `-m figure`).
  - _ALL: add a variant under a new key, e.g. a higher-effort copy of a model.
    Results are stored per key, so variants never overwrite each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# --- thinking settings -------------------------------------------------------------
# Models that think by default but accept an explicit off switch.
THINKING_OFF: dict[str, Any] = {"thinking": {"type": "disabled"}}
# Sonnet 5.5 rejects "disabled"; "between_tools" only thinks between tool calls,
# and this benchmark sends no tools, so it never thinks.
SONNET_55_OFF: dict[str, Any] = {"thinking": {"type": "between_tools"}}
# Fable 5, Fable 5.1 and Opus 5.5 always think (the API rejects turning it off).
# Effort is the only control; "low" keeps the thinking short.
EFFORT_LOW: dict[str, Any] = {"output_config": {"effort": "low"}}


@dataclass(frozen=True)
class ModelSpec:
    key: str  # short CLI name, also the results folder name
    model_id: str  # exact API model ID
    label: str  # name shown on the plot
    price_in: float  # USD per 1M input tokens (standard rate; batch is half)
    price_out: float  # USD per 1M output tokens
    # Extra request params. Empty = thinking is already off when `thinking` is omitted.
    params: dict[str, Any] = field(default_factory=dict)
    # True for models whose thinking can't be turned off: starred on the plot, given
    # room for thinking tokens, and exempt from the "no thinking" check.
    thinks: bool = False

    @property
    def max_tokens(self) -> int:
        # Thinking tokens count toward max_tokens. Billing is on tokens actually generated.
        return 4096 if self.thinks else 64

    @property
    def thinking_note(self) -> str:
        if self.thinks:
            effort = self.params.get("output_config", {}).get("effort", "default")
            return f"always on (effort={effort})"
        if "thinking" in self.params:
            return f"off (thinking={self.params['thinking']['type']})"
        return "off (default)"


# Registry order = plot order when no -m is given.
_ALL = [
    ModelSpec("sonnet-4.5", "claude-sonnet-4-5", "Sonnet 4.5", 3, 15),  # retires 2026-11-30
    ModelSpec("haiku-4.5", "claude-haiku-4-5", "Haiku 4.5", 1, 5),
    ModelSpec("opus-4.5", "claude-opus-4-5", "Opus 4.5", 5, 25),
    ModelSpec("opus-4.6", "claude-opus-4-6", "Opus 4.6", 5, 25),
    ModelSpec("sonnet-4.6", "claude-sonnet-4-6", "Sonnet 4.6", 3, 15),
    ModelSpec("opus-4.7", "claude-opus-4-7", "Opus 4.7", 5, 25),
    ModelSpec("opus-4.8", "claude-opus-4-8", "Opus 4.8", 5, 25),
    ModelSpec("fable-5", "claude-fable-5", "Fable 5", 10, 50, EFFORT_LOW, thinks=True),
    ModelSpec("sonnet-5", "claude-sonnet-5", "Sonnet 5", 2, 10, THINKING_OFF),
    ModelSpec("opus-5", "claude-opus-5", "Opus 5", 5, 25, THINKING_OFF),
    ModelSpec("fable-5.1", "claude-fable-5-1", "Fable 5.1", 10, 50, EFFORT_LOW, thinks=True),
    ModelSpec("opus-5.5", "claude-opus-5-5", "Opus 5.5", 4, 20, EFFORT_LOW, thinks=True),
    ModelSpec("sonnet-5.5", "claude-sonnet-5-5", "Sonnet 5.5", 2, 10, SONNET_55_OFF),
    ModelSpec("haiku-5.5", "claude-haiku-5-5", "Haiku 5.5", 0.10, 0.50, THINKING_OFF),
    # Example variant (uncomment to try): same model, more thinking, separate results.
    # ModelSpec("opus-5.5-high", "claude-opus-5-5", "Opus 5.5 high", 4, 20,
    #           {"output_config": {"effort": "high"}}, thinks=True),
]

MODELS: dict[str, ModelSpec] = {m.key: m for m in _ALL}

# Models `landwater run` uses when -m is not given. Edit this list to change the default.
DEFAULT_MODELS = ["haiku-4.5", "sonnet-4.6", "opus-4.8"]

PRESETS: dict[str, list[str]] = {
    "default": DEFAULT_MODELS,
    # The 12-model lineup of the "How 12 blind Claudes see the Earth" figure.
    "figure": [
        "sonnet-4.5", "haiku-4.5", "opus-4.5",
        "opus-4.6", "sonnet-4.6", "opus-4.7",
        "opus-4.8", "fable-5", "sonnet-5",
        "opus-5", "fable-5.1", "opus-5.5",
    ],
    "no-thinking": [m.key for m in _ALL if not m.thinks],
    "thinking": [m.key for m in _ALL if m.thinks],
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
