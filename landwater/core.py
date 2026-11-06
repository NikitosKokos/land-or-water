"""Grid, prompt, answer parsing, ground truth and scoring. No network here."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from .models import ModelSpec

PROMPT = (
    "Is the point at latitude {lat}, longitude {lon} on land or in water?\n"
    "Answer with exactly one word: Land or Water."
)

# Cell codes used in prediction grids.
WATER, LAND, UNKNOWN, MISSING = 0, 1, 2, 3


# --------------------------------------------------------------------------- grid


def grid(step: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre latitudes (north to south) and longitudes (west to east).

    step=2 gives 90 x 180 = 16,200 points: lat 89..-89, lon -179..179.
    """
    n_lat = 180 / step if step > 0 else 0.5
    if abs(n_lat - round(n_lat)) > 1e-9 or n_lat < 1:
        raise ValueError(f"step must divide 180 evenly, got {step}")
    n_lat = round(n_lat)
    lats = 90 - step / 2 - step * np.arange(n_lat)
    lons = -180 + step / 2 + step * np.arange(2 * n_lat)
    return lats, lons


def point_id(i: int, j: int) -> str:
    """Batch custom_id for grid cell (row i, col j). Must match ^[a-zA-Z0-9_-]{1,64}$."""
    return f"r{i}c{j}"


def parse_point_id(pid: str) -> tuple[int, int]:
    row, col = pid[1:].split("c")
    return int(row), int(col)


def points(step: float) -> list[tuple[str, float, float]]:
    lats, lons = grid(step)
    return [
        (point_id(i, j), float(lat), float(lon))
        for i, lat in enumerate(lats)
        for j, lon in enumerate(lons)
    ]


def fmt_num(x: float) -> str:
    return f"{round(x, 6):g}"


# ------------------------------------------------------------------------ request


def prompt(lat: float, lon: float) -> str:
    return PROMPT.format(lat=fmt_num(lat), lon=fmt_num(lon))


def request_params(spec: ModelSpec, lat: float, lon: float) -> dict[str, Any]:
    """Messages API params for one question. No system prompt, no tools, no images."""
    return {
        "model": spec.model_id,
        "max_tokens": spec.max_tokens,
        "messages": [{"role": "user", "content": prompt(lat, lon)}],
        **spec.params,
    }


def used_thinking(message: Any) -> bool:
    """True if the reply has any thinking block (even an empty, display-omitted one)."""
    return any(block.type in ("thinking", "redacted_thinking") for block in message.content)


def message_record(pid: str, message: Any, batch: bool) -> dict[str, Any]:
    """Compact JSON-able record of one answer. Raw text is kept so parsing can change later."""
    text = "".join(block.text for block in message.content if block.type == "text")
    return {
        "id": pid,
        "text": text,
        "thought": used_thinking(message),
        "stop": message.stop_reason,
        "in": message.usage.input_tokens,
        "out": message.usage.output_tokens,
        "batch": batch,
    }


# ------------------------------------------------------------------------ parsing

_WORD = re.compile(r"\b(land|water)\b", re.IGNORECASE)
_THINK_TAGS = re.compile(r"<thinking>.*?(?:</thinking>|$)", re.IGNORECASE | re.DOTALL)


def _code(word: str) -> int:
    return LAND if word.lower() == "land" else WATER


def parse_answer(text: str | None) -> int:
    """Map a reply to LAND / WATER / UNKNOWN."""
    if not text:
        return UNKNOWN
    text = _THINK_TAGS.sub(" ", text)
    found = {w.lower() for w in _WORD.findall(text)}
    if len(found) == 1:
        return _code(found.pop())
    if not found:
        return UNKNOWN
    # Both words appear (e.g. a hedge, then a verdict): trust the last line naming only one.
    for line in reversed(text.strip().splitlines()):
        found = {w.lower() for w in _WORD.findall(line)}
        if len(found) == 1:
            return _code(found.pop())
    return UNKNOWN


def prediction_grid(records: dict[str, dict], step: float) -> np.ndarray:
    lats, lons = grid(step)
    pred = np.full((len(lats), len(lons)), MISSING, dtype=np.int8)
    for pid, rec in records.items():
        i, j = parse_point_id(pid)
        pred[i, j] = parse_answer(rec.get("text"))
    return pred


# -------------------------------------------------------------------- ground truth


@lru_cache(maxsize=8)
def truth(step: float) -> np.ndarray:
    """Boolean land mask at the grid points, from the 1-km GEBCO-derived `global-land-mask`."""
    from global_land_mask import globe

    lats, lons = grid(step)
    lon2, lat2 = np.meshgrid(lons, lats)
    return np.asarray(globe.is_land(lat2, lon2), dtype=bool)


# ------------------------------------------------------------------------ scoring


@dataclass(frozen=True)
class Score:
    accuracy: float  # area-weighted (cos lat), over asked cells; unclear answers count wrong
    accuracy_flat: float  # every cell weighs the same
    land_recall: float  # share of true-land area answered "land"
    water_recall: float  # share of true-water area answered "water"
    coverage: float  # share of grid cells asked so far
    n_asked: int
    n_unknown: int


def area_weights(step: float) -> np.ndarray:
    lats, lons = grid(step)
    return np.repeat(np.cos(np.radians(lats))[:, None], len(lons), axis=1)


def score(pred: np.ndarray, step: float) -> Score | None:
    is_land = truth(step)
    w = area_weights(step)
    asked = pred != MISSING
    if not asked.any():
        return None
    correct = asked & (pred == np.where(is_land, LAND, WATER))

    def frac(mask: np.ndarray, of: np.ndarray, weights: np.ndarray) -> float:
        total = (weights * of).sum()
        return float((weights * mask).sum() / total) if total else float("nan")

    ones = np.ones_like(w)
    return Score(
        accuracy=frac(correct, asked, w),
        accuracy_flat=frac(correct, asked, ones),
        land_recall=frac(correct & is_land, asked & is_land, w),
        water_recall=frac(correct & ~is_land, asked & ~is_land, w),
        coverage=float(asked.mean()),
        n_asked=int(asked.sum()),
        n_unknown=int((pred == UNKNOWN).sum()),
    )
