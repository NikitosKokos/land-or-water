import os
import re

import numpy as np
import pytest

from landwater.core import (
    LAND,
    MISSING,
    UNKNOWN,
    WATER,
    grid,
    parse_answer,
    parse_point_id,
    point_id,
    points,
    prediction_grid,
    prompt,
    request_params,
    score,
    truth,
)
from landwater.models import MODELS, PRESETS, resolve


def test_grid_step2_is_16200_points():
    lats, lons = grid(2)
    assert lats.size * lons.size == 16_200
    assert (lats[0], lats[-1]) == (89, -89)
    assert (lons[0], lons[-1]) == (-179, 179)


@pytest.mark.parametrize("step", [0, -2, 7])
def test_grid_rejects_bad_step(step):
    with pytest.raises(ValueError):
        grid(step)


def test_point_ids_roundtrip_and_are_valid_custom_ids():
    for pid, _, _ in points(10):
        assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", pid)
    assert parse_point_id(point_id(89, 179)) == (89, 179)


def test_prompt_format():
    assert prompt(-89.0, 179.0) == (
        "Is the point at latitude -89, longitude 179 on land or in water?\n"
        "Answer with exactly one word: Land or Water."
    )
    assert "latitude 0.5," in prompt(0.5, 1.5)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Land", LAND),
        ("water", WATER),
        ("**Water.**", WATER),
        ("Water (Pacific Ocean)", WATER),
        ("Greenland", UNKNOWN),  # word boundary: not "land"
        ("", UNKNOWN),
        (None, UNKNOWN),
        ("I can't tell.", UNKNOWN),
        ("It could be land or water.\nWater", WATER),
        ("<thinking>maybe land</thinking>Water", WATER),
        ("Land or water", UNKNOWN),
    ],
)
def test_parse_answer(text, expected):
    assert parse_answer(text) == expected


def test_truth_known_points():
    t = truth(2)
    lats, lons = grid(2)
    i = int(np.where(lats == 1)[0][0])
    # (1N, 179W) mid Pacific -> water; (1N, 21E) Congo basin -> land.
    assert not t[i, int(np.where(lons == -179)[0][0])]
    assert t[i, int(np.where(lons == 21)[0][0])]


def test_score_perfect_and_inverted():
    t = truth(10)
    perfect = np.where(t, LAND, WATER).astype(np.int8)
    assert score(perfect, 10).accuracy == pytest.approx(1.0)
    inverted = np.where(t, WATER, LAND).astype(np.int8)
    assert score(inverted, 10).accuracy == pytest.approx(0.0)


def test_score_area_weighting_and_unknowns():
    t = truth(10)
    pred = np.where(t, LAND, WATER).astype(np.int8)
    pred[0, :] = UNKNOWN  # polar row: small area, many cells
    s = score(pred, 10)
    assert s.accuracy > s.accuracy_flat  # wrong cells are near the pole, so they weigh less
    assert s.n_unknown == pred.shape[1]


def test_score_ignores_missing_cells():
    pred = np.full(grid(10)[0].shape + grid(10)[1].shape, MISSING, dtype=np.int8)
    assert score(pred, 10) is None
    t = truth(10)
    pred[5] = np.where(t[5], LAND, WATER)
    s = score(pred, 10)
    assert s.accuracy == pytest.approx(1.0)
    assert s.coverage == pytest.approx(1 / pred.shape[0])


def test_prediction_grid_from_records():
    recs = {"r0c0": {"id": "r0c0", "text": "Water"}, "r1c2": {"id": "r1c2", "text": "Land"}}
    pred = prediction_grid(recs, 10)
    assert pred[0, 0] == WATER and pred[1, 2] == LAND and pred[0, 1] == MISSING


def test_request_params_per_model():
    for spec in MODELS.values():
        p = request_params(spec, 1.0, 2.0)
        assert p["model"] == spec.model_id
        assert "temperature" not in p and "system" not in p and "tools" not in p
        if spec.thinks:
            # Thinking can't be disabled: lowest effort, room for thinking tokens.
            assert p["output_config"] == {"effort": "low"} and "thinking" not in p
            assert p["max_tokens"] >= 1024
        else:
            assert p["max_tokens"] <= 64 and "output_config" not in p
            assert p.get("thinking", {"type": "disabled"})["type"] in {"disabled", "between_tools"}
    assert request_params(MODELS["sonnet-5.5"], 0, 0)["thinking"] == {"type": "between_tools"}
    assert "thinking" not in request_params(MODELS["opus-4.8"], 0, 0)  # off when omitted


def test_only_models_that_cannot_disable_thinking_think():
    assert {m.model_id for m in MODELS.values() if m.thinks} == {
        "claude-fable-5", "claude-fable-5-1", "claude-opus-5-5"}


def test_presets_resolve():
    default = ["haiku-4.5", "sonnet-4.6", "opus-4.8"]
    assert [m.key for m in resolve([])] == []
    assert [m.key for m in resolve(["default"])] == default
    assert [m.key for m in resolve(["haiku-4.5", "default"])] == default
    figure = resolve(["figure"])
    assert len(figure) == 12 and figure[0].key == "sonnet-4.5" and figure[-1].key == "opus-5.5"
    assert not any(m.thinks for m in resolve(["no-thinking"]))
    assert all(m.thinks for m in resolve(["thinking"]))
    assert set(PRESETS["all"]) == set(MODELS)
    with pytest.raises(SystemExit):
        resolve(["gpt-4"])


def test_dotenv(tmp_path, monkeypatch):
    from landwater import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LW_TEST", raising=False)
    (tmp_path / ".env").write_text('# comment\nANTHROPIC_API_KEY=\nLW_TEST="sk-ant-xyz"\n', encoding="utf-8")
    cli.load_dotenv()
    assert "ANTHROPIC_API_KEY" not in os.environ  # empty value ignored
    assert os.environ["LW_TEST"] == "sk-ant-xyz"
