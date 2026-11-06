# Land or Water?

Ask a Claude model "Land or Water?" for one latitude/longitude at a time, typed as text, at
every 2° of the globe (16,200 points). Plot the answers. What comes out is the world map the
model learned from text alone.

No images, no tools, no system prompt, **no thinking**. Each question is a separate request:

```
Is the point at latitude 41, longitude -73 on land or in water?
Answer with exactly one word: Land or Water.
```

## Setup

```bash
uv sync                                   # or: pip install -e .
```

Paste your API key into the `.env` file in the project root (it is git-ignored):

```
ANTHROPIC_API_KEY=sk-ant-...
```

An `ANTHROPIC_API_KEY` environment variable also works and takes priority over `.env`.

## Usage

```bash
uv run landwater models                       # list models; [default] marks the 3 test models
uv run landwater run --step 10 --live         # quick smoke test: 648 points x 3 models, a few cents
uv run landwater run --dry-run                # plan + cost estimate for the full 16,200-point grid
uv run landwater run                          # full run via the Batch API (asks before spending)
uv run landwater report                       # accuracy table + actual spend
uv run landwater plot --truth                 # plots/earth_step2.png
uv run landwater plot --diff                  # error maps: orange = said land, blue = said water
```

- Default models: **Haiku 4.5, Sonnet 4.6, Opus 4.8** (`DEFAULT_MODELS` in
  [landwater/models.py](landwater/models.py)). Pick others with `-m`, e.g. `-m opus-5 sonnet-5.5`,
  or `-m all`.
- `--step` sets the grid spacing (must divide 180). `2` gives 16,200 points, `1` gives 64,800, `10` gives 648.
- `run` defaults to the **Message Batches API**: half price, usually done within an hour
  (up to 24h). `--no-wait` submits and exits; run the same command again later to collect.
- `--live` sends requests directly (full price, `--concurrency 16`). Good for small grids.
- Everything is resumable. Answers go to `results/<model>/step<N>.jsonl` as they arrive.
  Ctrl+C, then run the same command again: only unanswered points are asked.

## Thinking is off

| Models | How thinking is turned off |
|---|---|
| Sonnet 4.5, Haiku 4.5, Opus 4.5 to 4.8, Sonnet 4.6 | off by default when `thinking` is omitted |
| Sonnet 5, Opus 5, Haiku 5.5 | `thinking: {type: "disabled"}` |
| Sonnet 5.5 | `thinking: {type: "between_tools"}` (no tools are sent, so it never thinks) |

**Not supported:** Fable 5, Fable 5.1 and Opus 5.5. Their thinking can't be turned off (the API
rejects it), so they are left out.

Checks: the first request for each model is a live probe; if the reply contains a thinking block,
that model is skipped. Every saved answer records `"thought": true/false`, and `report` shows the
count in the `thought` column (should be 0).

## Other choices

- **Sampling**: default temperature for every model (the current API version doesn't allow
  changing it). Each point is asked once, so maps include sampling noise.
- **Refusal fallbacks are off on purpose.** A fallback answers with a different model, which
  would contaminate that model's map. Refusals are recorded and shown as gray "no clear answer" cells.
- **Probe request**: before a model's batch is submitted, one live request checks that the model
  ID and params are valid, so a bad config fails once, not 16,200 times.
- **Parsing**: raw reply text is stored; `land`/`water` is parsed when reports and plots are made, so
  the parser can change without re-asking. Replies naming neither or both words count as unclear.

## Scoring

Ground truth is [`global-land-mask`](https://pypi.org/project/global-land-mask/) (1-km grid from
GEBCO), sampled at each exact grid point. Accuracy is **area-weighted** by cos(latitude), since a
2° cell near the pole covers much less of the Earth than one at the equator. Unclear answers and
refusals count as wrong. `report` also shows unweighted accuracy, land/water recall and actual cost
from logged token usage.

## Cost

Actual API bill for this project with the three default models (Haiku 4.5, Sonnet 4.6, Opus 4.8),
thinking off: **$5 in total**.

![Anthropic Console billing: Opus 4.8 $2.93, Sonnet 4.6 $1.57, Haiku 4.5 $0.49, $5 billed total](assets/cost.png)

That covers two runs: a `--step 10 --live` smoke test, then the full 2° grid through the Batch API.
Per-run split, from the token usage `landwater report` logs:

| Model | Full grid (16,200 points, batch) | Smoke test (648 points, live) | Billed |
|---|---|---|---|
| Opus 4.8 | $2.72 | $0.22 | $2.93 |
| Sonnet 4.6 | $1.46 | $0.12 | $1.57 |
| Haiku 4.5 | $0.46 | $0.04 | $0.49 |
| **Total** | **$4.64** | **$0.38** | **$5** |

Each answer is about 36-46 input tokens and 4-5 output tokens. `run` prints an estimate and asks
before spending; `uv run landwater run --dry-run` shows the estimate only.

## Layout

```
landwater/
  models.py   model registry, thinking-off params, default models
  core.py     grid, prompt, answer parsing, ground truth, scoring (no network)
  runner.py   batch + live runners, probe, cost estimate
  store.py    JSONL results + batch bookkeeping
  plot.py     small-multiples figure, error view, per-model pixel maps
  cli.py      `landwater` command, .env loading
tests/        offline tests (uv run pytest)
```
