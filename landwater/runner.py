"""Ask the questions: Message Batches API (default, half price) or live async requests.

Both paths are resumable: answered points live in results/<model>/stepN.jsonl and are
never asked again. Re-run the same command to retry failures or resume after Ctrl+C.

Thinking is off for every model except the ones marked `thinks=True` in models.py (they
can't turn it off). Each record stores "thought": true/false so `landwater report` can prove it.

No server-side refusal fallbacks are used on purpose: a fallback would answer with a
different model and contaminate that model's map. Refusals are recorded as-is and
show up as "unclear" cells.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections import Counter
from dataclasses import dataclass, field

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from . import store
from .core import message_record, points, request_params, used_thinking
from .models import ModelSpec

BATCH_CHUNK = 50_000  # API limit is 100k requests / 256 MB per batch
# Measured on the step-10 smoke test: 36 input tokens (46 on Opus 4.7+ tokenizer), ~4-5 output.
EST_IN_TOKENS = 46
EST_OUT_TOKENS = 5
EST_THINKING_TOKENS = 250  # guess for effort=low on thinking models; run a small --step first

# Errors that mean "this model/request config is wrong" - retrying won't help.
_CONFIG_ERRORS = (anthropic.BadRequestError, anthropic.PermissionDeniedError, anthropic.NotFoundError)


@dataclass
class Plan:
    spec: ModelSpec
    todo: list[tuple[str, float, float]]
    pending: list[str] = field(default_factory=list)  # uncollected batch IDs
    has_records: bool = False


def make_plans(specs: list[ModelSpec], step: float) -> list[Plan]:
    all_points = points(step)
    plans = []
    for spec in specs:
        done = store.load_records(spec.key, step)
        todo = [p for p in all_points if p[0] not in done]
        pending = [b["id"] for b in store.load_batches(spec.key, step) if not b["collected"]]
        plans.append(Plan(spec, todo, pending, has_records=bool(done)))
    return plans


def estimate_usd(spec: ModelSpec, n: int, batch: bool) -> float:
    out = EST_OUT_TOKENS + (EST_THINKING_TOKENS if spec.thinks else 0)
    usd = n * (EST_IN_TOKENS * spec.price_in + out * spec.price_out) / 1e6
    return usd * (0.5 if batch else 1.0)


def print_plan(plans: list[Plan], batch: bool) -> float:
    total = 0.0
    print(f"{'model':<12} {'to ask':>8} {'est. USD':>9}  note")
    for p in plans:
        if p.pending:
            print(f"{p.spec.key:<12} {'-':>8} {'-':>9}  {len(p.pending)} batch(es) in flight, will collect")
            continue
        usd = estimate_usd(p.spec, len(p.todo), batch)
        total += usd
        note = "always thinks: rough estimate" if p.spec.thinks else ""
        print(f"{p.spec.key:<12} {len(p.todo):>8,} {usd:>9.2f}  {note}")
    mode = "batch, 50% off" if batch else "live"
    print(f"{'total':<12} {'':>8} {total:>9.2f}  ({mode})")
    return total


def probe(client: anthropic.Anthropic, plan: Plan, step: float) -> bool:
    """Send one real request first so a bad config fails fast instead of 16k times.

    Also enforces "thinking off": a model not marked `thinks=True` that returns a thinking
    block is skipped. A good answer is saved, so it isn't wasted.
    """
    pid, lat, lon = plan.todo[0]
    try:
        msg = client.messages.create(**request_params(plan.spec, lat, lon))
    except _CONFIG_ERRORS as e:
        print(f"  {plan.spec.key}: skipped - {type(e).__name__}: {e.message}")
        return False
    if used_thinking(msg) and not plan.spec.thinks:
        print(f"  {plan.spec.key}: skipped - model returned a thinking block; thinking must be off. "
              "Fix its params in landwater/models.py.")
        return False
    store.append_records(plan.spec.key, step, [message_record(pid, msg, batch=False)])
    plan.todo = plan.todo[1:]
    plan.has_records = True
    return True


# -------------------------------------------------------------------------- batch


def _submit(client: anthropic.Anthropic, plan: Plan, step: float) -> bool:
    """Submit plan.todo as one or more batches. Returns False if the batch queue is full."""
    batches = store.load_batches(plan.spec.key, step)
    for start in range(0, len(plan.todo), BATCH_CHUNK):
        chunk = plan.todo[start : start + BATCH_CHUNK]
        requests = [
            Request(custom_id=pid, params=MessageCreateParamsNonStreaming(**request_params(plan.spec, lat, lon)))
            for pid, lat, lon in chunk
        ]
        try:
            batch = client.messages.batches.create(requests=requests)
        except anthropic.RateLimitError as e:
            print(f"  {plan.spec.key}: batch queue limit reached ({e.message}). "
                  "Re-run after current batches finish.")
            return False
        batches.append({"id": batch.id, "collected": False, "size": len(chunk)})
        store.save_batches(plan.spec.key, step, batches)  # save right away: survives Ctrl+C
        plan.pending.append(batch.id)
        print(f"  {plan.spec.key}: submitted {batch.id} ({len(chunk):,} requests)")
    plan.todo = []
    return True


def _collect(client: anthropic.Anthropic, plan: Plan, step: float, batch_id: str) -> None:
    outcomes: Counter[str] = Counter()
    records = []
    for result in client.messages.batches.results(batch_id):
        outcomes[result.result.type] += 1
        if result.result.type == "succeeded":
            records.append(message_record(result.custom_id, result.result.message, batch=True))
    store.append_records(plan.spec.key, step, records)

    batches = store.load_batches(plan.spec.key, step)
    for b in batches:
        if b["id"] == batch_id:
            b["collected"] = True
    store.save_batches(plan.spec.key, step, batches)

    failed = sum(n for kind, n in outcomes.items() if kind != "succeeded")
    msg = f"  {plan.spec.key}: collected {batch_id}: {outcomes['succeeded']:,} answers"
    if failed:
        msg += f", {failed:,} not answered ({dict(outcomes)}) - re-run to retry them"
    print(msg)


def run_batch(plans: list[Plan], step: float, wait: bool, poll_seconds: int) -> None:
    client = anthropic.Anthropic()

    for plan in plans:
        if plan.pending or not plan.todo:
            continue
        if not plan.has_records and not probe(client, plan, step):
            continue
        if plan.todo and not _submit(client, plan, step):
            break  # queue full: don't try the remaining models now

    in_flight = [(plan, bid) for plan in plans for bid in plan.pending]
    if not in_flight:
        print("Nothing in flight.")
        return
    if not wait:
        print("Submitted. Re-run the same command later to collect results.")
        return

    print(f"Waiting for {len(in_flight)} batch(es), checking every {poll_seconds}s.\n"
          "Batches queue first, so counts can stay at 0 for a while; most finish within an hour,\n"
          "at most 24h. Ctrl+C is safe: the batches keep running, re-run this command to collect.")
    started = time.monotonic()
    while in_flight:
        still, progress = [], []
        for plan, bid in in_flight:
            batch = client.messages.batches.retrieve(bid)
            if batch.processing_status == "ended":
                _collect(client, plan, step, bid)
            else:
                c = batch.request_counts
                done = c.succeeded + c.errored + c.canceled + c.expired
                progress.append(f"{plan.spec.key} {done:,}/{done + c.processing:,}")
                still.append((plan, bid))
        in_flight = still
        if in_flight:
            minutes = (time.monotonic() - started) / 60
            print(f"  [{minutes:4.0f} min] " + " | ".join(progress))
            time.sleep(poll_seconds)
    print("All batches collected.")


# --------------------------------------------------------------------------- live


async def _live_model(client: anthropic.AsyncAnthropic, plan: Plan, step: float, concurrency: int) -> None:
    todo = list(plan.todo)
    # Random order: an interrupted run still gives an evenly sampled (noisy) map.
    random.Random(0).shuffle(todo)
    sem = asyncio.Semaphore(concurrency)
    errors: Counter[str] = Counter()
    done = 0
    path = store.results_path(plan.spec.key, step)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:

        async def ask(pid: str, lat: float, lon: float) -> None:
            nonlocal done
            async with sem:
                try:
                    msg = await client.messages.create(**request_params(plan.spec, lat, lon))
                except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError) as e:
                    errors[type(e).__name__] += 1  # SDK already retried; leave for next run
                    return
            store.write_record(f, message_record(pid, msg, batch=False))
            done += 1
            if done % 250 == 0:
                f.flush()
                print(f"  {plan.spec.key}: {done:,}/{len(todo):,}")

        await asyncio.gather(*(ask(*p) for p in todo))

    msg = f"  {plan.spec.key}: done, {done:,} answers"
    if errors:
        msg += f", {sum(errors.values()):,} failed ({dict(errors)}) - re-run to retry them"
    print(msg)


def run_live(plans: list[Plan], step: float, concurrency: int) -> None:
    sync_client = anthropic.Anthropic()
    ready = [p for p in plans if p.todo and (p.has_records or probe(sync_client, p, step))]

    async def main() -> None:
        async with anthropic.AsyncAnthropic(max_retries=6) as client:
            for plan in ready:
                if plan.todo:
                    await _live_model(client, plan, step, concurrency)

    asyncio.run(main())
