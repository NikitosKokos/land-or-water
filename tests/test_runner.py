"""Batch and live orchestration against a fake client (no network)."""

from types import SimpleNamespace

import anthropic
import pytest

from landwater import runner, store
from landwater.models import MODELS

STEP = 30  # 6 x 12 = 72 points


def fake_message(text="Water", thinking=False):
    content = [SimpleNamespace(type="text", text=text)]
    if thinking:
        content.insert(0, SimpleNamespace(type="thinking", thinking=""))
    return SimpleNamespace(
        content=content,
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=40, output_tokens=1),
    )


class FakeBatches:
    def __init__(self):
        self.created = []
        self.polls = 0

    def create(self, requests):
        self.created.append(requests)
        return SimpleNamespace(id=f"msgbatch_{len(self.created)}", processing_status="in_progress")

    def retrieve(self, batch_id):
        self.polls += 1
        status = "ended" if self.polls > 1 else "in_progress"
        counts = SimpleNamespace(succeeded=0, errored=0, canceled=0, expired=0, processing=1)
        return SimpleNamespace(id=batch_id, processing_status=status, request_counts=counts)

    def results(self, batch_id):
        requests = self.created[int(batch_id.split("_")[1]) - 1]
        for k, req in enumerate(requests):
            if k == 0:  # one failure, to be retried on the next run
                yield SimpleNamespace(custom_id=req["custom_id"], result=SimpleNamespace(type="errored"))
            else:
                yield SimpleNamespace(custom_id=req["custom_id"],
                                      result=SimpleNamespace(type="succeeded", message=fake_message("Land")))


class FakeClient:
    def __init__(self, *a, **kw):
        self.messages = SimpleNamespace(create=self._create, batches=FAKE_BATCHES)
        self.calls = []

    def _create(self, **params):
        self.calls.append(params)
        return fake_message()


FAKE_BATCHES = FakeBatches()


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    global FAKE_BATCHES
    FAKE_BATCHES = FakeBatches()
    store.set_root(tmp_path)
    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    monkeypatch.setattr(runner.time, "sleep", lambda s: None)


def test_batch_probe_submit_collect_and_resume():
    spec = MODELS["opus-5"]
    plans = runner.make_plans([spec], STEP)
    assert len(plans[0].todo) == 72

    runner.run_batch(plans, STEP, wait=True, poll_seconds=0)

    sent = FAKE_BATCHES.created[0]
    assert len(sent) == 71  # first point went through the live probe
    assert sent[0]["params"]["thinking"] == {"type": "disabled"}
    assert sent[0]["params"]["model"] == "claude-opus-5"

    recs = store.load_records(spec.key, STEP)
    assert len(recs) == 71  # probe + 70 successes; 1 errored
    assert not any(r["thought"] for r in recs.values())
    assert all(b["collected"] for b in store.load_batches(spec.key, STEP))

    # Re-run asks only the failed point.
    plans = runner.make_plans([spec], STEP)
    assert len(plans[0].todo) == 1 and not plans[0].pending


def test_batch_no_wait_then_collect_later():
    spec = MODELS["haiku-4.5"]
    runner.run_batch(runner.make_plans([spec], STEP), STEP, wait=False, poll_seconds=0)
    plans = runner.make_plans([spec], STEP)
    assert plans[0].pending == ["msgbatch_1"]
    runner.run_batch(plans, STEP, wait=True, poll_seconds=0)
    assert len(FAKE_BATCHES.created) == 1  # collected, not resubmitted
    assert len(store.load_records(spec.key, STEP)) == 71


def test_probe_config_error_skips_model(monkeypatch):
    def bad_create(self, **params):
        raise anthropic.NotFoundError(
            "model not found", response=SimpleNamespace(status_code=404, headers={}, request=None), body=None)

    monkeypatch.setattr(FakeClient, "_create", bad_create)
    runner.run_batch(runner.make_plans([MODELS["sonnet-4.5"]], STEP), STEP, wait=True, poll_seconds=0)
    assert FAKE_BATCHES.created == []
    assert store.load_records("sonnet-4.5", STEP) == {}


def test_probe_skips_model_that_thinks(monkeypatch):
    monkeypatch.setattr(FakeClient, "_create", lambda self, **params: fake_message("Land", thinking=True))
    runner.run_batch(runner.make_plans([MODELS["sonnet-5"]], STEP), STEP, wait=True, poll_seconds=0)
    assert FAKE_BATCHES.created == []
    assert store.load_records("sonnet-5", STEP) == {}


def test_live_mode(monkeypatch):
    class FakeAsync:
        def __init__(self, *a, **kw):
            self.messages = SimpleNamespace(create=self._create)

        async def _create(self, **params):
            return fake_message("Land")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(anthropic, "AsyncAnthropic", FakeAsync)
    spec = MODELS["opus-4.8"]
    runner.run_live(runner.make_plans([spec], STEP), STEP, concurrency=4)
    assert len(store.load_records(spec.key, STEP)) == 72
    assert runner.make_plans([spec], STEP)[0].todo == []
