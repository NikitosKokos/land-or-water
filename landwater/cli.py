"""Command line: landwater {models,run,report,plot}."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import anthropic

from . import store
from .core import fmt_num, grid, prediction_grid, score, truth
from .models import DEFAULT_MODELS, MODELS, PRESETS, resolve

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv() -> Path | None:
    """Read KEY=VALUE lines from .env (current dir, else project root) into os.environ.

    Variables already set in the real environment win. Empty values are ignored.
    """
    for path in (Path.cwd() / ".env", PROJECT_ROOT / ".env"):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip().removeprefix("export ").strip()
            value = value.strip().strip("\"'")
            if key and value:
                os.environ.setdefault(key, value)
        return path
    return None


def _confirm(yes: bool) -> bool:
    if yes:
        return True
    if not sys.stdin.isatty():
        print("Not a terminal: re-run with --yes to confirm.")
        return False
    return input("Proceed? [y/N] ").strip().lower() in {"y", "yes"}


# ----------------------------------------------------------------------- commands


def cmd_models(args: argparse.Namespace) -> None:
    print(f"{'key':<12} {'model id':<18} {'$in/$out per 1M':<16} {'thinking':<28}")
    for m in MODELS.values():
        default = "[default]" if m.key in DEFAULT_MODELS else ""
        print(f"{m.key:<12} {m.model_id:<18} {f'{m.price_in:g}/{m.price_out:g}':<16} {m.thinking_note:<28} {default}")
    print("\npresets (use with -m, e.g. `-m figure`):")
    for name, keys in PRESETS.items():
        print(f"  {name:<12} {' '.join(keys)}")
    print("\nChange the default in landwater/models.py (DEFAULT_MODELS), or pick per run with -m.")


def cmd_run(args: argparse.Namespace) -> None:
    from .runner import make_plans, print_plan, run_batch, run_live

    specs = resolve(args.models or DEFAULT_MODELS)
    print(f"Models: {', '.join(s.key for s in specs)}")
    thinkers = [s.key for s in specs if s.thinks]
    if thinkers:
        print(f"Note: {', '.join(thinkers)} always think (thinking can't be turned off on them); "
              "they run at effort=low and are starred on the plot. All others: thinking off.")
    else:
        print("Thinking: off for all models.")
    lats, lons = grid(args.step)
    print(f"Grid: every {fmt_num(args.step)} deg -> {len(lats)} x {len(lons)} = {lats.size * lons.size:,} points per model\n")
    plans = make_plans(specs, args.step)
    batch = not args.live
    if args.live and any(p.pending for p in plans):
        print("Note: some models have batches in flight; run without --live to collect them first.")
        plans = [p for p in plans if not p.pending]
    total = print_plan(plans, batch)

    if args.dry_run:
        return
    if not any(p.todo or p.pending for p in plans):
        print("\nEverything already answered. Next: landwater report / landwater plot")
        return
    if total > 0 and not _confirm(args.yes):
        return

    if batch:
        run_batch(plans, args.step, wait=not args.no_wait, poll_seconds=args.poll)
    else:
        run_live(plans, args.step, args.concurrency)


def _selected(args: argparse.Namespace) -> list:
    if args.models:
        return resolve(args.models)
    have = set(store.models_with_results(args.step))
    return [m for m in MODELS.values() if m.key in have]


def cmd_report(args: argparse.Namespace) -> None:
    specs = _selected(args)
    if not specs:
        print(f"No results for step {fmt_num(args.step)} yet. Run `landwater run` first.")
        return
    land_share = truth(args.step).mean()
    print(f"step {fmt_num(args.step)} deg; ground truth: {land_share:.1%} of grid points are land\n")
    head = (f"{'model':<12} {'asked':>7} {'acc(area)':>9} {'acc(flat)':>9} {'land rec':>8} "
            f"{'water rec':>9} {'unclear':>7} {'refusal':>7} {'thought':>7} {'out tok':>9} {'USD':>7}")
    print(head)
    print("-" * len(head))
    for spec in specs:
        recs = store.load_records(spec.key, args.step)
        s = score(prediction_grid(recs, args.step), args.step)
        if s is None:
            print(f"{spec.key:<12} {0:>7}")
            continue
        refusals = sum(1 for r in recs.values() if r.get("stop") == "refusal")
        thought = sum(1 for r in recs.values() if r.get("thought"))  # stays 0 unless spec.thinks
        out_tok = sum(r.get("out", 0) for r in recs.values())
        usd = sum(
            (r.get("in", 0) * spec.price_in + r.get("out", 0) * spec.price_out) / 1e6 * (0.5 if r.get("batch") else 1)
            for r in recs.values()
        )
        print(f"{spec.key:<12} {s.n_asked:>7,} {s.accuracy:>9.1%} {s.accuracy_flat:>9.1%} {s.land_recall:>8.1%} "
              f"{s.water_recall:>9.1%} {s.n_unknown:>7,} {refusals:>7,} {thought:>7,} {out_tok:>9,} {usd:>7.2f}")


def cmd_plot(args: argparse.Namespace) -> None:
    from .plot import Panel, render, save_pixels, truth_grid

    specs = _selected(args)
    if not specs:
        print(f"No results for step {fmt_num(args.step)} yet. Run `landwater run` first.")
        return
    panels = []
    for spec in specs:
        pred = prediction_grid(store.load_records(spec.key, args.step), args.step)
        panels.append(Panel(spec.label, pred, score(pred, args.step), thinks=spec.thinks))
    if args.truth:
        panels.append(Panel("Ground truth (1-km mask)", truth_grid(args.step), None, is_truth=True))

    # Default layout: 2 per row up to 4 maps (2x2), else 3 per row like the reference figure.
    ncols = args.cols or (2 if len(panels) <= 4 else 3)
    suffix = "_errors" if args.diff else ""
    out = Path(args.out) if args.out else Path("plots") / f"earth_step{fmt_num(args.step)}{suffix}.png"
    print(f"wrote {render(panels, args.step, out, diff=args.diff, title=args.title, ncols=ncols)}")

    if args.each:
        for spec, panel in zip(specs, panels):  # truth panel is last, so zip skips it
            path = out.parent / f"{spec.key}_step{fmt_num(args.step)}{suffix}.png"
            print(f"wrote {save_pixels(panel.pred, path, diff=args.diff, step=args.step)}")


# -------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="landwater", description="Ask Claude 'Land or Water?' across the globe.")
    p.add_argument("--results", default="results", help="results directory (default: results)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("models", help="list models and presets").set_defaults(func=cmd_models)

    def add_common(sp: argparse.ArgumentParser, models_required: bool) -> None:
        sp.add_argument("-m", "--models", nargs="+", required=models_required,
                        help="model keys and/or presets: default, figure, no-thinking, thinking, all "
                             "(see `landwater models`)")
        sp.add_argument("--step", type=float, default=2.0,
                        help="grid spacing in degrees; must divide 180 (default 2 -> 16,200 points)")

    r = sub.add_parser("run", help="ask the questions (resumable)")
    add_common(r, models_required=False)  # default: DEFAULT_MODELS in models.py
    r.add_argument("--live", action="store_true", help="live async requests instead of the Batch API (full price)")
    r.add_argument("--concurrency", type=int, default=16, help="parallel requests in --live mode")
    r.add_argument("--no-wait", action="store_true", help="submit batches and exit; re-run later to collect")
    r.add_argument("--poll", type=int, default=60, help="seconds between batch status checks")
    r.add_argument("--dry-run", action="store_true", help="show plan and cost estimate only")
    r.add_argument("-y", "--yes", action="store_true", help="skip the spend confirmation")
    r.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="accuracy table")
    add_common(rep, models_required=False)
    rep.set_defaults(func=cmd_report)

    pl = sub.add_parser("plot", help="render the maps")
    add_common(pl, models_required=False)
    pl.add_argument("-o", "--out", help="output PNG (default plots/earth_step<N>.png)")
    pl.add_argument("--diff", action="store_true", help="color errors instead of land/water")
    pl.add_argument("--truth", action="store_true", help="add a ground-truth panel last")
    pl.add_argument("--cols", type=int, help="maps per row (default: 2 for up to 4 maps, else 3)")
    pl.add_argument("--each", action="store_true", help="also save one pixel map per model")
    pl.add_argument("--title", help="override the figure title")
    pl.set_defaults(func=cmd_plot)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    store.set_root(args.results)
    load_dotenv()
    try:
        args.func(args)
    except anthropic.AuthenticationError:
        sys.exit("Authentication failed: the API key was rejected. Check ANTHROPIC_API_KEY in .env.")
    except TypeError as e:
        if "authentication method" not in str(e):
            raise
        sys.exit(f"No API key: paste it into {PROJECT_ROOT / '.env'} as ANTHROPIC_API_KEY=sk-ant-...")
    except anthropic.AnthropicError as e:
        sys.exit(f"{type(e).__name__}: {e}")
    except KeyboardInterrupt:
        sys.exit("\nInterrupted. Progress is saved; re-run the same command to resume.")


if __name__ == "__main__":
    main()
