"""Command-line entry point."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from llm_eval import __version__
from llm_eval.dataset import DatasetError, file_sha256, load_jsonl
from llm_eval.metrics.tool_calls import MATCH_MODES, TOOL_CALL_METRICS
from llm_eval.pricing import PriceTable, PricingError, load_pricing
from llm_eval.runner import default_run_dir, run_eval, write_run
from llm_eval.scoring import DEFAULT_KS, Scorecard, ScoringError, score_run
from llm_eval.serialization import RunFormatError, read_manifest, read_results
from llm_eval.system import BUILTIN_SYSTEMS, load_system


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-eval",
        description="Evaluate RAG and agent systems: retrieval, faithfulness, tool calls, "
        "latency and cost.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="run a system over a dataset and save raw results")
    run.add_argument("--dataset", required=True, type=Path, help="path to a JSONL dataset")
    run.add_argument(
        "--system",
        required=True,
        help=f"builtin ({', '.join(sorted(BUILTIN_SYSTEMS))}) or 'module:ClassName'",
    )
    run.add_argument("--out", type=Path, default=None, help="output dir (default: runs/<utc>)")

    score = sub.add_parser("score", help="score a saved run against its dataset")
    score.add_argument("run_dir", type=Path, help="directory written by `llm-eval run`")
    score.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="dataset path (default: the one recorded in the manifest); must match its sha256",
    )
    score.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=list(DEFAULT_KS),
        help=f"cutoffs for @k metrics (default: {' '.join(map(str, DEFAULT_KS))})",
    )
    score.add_argument(
        "--pricing",
        type=Path,
        default=None,
        help="TOML price table (see examples/pricing.example.toml); omit to skip cost",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _cmd_run(args.dataset, args.system, args.out)
    if args.command == "score":
        return _cmd_score(args.run_dir, args.dataset, args.k, args.pricing)
    parser.print_help()
    return 0


def _cmd_run(dataset: Path, system_spec: str, out: Path | None) -> int:
    try:
        cases = load_jsonl(dataset)
    except (OSError, DatasetError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        system = load_system(system_spec)
    except (ImportError, AttributeError, ValueError, TypeError) as exc:
        print(f"error: could not load system: {exc}", file=sys.stderr)
        return 2

    run = run_eval(system, cases)
    out_dir = write_run(
        run,
        out or default_run_dir(),
        extra_manifest={
            "system_spec": system_spec,
            "dataset": str(dataset),
            "dataset_sha256": file_sha256(dataset),
        },
    )
    print(f"{len(run.results)} cases, {run.error_count} errors -> {out_dir}")
    return 1 if run.error_count else 0


def _cmd_score(
    run_dir: Path, dataset: Path | None, ks: list[int], pricing_path: Path | None
) -> int:
    if any(k < 1 for k in ks):
        print("error: --k values must be >= 1", file=sys.stderr)
        return 2
    try:
        manifest = read_manifest(run_dir)
        results = read_results(run_dir)
        if dataset is None:
            recorded = manifest.get("dataset")
            if not isinstance(recorded, str):
                print("error: manifest has no dataset path; pass --dataset", file=sys.stderr)
                return 2
            dataset = Path(recorded)
        expected_sha = manifest.get("dataset_sha256")
        if isinstance(expected_sha, str) and file_sha256(dataset) != expected_sha:
            print(
                f"error: {dataset} does not match the dataset this run used "
                f"(sha256 {expected_sha[:12]}...)",
                file=sys.stderr,
            )
            return 2
        cases = load_jsonl(dataset)
        pricing: PriceTable | None = load_pricing(pricing_path) if pricing_path else None
        card = score_run(cases, results, sorted(set(ks)), pricing=pricing)
    except (OSError, DatasetError, RunFormatError, ScoringError, PricingError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    out = run_dir / "scores.json"
    out.write_text(json.dumps(dataclasses.asdict(card), indent=2) + "\n", encoding="utf-8")
    print(_format_scorecard(card))
    print(f"\nwrote {out}")
    return 0


def _format_scorecard(card: Scorecard) -> str:
    sections = [
        _format_retrieval(card),
        _format_tool_calls(card),
        _format_latency(card),
        _format_cost(card),
    ]
    return "\n\n".join(s for s in sections if s)


def _format_retrieval(card: Scorecard) -> str:
    if not card.retrieval:
        return "retrieval: no cases with expected_doc_ids"
    first = next(iter(card.retrieval.values()))
    lines = [
        f"retrieval (n={first.n}, errored={first.n_errored}, errored cases score 0)",
        f"  {'metric':<16}{'mean':>8}",
    ]
    lines += [f"  {name:<16}{agg.mean:>8.3f}" for name, agg in card.retrieval.items()]
    return "\n".join(lines)


def _format_tool_calls(card: Scorecard) -> str:
    if not card.tool_calls:
        return "tool calls: no cases with expected_tool_calls"
    first = next(iter(card.tool_calls.values()))
    header = "".join(f"{mode:>8}" for mode in MATCH_MODES)
    lines = [
        f"tool calls (n={first.n}, errored={first.n_errored}, errored cases score 0)",
        f"  {'metric':<16}{header}",
    ]
    for metric in TOOL_CALL_METRICS:
        row = "".join(f"{card.tool_calls[f'{mode}/{metric}'].mean:>8.3f}" for mode in MATCH_MODES)
        lines.append(f"  {metric:<16}{row}")
    return "\n".join(lines)


def _format_latency(card: Scorecard) -> str:
    lat = card.latency
    if lat is None:
        return ""
    pcts = "  ".join(
        f"{name}={'n/a' if v is None else f'{v:.1f}'}" for name, v in lat.percentiles_ms.items()
    )
    lines = [
        f"latency ms (n={lat.n}, all requests incl. errors)",
        f"  mean={lat.mean_ms:.1f}  min={lat.min_ms:.1f}  max={lat.max_ms:.1f}",
        f"  {pcts}",
    ]
    lines += [f"  note: {w}" for w in lat.warnings]
    return "\n".join(lines)


def _format_cost(card: Scorecard) -> str:
    cost = card.cost
    if cost is None:
        return "cost: skipped (pass --pricing)"
    mean = "n/a" if cost.mean_per_request is None else f"{cost.mean_per_request:.6f}"
    lines = [
        f"cost {cost.currency} (prices as of {cost.pricing_as_of}, source: {cost.pricing_source})",
        f"  priced={cost.n_priced}  unpriced={cost.n_unpriced}",
        f"  mean/request={mean}  total={cost.total:.6f} (priced requests only)",
    ]
    if cost.unpriced_models:
        lines.append(f"  models missing from price table: {', '.join(cost.unpriced_models)}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
