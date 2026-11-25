"""Command-line entry point."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from llm_eval import __version__
from llm_eval.calibration import CalibrationError, CalibrationReport, calibrate, load_labels
from llm_eval.dataset import DatasetError, file_sha256, load_jsonl
from llm_eval.judges import BUILTIN_JUDGES, Judge, load_judge
from llm_eval.metrics.tool_calls import MATCH_MODES, TOOL_CALL_METRICS
from llm_eval.pricing import PriceTable, PricingError, load_pricing
from llm_eval.report import ReportInputs, render_report
from llm_eval.runner import default_run_dir, run_eval, write_run
from llm_eval.scoring import DEFAULT_KS, Scorecard, ScoringError, score_run
from llm_eval.serialization import RunFormatError, read_manifest, read_results
from llm_eval.system import BUILTIN_SYSTEMS, load_system
from llm_eval.types import CaseResult, EvalCase, JSONObject


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
    system_help = f"builtin ({', '.join(sorted(BUILTIN_SYSTEMS))}) or 'module:ClassName'"
    run.add_argument("--system", required=True, help=system_help)
    run.add_argument("--out", type=Path, default=None, help="output dir (default: runs/<utc>)")

    score = sub.add_parser("score", help="score a saved run against its dataset")
    score.add_argument("run_dir", type=Path, help="directory written by `llm-eval run`")
    score.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="dataset path (default: the one recorded in the manifest); must match its sha256",
    )
    _add_score_options(score)

    bench = sub.add_parser("bench", help="run a system over a dataset, then score it")
    bench.add_argument("--dataset", required=True, type=Path, help="path to a JSONL dataset")
    bench.add_argument("--system", required=True, help=system_help)
    bench.add_argument("--out", type=Path, default=None, help="output dir (default: runs/<utc>)")
    _add_score_options(bench)

    rep = sub.add_parser("report", help="score a saved run and write a Markdown report")
    rep.add_argument("run_dir", type=Path, help="directory written by `llm-eval run`")
    rep.add_argument("--dataset", type=Path, default=None, help="as for `score`")
    _add_score_options(rep)
    rep.add_argument(
        "--calibration-labels",
        type=Path,
        default=None,
        help="labelled claims; calibrates --judge and includes the result in the report",
    )
    rep.add_argument(
        "--out", type=Path, default=None, help="report path (default: RUN_DIR/report.md)"
    )
    rep.add_argument("--worst", type=int, default=5, help="worst cases shown per metric")

    cal = sub.add_parser("calibrate", help="measure a judge's agreement with human labels")
    cal.add_argument("--judge", required=True, help=JUDGE_HELP)
    cal.add_argument("--labels", required=True, type=Path, help="JSONL of labelled claims")
    cal.add_argument("--out", type=Path, default=None, help="write the report as JSON here")
    return parser


JUDGE_HELP = f"faithfulness judge: builtin ({', '.join(sorted(BUILTIN_JUDGES))}) or module:Class"


def _add_score_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=list(DEFAULT_KS),
        help=f"cutoffs for @k metrics (default: {' '.join(map(str, DEFAULT_KS))})",
    )
    parser.add_argument(
        "--pricing",
        type=Path,
        default=None,
        help="TOML price table (see examples/pricing.example.toml); omit to skip cost",
    )
    parser.add_argument("--judge", default=None, help=JUDGE_HELP + "; omit to skip faithfulness")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _cmd_run(args.dataset, args.system, args.out)[0]
    if args.command == "bench":
        code, out_dir = _cmd_run(args.dataset, args.system, args.out)
        if out_dir is None:
            return code
        print()
        score_code = _cmd_score(out_dir, None, args.k, args.pricing, args.judge)
        return score_code or code
    if args.command == "score":
        return _cmd_score(args.run_dir, args.dataset, args.k, args.pricing, args.judge)
    if args.command == "report":
        return _cmd_report(
            args.run_dir,
            args.dataset,
            args.k,
            args.pricing,
            args.judge,
            args.calibration_labels,
            args.out,
            args.worst,
        )
    if args.command == "calibrate":
        return _cmd_calibrate(args.judge, args.labels, args.out)
    parser.print_help()
    return 0


def _cmd_run(dataset: Path, system_spec: str, out: Path | None) -> tuple[int, Path | None]:
    """Returns (exit code, run directory or None if nothing was written)."""
    try:
        cases = load_jsonl(dataset)
    except (OSError, DatasetError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2, None
    try:
        system = load_system(system_spec)
    except (OSError, ImportError, AttributeError, ValueError, TypeError) as exc:
        print(f"error: could not load system: {exc}", file=sys.stderr)
        return 2, None

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
    return (1 if run.error_count else 0), out_dir


@dataclasses.dataclass(frozen=True)
class _Scored:
    card: Scorecard
    cases: list[EvalCase]
    results: list[CaseResult]
    manifest: JSONObject
    ks: list[int]


def _load_and_score(
    run_dir: Path,
    dataset: Path | None,
    ks: list[int],
    pricing_path: Path | None,
    judge_spec: str | None,
) -> _Scored | int:
    """Load a run, verify its dataset, and score it. Returns an exit code on failure."""
    if any(k < 1 for k in ks):
        print("error: --k values must be >= 1", file=sys.stderr)
        return 2
    try:
        judge: Judge | None = load_judge(judge_spec) if judge_spec else None
    except (ImportError, AttributeError, ValueError, TypeError) as exc:
        print(f"error: could not load judge: {exc}", file=sys.stderr)
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
        sorted_ks = sorted(set(ks))
        card = score_run(cases, results, sorted_ks, pricing=pricing, judge=judge)
    except (OSError, DatasetError, RunFormatError, ScoringError, PricingError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return _Scored(card, cases, results, manifest, sorted_ks)


def _write_scores(run_dir: Path, card: Scorecard) -> Path:
    out = run_dir / "scores.json"
    out.write_text(json.dumps(dataclasses.asdict(card), indent=2) + "\n", encoding="utf-8")
    return out


def _cmd_score(
    run_dir: Path,
    dataset: Path | None,
    ks: list[int],
    pricing_path: Path | None,
    judge_spec: str | None,
) -> int:
    scored = _load_and_score(run_dir, dataset, ks, pricing_path, judge_spec)
    if isinstance(scored, int):
        return scored
    out = _write_scores(run_dir, scored.card)
    print(_format_scorecard(scored.card))
    print(f"\nwrote {out}")
    return 0


def _cmd_report(
    run_dir: Path,
    dataset: Path | None,
    ks: list[int],
    pricing_path: Path | None,
    judge_spec: str | None,
    calibration_labels: Path | None,
    out: Path | None,
    worst_n: int,
) -> int:
    if calibration_labels is not None and judge_spec is None:
        print("error: --calibration-labels requires --judge", file=sys.stderr)
        return 2
    scored = _load_and_score(run_dir, dataset, ks, pricing_path, judge_spec)
    if isinstance(scored, int):
        return scored
    calibration: CalibrationReport | None = None
    if calibration_labels is not None and judge_spec is not None:
        try:
            calibration = calibrate(load_judge(judge_spec), load_labels(calibration_labels))
        except (OSError, CalibrationError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    _write_scores(run_dir, scored.card)
    text = render_report(
        ReportInputs(
            card=scored.card,
            cases=scored.cases,
            results=scored.results,
            manifest=scored.manifest,
            ks=scored.ks,
            calibration=calibration,
            calibration_source=str(calibration_labels) if calibration_labels else None,
            worst_n=worst_n,
        )
    )
    path = out or run_dir / "report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path}")
    return 0


def _format_scorecard(card: Scorecard) -> str:
    sections = [
        _format_retrieval(card),
        _format_tool_calls(card),
        _format_faithfulness(card),
        _format_correctness(card),
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


def _format_faithfulness(card: Scorecard) -> str:
    if card.judge is None:
        return "faithfulness: skipped (pass --judge; calibrate it first)"
    if not card.faithfulness:
        return "faithfulness: no RAG cases (expected_doc_ids or retrieved contexts)"
    faith, n_claims = card.faithfulness["faithfulness"], card.faithfulness["n_claims"]
    return "\n".join(
        [
            f"faithfulness (judge={card.judge}, n={faith.n}, errored={faith.n_errored})",
            f"  mean={faith.mean:.3f}  mean claims/answer={n_claims.mean:.1f}",
            "  note: only as trustworthy as the judge's calibration (llm-eval calibrate)",
        ]
    )


def _format_correctness(card: Scorecard) -> str:
    if not card.correctness:
        return "answer correctness: no cases with expected_answer"
    first = next(iter(card.correctness.values()))
    lines = [
        f"answer correctness vs expected_answer (n={first.n}, errored={first.n_errored})",
        f"  {'metric':<16}{'mean':>8}",
    ]
    lines += [f"  {name:<16}{agg.mean:>8.3f}" for name, agg in card.correctness.items()]
    lines.append("  note: token overlap, not meaning; see README for limitations")
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


def _cmd_calibrate(judge_spec: str, labels_path: Path, out: Path | None) -> int:
    try:
        labelled = load_labels(labels_path)
    except (OSError, CalibrationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        judge = load_judge(judge_spec)
    except (ImportError, AttributeError, ValueError, TypeError) as exc:
        print(f"error: could not load judge: {exc}", file=sys.stderr)
        return 2
    report = calibrate(judge, labelled)
    print(_format_calibration(report))
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dataclasses.asdict(report), indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _format_calibration(r: CalibrationReport) -> str:
    su, us = r.human_supported_judge_unsupported, r.human_unsupported_judge_supported
    lines = [
        f"calibration: judge={r.judge}  n={r.n}",
        "",
        f"  {'':<22}{'judge: sup':>12}{'judge: unsup':>14}",
        f"  {'human: supported':<22}{r.both_supported:>12}{su:>14}",
        f"  {'human: unsupported':<22}{us:>12}{r.both_unsupported:>14}",
        "",
        f"  accuracy             {r.accuracy:.3f}",
        f"  cohen's kappa        {_fmt(r.cohens_kappa)}",
        f"  unsupported recall   {_fmt(r.unsupported_recall)}  (hallucinations caught)",
        f"  unsupported precision {_fmt(r.unsupported_precision)}",
    ]
    if r.disagreements:
        lines += ["", f"disagreements ({len(r.disagreements)}):"]
        for d in r.disagreements:
            human = "supported" if d.human_supported else "unsupported"
            lines.append(f"  [{d.id}] human={human}: {d.claim}")
            if d.judge.rationale:
                lines.append(f"      judge: {d.judge.rationale}")
            if d.note:
                lines.append(f"      note: {d.note}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
