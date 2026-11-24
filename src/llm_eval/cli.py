"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from llm_eval import __version__
from llm_eval.dataset import DatasetError, file_sha256, load_jsonl
from llm_eval.runner import default_run_dir, run_eval, write_run
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _cmd_run(args.dataset, args.system, args.out)
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


if __name__ == "__main__":
    raise SystemExit(main())
