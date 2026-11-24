"""Command-line entry point. Subcommands are added as slices land."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from llm_eval import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-eval",
        description="Evaluate RAG and agent systems: retrieval, faithfulness, tool calls, "
        "latency and cost.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
