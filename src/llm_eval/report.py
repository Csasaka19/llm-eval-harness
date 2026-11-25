"""Render a scored run as a self-contained Markdown report.

Every mean is shown with a bootstrap confidence interval and its sample size. Results are
broken down by ``metadata.type`` when the dataset provides it, the worst cases are shown
with their evidence, and a methodology section states every convention the numbers depend
on.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

from llm_eval import __version__
from llm_eval.calibration import CalibrationReport
from llm_eval.metrics.latency import MIN_SAMPLES, percentile
from llm_eval.scoring import CaseScores, Scorecard
from llm_eval.stats import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    Interval,
    bootstrap_ci,
)
from llm_eval.types import CaseResult, EvalCase, JSONObject, ToolCall

Family = Callable[[CaseScores], dict[str, float] | None]
FAMILIES: dict[str, Family] = {
    "retrieval": lambda c: c.retrieval,
    "tool_calls": lambda c: c.tool_calls,
    "faithfulness": lambda c: c.faithfulness,
    "correctness": lambda c: c.correctness,
}
SMALL_N = 100


@dataclass(frozen=True)
class ReportInputs:
    card: Scorecard
    cases: Sequence[EvalCase]
    results: Sequence[CaseResult]
    manifest: JSONObject
    ks: Sequence[int]
    calibration: CalibrationReport | None = None
    calibration_source: str | None = None
    worst_n: int = 5


def headline_metrics(card: Scorecard, ks: Sequence[int]) -> list[tuple[str, str, str]]:
    """(family, key, label) rows for the summary table, in display order."""
    rows: list[tuple[str, str, str]] = []
    if card.retrieval:
        rows.append(("retrieval", "mrr", "Retrieval MRR"))
        for k in ks:
            rows.append(("retrieval", f"recall@{k}", f"Retrieval recall@{k}"))
        rows.append(("retrieval", f"ndcg@{max(ks)}", f"Retrieval nDCG@{max(ks)}"))
    if card.tool_calls:
        for mode in ("exact", "subset", "name"):
            rows.append(("tool_calls", f"{mode}/f1", f"Tool-call F1 ({mode})"))
        rows.append(("tool_calls", "exact/sequence_match", "Tool-call sequence match (exact)"))
    if card.faithfulness:
        rows.append(("faithfulness", "faithfulness", "Faithfulness"))
    if card.correctness:
        rows.append(("correctness", "token_recall", "Answer token recall"))
        rows.append(("correctness", "token_f1", "Answer token F1"))
        rows.append(("correctness", "exact_match", "Answer exact match"))
    return rows


def render_report(inp: ReportInputs) -> str:
    sections = [
        _header(inp),
        _summary(inp),
        _by_type(inp),
        _latency_cost(inp),
        _calibration(inp),
        _worst_cases(inp),
        _methodology(inp),
        _limitations(inp),
    ]
    return "\n\n".join(s for s in sections if s).rstrip() + "\n"


# --- sections --------------------------------------------------------------------


def _header(inp: ReportInputs) -> str:
    m = inp.manifest
    n_err = sum(1 for r in inp.results if not r.ok)
    rows = [
        ("System", _code(_mstr(m, "system"))),
        ("System spec", _code(_mstr(m, "system_spec"))),
        ("Dataset", _code(_mstr(m, "dataset"))),
        ("Dataset SHA-256", _code(_mstr(m, "dataset_sha256")[:16] + "...")),
        ("Run started", _mstr(m, "started_at")),
        ("Cases", f"{len(inp.cases)} ({n_err} errored)"),
        ("Faithfulness judge", _code(inp.card.judge) if inp.card.judge else "not used"),
        ("Harness version", f"run {_mstr(m, 'harness_version')}, report {__version__}"),
    ]
    table = _table(["", ""], [[k, v] for k, v in rows])
    return f"# Evaluation report: {_mstr(m, 'system')}\n\n{table}"


def _summary(inp: ReportInputs) -> str:
    pct = int(DEFAULT_CONFIDENCE * 100)
    rows = []
    for family, key, label in headline_metrics(inp.card, inp.ks):
        values, n_err = _values(inp.card, FAMILIES[family], key)
        ci = bootstrap_ci(values)
        rows.append(
            [label, f"{ci.estimate:.3f}", f"[{ci.low:.3f}, {ci.high:.3f}]", str(ci.n), str(n_err)]
        )
    if not rows:
        return "## Summary\n\nNo labelled metrics to report."
    table = _table(["Metric", "Mean", f"{pct}% CI", "n", "errored"], rows)
    note = (
        "Errored cases are included and score 0. Each metric's *n* counts only the cases "
        "that have that metric's labels."
    )
    return f"## Summary\n\n{table}\n\n{note}"


def _by_type(inp: ReportInputs) -> str:
    types = {c.id: t for c in inp.cases if isinstance(t := c.metadata.get("type"), str)}
    if not types:
        return ""
    primary = [
        row
        for row in headline_metrics(inp.card, inp.ks)
        if row[1] in {f"recall@{min(inp.ks)}", "mrr", "exact/f1", "faithfulness", "token_recall"}
    ]
    ordered_types = sorted(set(types.values()))
    header = ["Type", "n", *[label for _, _, label in primary]]
    rows = []
    for t in ordered_types:
        cells = [t, str(sum(1 for v in types.values() if v == t))]
        for family, key, _ in primary:
            vals = [
                s[key]
                for c in inp.card.cases
                if types.get(c.case_id) == t and (s := FAMILIES[family](c)) is not None
            ]
            cells.append(f"{sum(vals) / len(vals):.3f}" if vals else "-")
        rows.append(cells)
    note = "Per-type means have small *n*; treat differences of a few cases as noise."
    return f"## Results by case type\n\n{_table(header, rows)}\n\n{note}"


def _latency_cost(inp: ReportInputs) -> str:
    lat = inp.card.latency
    parts = ["## Latency and cost"]
    if lat is not None:
        values = [c.latency_ms for c in inp.card.cases]
        rows = [["mean", f"{lat.mean_ms:.1f}", _ci_str(bootstrap_ci(values))]]
        for name, value in lat.percentiles_ms.items():
            p = int(name[1:])
            if value is None:
                rows.append([name, "withheld", f"needs n >= {MIN_SAMPLES[p]}, have {lat.n}"])
            else:
                ci = bootstrap_ci(values, partial(_pct, p=p))
                rows.append([name, f"{value:.1f}", _ci_str(ci)])
        rows.append(["max", f"{lat.max_ms:.1f}", ""])
        parts.append(
            f"Latency (ms, n={lat.n}, all requests including errors):\n\n"
            + _table(["Statistic", "Value", "95% CI"], rows)
        )
    cost = inp.card.cost
    if cost is None:
        parts.append("Cost: not computed (no price table supplied).")
    else:
        mean = "n/a" if cost.mean_per_request is None else f"{cost.mean_per_request:.6f}"
        parts.append(
            _table(
                ["Cost", "Value"],
                [
                    ["Mean per priced request", f"{mean} {cost.currency}"],
                    ["Total (priced requests)", f"{cost.total:.6f} {cost.currency}"],
                    ["Priced / unpriced requests", f"{cost.n_priced} / {cost.n_unpriced}"],
                    ["Prices as of", cost.pricing_as_of],
                    ["Price source", cost.pricing_source],
                ],
            )
        )
        if cost.unpriced_models:
            parts.append(f"Models missing from the price table: {', '.join(cost.unpriced_models)}.")
    return "\n\n".join(parts)


def _calibration(inp: ReportInputs) -> str:
    if inp.card.judge is None:
        return ""
    r = inp.calibration
    if r is None:
        return (
            "## Judge calibration\n\n**Not provided.** The faithfulness scores above come from "
            "a judge whose agreement with human labels was not measured for this report. "
            "Treat them as unvalidated."
        )
    rows = [
        ["Labelled claims", str(r.n)],
        ["Accuracy", f"{r.accuracy:.3f}"],
        ["Cohen's kappa", _opt(r.cohens_kappa)],
        ["Unsupported recall (hallucinations caught)", _opt(r.unsupported_recall)],
        ["Unsupported precision", _opt(r.unsupported_precision)],
    ]
    matrix = _table(
        ["", "judge: supported", "judge: unsupported"],
        [
            ["human: supported", str(r.both_supported), str(r.human_supported_judge_unsupported)],
            [
                "human: unsupported",
                str(r.human_unsupported_judge_supported),
                str(r.both_unsupported),
            ],
        ],
    )
    return (
        f"## Judge calibration\n\nJudge {_code(r.judge)} against labelled claims from "
        f"{_code(inp.calibration_source or 'unknown source')}. Agreement is only as good as "
        "those labels; check who wrote them.\n\n"
        f"{_table(['', ''], rows)}\n\n{matrix}\n\n"
        f"Read the faithfulness score in light of these numbers: {_kappa_words(r.cohens_kappa)}"
    )


def _worst_cases(inp: ReportInputs) -> str:
    by_case = {c.id: c for c in inp.cases}
    by_result = {r.case_id: r for r in inp.results}
    blocks = []
    specs: list[tuple[str, str, str]] = [
        ("retrieval", f"recall@{min(inp.ks)}", f"Retrieval (lowest recall@{min(inp.ks)})"),
        ("tool_calls", "exact/f1", "Tool calls (lowest exact F1)"),
        ("faithfulness", "faithfulness", "Faithfulness (lowest)"),
        ("correctness", "token_recall", "Answer correctness (lowest token recall)"),
    ]
    for family, key, title in specs:
        scored = [(s[key], c) for c in inp.card.cases if (s := FAMILIES[family](c)) is not None]
        failing = sorted((x for x in scored if x[0] < 1.0), key=lambda x: (x[0], x[1].case_id))
        if not scored:
            continue
        if not failing:
            blocks.append(f"### {title}\n\nNo case scored below 1.0.")
            continue
        shown = failing[: inp.worst_n]
        items = [
            _case_block(score, cs, by_case[cs.case_id], by_result[cs.case_id], family)
            for score, cs in shown
        ]
        more = (
            f"\n\n{len(failing) - len(shown)} more below 1.0 not shown."
            if len(failing) > len(shown)
            else ""
        )
        blocks.append(
            f"### {title}\n\n{len(failing)} of {len(scored)} cases scored below 1.0.\n\n"
            + "\n\n".join(items)
            + more
        )
    if not blocks:
        return ""
    return "## Worst cases\n\n" + "\n\n".join(blocks)


def _case_block(
    score: float, cs: CaseScores, case: EvalCase, result: CaseResult, family: str
) -> str:
    t = case.metadata.get("type")
    lines = [f"**{case.id}**" + (f" ({t})" if isinstance(t, str) else "") + f": {score:.3f}"]
    lines.append(f"- Input: {_q(case.input)}")
    resp = result.response
    if result.error:
        lines.append(f"- **Error:** {_q(result.error)}")
    if family == "retrieval":
        lines.append(f"- Expected docs: {_ids(case.expected_doc_ids)}")
        lines.append(f"- Retrieved: {_ids(resp.retrieved_doc_ids[:5]) if resp else '-'}")
    elif family == "tool_calls":
        lines.append(f"- Expected calls: {_calls(case.expected_tool_calls or ())}")
        lines.append(f"- Made calls: {_calls(resp.tool_calls) if resp else '-'}")
    else:
        if case.expected_answer is not None:
            lines.append(f"- Reference: {_q(case.expected_answer)}")
        if resp:
            lines.append(f"- Answer: {_q(resp.answer)}")
        if family == "faithfulness" and cs.claims:
            for v in cs.claims:
                mark = "supported" if v.supported else "**unsupported**"
                why = f" ({_q(v.rationale)})" if v.rationale else ""
                lines.append(f"  - {mark}: {_q(v.claim)}{why}")
    return "\n".join(lines)


def _methodology(inp: ReportInputs) -> str:
    pct = int(DEFAULT_CONFIDENCE * 100)
    ks = ", ".join(map(str, inp.ks))
    items = [
        f"**Confidence intervals:** {pct}% percentile bootstrap over cases, "
        f"{DEFAULT_RESAMPLES} resamples, seed {DEFAULT_SEED}. "
        "Cases are treated as independent draws.",
        "**Errors:** a case whose system call raised stays in every denominator it "
        "belongs to and scores 0.",
        "**Retrieval:** binary relevance against `expected_doc_ids`; retrieved IDs "
        "de-duplicated; precision@k divides by k; nDCG uses gain 1 and discount "
        f"1/log2(rank+1); cutoffs k = {ks}.",
        "**Tool calls:** maximum bipartite matching between expected and made calls; "
        "argument equality with JSON semantics (1 == 1.0, true != 1). Modes: `exact` "
        "(identical arguments), `subset` (expected arguments present, extras allowed), "
        '`name` (arguments ignored). `"expected_tool_calls": []` means the system should '
        "make no calls.",
        "**Faithfulness:** answers split into sentence-level claims; the share of claims "
        "the judge marks as supported by the retrieved contexts. Empty answers score 1.0 "
        "vacuously.",
        "**Answer correctness:** SQuAD-style exact match, token F1 and token recall "
        "against `expected_answer`, with numbers kept intact during normalisation.",
        "**Latency:** wall-clock time around each system call, including errored calls. "
        "Tail percentiles are withheld below n = 1/(1-p).",
        "**Cost:** token usage reported by the system times a dated, sourced price table. "
        "Requests with unknown usage or model are counted as unpriced, never as free.",
    ]
    return "## Methodology\n\n" + "\n".join(f"- {i}" for i in items)


def _limitations(inp: ReportInputs) -> str:
    items = [
        "Token-overlap metrics (answer correctness, the `overlap` judge) measure wording, "
        "not meaning.",
        "Faithfulness is not correctness: an answer can faithfully repeat a wrong document.",
    ]
    if len(inp.cases) < SMALL_N:
        items.insert(
            0,
            f"**Small dataset (n={len(inp.cases)}).** One case moves a mean by "
            f"{100 / len(inp.cases):.1f} points, and bootstrap intervals are optimistic at "
            "this size.",
        )
    if inp.card.judge and inp.calibration is None:
        items.append("The faithfulness judge was not calibrated for this report.")
    return "## Limitations\n\n" + "\n".join(f"- {i}" for i in items)


# --- helpers ---------------------------------------------------------------------


def _values(card: Scorecard, get: Family, key: str) -> tuple[list[float], int]:
    vals: list[float] = []
    n_err = 0
    for c in card.cases:
        scores = get(c)
        if scores is not None:
            vals.append(scores[key])
            n_err += c.errored
    return vals, n_err


def _pct(values: Sequence[float], p: int) -> float:
    return percentile(values, p)


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _q(text: str, limit: int = 200) -> str:
    text = " ".join(text.split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _code(text: str) -> str:
    return f"`{text.replace('`', '')}`"


def _ids(ids: Sequence[str]) -> str:
    return ", ".join(_code(i) for i in ids) if ids else "(none)"


def _calls(calls: Sequence[ToolCall]) -> str:
    if not calls:
        return "(none)"
    return ", ".join(
        _code(f"{c.name}({json.dumps(c.arguments, ensure_ascii=False)})") for c in calls
    )


def _mstr(m: JSONObject, key: str) -> str:
    value = m.get(key)
    return value if isinstance(value, str) else "unknown"


def _opt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _ci_str(ci: Interval) -> str:
    return f"[{ci.low:.1f}, {ci.high:.1f}]"


# Landis & Koch (1977) bands: a common, if arbitrary, convention.
_KAPPA_BANDS: tuple[tuple[float, str], ...] = (
    (0.0, "no better than chance"),
    (0.2, "slight"),
    (0.4, "fair"),
    (0.6, "moderate"),
    (0.8, "substantial"),
)


def _kappa_words(kappa: float | None) -> str:
    if kappa is None:
        return "kappa is undefined for this label set."
    label = next((name for upper, name in _KAPPA_BANDS if kappa <= upper), "almost perfect")
    return f"kappa {kappa:.2f} is *{label}* agreement (Landis & Koch bands)."
