# llm-eval-harness

A small, honest evaluation suite for RAG and agent systems: retrieval accuracy, answer
faithfulness, tool-call correctness, latency and cost per request.

> **Status:** early. Built in public, one slice at a time. See the roadmap below.

## Principles

- **Report uncertainty.** Every headline number comes with a confidence interval and sample size.
- **Validate the judges.** LLM-as-judge scores are only reported alongside measured agreement
  with human labels.
- **Show failures.** Reports include concrete failing examples, not just averages.
- **Provider-agnostic.** Systems under test and judges sit behind small interfaces; no vendor
  lock-in.
- **Reproducible.** Datasets, configs, and pricing tables are versioned and dated.

## Roadmap

- [x] 1. Project scaffold (packaging, lint, types, tests, CI)
- [x] 2. Core types and runner (cases, results, system adapter, JSONL datasets)
- [x] 3. Retrieval metrics (recall@k, precision@k, MRR, nDCG)
- [x] 4. Latency and cost tracking (percentiles, token-based cost)
- [x] 5. Tool-call correctness (name, arguments, sequence)
- [ ] 6. Answer faithfulness (claim-level judge + human-agreement check)
- [ ] 7. Sample datasets and a baseline system
- [ ] 8. Report generator (tables, bootstrap CIs, failure examples)
- [ ] 9. First public eval report with full methodology

## Usage

```bash
llm-eval run --dataset examples/smoke.jsonl --system echo --out runs/demo
# -> runs/demo/{manifest.json, results.jsonl}

llm-eval score runs/demo --k 1 5 --pricing examples/pricing.example.toml
# -> prints a scorecard and writes runs/demo/scores.json
```

`--system` takes a builtin alias (`echo`) or any `package.module:ClassName` whose class
has a `name` and a `run(case) -> SystemResponse` method (see `llm_eval.system.System`).

Raw per-case results are the source of truth. Metrics and reports are computed from
`results.jsonl` afterwards, so scoring can change without re-running (and re-paying for)
the system under test. The manifest records the dataset's SHA-256 so results can be tied to
an exact dataset version.

### Dataset format (JSONL, one case per line)

```json
{"id": "rag-001", "input": "What year was the Eiffel Tower completed?",
 "expected_doc_ids": ["eiffel"], "expected_answer": "1889"}
{"id": "agent-001", "input": "Weather in Nairobi in celsius?",
 "expected_tool_calls": [{"name": "get_weather", "arguments": {"city": "Nairobi", "unit": "celsius"}}]}
```

Only `id` and `input` are required. A case counts toward a metric only if it has that
metric's gold label. For tool calls, omitting `expected_tool_calls` means "not labelled",
while `"expected_tool_calls": []` means "the system should make **no** tool calls". That
lets you test for over-eager tool use. Unknown keys, duplicate ids, and type errors are rejected with the
line number.

## Metrics

### Retrieval

Computed for every case with `expected_doc_ids`, using binary relevance:
`recall@k`, `precision@k`, `hit_rate@k`, `ndcg@k`, and `mrr`.

Conventions, since libraries disagree:

- Retrieved IDs are **de-duplicated** (first occurrence kept) before scoring, so
  repeating a relevant document can't inflate precision.
- `precision@k` divides by **k**, not by the number of documents returned.
- nDCG uses gain 1 and discount `1/log2(rank+1)`. The ideal ranking puts `min(k, |relevant|)`
  relevant docs first.
- **Errored cases score 0** and stay in the denominator. The scorecard reports `n` and
  how many cases errored.
- `score` refuses to run if the dataset's SHA-256 differs from the one in the run
  manifest, or if result and dataset case IDs don't match exactly.

### Tool-call correctness

Computed for every case with `expected_tool_calls`. Each metric is reported at three
argument-matching strictness levels, side by side, so you can see how much a score
depends on leniency:

| Mode | A call matches when |
|---|---|
| `exact` | Same name; arguments deep-equal (no extra or missing keys) |
| `subset` | Same name; every expected argument present and equal; extras allowed |
| `name` | Same name; arguments ignored |

| Metric | Meaning |
|---|---|
| `precision` | Matched / calls made (1.0 if none were made) |
| `recall` | Matched / calls expected (1.0 if none were expected) |
| `f1` | Harmonic mean. **Use this as the headline number:** a system that makes no calls gets precision 1.0 |
| `ordered_recall` | Longest in-order common subsequence / calls expected |
| `sequence_match` | 1 if the calls made equal the expected calls one-to-one and in order |

- Expected and made calls are paired by **maximum bipartite matching**, not a greedy
  scan, so a lenient match can't "steal" a call another expectation needed.
- Calling a tool twice when it was expected once counts as one match and one extra call.
- Values compare with **JSON semantics**: `1 == 1.0`, but `true != 1`. No normalization
  (case, whitespace, units) is applied.
- **Errored cases score 0** on every tool-call metric, even when no calls were expected.
  A crash is not a correct "no tool needed" decision.

### Latency

Wall-clock time around each `System.run` call, in milliseconds. It covers **all
requests, including errors**, because timeouts are part of what users experience.
Percentiles use linear interpolation (same as numpy's default). A tail percentile is
**withheld** (shown as `n/a` with a note) unless `n >= 1/(1-p)`, meaning 20 requests
for p95 and 100 for p99. Below that, "p99" is just the maximum under another name.

### Cost per request

The harness ships **no prices**. Pass a TOML price table with `--pricing` (see
`examples/pricing.example.toml`). It must include an `as_of` date and a `source`, and both
are printed with every cost figure.

- Cost = `input_tokens x input price + output_tokens x output price` (per 1M tokens),
  using the `model` and `usage` the system reports.
- A request with no usage, or with a model missing from the table, is **unpriced**. It's
  counted separately and never treated as free. Missing models are listed by name.
- Errored requests that report usage are priced, because failed calls still cost money.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

ruff check . && ruff format --check .
mypy
pytest
```

## License

MIT
