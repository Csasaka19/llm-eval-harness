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
- [ ] 3. Retrieval metrics (recall@k, precision@k, MRR, nDCG)
- [ ] 4. Latency and cost tracking (percentiles, token-based cost)
- [ ] 5. Tool-call correctness (name, arguments, sequence)
- [ ] 6. Answer faithfulness (claim-level judge + human-agreement check)
- [ ] 7. Sample datasets and a baseline system
- [ ] 8. Report generator (tables, bootstrap CIs, failure examples)
- [ ] 9. First public eval report with full methodology

## Usage

```bash
llm-eval run --dataset examples/smoke.jsonl --system echo
# -> runs/<utc-timestamp>/{manifest.json, results.jsonl}
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
metric's gold label. Unknown keys, duplicate ids, and type errors are rejected with the
line number.

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
