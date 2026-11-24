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
- [ ] 2. Core types and runner (cases, results, system adapter, JSONL datasets)
- [ ] 3. Retrieval metrics (recall@k, precision@k, MRR, nDCG)
- [ ] 4. Latency and cost tracking (percentiles, token-based cost)
- [ ] 5. Tool-call correctness (name, arguments, sequence)
- [ ] 6. Answer faithfulness (claim-level judge + human-agreement check)
- [ ] 7. Sample datasets and a baseline system
- [ ] 8. Report generator (tables, bootstrap CIs, failure examples)
- [ ] 9. First public eval report with full methodology

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
