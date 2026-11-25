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
- [x] 6. Answer faithfulness (claim-level judge + human-agreement check)
- [x] 7. Sample datasets and a baseline system
- [x] 7.5. Answer correctness (exact match, token F1/recall vs reference)
- [ ] 8. Report generator (tables, bootstrap CIs, failure examples)
- [ ] 9. First public eval report with full methodology

## Quick start

```bash
# RAG baseline: BM25 retrieval + extractive answer, scored with the overlap judge
llm-eval bench --dataset datasets/rag/questions.jsonl --system bm25 --judge overlap

# Agent baseline: regex rules -> tool calls
llm-eval bench --dataset datasets/agent/tasks.jsonl --system rules-agent
```

`bench` = `run` + `score`. Both baselines are deterministic and need no API keys.

## Usage

```bash
llm-eval run --dataset examples/smoke.jsonl --system echo --out runs/demo
# -> runs/demo/{manifest.json, results.jsonl}

llm-eval score runs/demo --k 1 5 --pricing examples/pricing.example.toml --judge overlap
# -> prints a scorecard and writes runs/demo/scores.json

llm-eval calibrate --judge overlap --labels examples/faithfulness_calibration.jsonl
# -> agreement between the judge and human labels, plus every disagreement
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

## Datasets

| Path | Contents |
|---|---|
| `datasets/rag/corpus.jsonl` | 30 short internal documents of a **fictional** company, Kestrel Robotics |
| `datasets/rag/questions.jsonl` | 40 questions with gold doc IDs and reference answers. Types: direct (26), paraphrase (8), multi-doc (4), distractor (2) |
| `datasets/agent/tasks.jsonl` | 20 tool-use tasks. Types: single (7), paraphrase (4), multi-step (3), **no-tool** (6) |
| `datasets/agent/tools.json` | Schemas for the 5 tools the tasks use |

The company is fictional so that a model **cannot answer from what it learned in
training**. It has to retrieve, which keeps retrieval and faithfulness scores meaningful.
The corpus includes a superseded policy as a distractor.

Known limitations, so the numbers below aren't over-read:

- **Small.** With 40 and 20 cases, one case moves a mean by 2.5 to 5 points. Slice 8 adds
  confidence intervals.
- **Easy on direct questions.** BM25 gets all 26 direct questions right at rank 1. The
  hard cases are the paraphrase and multi-doc ones, so read results **by question type**,
  not just the overall mean.
- On multi-doc questions (2 gold docs), `recall@1` can't exceed 0.5.
- The labels were written by the maintainer and are not independently reviewed.

## Baselines

| System | What it does |
|---|---|
| `bm25` | Okapi BM25 (k1=1.5, b=0.75) over title + text; top 10 doc IDs returned, top 3 passed as context. The answer is the sentence from the top document that shares the most words with the question. Corpus from `$LLM_EVAL_CORPUS`, defaulting to `datasets/rag/corpus.jsonl`. |
| `rules-agent` | Regular expressions mapping known phrasings to tool calls. No understanding, so it fails on paraphrases. |

Results (deterministic, pinned in tests):

| Baseline | Metric | Overall | By type |
|---|---|---|---|
| `bm25` | recall@1 | 0.863 | direct 1.00 · distractor 1.00 · paraphrase 0.62 · multi-doc 0.38 |
| `bm25` | MRR | 0.929 | |
| `bm25` | faithfulness (overlap) | 1.000 | extractive, so faithful by construction |
| `bm25` | answer token recall | 0.621 | exact match 0.000 (extractive sentences never equal short references) |
| `rules-agent` | F1 (all modes) | 0.800 | fails exactly the 4 paraphrase tasks; 6/6 no-tool correct |

**Faithful is not the same as correct.** For *"What is the flight time of the K-200?"*
the BM25 baseline answers *"The K-200 is the Kestrel survey drone."* That sentence comes
from the right document, so faithfulness is 1.0, but it doesn't answer the question.
Answer correctness catches this: token recall is 0. That is why both are reported.

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

### Answer faithfulness

How much of an answer is backed by the context the system retrieved. The answer is split
into sentences ("claims"), a **judge** decides whether each claim is supported, and
`faithfulness = supported claims / claims`.

- Scored only with `--judge`, for RAG cases (those with `expected_doc_ids` or retrieved
  contexts). An errored RAG case scores 0. A case that retrieved nothing is judged against
  empty context.
- An empty answer scores 1.0 by default (there are no claims to be wrong), so **mean
  claims per answer** is shown next to the score.
- Every claim's verdict and the judge's reasoning are saved in `scores.json`.
- Faithfulness is not correctness: a claim can faithfully repeat a wrong document.

Judges implement `name` and `judge_claim(claim, contexts) -> ClaimVerdict` and are
loaded like systems (`--judge overlap` or `--judge module:Class`). The built-in
`overlap` judge is deterministic and dependency-free. It is a **weak baseline**: every
number in the claim must appear in the context, and at least 75% of its content words
must.

#### Calibrate the judge before trusting it

`llm-eval calibrate` runs a judge over human-labelled claims and reports a confusion
matrix, accuracy, **Cohen's kappa**, and recall on unsupported claims (the share of
hallucinations caught). Accuracy alone can mislead: a judge that always says "supported"
gets 80% on a set that is 80% supported, with kappa 0.

Built-in `overlap` judge on `examples/faithfulness_calibration.jsonl` (n=24):

| accuracy | Cohen's kappa | unsupported recall | unsupported precision |
|---|---|---|---|
| 0.583 | 0.172 | 0.636 | 0.538 |

That is only slight agreement. Overlap misses paraphrases ("Danish" vs "Denmark"),
swapped entities and roles ("Guido van Rossum was created by Python"), and some
negations. It does catch changed numbers. Treat its scores as a smoke test, not a
measurement. Any judge used in a published report must show its own calibration numbers
next to its scores.

> The labels in `examples/faithfulness_calibration.jsonl` are a **draft**. They must be
> reviewed by a human before a published report relies on them.

### Answer correctness

Computed for every case with `expected_answer`, using standard question-answering token
metrics:

| Metric | Meaning |
|---|---|
| `exact_match` | Normalized answer == normalized reference |
| `token_f1` | F1 over normalized tokens (SQuAD-style) |
| `token_recall` | Share of reference tokens that appear in the answer |

F1 penalizes long answers even when they contain the right fact, and recall doesn't, so
both are shown. Normalization lowercases and drops articles and punctuation, but **keeps
numbers intact** (SQuAD would turn `1.8` into `18`). An errored case scores 0.

These metrics measure word overlap, not meaning. A real false negative from the BM25
baseline: the question is *"Can I fly premium economy on a 9-hour flight?"*, the
reference is *"Yes"*, and the answer *"Premium economy is allowed for flights of 6 hours or
longer"* is correct but scores 0. Of the 10 BM25 answers scoring 0, the other 9 really
are wrong. A meaning-based grader (an LLM, for example) would need the same human-label
calibration as the faithfulness judge.

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
