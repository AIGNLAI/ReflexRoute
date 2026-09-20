# Small benchmark

This is a deliberately small, low-cost measurement of ReflexRoute itself. It
shows the cold-start and few-shot modes without claiming a comparison against
other routing systems.

## Result

| Mode | Mean quality | Router latency P50 / P95 | Router cost / 1,000 decisions | Selected-model cost / 1,000 queries |
| --- | ---: | ---: | ---: | ---: |
| Zero-shot | 0.5497 | 722 / 1,376 ms | $0.108 | $11.786 |
| Few-shot, K=4 | 0.6832 | 1,333 / 2,033 ms | $0.403 | $18.111 |

Adding four retrieved observations per candidate increased mean quality by
0.1335, or 24.3% relative to the zero-shot result. The complete run made 382
Jev decisions and incurred $0.09757 in actual routing charges.

These are exploratory measurements, not broad quality or latency guarantees.

## Setup

- Source: [LLMRouterBench](https://github.com/ynulihao/LLMRouterBench) cached
  model outcomes.
- Evaluation: 191 validation queries with complete quality and cost records for
  all four candidates.
- Candidates: Llama 3.1 8B Instruct, DeepSeek R1 0528, Gemini 2.5 Flash, and
  Gemini 2.5 Pro.
- Few-shot context: 256 training queries, producing 1,024 observations; local
  TF-IDF retrieves K=4 observations independently for each candidate.
- Quality: mean of the benchmark's per-query score, normalized to [0, 1].
- Cost: the router column is the amount billed for Jev decisions; selected-model
  cost comes from the benchmark's cached executions. No candidate model was
  called again for this run.
- Latency: wall-clock routing latency only. It excludes candidate-model
  generation latency and was measured from one network location on 2026-09-20.

The machine-readable aggregate is in
[`results/mini_v1/summary.json`](results/mini_v1/summary.json). Per-query
checkpoints are intentionally gitignored.

## Reproduce

Install the benchmark dependency:

```bash
python -m pip install -e '.[benchmark]'
```

Prepare `train.pkl` and `val.pkl` dense tables under `data/llmrouterbench`, or
pass another directory. Each row needs `sample_id`, `eval_name`, `query_text`,
and the performance/cost columns used in the script.

```bash
export OPENROUTER_API_KEY="your-key"
python benchmarks/mini_llmrouterbench.py \
  --data-dir data/llmrouterbench \
  --output-dir benchmarks/results/mini_v1
```

Completed decisions are appended to `decisions.jsonl`, so rerunning the same
command resumes without paying for successful calls again.
