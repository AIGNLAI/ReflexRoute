# ReflexRoute

[![CI](https://github.com/AIGNLAI/ReflexRoute/actions/workflows/ci.yml/badge.svg)](https://github.com/AIGNLAI/ReflexRoute/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Route on priors. Adapt with examples.**

Fast, training-free LLM routing powered by Jev. Zero-shot by default; few-shot
when you bring history.

ReflexRoute is a lightweight Python router for fast, inexpensive model
selection. It combines versioned model priors with your own routing outcomes,
then asks [`typesafe/jev-1.13`](https://openrouter.ai/typesafe/jev-1.13/) one
structured `Choice` question through OpenRouter's Decisions API. It never uses
chat completion to make the routing decision.

> [!NOTE]
> ReflexRoute is an early-stage project. OpenRouter's Decisions endpoint is
> experimental, and the bundled profiles are starting points—not benchmark
> guarantees or live pricing data.

## Why ReflexRoute?

- **Zero-shot from the first request.** Built-in priors provide a useful cold
  start without router training.
- **Few-shot adaptation without fitting.** Add past outcomes and relevant
  evidence overrides generic priors in context.
- **Fast and inexpensive.** Retrieval is local, and Jev makes one compact,
  structured decision.
- **Hard budget enforcement.** Python filters ineligible models before Jev sees
  the candidates.
- **Small dependency surface.** The runtime uses only the Python standard
  library.
- **Auditable decisions.** Inspect the complete state, question, and opaque
  choice mapping before sending anything.

## Install

ReflexRoute requires Python 3.10 or newer.

```bash
python -m pip install "git+https://github.com/AIGNLAI/ReflexRoute.git"
```

For local development:

```bash
git clone https://github.com/AIGNLAI/ReflexRoute.git
cd ReflexRoute
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Set your OpenRouter API key:

```bash
export OPENROUTER_API_KEY="your-key"
```

## Quick start

### Zero-shot routing

```python
from reflexroute import Router

router = Router(
    models=[
        "openai/gpt-5.6",
        "google/gemini-flash",
        "anthropic/claude-sonnet",
    ]
)

result = router.route("Implement a distributed rate limiter.")

print(result.model)
print(result.probabilities)
print(result.confidence)
```

### Few-shot / in-context routing

```python
router.load_history("examples/history.jsonl")
result = router.route("Implement a distributed rate limiter.")
```

History can be JSONL or CSV. Its canonical shape is:

```json
{"query":"Solve this math problem...","model":"openai/gpt-5.6","performance":1.0,"cost":0.018,"latency":1.42}
```

`query`, `model`, and numeric `performance` are required. `cost` and `latency`
are optional. Performance is higher-is-better; its scale is yours, but it
should be consistent within a history file.

For each candidate, ReflexRoute independently retrieves up to
`history_top_k` similar observations. This per-model balance prevents a model
with many logs from consuming the context:

```python
router = Router(models=["model-a", "model-b"], history_top_k=4)
router.load_history("history.csv")
```

Retrieval uses deterministic TF-IDF cosine similarity locally, including basic
CJK bigram support. No embedding service is required.

### Hard budgets

```python
result = router.route("Summarize this report", budget=0.02)
```

The budget is USD per request. ReflexRoute filters candidates using the
profile's `estimated_cost` **before** building or sending the Jev request. Jev
is never asked to perform budget arithmetic.

- Models above the limit are excluded.
- Models without a known estimated cost are excluded when a budget is active.
- If nothing remains, ReflexRoute raises `BudgetError`.
- If one candidate remains, ReflexRoute returns it locally without an API call.

Bundled costs are illustrative. Production deployments should maintain a
profile file with estimates appropriate to their workloads.

## CLI

```bash
reflexroute route "Write a CUDA kernel"

reflexroute route \
  --models openai/gpt-5.6 google/gemini-flash \
  "Write a CUDA kernel"

reflexroute route \
  --history examples/history.jsonl \
  --budget 0.02 \
  "Write a CUDA kernel"
```

Use `--json` for machine-readable output and `--top-k` to change the evidence
limit. Quote multi-word queries.

Typical output:

```text
Selected: openai/gpt-5.6

openai/gpt-5.6       0.67
google/gemini-flash  0.21
model-c              0.12

Confidence: 0.74
```

## How it works

```text
Built-in model priors ─┐
User routing history ──┼─> context builder ─> Jev Choice ─> selected model
Target query ──────────┤
Optional hard budget ──┘  (budget filtering happens first, in Python)
```

Jev receives structured state rather than a chat prompt:

```json
{
  "target_query": "...",
  "candidate_models": {
    "model-a": {
      "prior": {"description": "...", "profile_version": "0.1.0"},
      "historical_evidence": []
    }
  }
}
```

The Choice instruction explicitly tells Jev that strong, relevant user
observations override generic priors when they conflict. Unknown models remain
eligible without a budget and receive an honest “unknown capabilities” prior
rather than invented strengths.

## Model profiles

Bundled profiles live in
[`reflexroute/profiles/models.yaml`](reflexroute/profiles/models.yaml). Supply a
deployment-specific file with `Router(profile_path="my-models.yaml")`:

```yaml
profile_version: "2026-09-20"
source: "internal evaluation"

models:
  provider/model-name:
    description: "A concise, fallible model prior."
    strengths:
      - coding
      - reasoning
    cost_level: medium
    latency_level: low
    estimated_cost: 0.01
```

`profile_version`, `source`, and unknown extra fields are preserved in the Jev
context, making profiles traceable and extensible.

## Python API

| API | Purpose |
| --- | --- |
| `Router(models, history_top_k=4)` | Create a router; omit `models` to use every bundled profile. |
| `router.load_history(path)` | Replace history from a JSONL or CSV file. |
| `router.set_history(records)` | Replace history from Python objects. |
| `router.eligible_models(budget)` | Inspect the result of hard budget filtering. |
| `router.build_decision(query, budget=None)` | Inspect Jev state without a network request. |
| `router.route(query, budget=None)` | Return a `RoutingResult`. |

`RoutingResult` contains `model`, `probabilities`, `confidence`, and `usage`.
Failures use typed `ReflexRouteError` subclasses for configuration, history,
budget, and API errors.

## Development

```bash
python -m pip install -e '.[dev]'
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow. The
initial scope intentionally excludes router training, embedding services, web
UIs, proxy servers, automatic benchmark downloads, and live price syncing.

## License

[MIT](LICENSE)
