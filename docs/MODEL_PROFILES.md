# Model profile methodology

ReflexRoute ships a broad but bounded snapshot of the models available through
OpenRouter. Profiles are routing priors—not benchmark verdicts, guarantees, or
live service telemetry.

## Current coverage

The `2026-09-20.1` snapshot contains 130 non-batch, non-free model IDs across
23 providers:

| Provider | Profiles |
| --- | ---: |
| `openai` | 18 |
| `qwen` | 16 |
| `google` | 12 |
| `anthropic` | 10 |
| `deepseek` | 8 |
| `mistralai` | 8 |
| `z-ai` | 8 |
| `meta-llama` | 5 |
| `moonshotai` | 5 |
| `perplexity` | 5 |
| `x-ai` | 5 |
| `amazon` | 4 |
| `bytedance-seed` | 4 |
| `cohere` | 4 |
| `minimax` | 4 |
| `nvidia` | 4 |
| `inclusionai` | 2 |
| `inference-net` | 2 |
| `sakana` | 2 |
| `inception` | 1 |
| `microsoft` | 1 |
| `prism-ml` | 1 |
| `unbiased` | 1 |

Batch variants and `:free` duplicates are omitted to keep model choices
distinct. Dynamic OpenRouter meta-routers are also omitted because their costs
cannot be represented as one hard-budget estimate.

## Field provenance

Directly sourced from the official OpenRouter Models API snapshot:

- Model ID and display name.
- Context length.
- Input and output modalities.
- Supported request parameters.
- Prompt and completion token prices.
- Official description, used only as input to conservative tag inference.

Derived by ReflexRoute:

- `description`: a compact summary of structured metadata and inferred focus.
- `strengths`: keyword- and capability-based routing tags.
- `latency_level`: a family-name heuristic, not measured latency.
- `cost_level`: a tier derived from the reference-request estimate.
- `estimated_cost`: token-price estimate for 2,000 input plus 1,000 output
  tokens.
- `default_candidate`: membership in the bounded cold-start shortlist.

Cost tiers use that reference-request estimate: `low` is at most $0.001,
`medium` at most $0.01, `high` at most $0.05, and `premium` above $0.05.
Zero-priced models would use `free`, although free duplicates are not part of
this snapshot.

The estimate excludes prompt caching, images, audio, web search, and
provider-specific fees. Users with different traffic shapes should maintain a
deployment-specific profile file.

## Updating the snapshot

Profile updates are explicit and reviewable. The generator never fetches the
network itself:

```bash
curl -fsSL https://openrouter.ai/api/v1/models -o /tmp/openrouter-models.json
python scripts/build_model_profiles.py \
  --catalog /tmp/openrouter-models.json
pytest
```

To verify deterministic regeneration against the same snapshot:

```bash
python scripts/build_model_profiles.py \
  --catalog /tmp/openrouter-models.json \
  --check
```

When updating, review `SELECTED_MODELS`, `DEFAULT_MODELS`, all inferred tags,
the snapshot date/version, and the generated diff before committing.

## Precedence

Profiles intentionally remain weak priors. When relevant user history conflicts
with a bundled profile, the Jev instruction tells the model to prefer the user
evidence.
