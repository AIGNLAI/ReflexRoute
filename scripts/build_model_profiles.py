#!/usr/bin/env python3
"""Build the checked-in model priors from an OpenRouter Models API snapshot.

This script deliberately consumes a local JSON snapshot instead of fetching the
network itself. Updating the catalog is therefore an explicit, reviewable act:

    curl -fsSL https://openrouter.ai/api/v1/models -o /tmp/models.json
    python scripts/build_model_profiles.py --catalog /tmp/models.json
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path
from typing import Any, Mapping


SNAPSHOT_DATE = "2026-09-20"
PROFILE_VERSION = "2026-09-20.1"
ASSUMED_INPUT_TOKENS = 2_000
ASSUMED_OUTPUT_TOKENS = 1_000

# Non-batch, non-free entries selected for provider breadth and common routing
# use cases. Existence is checked against every input snapshot.
SELECTED_MODELS = [
    # OpenAI (18)
    "openai/gpt-6-astra",
    "openai/gpt-6-astra-pro",
    "openai/gpt-5.6-luna-pro",
    "openai/gpt-5.6-luna",
    "openai/gpt-5.6-terra-pro",
    "openai/gpt-5.6-terra",
    "openai/gpt-5.6-sol-pro",
    "openai/gpt-5.6-sol",
    "openai/gpt-5.5-pro",
    "openai/gpt-5.5",
    "openai/gpt-5.4-nano",
    "openai/gpt-5.4-mini",
    "openai/gpt-5.4-pro",
    "openai/gpt-5.4",
    "openai/gpt-5.3-codex",
    "openai/gpt-5.2-codex",
    "openai/gpt-5.1-codex-mini",
    "openai/gpt-oss-120b",
    # Anthropic (10)
    "anthropic/claude-fable-5.1",
    "anthropic/claude-opus-5",
    "anthropic/claude-sonnet-5",
    "anthropic/claude-fable-5",
    "anthropic/claude-opus-4.8",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-opus-4.6",
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-opus-4.5",
    # Google (12)
    "google/gemini-3.8-flash",
    "google/gemini-3.7-flash",
    "google/gemini-3.6-flash",
    "google/gemini-3.5-flash-lite",
    "google/gemini-3.5-flash",
    "google/gemini-3.1-flash-lite",
    "google/gemini-3.1-pro-preview",
    "google/gemini-3-flash-preview",
    "google/gemini-2.5-flash-lite",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "google/gemma-4-31b-it",
    # Qwen (16)
    "qwen/qwen3.8-max-0902",
    "qwen/qwen3.8-flash",
    "qwen/qwen3.8-27b",
    "qwen/qwen3.8-2.4t-a95b",
    "qwen/qwen3.7-flash",
    "qwen/qwen3.7-plus",
    "qwen/qwen3.7-max",
    "qwen/qwen3.6-flash",
    "qwen/qwen3.6-35b-a3b",
    "qwen/qwen3.6-plus",
    "qwen/qwen3.5-122b-a10b",
    "qwen/qwen3.5-flash-02-23",
    "qwen/qwen3-max-thinking",
    "qwen/qwen3-coder-next",
    "qwen/qwen3-vl-235b-a22b-thinking",
    "qwen/qwen3-coder-plus",
    # DeepSeek (8)
    "deepseek/deepseek-v4.1-flash",
    "deepseek/deepseek-v4-pro-0813",
    "deepseek/deepseek-v4-flash-0731",
    "deepseek/deepseek-v4-pro",
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v3.2",
    "deepseek/deepseek-r1-0528",
    "deepseek/deepseek-r1",
    # Mistral (8)
    "mistralai/mistral-medium-3-5",
    "mistralai/mistral-small-2603",
    "mistralai/devstral-2512",
    "mistralai/ministral-14b-2512",
    "mistralai/mistral-medium-3.1",
    "mistralai/codestral-2508",
    "mistralai/mistral-small-3.2-24b-instruct",
    "mistralai/mistral-nemo",
    # Z.ai (8)
    "z-ai/glm-5.3-flashx",
    "z-ai/glm-5.3-flash",
    "z-ai/glm-5.3",
    "z-ai/glm-5.2",
    "z-ai/glm-5.1",
    "z-ai/glm-5v-turbo",
    "z-ai/glm-5",
    "z-ai/glm-4.7",
    # Meta (5)
    "meta-llama/llama-4-maverick",
    "meta-llama/llama-4-scout",
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    # xAI (5)
    "x-ai/grok-4.6",
    "x-ai/grok-4.5",
    "x-ai/grok-build-0.1",
    "x-ai/grok-4.3",
    "x-ai/grok-4.20",
    # Moonshot (5)
    "moonshotai/kimi-k3",
    "moonshotai/kimi-k2.7-code",
    "moonshotai/kimi-k2.6",
    "moonshotai/kimi-k2.5",
    "moonshotai/kimi-k2-thinking",
    # MiniMax (4)
    "minimax/minimax-m3",
    "minimax/minimax-m2.7",
    "minimax/minimax-m2.5",
    "minimax/minimax-m1",
    # Amazon (4)
    "amazon/nova-2-lite-v1",
    "amazon/nova-premier-v1",
    "amazon/nova-micro-v1",
    "amazon/nova-pro-v1",
    # NVIDIA (4)
    "nvidia/nemotron-3.5-lightning",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/nemotron-3-nano-30b-a3b",
    # Perplexity (5)
    "perplexity/sonar-pro-search",
    "perplexity/sonar-reasoning-pro",
    "perplexity/sonar-pro",
    "perplexity/sonar-deep-research",
    "perplexity/sonar",
    # Cohere (4)
    "cohere/command-a",
    "cohere/command-r7b-12-2024",
    "cohere/command-r-08-2024",
    "cohere/command-r-plus-08-2024",
    # ByteDance Seed (4)
    "bytedance-seed/seed-2-1-turbo",
    "bytedance-seed/seed-2.0-code",
    "bytedance-seed/seed-2.0-lite",
    "bytedance-seed/seed-2.0-mini",
    # Additional useful specialists and provider coverage (10)
    "microsoft/phi-4",
    "inception/mercury-2.5",
    "inference-net/schematron-v2-turbo",
    "inference-net/schematron-v2-small",
    "sakana/fugu-ultra-v2",
    "sakana/fugu-max",
    "unbiased/pareto",
    "prism-ml/ternary-bonsai-2-27b",
    "inclusionai/ling-3.0-flash-vl",
    "inclusionai/ling-3.0-flash",
]

DEFAULT_MODELS = {
    "openai/gpt-5.6-sol",
    "openai/gpt-5.6-luna",
    "anthropic/claude-sonnet-5",
    "anthropic/claude-haiku-4.5",
    "google/gemini-3.8-flash",
    "qwen/qwen3.8-max-0902",
    "deepseek/deepseek-v4.1-flash",
    "mistralai/mistral-medium-3-5",
    "z-ai/glm-5.3",
    "meta-llama/llama-4-maverick",
    "x-ai/grok-4.6",
    "moonshotai/kimi-k3",
    "minimax/minimax-m3",
    "amazon/nova-2-lite-v1",
}


def _decimal(value: Any) -> Decimal:
    try:
        return Decimal(str(value or "0"))
    except Exception as exc:
        raise ValueError(f"Invalid price value: {value!r}") from exc


def _contains(text: str, *needles: str) -> bool:
    return any(needle in text for needle in needles)


def infer_strengths(model: Mapping[str, Any]) -> list[str]:
    """Conservatively infer routing tags from official metadata."""

    model_id = str(model["id"])
    text = " ".join(
        (model_id, str(model.get("name", "")), str(model.get("description", "")))
    ).casefold()
    architecture = model.get("architecture") or {}
    inputs = set(architecture.get("input_modalities") or [])
    parameters = set(model.get("supported_parameters") or [])
    strengths: list[str] = []

    def add(value: str) -> None:
        if value not in strengths:
            strengths.append(value)

    if _contains(text, "safety", "guard", "moderation"):
        add("safety classification")
    if _contains(text, "html-to-json", "extraction", "schema extraction"):
        add("structured extraction")
    if _contains(text, "code", "coder", "codex", "devstral", "software engineering", "programming"):
        add("coding")
    if model.get("reasoning") or _contains(text, "reasoning", "thinking", "math", "scientific", "analysis"):
        add("reasoning")
    if _contains(text, "math", "mathematics"):
        add("mathematics")
    if "image" in inputs:
        add("vision")
    if "video" in inputs:
        add("video understanding")
    if "audio" in inputs or _contains(text, "audio", "speech", "transcription"):
        add("audio understanding")
    if model_id.startswith("perplexity/") or _contains(text, "web search", "deep research", "search model"):
        add("web research")
    if "tools" in parameters or "tool_choice" in parameters:
        add("tool use")
    if "structured_outputs" in parameters:
        add("structured output")
    if _contains(text, "multilingual", "translation", "many languages"):
        add("multilingual tasks")
    if _contains(text, "agentic", "agent", "multi-agent"):
        add("agentic tasks")
    if _contains(text, "creative", "writing", "roleplay"):
        add("writing")
    if int(model.get("context_length") or 0) >= 250_000:
        add("long-context tasks")
    if _contains(text, "flash", "nano", "mini", "lite", "haiku", "lightning", "turbo", "micro", "small"):
        add("fast responses")

    prompt = _decimal((model.get("pricing") or {}).get("prompt"))
    completion = _decimal((model.get("pricing") or {}).get("completion"))
    if prompt * 1_000_000 <= Decimal("0.25") and completion * 1_000_000 <= Decimal("1"):
        add("cost-efficient workloads")
    if len(strengths) < 3:
        add("general instruction following")
    return strengths[:7]


def infer_latency(model: Mapping[str, Any]) -> str:
    text = f"{model['id']} {model.get('name', '')}".casefold()
    if _contains(text, "deep-research", "reasoning-pro", "ultra", "astra", "opus", "thinking", "-pro", " max"):
        return "high"
    if _contains(text, "flash", "nano", "mini", "lite", "haiku", "lightning", "turbo", "micro", "small"):
        return "low"
    return "medium"


def cost_data(model: Mapping[str, Any]) -> tuple[Decimal, Decimal, Decimal, str]:
    pricing = model.get("pricing") or {}
    prompt = _decimal(pricing.get("prompt"))
    completion = _decimal(pricing.get("completion"))
    if prompt < 0 or completion < 0:
        raise ValueError(f"Dynamic/negative pricing is unsupported for {model['id']}")
    estimated = prompt * ASSUMED_INPUT_TOKENS + completion * ASSUMED_OUTPUT_TOKENS
    if estimated == 0:
        level = "free"
    elif estimated <= Decimal("0.001"):
        level = "low"
    elif estimated <= Decimal("0.01"):
        level = "medium"
    elif estimated <= Decimal("0.05"):
        level = "high"
    else:
        level = "premium"
    return prompt * 1_000_000, completion * 1_000_000, estimated, level


def build_description(model: Mapping[str, Any], strengths: list[str]) -> str:
    architecture = model.get("architecture") or {}
    inputs = architecture.get("input_modalities") or ["text"]
    context = int(model.get("context_length") or 0)
    focus = ", ".join(strengths[:4])
    return (
        f"{model.get('name') or model['id']} is an OpenRouter model accepting "
        f"{', '.join(inputs)} input with a {context:,}-token context window. "
        f"This routing prior emphasizes {focus}."
    )


def yaml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def yaml_number(value: Decimal) -> str:
    rendered = format(value.quantize(Decimal("0.00000001")), "f").rstrip("0").rstrip(".")
    return rendered or "0"


def render(catalog: Mapping[str, Any]) -> str:
    data = catalog.get("data")
    if not isinstance(data, list):
        raise ValueError("Catalog must contain a data list.")
    by_id = {str(item["id"]): item for item in data}
    missing = [model_id for model_id in SELECTED_MODELS if model_id not in by_id]
    if missing:
        raise ValueError(f"Selected models missing from snapshot: {', '.join(missing)}")
    if len(SELECTED_MODELS) != len(set(SELECTED_MODELS)):
        raise ValueError("SELECTED_MODELS contains duplicates.")
    if len(SELECTED_MODELS) < 100:
        raise ValueError("At least 100 model profiles are required.")

    providers = {model_id.split("/", 1)[0] for model_id in SELECTED_MODELS}
    lines = [
        "# Generated by scripts/build_model_profiles.py; edit the generator, not this file.",
        f"profile_version: {yaml_string(PROFILE_VERSION)}",
        "source:",
        f"  type: {yaml_string('OpenRouter Models API snapshot plus ReflexRoute heuristic annotations')}",
        f"  url: {yaml_string('https://openrouter.ai/api/v1/models')}",
        f"  retrieved_at: {yaml_string(SNAPSHOT_DATE)}",
        f"  catalog_models: {len(data)}",
        f"  selected_models: {len(SELECTED_MODELS)}",
        f"  selected_providers: {len(providers)}",
        "cost_estimate:",
        "  currency: \"USD\"",
        f"  assumed_input_tokens: {ASSUMED_INPUT_TOKENS}",
        f"  assumed_output_tokens: {ASSUMED_OUTPUT_TOKENS}",
        "  excludes: \"cache, images, audio, web search, and provider-specific fees\"",
        "inference_notes:",
        "  strengths: \"Conservative tags inferred from official descriptions, modalities, and supported parameters.\"",
        "  latency_level: \"Heuristic tier inferred from model family naming; not measured latency.\"",
        "",
        "models:",
    ]

    for model_id in SELECTED_MODELS:
        model = by_id[model_id]
        strengths = infer_strengths(model)
        prompt_million, completion_million, estimated, cost_level = cost_data(model)
        architecture = model.get("architecture") or {}
        inputs = architecture.get("input_modalities") or ["text"]
        outputs = architecture.get("output_modalities") or ["text"]
        parameters = set(model.get("supported_parameters") or [])
        lines.extend(
            [
                f"  {model_id}:",
                f"    display_name: {yaml_string(str(model.get('name') or model_id))}",
                f"    description: {yaml_string(build_description(model, strengths))}",
                "    strengths:",
                *[f"      - {yaml_string(value)}" for value in strengths],
                f"    cost_level: {cost_level}",
                f"    latency_level: {infer_latency(model)}",
                f"    estimated_cost: {yaml_number(estimated)}",
                f"    context_length: {int(model.get('context_length') or 0)}",
                "    input_modalities:",
                *[f"      - {yaml_string(str(value))}" for value in inputs],
                "    output_modalities:",
                *[f"      - {yaml_string(str(value))}" for value in outputs],
                f"    supports_tools: {str('tools' in parameters or 'tool_choice' in parameters).lower()}",
                f"    supports_structured_outputs: {str('structured_outputs' in parameters).lower()}",
                f"    supports_reasoning: {str(bool(model.get('reasoning'))).lower()}",
                f"    prompt_cost_per_million: {yaml_number(prompt_million)}",
                f"    completion_cost_per_million: {yaml_number(completion_million)}",
                f"    default_candidate: {str(model_id in DEFAULT_MODELS).lower()}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path, help="OpenRouter models JSON snapshot")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reflexroute/profiles/models.yaml"),
        help="generated YAML path",
    )
    parser.add_argument("--check", action="store_true", help="fail when output is not up to date")
    args = parser.parse_args()

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    rendered = render(catalog)
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"{args.output} is not up to date")
        print(f"{args.output}: up to date ({len(SELECTED_MODELS)} profiles)")
        return 0
    args.output.write_text(rendered, encoding="utf-8")
    print(f"Wrote {len(SELECTED_MODELS)} profiles to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
