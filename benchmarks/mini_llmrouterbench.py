#!/usr/bin/env python3
"""Run a small, resumable ReflexRoute benchmark on LLMRouterBench.

Only ReflexRoute's Jev decisions make network requests. Model quality and model
execution cost are read from LLMRouterBench's precomputed outcomes, which keeps
this benchmark inexpensive and reproducible.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import statistics
import sys
import threading
import time
from typing import Any, Iterable, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - exercised by benchmark users
    raise SystemExit(
        "This benchmark needs pandas. Install it with `python -m pip install pandas`."
    ) from exc

from reflexroute import HistoryRecord, Router


SEED = 20260920
DEFAULT_DATA_DIR = Path("data/llmrouterbench")
MODEL_SPECS = (
    (6, "meta-llama/llama-3.1-8b-instruct"),
    (18, "deepseek/deepseek-r1-0528"),
    (21, "google/gemini-2.5-flash"),
    (22, "google/gemini-2.5-pro"),
)
MODE_TO_METHOD = {
    "zero-shot": "reflexroute_zero_shot",
    "few-shot": "reflexroute_few_shot_k4",
}


@dataclass(frozen=True)
class Example:
    sample_id: str
    eval_name: str
    query: str
    performance: dict[str, float]
    cost: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmarks/results/mini_v1")
    )
    parser.add_argument("--context-size", type=int, default=256)
    parser.add_argument("--history-top-k", type=int, default=4)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=tuple(MODE_TO_METHOD),
        default=list(MODE_TO_METHOD),
    )
    return parser.parse_args()


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def complete_examples(frame: Any) -> list[Example]:
    required = [
        f"model_{index}_{metric}"
        for index, _ in MODEL_SPECS
        for metric in ("performance", "cost")
    ]
    mask = frame[required].notna().all(axis=1)
    examples: list[Example] = []
    for _, row in frame.loc[mask].iterrows():
        examples.append(
            Example(
                sample_id=str(row["sample_id"]),
                eval_name=str(row["eval_name"]),
                query=str(row["query_text"]),
                performance={
                    model: float(row[f"model_{index}_performance"])
                    for index, model in MODEL_SPECS
                },
                cost={
                    model: float(row[f"model_{index}_cost"])
                    for index, model in MODEL_SPECS
                },
            )
        )
    return examples


def sample_context(examples: Sequence[Example], size: int) -> list[Example]:
    if size <= 0:
        raise ValueError("context-size must be positive")
    if size >= len(examples):
        return list(examples)

    # Preserve the source benchmark's task mixture while keeping selection stable.
    rng = random.Random(SEED)
    groups: dict[str, list[Example]] = {}
    for example in examples:
        groups.setdefault(example.eval_name, []).append(example)
    for group in groups.values():
        rng.shuffle(group)

    exact = {name: size * len(group) / len(examples) for name, group in groups.items()}
    quota = {name: min(len(groups[name]), math.floor(value)) for name, value in exact.items()}
    remaining = size - sum(quota.values())
    order = sorted(groups, key=lambda name: (-(exact[name] - quota[name]), name))
    for name in order:
        if remaining == 0:
            break
        if quota[name] < len(groups[name]):
            quota[name] += 1
            remaining -= 1
    return [item for name in sorted(groups) for item in groups[name][: quota[name]]]


def make_history(context: Iterable[Example]) -> list[HistoryRecord]:
    return [
        HistoryRecord(
            query=example.query,
            model=model,
            performance=example.performance[model],
            cost=example.cost[model],
        )
        for example in context
        for _, model in MODEL_SPECS
    ]


def percentile(values: Sequence[float], proportion: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * proportion
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def usage_cost(usage: Mapping[str, Any]) -> float | None:
    """Read a billed dollar amount without guessing from token counts."""

    for key in ("cost", "total_cost"):
        value = usage.get(key)
        if finite(value):
            return float(value)
    return None


def append_jsonl(path: Path, value: Mapping[str, Any], lock: threading.Lock) -> None:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid checkpoint line {line_number}: {exc}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def route_once(
    router: Router,
    mode: str,
    example: Example,
    retries: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        started = time.perf_counter()
        try:
            result = router.route(example.query)
            elapsed = time.perf_counter() - started
            return {
                "mode": mode,
                "method": MODE_TO_METHOD[mode],
                "sample_id": example.sample_id,
                "eval_name": example.eval_name,
                "selected_model": result.model,
                "probabilities": result.probabilities,
                "confidence": result.confidence,
                "usage": result.usage,
                "router_cost": usage_cost(result.usage),
                "router_latency_seconds": elapsed,
                "realized_quality": example.performance[result.model],
                "selected_model_cost": example.cost[result.model],
                "attempts": attempt,
            }
        except Exception as exc:  # API errors are retried and checkpointed by caller
            last_error = exc
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 4))
    assert last_error is not None
    raise last_error


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    methods = sorted({str(row["method"]) for row in rows})
    result: dict[str, Any] = {}
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        latencies = [float(row["router_latency_seconds"]) for row in subset]
        known_router_costs = [
            float(row["router_cost"])
            for row in subset
            if finite(row.get("router_cost"))
        ]
        router_cost_total = sum(known_router_costs)
        selected_cost_total = sum(float(row["selected_model_cost"]) for row in subset)
        count = len(subset)
        selections: dict[str, int] = {}
        for row in subset:
            model = str(row["selected_model"])
            selections[model] = selections.get(model, 0) + 1
        result[method] = {
            "n": count,
            "quality_mean": statistics.fmean(float(row["realized_quality"]) for row in subset),
            "router_latency_p50_ms": percentile(latencies, 0.50) * 1000,
            "router_latency_p95_ms": percentile(latencies, 0.95) * 1000,
            "router_cost_total_usd": router_cost_total,
            "router_cost_per_1000_usd": router_cost_total / count * 1000,
            "router_cost_coverage": len(known_router_costs) / count,
            "selected_model_cost_mean_usd": selected_cost_total / count,
            "selected_model_cost_per_1000_usd": selected_cost_total / count * 1000,
            "total_cost_per_1000_usd": (router_cost_total + selected_cost_total) / count * 1000,
            "selections": dict(sorted(selections.items())),
        }
    return result


def print_table(summary: Mapping[str, Mapping[str, Any]]) -> None:
    print("\nmethod                         n   quality   p50 ms   p95 ms   route $/1k   model $/1k   total $/1k")
    print("-" * 104)
    for method, metrics in summary.items():
        print(
            f"{method:28} {metrics['n']:3d}   {metrics['quality_mean']:.4f}"
            f"   {metrics['router_latency_p50_ms']:7.1f}"
            f"   {metrics['router_latency_p95_ms']:7.1f}"
            f"   {metrics['router_cost_per_1000_usd']:11.4f}"
            f"   {metrics['selected_model_cost_per_1000_usd']:11.4f}"
            f"   {metrics['total_cost_per_1000_usd']:11.4f}"
        )


def main() -> None:
    args = parse_args()
    if args.workers <= 0 or args.retries <= 0 or args.history_top_k <= 0:
        raise SystemExit("workers, retries, and history-top-k must be positive")

    train_path = args.data_dir / "train.pkl"
    val_path = args.data_dir / "val.pkl"
    if not train_path.exists() or not val_path.exists():
        raise SystemExit(f"LLMRouterBench train.pkl/val.pkl not found under {args.data_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "decisions.jsonl"
    summary_path = args.output_dir / "summary.json"

    context_pool = complete_examples(pd.read_pickle(train_path))
    targets = complete_examples(pd.read_pickle(val_path))
    context = sample_context(context_pool, args.context_size)
    if args.limit is not None:
        targets = targets[: max(0, args.limit)]
    if not targets:
        raise SystemExit("No complete validation examples selected")

    history = make_history(context)
    checkpoint_rows = load_jsonl(checkpoint_path)
    completed = {
        (str(row.get("mode")), str(row.get("sample_id")))
        for row in checkpoint_rows
        if row.get("method") in MODE_TO_METHOD.values()
    }
    target_ids = {example.sample_id for example in targets}
    router_rows = [
        row
        for row in checkpoint_rows
        if str(row.get("sample_id")) in target_ids and row.get("mode") in args.modes
    ]

    routers: dict[str, Router] = {}
    model_names = [model for _, model in MODEL_SPECS]
    for mode in args.modes:
        router = Router(models=model_names, history_top_k=args.history_top_k, timeout=30.0)
        if mode == "few-shot":
            router.set_history(history)
        routers[mode] = router

    jobs = [
        (mode, example)
        for mode in args.modes
        for example in targets
        if (mode, example.sample_id) not in completed
    ]
    print(
        f"context={len(context)} rows ({len(history)} observations), "
        f"targets={len(targets)}, cached={len(router_rows)}, pending={len(jobs)}"
    )
    lock = threading.Lock()
    errors: list[tuple[str, str, str]] = []
    if jobs:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_job = {
                executor.submit(route_once, routers[mode], mode, example, args.retries): (
                    mode,
                    example,
                )
                for mode, example in jobs
            }
            finished = 0
            for future in as_completed(future_to_job):
                mode, example = future_to_job[future]
                try:
                    row = future.result()
                except Exception as exc:
                    errors.append((mode, example.sample_id, str(exc)))
                else:
                    append_jsonl(checkpoint_path, row, lock)
                    router_rows.append(row)
                finished += 1
                if finished % 10 == 0 or finished == len(jobs):
                    print(f"completed {finished}/{len(jobs)} new decisions; errors={len(errors)}", flush=True)

    summary = summarize(router_rows)
    payload = {
        "benchmark": "LLMRouterBench mini",
        "run_date": "2026-09-20",
        "seed": SEED,
        "context_size": len(context),
        "history_records": len(history),
        "history_top_k": args.history_top_k,
        "target_count": len(targets),
        "models": [
            {"dataset_index": index, "openrouter_id": model}
            for index, model in MODEL_SPECS
        ],
        "methods": summary,
        "errors": [
            {"mode": mode, "sample_id": sample_id, "error": error}
            for mode, sample_id, error in errors
        ],
    }
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print_table(summary)
    print(f"\nWrote {summary_path} and {checkpoint_path}")
    if errors:
        raise SystemExit(f"{len(errors)} routing decisions failed; rerun to resume them")


if __name__ == "__main__":
    main()
