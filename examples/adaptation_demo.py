"""Show the same routing decision before and after adding a few observations.

Run with:
    OPENROUTER_API_KEY=... python examples/adaptation_demo.py

This demo makes two Jev Decisions API calls. It only selects a model; it does
not invoke the selected model.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from reflexroute import HistoryRecord, Router, RoutingResult


MODELS = [
    "meta-llama/llama-3.1-8b-instruct",
    "deepseek/deepseek-r1-0528",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
]
QUERY = "Implement a Python async rate limiter and explain its failure modes."

# Illustrative observations from an application's own evaluation loop.
HISTORY = [
    HistoryRecord(
        query="Debug a concurrent Python worker pool with intermittent deadlocks.",
        model="meta-llama/llama-3.1-8b-instruct",
        performance=0.42,
        cost=0.0001,
    ),
    HistoryRecord(
        query="Debug a concurrent Python worker pool with intermittent deadlocks.",
        model="deepseek/deepseek-r1-0528",
        performance=0.61,
        cost=0.017,
    ),
    HistoryRecord(
        query="Debug a concurrent Python worker pool with intermittent deadlocks.",
        model="google/gemini-2.5-flash",
        performance=0.97,
        cost=0.006,
    ),
    HistoryRecord(
        query="Debug a concurrent Python worker pool with intermittent deadlocks.",
        model="google/gemini-2.5-pro",
        performance=0.73,
        cost=0.074,
    ),
    HistoryRecord(
        query="Summarize a customer support email in three bullets.",
        model="meta-llama/llama-3.1-8b-instruct",
        performance=0.76,
        cost=0.0001,
    ),
    HistoryRecord(
        query="Summarize a customer support email in three bullets.",
        model="deepseek/deepseek-r1-0528",
        performance=0.85,
        cost=0.012,
    ),
    HistoryRecord(
        query="Summarize a customer support email in three bullets.",
        model="google/gemini-2.5-flash",
        performance=0.95,
        cost=0.004,
    ),
    HistoryRecord(
        query="Summarize a customer support email in three bullets.",
        model="google/gemini-2.5-pro",
        performance=0.94,
        cost=0.052,
    ),
]


def route_and_time(router: Router) -> tuple[RoutingResult, float]:
    started = time.perf_counter()
    result = router.route(QUERY)
    return result, time.perf_counter() - started


def show(label: str, result: RoutingResult, elapsed: float) -> None:
    billed = result.usage.get("cost")
    billed_text = f"${float(billed):.6f}" if isinstance(billed, (int, float)) else "n/a"
    print(f"\n{label}")
    print(f"  selected   {result.model}")
    print(f"  confidence {result.confidence:.2f}")
    print(f"  latency    {elapsed * 1000:.0f} ms")
    print(f"  route cost {billed_text}")
    print("  probabilities")
    for model, probability in sorted(
        result.probabilities.items(), key=lambda item: item[1], reverse=True
    ):
        print(f"    {probability:5.1%}  {model}")


def main() -> None:
    router = Router(models=MODELS, history_top_k=2)

    zero_shot, zero_shot_time = route_and_time(router)
    show("Zero-shot · built-in priors", zero_shot, zero_shot_time)

    router.set_history(HISTORY)
    few_shot, few_shot_time = route_and_time(router)
    show("Few-shot · priors + 8 user observations", few_shot, few_shot_time)


if __name__ == "__main__":
    main()
