"""Command-line interface for ReflexRoute."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from .router import Router
from .schemas import ReflexRouteError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reflexroute",
        description="Fast zero-shot and few-shot LLM routing powered by Jev.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    route = subparsers.add_parser("route", help="route one query")
    route.add_argument("query", nargs="?", help="query to route (quote multi-word queries)")
    route.add_argument("--models", nargs="+", metavar="MODEL", help="candidate model IDs")
    route.add_argument("--history", help="routing history (.jsonl or .csv)")
    route.add_argument("--budget", type=float, help="hard USD/request budget")
    route.add_argument("--top-k", type=int, default=4, help="history records per model (default: 4)")
    route.add_argument("--json", action="store_true", help="print machine-readable JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # With argparse, a variable-length --models placed before the final query
    # consumes that quoted query. Recover the documented, ergonomic form.
    if args.command == "route" and args.query is None and args.models:
        if len(args.models) < 2:
            parser.error("a query is required")
        args.query = args.models.pop()

    try:
        router = Router(models=args.models, history_top_k=args.top_k)
        if args.history:
            router.load_history(args.history)
        result = router.route(args.query, budget=args.budget)
    except (ReflexRouteError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0
    print(f"Selected: {result.model}\n")
    width = max(len(model) for model in result.probabilities)
    for model, probability in sorted(
        result.probabilities.items(), key=lambda item: item[1], reverse=True
    ):
        print(f"{model:<{width}}  {probability:.2f}")
    print(f"\nConfidence: {result.confidence:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
