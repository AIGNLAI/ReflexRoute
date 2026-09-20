#!/usr/bin/env python3
"""Serve the dependency-free ReflexRoute Playground.

Run from the repository root:

    python demo/app.py

Set OPENROUTER_API_KEY to enable live decisions. Without a key, the interface
still works in replay mode using a previously recorded two-decision demo.
"""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping
from urllib.parse import urlsplit


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = Path(__file__).resolve().parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.adaptation_demo import HISTORY, MODELS, QUERY
from reflexroute import Router
from reflexroute.retrieval import retrieve_balanced


MODEL_LABELS = {
    "meta-llama/llama-3.1-8b-instruct": "Llama 3.1 8B",
    "deepseek/deepseek-r1-0528": "DeepSeek R1",
    "google/gemini-2.5-flash": "Gemini 2.5 Flash",
    "google/gemini-2.5-pro": "Gemini 2.5 Pro",
}
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
MAX_BODY_BYTES = 64 * 1024


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def evidence_for(query: str, top_k: int = 2) -> list[dict[str, Any]]:
    retrieved = retrieve_balanced(query, MODELS, HISTORY, top_k)
    evidence = [
        {
            "model": model,
            "model_label": MODEL_LABELS[model],
            "query": item.record.query,
            "performance": item.record.performance,
            "cost": item.record.cost,
            "similarity": item.similarity,
        }
        for model, records in retrieved.items()
        for item in records
    ]
    return sorted(evidence, key=lambda item: item["similarity"], reverse=True)


def config_payload() -> dict[str, Any]:
    router = Router(models=MODELS, history_top_k=2)
    models = []
    for model in MODELS:
        profile = router.profiles[model]
        models.append(
            {
                "id": model,
                "label": MODEL_LABELS[model],
                "description": profile.description,
                "strengths": list(profile.strengths[:3]),
                "cost_level": profile.cost_level,
                "latency_level": profile.latency_level,
            }
        )
    return {
        "live_available": bool(os.environ.get("OPENROUTER_API_KEY", "").strip()),
        "default_query": QUERY,
        "history_count": len(HISTORY),
        "history_top_k": 2,
        "models": models,
    }


def benchmark_payload() -> dict[str, Any]:
    summary_path = REPOSITORY_ROOT / "benchmarks/results/mini_v1/summary.json"
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        methods = payload["methods"]
        return {
            "target_count": payload["target_count"],
            "zero_shot": methods["reflexroute_zero_shot"],
            "few_shot": methods["reflexroute_few_shot_k4"],
        }
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return {
            "target_count": 191,
            "zero_shot": {
                "quality_mean": 0.5497,
                "router_latency_p50_ms": 721.8,
                "router_cost_per_1000_usd": 0.1076,
            },
            "few_shot": {
                "quality_mean": 0.6832,
                "router_latency_p50_ms": 1333.2,
                "router_cost_per_1000_usd": 0.4032,
            },
        }


def replay_payload() -> dict[str, Any]:
    return {
        "query": QUERY,
        "recorded": True,
        "results": [
            {
                "mode": "zero-shot",
                "selected_model": "deepseek/deepseek-r1-0528",
                "selected_label": "DeepSeek R1",
                "confidence": 0.71,
                "probabilities": {
                    "meta-llama/llama-3.1-8b-instruct": 0.10,
                    "deepseek/deepseek-r1-0528": 0.78,
                    "google/gemini-2.5-flash": 0.12,
                    "google/gemini-2.5-pro": 0.0,
                },
                "latency_ms": 640.0,
                "route_cost_usd": 0.000090,
                "input_tokens": None,
                "output_tokens": None,
                "evidence": [],
            },
            {
                "mode": "few-shot",
                "selected_model": "google/gemini-2.5-flash",
                "selected_label": "Gemini 2.5 Flash",
                "confidence": 0.99,
                "probabilities": {
                    "meta-llama/llama-3.1-8b-instruct": 0.0,
                    "deepseek/deepseek-r1-0528": 0.0,
                    "google/gemini-2.5-flash": 1.0,
                    "google/gemini-2.5-pro": 0.0,
                },
                "latency_ms": 666.0,
                "route_cost_usd": 0.000115,
                "input_tokens": None,
                "output_tokens": None,
                "evidence": evidence_for(QUERY),
            },
        ],
    }


def route_payload(query: str, mode: str) -> dict[str, Any]:
    if mode not in {"zero-shot", "few-shot"}:
        raise ValueError("mode must be zero-shot or few-shot")
    normalized = query.strip()
    if not normalized:
        raise ValueError("query must not be empty")
    if len(normalized) > 12_000:
        raise ValueError("query must be at most 12,000 characters")
    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        raise PermissionError("Set OPENROUTER_API_KEY on the server to enable live routing.")

    router = Router(models=MODELS, history_top_k=2, timeout=30.0)
    if mode == "few-shot":
        router.set_history(HISTORY)
    started = time.perf_counter()
    result = router.route(normalized)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "mode": mode,
        "selected_model": result.model,
        "selected_label": MODEL_LABELS[result.model],
        "confidence": result.confidence,
        "probabilities": result.probabilities,
        "latency_ms": elapsed_ms,
        "route_cost_usd": _number(result.usage.get("cost")),
        "input_tokens": _number(result.usage.get("input_tokens")),
        "output_tokens": _number(result.usage.get("output_tokens")),
        "evidence": evidence_for(normalized) if mode == "few-shot" else [],
    }


class PlaygroundHandler(BaseHTTPRequestHandler):
    server_version = "ReflexRoutePlayground/0.1"

    def _headers(self, content_type: str, length: int, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'",
        )
        self.end_headers()

    def _json(self, payload: Mapping[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._headers("application/json; charset=utf-8", len(body), status)
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlsplit(self.path).path
        if path == "/api/config":
            self._json(config_payload())
            return
        if path == "/api/benchmark":
            self._json(benchmark_payload())
            return
        if path == "/api/replay":
            self._json(replay_payload())
            return
        if path not in STATIC_FILES:
            self._json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return
        name, content_type = STATIC_FILES[path]
        try:
            body = (DEMO_ROOT / name).read_bytes()
        except OSError:
            self._json({"error": "Static asset unavailable"}, HTTPStatus.NOT_FOUND)
            return
        self._headers(content_type, len(body))
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlsplit(self.path).path
        if path != "/api/route":
            self._json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return
        if self.headers.get_content_type() != "application/json":
            self._json(
                {"error": "Content-Type must be application/json"},
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            )
            return
        origin = self.headers.get("Origin")
        expected_origin = f"http://{self.headers.get('Host', '')}"
        if origin and origin != expected_origin:
            self._json({"error": "Cross-origin requests are not allowed"}, HTTPStatus.FORBIDDEN)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY_BYTES:
            self._json({"error": "Invalid request size"}, HTTPStatus.BAD_REQUEST)
            return
        try:
            value = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(value, Mapping):
                raise ValueError("request body must be an object")
            query = value.get("query")
            mode = value.get("mode")
            if not isinstance(query, str) or not isinstance(mode, str):
                raise ValueError("query and mode must be strings")
            payload = route_payload(query, mode)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        except PermissionError as exc:
            self._json({"error": str(exc)}, HTTPStatus.SERVICE_UNAVAILABLE)
            return
        except Exception as exc:
            self._json(
                {"error": f"Routing failed: {str(exc)[:500]}"},
                HTTPStatus.BAD_GATEWAY,
            )
            return
        self._json(payload)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise SystemExit("port must be between 1 and 65535")
    server = ThreadingHTTPServer((args.host, args.port), PlaygroundHandler)
    print(f"ReflexRoute Playground: http://{args.host}:{args.port}")
    print(
        "Live routing enabled."
        if os.environ.get("OPENROUTER_API_KEY", "").strip()
        else "Replay mode ready. Set OPENROUTER_API_KEY to enable live routing."
    )
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print("Warning: this demo has no user authentication; prefer the default localhost bind.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Playground.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
