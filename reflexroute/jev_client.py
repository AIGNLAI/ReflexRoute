"""Small standard-library client for OpenRouter's Jev Decisions API."""

from __future__ import annotations

import json
import math
import os
import socket
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .context_builder import QUESTION_ID
from .schemas import JevAPIError, RoutingResult


DEFAULT_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
JEV_MODEL = "typesafe/jev-1.13"


class JevClient:
    """Call Jev's structured decision endpoint (never chat completions)."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout: float = 30.0,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive.")
        self.api_key = api_key
        self.endpoint = endpoint
        self.timeout = timeout

    def choose(
        self,
        state: Mapping[str, object],
        question: Mapping[str, object],
        choice_to_model: Mapping[str, str],
    ) -> RoutingResult:
        api_key = (self.api_key or os.environ.get("OPENROUTER_API_KEY", "")).strip()
        if not api_key:
            raise JevAPIError(
                "OPENROUTER_API_KEY is not set. Export it before routing two or more models."
            )
        payload = {
            "model": JEV_MODEL,
            "state": dict(state),
            "questions": {QUESTION_ID: dict(question)},
        }
        response = self._post(payload, api_key)
        return self.parse_response(response, choice_to_model)

    def _post(self, payload: Mapping[str, object], api_key: str) -> Mapping[str, Any]:
        request = Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-OpenRouter-Title": "ReflexRoute",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise JevAPIError(
                f"OpenRouter Decisions API returned HTTP {exc.code}: {detail}"
            ) from exc
        except (URLError, TimeoutError, socket.timeout, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise JevAPIError(f"OpenRouter Decisions API request failed: {reason}") from exc
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as exc:
            raise JevAPIError("OpenRouter returned a non-JSON response.") from exc
        if not isinstance(decoded, Mapping):
            raise JevAPIError("OpenRouter response must be a JSON object.")
        return decoded

    @staticmethod
    def parse_response(
        response: Mapping[str, Any], choice_to_model: Mapping[str, str]
    ) -> RoutingResult:
        try:
            answer = response["answers"][QUESTION_ID]
            selected_id = answer["choice"]
            raw_probabilities = answer["probabilities"]
        except (KeyError, TypeError) as exc:
            raise JevAPIError("Jev response is missing the structured Choice answer.") from exc
        if selected_id not in choice_to_model:
            raise JevAPIError(f"Jev selected an unknown choice: {selected_id!r}.")
        if not isinstance(raw_probabilities, Mapping):
            raise JevAPIError("Jev probabilities must be an object.")
        try:
            probabilities = {
                model: float(raw_probabilities.get(choice_id, 0.0))
                for choice_id, model in choice_to_model.items()
            }
            default_confidence = probabilities[choice_to_model[selected_id]]
            confidence = float(answer.get("confidence", default_confidence))
        except (TypeError, ValueError) as exc:
            raise JevAPIError("Jev returned non-numeric probabilities.") from exc
        values = [*probabilities.values(), confidence]
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in values):
            raise JevAPIError("Jev probabilities and confidence must be between 0 and 1.")
        raw_usage = response.get("usage", {})
        usage = dict(raw_usage) if isinstance(raw_usage, Mapping) else {}
        return RoutingResult(
            model=choice_to_model[selected_id],
            probabilities=probabilities,
            confidence=confidence,
            usage=usage,
        )
