"""The high-level plug-and-play routing API."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .context_builder import ContextBuilder
from .jev_client import JevClient
from .profiles import load_profiles
from .schemas import (
    BudgetError,
    ConfigurationError,
    HistoryFormatError,
    HistoryRecord,
    RoutingResult,
)


class Router:
    """Route queries with built-in priors and optional user history."""

    def __init__(
        self,
        models: Sequence[str] | None = None,
        *,
        history_top_k: int = 4,
        profile_path: str | Path | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
        client: JevClient | None = None,
    ) -> None:
        self.profiles = load_profiles(profile_path)
        selected = list(self.profiles) if models is None else _validate_models(models)
        if not selected:
            raise ConfigurationError("At least one candidate model is required.")
        self.models = tuple(selected)
        self.history_top_k = history_top_k
        self._history: list[HistoryRecord] = []
        self._builder = ContextBuilder(history_top_k=history_top_k)
        self.client = client or JevClient(api_key=api_key, timeout=timeout)

    @property
    def history(self) -> tuple[HistoryRecord, ...]:
        return tuple(self._history)

    def load_history(self, path: str | Path) -> int:
        """Load and replace routing history from a JSONL or CSV file.

        Returns the number of accepted records. Unknown/non-candidate model rows
        are retained so that changing the candidate set later stays lossless;
        retrieval simply ignores them for the current decision.
        """

        history_path = Path(path)
        suffix = history_path.suffix.casefold()
        try:
            if suffix == ".jsonl":
                records = _read_jsonl(history_path)
            elif suffix == ".csv":
                records = _read_csv(history_path)
            else:
                raise HistoryFormatError("History must be a .jsonl or .csv file.")
        except OSError as exc:
            raise HistoryFormatError(f"Cannot read history file: {history_path}") from exc
        self._history = records
        return len(records)

    def set_history(self, records: Sequence[HistoryRecord | Mapping[str, Any]]) -> None:
        """Replace history programmatically."""

        converted: list[HistoryRecord] = []
        for index, value in enumerate(records, start=1):
            try:
                record = value if isinstance(value, HistoryRecord) else HistoryRecord.from_mapping(value)
                record.validate()
            except (HistoryFormatError, TypeError) as exc:
                raise HistoryFormatError(f"Invalid history record {index}: {exc}") from exc
            converted.append(record)
        self._history = converted

    def eligible_models(self, budget: float | None = None) -> tuple[str, ...]:
        """Apply a hard per-request budget before any Jev request is built."""

        if budget is None:
            return self.models
        try:
            limit = float(budget)
        except (TypeError, ValueError) as exc:
            raise BudgetError("budget must be a finite non-negative number.") from exc
        if not math.isfinite(limit) or limit < 0:
            raise BudgetError("budget must be a finite non-negative number.")

        eligible = tuple(
            model
            for model in self.models
            if (profile := self.profiles.get(model)) is not None
            and profile.estimated_cost is not None
            and profile.estimated_cost <= limit
        )
        if not eligible:
            unknown = [
                model
                for model in self.models
                if model not in self.profiles
                or self.profiles[model].estimated_cost is None
            ]
            unknown_note = (
                f" Models without a known estimated_cost were excluded: {', '.join(unknown)}."
                if unknown
                else ""
            )
            raise BudgetError(
                f"No candidate model satisfies the hard budget of {limit:g} USD/request."
                + unknown_note
            )
        return eligible

    def build_decision(
        self, query: str, *, budget: float | None = None
    ) -> tuple[dict[str, object], dict[str, object], dict[str, str]]:
        """Build the exact Jev state/question without making a network request."""

        normalized_query = _validate_query(query)
        candidates = self.eligible_models(budget)
        return self._builder.build(
            normalized_query, candidates, self.profiles, self._history
        )

    def route(self, query: str, *, budget: float | None = None) -> RoutingResult:
        """Select a model for ``query`` using zero-shot or in-context routing."""

        normalized_query = _validate_query(query)
        candidates = self.eligible_models(budget)
        if len(candidates) == 1:
            model = candidates[0]
            return RoutingResult(
                model=model,
                probabilities={model: 1.0},
                confidence=1.0,
                usage={"local_short_circuit": True},
            )
        state, question, choice_to_model = self._builder.build(
            normalized_query, candidates, self.profiles, self._history
        )
        return self.client.choose(state, question, choice_to_model)


def _validate_models(models: Sequence[str]) -> list[str]:
    if isinstance(models, (str, bytes)):
        raise ConfigurationError("models must be a sequence of model names, not a string.")
    values = [str(model).strip() for model in models]
    if not values or any(not model for model in values):
        raise ConfigurationError("models must contain at least one non-empty name.")
    if len(values) != len(set(values)):
        raise ConfigurationError("models must not contain duplicates.")
    return values


def _validate_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ConfigurationError("query must be a non-empty string.")
    return query.strip()


def _read_jsonl(path: Path) -> list[HistoryRecord]:
    records: list[HistoryRecord] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HistoryFormatError(
                    f"Invalid JSON on history line {line_number}: {exc.msg}."
                ) from exc
            if not isinstance(value, Mapping):
                raise HistoryFormatError(
                    f"History line {line_number} must contain a JSON object."
                )
            try:
                records.append(HistoryRecord.from_mapping(value))
            except HistoryFormatError as exc:
                raise HistoryFormatError(f"Invalid history line {line_number}: {exc}") from exc
    return records


def _read_csv(path: Path) -> list[HistoryRecord]:
    records: list[HistoryRecord] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"query", "model", "performance"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise HistoryFormatError(
                "CSV history needs query, model, and performance columns."
            )
        for line_number, value in enumerate(reader, start=2):
            try:
                records.append(HistoryRecord.from_mapping(value))
            except HistoryFormatError as exc:
                raise HistoryFormatError(f"Invalid history line {line_number}: {exc}") from exc
    return records
