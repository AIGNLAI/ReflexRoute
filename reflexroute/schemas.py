"""Public data structures used by ReflexRoute."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping


class ReflexRouteError(RuntimeError):
    """Base class for errors raised by ReflexRoute."""


class ConfigurationError(ReflexRouteError):
    """Raised when router configuration is invalid."""


class HistoryFormatError(ReflexRouteError):
    """Raised when a history file or record is invalid."""


class BudgetError(ReflexRouteError):
    """Raised when no candidate has a known cost within the hard budget."""


class JevAPIError(ReflexRouteError):
    """Raised when the Jev Decisions API cannot return a usable decision."""


@dataclass(frozen=True)
class HistoryRecord:
    """One observed model outcome for a query.

    ``performance`` is a higher-is-better user metric. Cost is expressed in
    USD per request and latency in seconds when those optional values exist.
    """

    query: str
    model: str
    performance: float
    cost: float | None = None
    latency: float | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HistoryRecord":
        try:
            record = cls(
                query=str(value["query"]).strip(),
                model=str(value["model"]).strip(),
                performance=float(value["performance"]),
                cost=_optional_float(value.get("cost")),
                latency=_optional_float(value.get("latency")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise HistoryFormatError(
                "History records require query, model, and numeric performance; "
                "cost and latency are optional numeric fields."
            ) from exc
        record.validate()
        return record

    def validate(self) -> None:
        if not self.query or not self.model:
            raise HistoryFormatError("History query and model must be non-empty.")
        if not math.isfinite(self.performance):
            raise HistoryFormatError("History performance must be finite.")
        for name, value in (("cost", self.cost), ("latency", self.latency)):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise HistoryFormatError(
                    f"History {name} must be finite and non-negative when supplied."
                )

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class ModelProfile:
    """A versioned, non-authoritative prior for one model."""

    model: str
    description: str
    strengths: tuple[str, ...] = ()
    cost_level: str | None = None
    latency_level: str | None = None
    estimated_cost: float | None = None
    profile_version: str | None = None
    source: Any = None
    default_candidate: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_context(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "description": self.description,
            "strengths": list(self.strengths),
            "cost_level": self.cost_level,
            "latency_level": self.latency_level,
            "estimated_cost": self.estimated_cost,
            "profile_version": self.profile_version,
            "source": self.source,
        }
        result.update(self.extra)
        return {key: value for key, value in result.items() if value is not None}


@dataclass(frozen=True)
class RoutingResult:
    """The selected model and Jev's calibrated distribution."""

    model: str
    probabilities: dict[str, float]
    confidence: float
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def selected_model(self) -> str:
        """Compatibility alias for callers that prefer an explicit name."""

        return self.model

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
