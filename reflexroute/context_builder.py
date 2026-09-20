"""Build one structured Jev Choice decision from priors and history."""

from __future__ import annotations

from typing import Mapping, Sequence

from .profiles import unknown_profile
from .retrieval import retrieve_balanced
from .schemas import HistoryRecord, ModelProfile


QUESTION_ID = "selected_model"


class ContextBuilder:
    def __init__(self, *, history_top_k: int = 4) -> None:
        if history_top_k <= 0:
            raise ValueError("history_top_k must be positive.")
        self.history_top_k = history_top_k

    def build(
        self,
        query: str,
        candidate_models: Sequence[str],
        profiles: Mapping[str, ModelProfile],
        history: Sequence[HistoryRecord],
    ) -> tuple[dict[str, object], dict[str, object], dict[str, str]]:
        """Return ``state``, one Choice question, and opaque-id mapping."""

        retrieved = retrieve_balanced(
            query, candidate_models, history, self.history_top_k
        )
        candidates: dict[str, object] = {}
        choice_to_model: dict[str, str] = {}
        criteria: dict[str, str] = {}
        for index, model in enumerate(candidate_models):
            choice_id = f"model_{index}"
            choice_to_model[choice_id] = model
            profile = profiles.get(model, unknown_profile(model))
            evidence = [item.to_context() for item in retrieved[model]]
            candidates[model] = {
                "prior": profile.to_context(),
                "historical_evidence": evidence,
            }
            criteria[choice_id] = (
                f"Select {model} when its prior and the supplied user observations "
                "indicate it is the best fit for the target query."
            )

        state: dict[str, object] = {
            "description": (
                "A model-routing decision. Performance is higher-is-better; cost and "
                "latency are lower-is-better. Similarity is TF-IDF cosine similarity. "
                "Built-in profiles are fallible priors, not absolute facts."
            ),
            "target_query": query,
            "candidate_models": candidates,
        }
        question: dict[str, object] = {
            "type": "choice",
            "instructions": (
                "Based on the provided model priors, historical observations, and "
                "target query, select the candidate model most suitable for handling "
                "the query. Relevant user historical evidence has priority over and "
                "should override a generic built-in prior when they conflict. Do not "
                "perform budget arithmetic: any hard budget has already been applied "
                "by the caller before this decision. Choose exactly one candidate."
            ),
            "criteria": criteria,
        }
        return state, question, choice_to_model
