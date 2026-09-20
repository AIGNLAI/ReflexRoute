import json

import pytest

from reflexroute import BudgetError, HistoryFormatError, Router, RoutingResult


class FakeClient:
    def __init__(self):
        self.calls = []

    def choose(self, state, question, choice_to_model):
        self.calls.append((state, question, choice_to_model))
        first = next(iter(choice_to_model.values()))
        return RoutingResult(first, {model: 0.5 for model in choice_to_model.values()}, 0.5)


def test_zero_shot_route_builds_structured_decision():
    client = FakeClient()
    router = Router(
        ["openai/gpt-5.6", "google/gemini-flash"], client=client
    )
    result = router.route("Implement a rate limiter")
    assert result.model == "openai/gpt-5.6"
    state, question, choice_map = client.calls[0]
    assert "messages" not in state
    assert set(state["candidate_models"]) == set(router.models)
    assert question["type"] == "choice"
    assert set(choice_map.values()) == set(router.models)


def test_budget_filters_before_api_and_single_candidate_short_circuits():
    client = FakeClient()
    router = Router(
        ["openai/gpt-5.6", "google/gemini-flash"], client=client
    )
    result = router.route("Summarize", budget=0.01)
    assert result.model == "google/gemini-flash"
    assert result.probabilities == {"google/gemini-flash": 1.0}
    assert result.usage["local_short_circuit"] is True
    assert client.calls == []


def test_impossible_budget_and_unknown_cost_raise():
    with pytest.raises(BudgetError, match="No candidate"):
        Router(["openai/gpt-5.6"]).route("query", budget=0.001)
    with pytest.raises(BudgetError, match="unknown/private"):
        Router(["unknown/private"]).route("query", budget=10)


def test_one_candidate_needs_no_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    result = Router(["unknown/private"]).route("query")
    assert result.model == "unknown/private"
    assert result.confidence == 1.0


def test_load_jsonl_and_csv(tmp_path):
    jsonl = tmp_path / "history.jsonl"
    jsonl.write_text(
        json.dumps({"query": "q", "model": "m", "performance": 1.0}) + "\n",
        encoding="utf-8",
    )
    csv_file = tmp_path / "history.csv"
    csv_file.write_text(
        "query,model,performance,cost,latency\nq2,m2,0.8,0.2,1.4\n",
        encoding="utf-8",
    )
    router = Router(["m"])
    assert router.load_history(jsonl) == 1
    assert router.history[0].cost is None
    assert router.load_history(csv_file) == 1
    assert router.history[0].latency == pytest.approx(1.4)


def test_invalid_history_has_line_context(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"query":"missing fields"}\n', encoding="utf-8")
    with pytest.raises(HistoryFormatError, match="line 1"):
        Router(["m"]).load_history(path)


def test_build_decision_applies_budget_to_choice_map():
    router = Router([
        "openai/gpt-5.6",
        "google/gemini-flash",
        "meta-llama/llama-3.3-70b-instruct",
    ])
    state, _, choice_map = router.build_decision("query", budget=0.01)
    assert set(choice_map.values()) == {
        "google/gemini-flash",
        "meta-llama/llama-3.3-70b-instruct",
    }
    assert set(state["candidate_models"]) == set(choice_map.values())
