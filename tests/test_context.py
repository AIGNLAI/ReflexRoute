from reflexroute import HistoryRecord
from reflexroute.context_builder import ContextBuilder
from reflexroute.profiles import load_profiles


def test_context_has_priors_balanced_evidence_and_one_choice():
    profiles = load_profiles()
    models = ["openai/gpt-5.6", "google/gemini-flash"]
    history = [
        HistoryRecord("write concurrent code", models[0], 0.9),
        HistoryRecord("write simple code", models[0], 0.7),
        HistoryRecord("summarize code", models[1], 0.8),
    ]
    state, question, choice_map = ContextBuilder(history_top_k=1).build(
        "write concurrent code", models, profiles, history
    )

    assert choice_map == {"model_0": models[0], "model_1": models[1]}
    assert question["type"] == "choice"
    assert set(question["criteria"]) == set(choice_map)
    assert "override" in question["instructions"]
    assert state["target_query"] == "write concurrent code"
    candidates = state["candidate_models"]
    assert len(candidates[models[0]]["historical_evidence"]) == 1
    assert len(candidates[models[1]]["historical_evidence"]) == 1
    assert candidates[models[0]]["prior"]["profile_version"] == "0.1.0"


def test_unknown_model_gets_honest_fallback_prior():
    state, _, _ = ContextBuilder().build("query", ["private/model"], {}, [])
    prior = state["candidate_models"]["private/model"]["prior"]
    assert prior["profile_available"] is False
    assert "unknown capabilities" in prior["description"]
