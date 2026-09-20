import pytest

from reflexroute import JevAPIError
from reflexroute.context_builder import QUESTION_ID
from reflexroute.jev_client import JEV_MODEL, JevClient


def test_choose_uses_fixed_jev_model_and_decisions_shape(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    client = JevClient()
    seen = {}

    def fake_post(payload, api_key):
        seen.update(payload)
        assert api_key == "test-key"
        return {
            "answers": {
                QUESTION_ID: {
                    "choice": "model_1",
                    "probabilities": {"model_0": 0.2, "model_1": 0.8},
                    "confidence": 0.74,
                }
            },
            "usage": {"input_tokens": 123},
        }

    monkeypatch.setattr(client, "_post", fake_post)
    result = client.choose(
        {"target_query": "q"},
        {"type": "choice", "criteria": {}},
        {"model_0": "a", "model_1": "b"},
    )
    assert seen["model"] == JEV_MODEL == "typesafe/jev-1.13"
    assert "messages" not in seen
    assert result.model == "b"
    assert result.probabilities == {"a": 0.2, "b": 0.8}
    assert result.confidence == pytest.approx(0.74)
    assert result.usage["input_tokens"] == 123


def test_missing_key_and_bad_response_are_clear(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(JevAPIError, match="OPENROUTER_API_KEY"):
        JevClient().choose({}, {}, {"model_0": "a"})

    with pytest.raises(JevAPIError, match="missing"):
        JevClient.parse_response({}, {"model_0": "a"})


def test_confidence_defaults_to_selected_probability():
    result = JevClient.parse_response(
        {
            "answers": {
                QUESTION_ID: {
                    "choice": "model_0",
                    "probabilities": {"model_0": 0.9, "model_1": 0.1},
                }
            }
        },
        {"model_0": "a", "model_1": "b"},
    )
    assert result.confidence == pytest.approx(0.9)
