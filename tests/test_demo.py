import pytest

from demo.app import benchmark_payload, config_payload, replay_payload, route_payload


def test_demo_config_and_replay_are_complete(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config = config_payload()
    replay = replay_payload()

    assert config["live_available"] is False
    assert len(config["models"]) == 4
    assert config["history_count"] == 8
    assert [result["mode"] for result in replay["results"]] == [
        "zero-shot",
        "few-shot",
    ]
    assert replay["results"][0]["selected_model"] != replay["results"][1][
        "selected_model"
    ]
    assert len(replay["results"][1]["evidence"]) == 8


def test_demo_benchmark_reads_the_committed_summary():
    benchmark = benchmark_payload()

    assert benchmark["target_count"] == 191
    assert benchmark["few_shot"]["quality_mean"] > benchmark["zero_shot"][
        "quality_mean"
    ]


def test_live_route_requires_server_side_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(PermissionError, match="OPENROUTER_API_KEY"):
        route_payload("Route this query", "zero-shot")
