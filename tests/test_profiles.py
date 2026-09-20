import pytest

from reflexroute import Router
from reflexroute.profiles import load_profiles


def test_catalog_has_at_least_one_hundred_traceable_profiles():
    profiles = load_profiles()
    assert len(profiles) == 130
    assert len({model.split("/", 1)[0] for model in profiles}) >= 20

    for model, profile in profiles.items():
        assert "/" in model
        assert not model.endswith(":batch")
        assert not model.endswith(":free")
        assert profile.description
        assert profile.strengths
        assert profile.estimated_cost is not None
        assert profile.estimated_cost >= 0
        assert profile.profile_version == "2026-09-20.1"
        assert profile.source["url"] == "https://openrouter.ai/api/v1/models"
        assert profile.source["retrieved_at"] == "2026-09-20"
        assert profile.extra["context_length"] > 0
        assert profile.extra["input_modalities"]
        assert profile.extra["output_modalities"]


def test_estimated_cost_uses_documented_token_assumption():
    for profile in load_profiles().values():
        prompt = float(profile.extra["prompt_cost_per_million"])
        completion = float(profile.extra["completion_cost_per_million"])
        expected = prompt / 1_000_000 * 2_000 + completion / 1_000_000 * 1_000
        assert profile.estimated_cost == pytest.approx(expected, abs=1e-8)


def test_default_router_uses_bounded_cross_provider_shortlist():
    router = Router()
    assert 10 <= len(router.models) <= 20
    assert len({model.split("/", 1)[0] for model in router.models}) >= 10
    assert all(router.profiles[model].default_candidate for model in router.models)
    assert len(router.models) < len(router.profiles)
