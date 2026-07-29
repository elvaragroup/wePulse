from __future__ import annotations

import json
from dataclasses import replace

import pytest

from src.llm import (
    FakeLLMClient,
    LLMRequest,
    SchemaValidationFailure,
    structured_schema,
)
from src.models import JudgeVerdict, NaiveBaselineWire, PersonaReaction

VALID = json.dumps(
    {
        "reaction": "criticize",
        "categories": ["privacy"],
        "intensity": 0.8,
        "quote": "Opt-out is not consent.",
        "reasoning": "Defaults users in.",
    }
)
MALFORMED = "here is my answer: {not json"
WRONG_SHAPE = json.dumps({"reaction": "criticize", "categories": [], "intensity": 0.8})


def req(**overrides) -> LLMRequest:
    base = {
        "role": "persona",
        "model": "claude-haiku-4-5-20251001",
        "system": "You are a privacy hawk.",
        "user": "Acme announced a thing.",
        "max_tokens": 1024,
        "temperature": 0.7,
        "schema_name": "PersonaReaction",
        "schema": structured_schema(PersonaReaction),
    }
    return LLMRequest(**{**base, **overrides})


# --- cache key sensitivity (spec 6) ---


def test_cache_key_is_stable():
    assert req().cache_key() == req().cache_key()


@pytest.mark.parametrize(
    "change",
    [
        {"temperature": 0.0},
        {"temperature": None},
        {"system": "You are a labour advocate."},
        {"user": "Acme announced a different thing."},
        {"model": "claude-sonnet-5"},
        {"max_tokens": 2048},
        {"thinking": "disabled"},
    ],
)
def test_cache_key_changes_when_the_request_changes(change):
    assert req().cache_key() != req(**change).cache_key()


def test_cache_key_ignores_role_label():
    """role is bookkeeping, not a request parameter -- it must not fragment the cache."""
    assert req().cache_key() == req(role="naive").cache_key()


def test_cache_key_tracks_the_schema():
    assert req().cache_key() != req(schema=structured_schema(JudgeVerdict)).cache_key()


# --- schema retry (spec 6) ---


async def test_malformed_then_valid_succeeds():
    def responder(_request, attempt):
        return VALID if attempt == 3 else MALFORMED

    client = FakeLLMClient(responder, max_retries=3, backoff_base=0)
    result = await client.complete(req(), PersonaReaction)
    assert result.categories == ["privacy"]
    assert client.calls_made == 3


async def test_invalid_on_every_attempt_raises():
    client = FakeLLMClient(lambda _r, _a: MALFORMED, max_retries=3, backoff_base=0)
    with pytest.raises(SchemaValidationFailure, match="3 attempts"):
        await client.complete(req(), PersonaReaction)
    assert client.calls_made == 3


async def test_schema_violation_also_retries():
    """Parseable JSON that breaks an invariant must be retried, not accepted."""
    client = FakeLLMClient(lambda _r, _a: WRONG_SHAPE, max_retries=3, backoff_base=0)
    with pytest.raises(SchemaValidationFailure):
        await client.complete(req(), PersonaReaction)


async def test_valid_first_time_makes_one_call():
    client = FakeLLMClient(lambda _r, _a: VALID, max_retries=3, backoff_base=0)
    await client.complete(req(), PersonaReaction)
    assert client.calls_made == 1


# --- caching ---


async def test_second_identical_request_hits_cache(tmp_path):
    client = FakeLLMClient(lambda _r, _a: VALID, cache_dir=tmp_path, backoff_base=0)
    await client.complete(req(), PersonaReaction)
    await client.complete(req(), PersonaReaction)
    assert client.calls_made == 1
    assert client.cache_hits == 1


async def test_cache_survives_a_new_client(tmp_path):
    first = FakeLLMClient(lambda _r, _a: VALID, cache_dir=tmp_path, backoff_base=0)
    await first.complete(req(), PersonaReaction)

    def explode(_r, _a):
        raise AssertionError("should have been served from cache")

    second = FakeLLMClient(explode, cache_dir=tmp_path, backoff_base=0)
    result = await second.complete(req(), PersonaReaction)
    assert result.reaction == "criticize"
    assert second.calls_made == 0


async def test_changed_prompt_misses_cache(tmp_path):
    client = FakeLLMClient(lambda _r, _a: VALID, cache_dir=tmp_path, backoff_base=0)
    await client.complete(req(), PersonaReaction)
    await client.complete(req(user="Something else entirely."), PersonaReaction)
    assert client.calls_made == 2


async def test_invalid_responses_are_not_cached(tmp_path):
    calls = {"n": 0}

    def responder(_r, _a):
        calls["n"] += 1
        return MALFORMED

    client = FakeLLMClient(responder, cache_dir=tmp_path, max_retries=2, backoff_base=0)
    with pytest.raises(SchemaValidationFailure):
        await client.complete(req(), PersonaReaction)
    assert not list(tmp_path.glob("*.json"))


# --- transport retry ---


class Flaky(Exception):
    status_code = 429


async def test_retryable_transport_error_backs_off_then_succeeds():
    state = {"failures": 2}

    def responder(_r, _a):
        if state["failures"]:
            state["failures"] -= 1
            raise Flaky("rate limited")
        return VALID

    client = FakeLLMClient(responder, max_retries=3, backoff_base=0)
    result = await client.complete(req(), PersonaReaction)
    assert result.reaction == "criticize"


async def test_non_retryable_error_propagates():
    class BadRequest(Exception):
        status_code = 400

    client = FakeLLMClient(
        lambda _r, _a: (_ for _ in ()).throw(BadRequest("bad")), max_retries=3, backoff_base=0
    )
    with pytest.raises(BadRequest):
        await client.complete(req(), PersonaReaction)


# --- schema cleaning ---


def test_structured_schema_strips_unsupported_constraints():
    schema = structured_schema(PersonaReaction)
    serialised = json.dumps(schema)
    assert "minimum" not in serialised
    assert "maximum" not in serialised
    assert "maxItems" not in serialised


def test_structured_schema_forbids_additional_properties():
    schema = structured_schema(PersonaReaction)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])


def test_structured_schema_recurses_into_defs():
    schema = structured_schema(NaiveBaselineWire)
    for definition in schema.get("$defs", {}).values():
        if definition.get("type") == "object":
            assert definition["additionalProperties"] is False


def test_pydantic_still_enforces_stripped_constraints():
    """Constraints removed from the wire schema are still validated client-side,
    which is why stripping them is safe."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PersonaReaction.model_validate_json(
            json.dumps(
                {
                    "reaction": "criticize",
                    "categories": ["privacy"],
                    "intensity": 9.9,
                    "quote": "x",
                    "reasoning": "y",
                }
            )
        )


def test_request_is_frozen():
    original = req()
    changed = replace(original, temperature=0.1)
    assert original.temperature == 0.7
    assert changed.temperature == 0.1
