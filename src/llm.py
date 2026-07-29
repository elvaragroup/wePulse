"""LLM client layer: cache -> retry -> call.

Every response is cached by a hash of the full request, so a rerun costs nothing
and a run is byte-reproducible from its manifest plus its cache (spec 4.3). The
Anthropic API exposes no seed parameter, so the cache is the *only* thing
delivering reproducibility -- treat it as part of the run record, not a
performance optimisation.

Both the real client and FakeLLMClient return raw response text from `_raw()`;
validation happens once, in the base class, so the schema-retry path is the same
in tests as in production.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

# JSON Schema keywords structured outputs does not support. Left in place they
# risk a 400; pydantic still enforces them client-side after parsing.
UNSUPPORTED_KEYWORDS = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minProperties",
        "maxProperties",
        "default",
    }
)

SUPPORTED_FORMATS = frozenset(
    {"date-time", "time", "date", "duration", "email", "hostname", "uri", "ipv4", "ipv6", "uuid"}
)


class SchemaValidationFailure(RuntimeError):
    """The model produced output that would not validate, `max_retries` times."""


@dataclass(frozen=True)
class LLMRequest:
    role: str
    model: str
    system: str
    user: str
    max_tokens: int
    temperature: float | None = None
    thinking: str | None = None
    schema_name: str | None = None
    schema: dict[str, Any] | None = field(default=None, repr=False)

    def cache_key(self) -> str:
        """sha256 over a canonical rendering of everything that shapes the response.

        DEVIATION from spec 4.3, which specifies
        sha256(model + system + user + temperature + seed): there is no seed
        parameter on the Anthropic API, and temperature is absent for models that
        reject it. Hashing the whole request is strictly stronger -- max_tokens,
        thinking mode, and the output schema all change the response and all
        belong in the key.
        """
        payload = {
            "model": self.model,
            "system": self.system,
            "user": self.user,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "thinking": self.thinking,
            "schema": self.schema,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def structured_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic JSON schema, trimmed to what structured outputs accepts."""

    def clean(node: Any) -> Any:
        if isinstance(node, list):
            return [clean(item) for item in node]
        if not isinstance(node, dict):
            return node

        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in UNSUPPORTED_KEYWORDS:
                continue
            if key == "format" and value not in SUPPORTED_FORMATS:
                continue
            out[key] = clean(value)

        if out.get("type") == "object":
            out["additionalProperties"] = False
            props = out.get("properties")
            if isinstance(props, dict):
                # Structured outputs requires every property to be listed as
                # required; optionality is expressed with a nullable type.
                out["required"] = list(props)
        return out

    return clean(model.model_json_schema())


class LLMClient(ABC):
    """Cache, retry, and concurrency. Subclasses supply `_raw` only."""

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        concurrency: int = 8,
        max_retries: int = 3,
        backoff_base: float = 0.5,
    ) -> None:
        self.cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self._semaphore = asyncio.Semaphore(concurrency)
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.calls_made = 0
        self.cache_hits = 0

    @abstractmethod
    async def _raw(self, request: LLMRequest, attempt: int) -> str:
        """Return the model's raw text response. Attempt is 1-based."""

    # --- cache ---

    def _cache_path(self, key: str) -> Path | None:
        return None if self.cache_dir is None else self.cache_dir / f"{key}.json"

    def _cache_read(self, request: LLMRequest) -> str | None:
        path = self._cache_path(request.cache_key())
        if path is None or not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))["response"]

    def _cache_write(self, request: LLMRequest, response: str) -> None:
        path = self._cache_path(request.cache_key())
        if path is None:
            return
        path.write_text(
            json.dumps(
                {
                    "role": request.role,
                    "model": request.model,
                    "schema_name": request.schema_name,
                    "response": response,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    # --- public API ---

    async def complete_text(self, request: LLMRequest) -> str:
        cached = self._cache_read(request)
        if cached is not None:
            self.cache_hits += 1
            return cached

        async with self._semaphore:
            text = await self._call_with_backoff(request, attempt=1)

        self._cache_write(request, text)
        return text

    async def complete(self, request: LLMRequest, schema: type[T]) -> T:
        """Validated structured output, retrying up to max_retries on parse failure."""
        cached = self._cache_read(request)
        if cached is not None:
            self.cache_hits += 1
            return schema.model_validate_json(cached)

        errors: list[str] = []
        async with self._semaphore:
            for attempt in range(1, self.max_retries + 1):
                text = await self._call_with_backoff(request, attempt=attempt)
                try:
                    parsed = schema.model_validate_json(text)
                except ValidationError as exc:
                    errors.append(f"attempt {attempt}: {exc.error_count()} error(s)")
                    continue
                self._cache_write(request, text)
                return parsed

        raise SchemaValidationFailure(
            f"{request.role}/{request.model} failed to produce valid "
            f"{schema.__name__} in {self.max_retries} attempts: {'; '.join(errors)}"
        )

    # --- retry on transport errors ---

    async def _call_with_backoff(self, request: LLMRequest, *, attempt: int) -> str:
        last: Exception | None = None
        for transport_attempt in range(1, self.max_retries + 1):
            try:
                self.calls_made += 1
                return await self._raw(request, attempt)
            except Exception as exc:  # noqa: BLE001 - re-raised below
                if not _is_retryable(exc) or transport_attempt == self.max_retries:
                    raise
                last = exc
                delay = self.backoff_base * (2 ** (transport_attempt - 1))
                await asyncio.sleep(delay + random.uniform(0, self.backoff_base))
        raise last  # pragma: no cover - loop always returns or raises


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is not None:
        return status == 429 or status >= 500
    return exc.__class__.__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
    }


class FakeLLMClient(LLMClient):
    """Deterministic offline client. `responder(request, attempt)` returns raw text."""

    def __init__(
        self,
        responder: Callable[[LLMRequest, int], str],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.responder = responder
        self.requests: list[LLMRequest] = []

    async def _raw(self, request: LLMRequest, attempt: int) -> str:
        self.requests.append(request)
        return self.responder(request, attempt)


class AnthropicLLMClient(LLMClient):
    """Real client. Constructed lazily so tests never need an API key."""

    def __init__(self, *, api_key: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        from anthropic import AsyncAnthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in, "
                "or export the variable."
            )
        self._client = AsyncAnthropic(api_key=key, max_retries=0)

    async def _raw(self, request: LLMRequest, attempt: int) -> str:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "system": request.system,
            "messages": [{"role": "user", "content": request.user}],
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.thinking == "disabled":
            kwargs["thinking"] = {"type": "disabled"}
        elif request.thinking == "adaptive":
            kwargs["thinking"] = {"type": "adaptive"}
        if request.schema is not None:
            kwargs["output_config"] = {
                "format": {"type": "json_schema", "schema": request.schema}
            }
        if attempt > 1:
            kwargs["messages"].append(
                {
                    "role": "user",
                    "content": (
                        "Your previous response did not match the required schema. "
                        "Reply with JSON matching the schema exactly and nothing else."
                    ),
                }
            )

        response = await self._client.messages.create(**kwargs)
        return "".join(block.text for block in response.content if block.type == "text")

    async def aclose(self) -> None:
        await self._client.close()
