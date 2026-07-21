"""
LLM client wrapper for Groq API.

Provides:
- Retry with exponential backoff on transient failures
- Timeout handling
- JSON response parsing with markdown fence stripping
- Automatic pydantic validation
- Latency and token telemetry
- Optional caching keyed by prompt hash
"""

from __future__ import annotations

import hashlib
import json
import time
from   pathlib import Path
from   typing  import Any, TypeVar

from groq   import Groq
from groq   import APIStatusError, APITimeoutError, APIConnectionError
from pydantic import BaseModel, ValidationError

from astrorag.config     import Settings, get_settings
from astrorag.llm.models import LLMResponse
from astrorag.logger     import get_logger
from astrorag.paths      import get_paths

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


# ══════════════════════════════════════════════════════════
# LLM client
# ══════════════════════════════════════════════════════════

class LLMClient:
    """
    Wrapper around the Groq client with production-grade features.

    Usage:
        client = LLMClient()
        response = client.chat_json(
            system         = "Return valid JSON only.",
            user           = "Decompose this query: ...",
            schema         = QueryDecomposition,
            max_tokens     = 500,
            stage_name     = "stage0",
        )
        decomposition = response.data
    """

    def __init__(
        self,
        settings:  Settings | None = None,
        api_key:   str      | None = None,
        cache_dir: Path     | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        key = api_key or self.settings.groq_api_key
        if not key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file or set the environment variable."
            )
        self.client = Groq(api_key=key, timeout=self.settings.groq_timeout_seconds)

        # optional cache directory
        self.cache_dir = cache_dir or (get_paths().data_dir / "llm_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_enabled = False   # off by default; turn on for dev

    # ── caching primitives ──────────────────────────────
    def _cache_key(
        self,
        system:      str,
        user:        str,
        model:       str,
        temperature: float,
    ) -> str:
        raw = f"{model}||{temperature}||{system}||{user}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"llm_{key[:16]}.json"

    def _cache_load(self, key: str) -> dict | None:
        p = self._cache_path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _cache_save(self, key: str, payload: dict) -> None:
        try:
            self._cache_path(key).write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"LLM cache save failed: {e}")

    # ── low-level chat call ─────────────────────────────
    def chat(
        self,
        system:      str,
        user:        str,
        model:       str  | None = None,
        temperature: float | None = None,
        max_tokens:  int  = 500,
        stage_name:  str  = "llm",
    ) -> tuple[str, dict]:
        """
        Send a chat completion and return (raw text, telemetry dict).

        Retries on transient failures (timeout, rate limit, connection).
        Does not retry on other errors — those should surface.
        """
        model       = model       or self.settings.groq_model
        temperature = temperature if temperature is not None else self.settings.groq_temperature

        last_error: Exception | None = None
        for attempt in range(self.settings.groq_max_retries + 1):
            try:
                t0 = time.time()
                resp = self.client.chat.completions.create(
                    model       = model,
                    messages    = [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    temperature = temperature,
                    max_tokens  = max_tokens,
                )
                elapsed = time.time() - t0

                raw   = resp.choices[0].message.content or ""
                usage = getattr(resp, "usage", None)
                telemetry = {
                    "model":           model,
                    "latency_seconds": elapsed,
                    "input_tokens":    getattr(usage, "prompt_tokens", 0)     if usage else 0,
                    "output_tokens":   getattr(usage, "completion_tokens", 0) if usage else 0,
                    "retries":         attempt,
                }
                logger.debug(
                    f"[{stage_name}] LLM ok in {elapsed:.2f}s "
                    f"(attempt {attempt+1}) — "
                    f"tokens in={telemetry['input_tokens']} "
                    f"out={telemetry['output_tokens']}"
                )
                return raw, telemetry

            except (APITimeoutError, APIConnectionError) as e:
                last_error = e
                wait = 2 ** attempt
                logger.warning(
                    f"[{stage_name}] transient LLM error "
                    f"({type(e).__name__}) — retrying in {wait}s"
                )
                time.sleep(wait)

            except APIStatusError as e:
                # 429 rate limit — retry with backoff
                if e.status_code == 429 and attempt < self.settings.groq_max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        f"[{stage_name}] rate limited — waiting {wait}s"
                    )
                    time.sleep(wait)
                    last_error = e
                    continue
                # other status errors — do not retry
                logger.error(f"[{stage_name}] LLM status error: {e}")
                raise

            except Exception as e:
                logger.error(
                    f"[{stage_name}] LLM unexpected error: "
                    f"{type(e).__name__} — {e}"
                )
                raise

        # all retries exhausted
        assert last_error is not None
        raise last_error

    # ── JSON chat with schema validation ─────────────────
    def chat_json(
        self,
        system:      str,
        user:        str,
        schema:      type[T],
        model:       str  | None = None,
        temperature: float | None = None,
        max_tokens:  int  = 500,
        stage_name:  str  = "llm",
        use_cache:   bool = False,
    ) -> LLMResponse:
        """
        Send a chat completion expecting JSON, validate against schema.

        Args:
            system:      System prompt content.
            user:        User prompt content.
            schema:      Pydantic model class to validate against.
            max_tokens:  Max output tokens.
            stage_name:  Tag for telemetry logs.
            use_cache:   If True, cache by prompt hash.

        Returns:
            LLMResponse with .data validated as the schema type.

        Raises:
            ValueError on validation failure after retry.
            APIStatusError on non-retryable API failure.
        """
        model       = model       or self.settings.groq_model
        temperature = temperature if temperature is not None else self.settings.groq_temperature

        # ── cache path ──────────────────────────────────
        if use_cache and self.cache_enabled:
            cache_key = self._cache_key(system, user, model, temperature)
            cached    = self._cache_load(cache_key)
            if cached is not None:
                try:
                    data = schema(**cached["data"])
                    logger.debug(f"[{stage_name}] LLM cache hit")
                    return LLMResponse(
                        data       = data,
                        model      = cached.get("model", model),
                        from_cache = True,
                    )
                except ValidationError:
                    logger.debug(f"[{stage_name}] cached data invalid; regenerating")

        # ── call with parse retry ───────────────────────
        last_error: Exception | None = None
        for parse_attempt in range(2):
            raw, tele = self.chat(
                system      = system,
                user        = user,
                model       = model,
                temperature = temperature,
                max_tokens  = max_tokens,
                stage_name  = stage_name,
            )

            cleaned = _strip_json_fences(raw)
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(
                    f"[{stage_name}] LLM returned invalid JSON "
                    f"(attempt {parse_attempt+1}) — "
                    f"snippet: {cleaned[:200]}"
                )
                if parse_attempt == 0:
                    # retry once with reinforced prompt
                    user = (
                        user + "\n\nIMPORTANT: Return ONLY valid JSON. "
                        "No markdown, no code fences, no commentary."
                    )
                    continue
                break

            try:
                validated = schema(**parsed)
            except ValidationError as e:
                last_error = e
                logger.warning(
                    f"[{stage_name}] LLM output failed schema validation "
                    f"(attempt {parse_attempt+1}) — "
                    f"errors: {e.errors()[:3]}"
                )
                if parse_attempt == 0:
                    user = (
                        user + "\n\nIMPORTANT: Follow the exact JSON schema "
                        "specified. All required keys must be present."
                    )
                    continue
                break

            # ── success ─────────────────────────────────
            response = LLMResponse(
                data            = validated,
                model           = tele["model"],
                latency_seconds = tele["latency_seconds"],
                input_tokens    = tele["input_tokens"],
                output_tokens   = tele["output_tokens"],
                retries         = tele["retries"],
                from_cache      = False,
            )

            if use_cache and self.cache_enabled:
                self._cache_save(cache_key, {
                    "model": tele["model"],
                    "data":  validated.model_dump(),
                })

            return response

        assert last_error is not None
        raise ValueError(
            f"LLM call failed after retries — last error: {last_error}"
        )


# ══════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════

def _strip_json_fences(text: str) -> str:
    """
    Remove markdown code fences from LLM output.

    Handles ```json, ```, and leading/trailing whitespace.
    """
    t = text.strip()
    # remove leading fence
    for fence in ("```json", "```JSON", "```"):
        if t.startswith(fence):
            t = t[len(fence):].lstrip("\n")
            break
    # remove trailing fence
    if t.endswith("```"):
        t = t[:-3].rstrip()
    return t.strip()


# ══════════════════════════════════════════════════════════
# singleton accessor
# ══════════════════════════════════════════════════════════

_client_instance: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Return the singleton LLMClient instance."""
    global _client_instance
    if _client_instance is None:
        _client_instance = LLMClient()
    return _client_instance