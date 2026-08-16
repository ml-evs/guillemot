"""Model construction, with retries for rate limiting.

Providers return HTTP 429 when a quota is exhausted, usually with a hint about how
long to wait. Left alone that aborts the whole agent run, losing any refinement work
already done in that turn, so the retry belongs inside the HTTP client rather than
around `agent.run`: the request is retried in place and the run carries on.

Configured through the environment:

* ``GUILLEMOT_RETRY_ATTEMPTS`` — how many times to try a request (default 5).
* ``GUILLEMOT_RETRY_BACKOFF_S`` — seconds to wait before the first retry (default 15).
* ``GUILLEMOT_RETRY_MAX_WAIT_S`` — cap on any single wait (default 300).
* ``GUILLEMOT_OLLAMA_URL`` — the ollama server, when using an ``ollama:`` model.
"""

import logging
import os

from pydantic_ai.models import Model

logger = logging.getLogger(__name__)

# Set by the application once logfire has been configured, so failed model calls are
# reported alongside the agent run spans rather than into a separate log.
_logfire_enabled = False


def use_logfire(enabled: bool = True) -> None:
    """Send failure reports to logfire instead of the standard library logger."""
    global _logfire_enabled
    _logfire_enabled = enabled


def _warn(message: str, **attributes) -> None:
    if _logfire_enabled:
        import logfire

        logfire.warning(message, **attributes)
    else:
        logger.warning("%s %s", message, attributes)


DEFAULT_ATTEMPTS = 5
DEFAULT_BACKOFF_S = 15.0
DEFAULT_MAX_WAIT_S = 300.0
DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"

# Rate limiting, plus the transient server-side failures worth another go.
RETRY_STATUS_CODES = [429, 500, 502, 503, 504]


class RetrySettings:
    def __init__(self) -> None:
        self.attempts = int(os.getenv("GUILLEMOT_RETRY_ATTEMPTS", DEFAULT_ATTEMPTS))
        self.backoff_s = float(
            os.getenv("GUILLEMOT_RETRY_BACKOFF_S", DEFAULT_BACKOFF_S)
        )
        self.max_wait_s = float(
            os.getenv("GUILLEMOT_RETRY_MAX_WAIT_S", DEFAULT_MAX_WAIT_S)
        )

    def __repr__(self) -> str:
        return (
            f"RetrySettings(attempts={self.attempts}, backoff_s={self.backoff_s}, "
            f"max_wait_s={self.max_wait_s})"
        )


def _describe_failure(status_code: int, retry_after: str | None) -> str:
    reason = "rate limited" if status_code == 429 else f"HTTP {status_code}"
    if retry_after:
        return f"{reason}; provider asked for {retry_after}s"
    return reason


async def _log_failed_response(response) -> None:
    """httpx event hook: report every failed request, including retried ones.

    Without this a retry is invisible — the run simply pauses — so a rate-limited
    session looks indistinguishable from a hung one.
    """
    if response.status_code < 400:
        return
    # The body is not read here: it may be a streaming response that has not been
    # consumed yet, and reading it in a hook would break the caller.
    _warn(
        "Model request failed: {reason}",
        reason=_describe_failure(
            response.status_code, response.headers.get("retry-after")
        ),
        status_code=response.status_code,
        retry_after=response.headers.get("retry-after"),
        method=response.request.method,
        url=str(response.request.url),
    )


def _log_before_sleep(retry_state) -> None:
    """tenacity hook: say what failed and how long we are waiting before trying again."""
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    status_code = getattr(getattr(exception, "response", None), "status_code", "?")
    retry_after = None
    if exception is not None and getattr(exception, "response", None) is not None:
        retry_after = exception.response.headers.get("retry-after")

    sleep_s = retry_state.next_action.sleep if retry_state.next_action else 0.0
    _warn(
        "Model request failed (attempt {attempt}): {reason} — retrying in {sleep_s:.0f}s",
        attempt=retry_state.attempt_number,
        reason=(
            _describe_failure(status_code, retry_after)
            if isinstance(status_code, int)
            else str(exception)
        ),
        status_code=status_code,
        retry_after=retry_after,
        sleep_s=sleep_s,
    )


def _google_model(model_id: str, settings: RetrySettings) -> Model:
    """Gemini, with retries handled by the Google SDK's own HTTP layer.

    `GoogleProvider` takes a `google.genai.Client` rather than an httpx client, so the
    generic httpx retry transport cannot be used here; the SDK's `HttpRetryOptions`
    does the same job.
    """
    from google.genai import Client
    from google.genai.types import HttpOptions, HttpRetryOptions
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    client = Client(
        api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
        http_options=HttpOptions(
            # The SDK swallows the retried failures, so hook httpx to report them.
            async_client_args={"event_hooks": {"response": [_log_failed_response]}},
            retry_options=HttpRetryOptions(
                attempts=settings.attempts,
                initial_delay=settings.backoff_s,
                max_delay=settings.max_wait_s,
                exp_base=2,
                http_status_codes=RETRY_STATUS_CODES,
            ),
        ),
    )
    return GoogleModel(model_id, provider=GoogleProvider(client=client))


def _retrying_http_client(settings: RetrySettings):
    """An httpx client that retries rate-limited requests, honouring `Retry-After`."""
    import httpx
    from pydantic_ai.retries import (
        AsyncTenacityTransport,
        RetryConfig,
        wait_retry_after,
    )
    from tenacity import (
        retry_if_exception,
        stop_after_attempt,
        wait_exponential,
    )

    def is_retryable(exc: BaseException) -> bool:
        return (
            isinstance(exc, httpx.HTTPStatusError)
            and exc.response.status_code in RETRY_STATUS_CODES
        )

    transport = AsyncTenacityTransport(
        config=RetryConfig(
            retry=retry_if_exception(is_retryable),
            # Prefer the server's own Retry-After; fall back to our backoff otherwise.
            wait=wait_retry_after(
                fallback_strategy=wait_exponential(
                    multiplier=settings.backoff_s, max=settings.max_wait_s
                ),
                max_wait=settings.max_wait_s,
            ),
            stop=stop_after_attempt(settings.attempts),
            before_sleep=_log_before_sleep,
            reraise=True,
        ),
        validate_response=lambda response: response.raise_for_status(),
    )
    return httpx.AsyncClient(transport=transport, timeout=120)


def _openai_model(model_id: str, settings: RetrySettings) -> Model:
    """OpenAI, with retries in an httpx transport that honours `Retry-After`."""
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    return OpenAIChatModel(
        model_id,
        provider=OpenAIProvider(http_client=_retrying_http_client(settings)),
    )


def _ollama_model(model_id: str, settings: RetrySettings) -> Model:
    """A model served by ollama, locally or on a machine in the group.

    Ollama speaks the OpenAI API, so the same retry transport applies. Point
    ``GUILLEMOT_OLLAMA_URL`` at the server; it defaults to a local one.
    """
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.ollama import OllamaProvider

    return OpenAIChatModel(
        model_id,
        provider=OllamaProvider(
            base_url=os.getenv("GUILLEMOT_OLLAMA_URL", DEFAULT_OLLAMA_URL),
            http_client=_retrying_http_client(settings),
        ),
    )


def build_model(model_name: str) -> Model | str:
    """Build the configured model with rate-limit retries where we know how.

    Falls back to returning the name unchanged for providers we have not wired up, so
    an unfamiliar `GUILLEMOT_AI_MODEL` still works — just without the retry handling.
    """
    settings = RetrySettings()

    provider, _, model_id = model_name.partition(":")
    if not model_id:
        # A bare model name, e.g. "gemini-2.5-flash-lite"
        provider, model_id = "google-gla", model_name

    if provider in ("google-gla", "google-vertex"):
        return _google_model(model_id, settings)
    if provider == "openai":
        return _openai_model(model_id, settings)
    if provider == "ollama":
        return _ollama_model(model_id, settings)

    return model_name
