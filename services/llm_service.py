"""
LLM Service Module
Handles all communication with the LLM API.
Supports OpenAI, Groq, and OpenAI-compatible APIs via environment variables.
"""

import os
import time
import logging
from typing import Generator
from pathlib import Path

from dotenv import load_dotenv

# Explicitly resolve absolute path to .env file
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Configuration ─────────────────────────────────────────────────────────────

class LLMConfig:
    """Centralized configuration loaded from environment variables."""

    def __init__(self):
        load_dotenv(dotenv_path=ENV_PATH, override=True)
        self.api_provider  = os.getenv("LLM_PROVIDER", "groq").lower()          # groq | openai
        self.api_key       = os.getenv("LLM_API_KEY", "")
        self.model_name    = os.getenv("LLM_MODEL", self._default_model())
        self.max_tokens    = int(os.getenv("LLM_MAX_TOKENS", "3000"))
        self.temperature   = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        self.base_url      = os.getenv("LLM_BASE_URL", self._default_base_url())
        self.timeout       = int(os.getenv("LLM_TIMEOUT", "60"))
        self.max_retries   = int(os.getenv("LLM_MAX_RETRIES", "2"))

    def _default_model(self) -> str:
        provider = os.getenv("LLM_PROVIDER", "groq").lower()
        defaults = {
            "groq":   "openai/gpt-oss-120b",
            "openai": "gpt-4o-mini",
            "gemini": "gemini-3.8-flash",
        }
        return defaults.get(provider, "openai/gpt-oss-120b")

    def _default_base_url(self) -> str:
        provider = os.getenv("LLM_PROVIDER", "groq").lower()
        urls = {
            "groq":   "https://api.groq.com/openai/v1",
            "openai": "https://api.openai.com/v1",
            "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
        }
        return urls.get(provider, "https://api.groq.com/openai/v1")

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip())


# ─── LLM Service ───────────────────────────────────────────────────────────────

class LLMService:
    """
    Service class for LLM API communication.
    Supports streaming and non-streaming responses.
    Handles retries, error classification, and graceful failures.
    """

    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig()
        self._client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the OpenAI-compatible client."""
        if not self.config.is_configured():
            logger.warning("LLM API key not configured. Service will be unavailable.")
            return

        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                max_retries=0,  # We handle retries manually for better error messaging
            )
            logger.info(f"LLM client initialized: provider={self.config.api_provider}, model={self.config.model_name}")
        except ImportError:
            logger.error("openai package not installed. Run: pip install openai")
            raise

    def is_available(self) -> bool:
        """Check if the LLM service is properly configured."""
        return self._client is not None and self.config.is_configured()

    def get_response(self, messages: list[dict], max_continuations: int = 4) -> str:
        """
        Send messages to the LLM and return the complete response text.
        Automatically continues if output was truncated (finish_reason == 'length' or unclosed code block).

        Args:
            messages: List of {"role": ..., "content": ...} dicts
            max_continuations: Max number of auto-continuations for large tasks

        Returns:
            The assistant's complete response as a string
        """
        if not self.is_available():
            raise LLMServiceError(
                "API key not configured",
                error_type="configuration"
            )

        current_messages = list(messages)
        full_response = ""

        for cont_idx in range(max_continuations):
            turn_content = ""
            finish_reason = None
            max_attempts = self.config.max_retries + 2

            for attempt in range(max_attempts):
                try:
                    response = self._client.chat.completions.create(
                        model=self.config.model_name,
                        messages=current_messages,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        stream=False,
                    )
                    choice = response.choices[0]
                    turn_content = choice.message.content or ""
                    finish_reason = choice.finish_reason
                    break
                except Exception as e:
                    error_type = self._classify_error(e)
                    logger.warning(f"Attempt {attempt + 1} failed: {error_type} - {str(e)}")
                    if error_type in ("authentication", "invalid_request", "configuration"):
                        raise LLMServiceError(str(e), error_type=error_type)
                    if attempt < max_attempts - 1:
                        wait_time = 2.5 * (attempt + 1)
                        logger.info(f"Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    raise LLMServiceError(str(e), error_type=error_type)

            full_response += turn_content

            unclosed_code_fence = (full_response.count("```") % 2 != 0)
            is_cut_off = (finish_reason == "length") or (unclosed_code_fence and len(turn_content) > 100)

            if is_cut_off and cont_idx < max_continuations - 1:
                logger.info(f"Response detected as incomplete (length={finish_reason}, unclosed_code={unclosed_code_fence}). Auto-continuing turn {cont_idx + 2}...")
                current_messages.append({"role": "assistant", "content": turn_content})
                if unclosed_code_fence:
                    continue_prompt = "You were cut off inside a code block. Continue writing the remaining code immediately from the exact character where you stopped. Do not add markdown backticks at the start, do not repeat any code, and write all remaining methods until the code is 100% complete, then close the code block."
                else:
                    continue_prompt = "You were cut off before finishing the task. Continue writing the rest of the response immediately from where you stopped. Do not repeat anything already written."
                current_messages.append({"role": "user", "content": continue_prompt})
            else:
                break

        return full_response

    def get_streaming_response(
        self,
        messages: list[dict],
        max_continuations: int = 4
    ) -> Generator[str, None, None]:
        """
        Send messages and stream back response tokens as they arrive.
        Includes automatic retry with backoff for rate limits,
        AND automatic multi-chunk continuation if the task exceeds max_tokens
        or an unclosed code block is detected, until the response is 100% complete.

        Args:
            messages: List of {"role": ..., "content": ...} dicts
            max_continuations: Max number of auto-continuations for large tasks

        Yields:
            Response text chunks as they stream in
        """
        if not self.is_available():
            raise LLMServiceError(
                "API key not configured",
                error_type="configuration"
            )

        current_messages = list(messages)
        full_text = ""

        for cont_idx in range(max_continuations):
            turn_content = ""
            finish_reason = None
            max_attempts = self.config.max_retries + 2

            for attempt in range(max_attempts):
                try:
                    stream = self._client.chat.completions.create(
                        model=self.config.model_name,
                        messages=current_messages,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        stream=True,
                    )

                    for chunk in stream:
                        if chunk.choices:
                            if chunk.choices[0].finish_reason:
                                finish_reason = chunk.choices[0].finish_reason
                            delta = chunk.choices[0].delta
                            if delta and delta.content:
                                turn_content += delta.content
                                full_text += delta.content
                                yield delta.content
                    break

                except Exception as e:
                    error_type = self._classify_error(e)
                    logger.warning(f"Streaming attempt {attempt + 1} failed: {error_type} - {str(e)}")

                    if error_type == "rate_limit" and attempt < max_attempts - 1:
                        wait_time = 2.5 * (attempt + 1)
                        logger.info(f"Rate limit encountered. Auto-retrying stream in {wait_time}s...")
                        time.sleep(wait_time)
                        continue

                    if error_type in ("network", "timeout") and attempt < max_attempts - 1:
                        time.sleep(1.5 * (attempt + 1))
                        continue

                    raise LLMServiceError(str(e), error_type=error_type)

            # Check if generation stopped prematurely:
            # 1) API explicitly returned finish_reason == 'length'
            # 2) Unclosed code block (odd number of ``` fences)
            unclosed_code_fence = (full_text.count("```") % 2 != 0)
            is_cut_off = (finish_reason == "length") or (unclosed_code_fence and len(turn_content) > 100)

            if is_cut_off and cont_idx < max_continuations - 1:
                logger.info(f"Response detected as incomplete (length={finish_reason}, unclosed_code={unclosed_code_fence}). Auto-continuing chunk {cont_idx + 2}...")
                current_messages.append({"role": "assistant", "content": turn_content})
                if unclosed_code_fence:
                    continue_prompt = "You were cut off inside a code block. Continue writing the remaining code immediately from the exact character where you stopped. Do not add markdown backticks at the start, do not repeat any code, and write all remaining methods until the code is 100% complete, then close the code block."
                else:
                    continue_prompt = "You were cut off before finishing the task. Continue writing the rest of the response immediately from where you stopped. Do not repeat anything already written."
                current_messages.append({
                    "role": "user",
                    "content": continue_prompt
                })
            else:
                # Finished normally with "stop"
                break

    def _classify_error(self, error: Exception) -> str:
        """Classify an exception into a human-readable error type."""
        if error is None:
            return "unknown"

        error_str = str(error).lower()
        error_class = type(error).__name__.lower()

        if any(k in error_str for k in ("api key", "authentication", "unauthorized", "invalid_api_key", "401")):
            return "authentication"
        if any(k in error_str for k in ("rate limit", "429", "too many requests")):
            return "rate_limit"
        if any(k in error_str for k in ("timeout", "timed out")):
            return "timeout"
        if any(k in error_str for k in ("connection", "network", "dns", "503", "502")):
            return "network"
        if any(k in error_str for k in ("invalid", "bad request", "400")):
            return "invalid_request"
        if "model" in error_str and ("not found" in error_str or "404" in error_str):
            return "model_not_found"
        return "unknown"


# ─── Custom Exception ──────────────────────────────────────────────────────────

class LLMServiceError(Exception):
    """
    Custom exception for LLM service failures.
    Carries both the error message and a type for UI-friendly handling.
    """

    USER_MESSAGES = {
        "authentication":  "❌ **Invalid API Key** — Please check your `.env` file and ensure `LLM_API_KEY` is set correctly.",
        "rate_limit":      "⏳ **Rate Limit Reached** — Too many requests. Please wait a moment and try again.",
        "timeout":         "⌛ **Request Timed Out** — The API took too long to respond. Please try again.",
        "network":         "🌐 **Network Error** — Could not reach the API. Check your internet connection.",
        "invalid_request": "⚠️ **Invalid Request** — The message could not be processed. Try rephrasing.",
        "model_not_found": "🤖 **Model Not Found** — Check that `LLM_MODEL` in your `.env` is a valid model name.",
        "configuration":   "⚙️ **Not Configured** — Please add your API key to the `.env` file and restart the app.",
        "unknown":         "❗ **Unexpected Error** — Something went wrong. Please try again.",
    }

    def __init__(self, message: str, error_type: str = "unknown"):
        super().__init__(message)
        self.error_type = error_type
        self.user_friendly_message = self.USER_MESSAGES.get(error_type, self.USER_MESSAGES["unknown"])


# ─── Singleton Factory ─────────────────────────────────────────────────────────

_service_instance: LLMService | None = None

def get_llm_service() -> LLMService:
    """Get or create a singleton LLMService instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = LLMService()
    return _service_instance
