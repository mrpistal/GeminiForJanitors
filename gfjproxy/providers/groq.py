from typing import Any

import httpx2

from .._globals import PROCESS_TIMEOUT
from ..http_client import http_client
from ..logging import xlog
from ..models import JaiMessage, JaiResult, JaiResultMetadata, JaiResultTokenUsage
from ..statistics import track_stats
from ..xuiduser import XUID


def groq_generate_content(
    user: XUID,
    api_key: str,
    model: str,
    messages: list[JaiMessage],
    settings: dict[str, Any] | None = None,
) -> JaiResult:
    """Wrapper around Groq's Chat Completions API.

    User paramater is only used for logging."""

    groq_request = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "content": message.content,
                "role": message.role,
            }
            for message in messages
        ],
    }

    for key, value in (settings or {}).items():
        if key == "temperature":
            groq_request["temperature"] = value
        elif key == "max_tokens":
            groq_request["max_tokens"] = value
        elif key == "top_k":
            groq_request["top_k"] = value
        elif key == "top_p":
            groq_request["top_p"] = value
        elif key == "frequency_penalty":
            groq_request["frequency_penalty"] = value
        elif key == "repetition_penalty":
            groq_request["presence_penalty"] = value

    headers = {
        "Authorization": f"Bearer {api_key.removeprefix('groq/')}",
        "Content-Type": "application/json",
    }

    try:
        groq_response = http_client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=groq_request,
            headers=headers,
            timeout=PROCESS_TIMEOUT,
        )
        groq_response.raise_for_status()
        groq_result = groq_response.json()
    except httpx2.TimeoutException:
        track_stats("groq.time_out")
        return JaiResult(504, "Gateway Timeout")
    except httpx2.HTTPStatusError as e:
        message = "Error from Groq"

        error = e.response.json()
        if isinstance(error, dict):
            if "error" in error:
                error = error["error"]

            if error_code := error.get("code"):
                message += f" ({error_code})"
            if error_message := error.get("message"):
                message += f": {error_message}"
        else:
            xlog(user, f"{message}: {error!r}")

        if e.response.is_client_error:
            track_stats("groq.failed.client")
        elif e.response.is_server_error:
            track_stats("groq.failed.server")
        else:
            track_stats("groq.failed.unknown")

        return JaiResult(e.response.status_code, message)
    except Exception as e:  # ruff: ignore[BLE001]
        xlog(user, repr(e))
        track_stats("groq.failed.exception")
        return JaiResult(502, "Unhanded exception from Groq.")

    try:
        text = str(groq_result["choices"][0]["message"]["content"] or "")
    except (KeyError, IndexError, TypeError):
        text = ""

    metadata = JaiResultMetadata()
    if usage := groq_result.get("usage"):
        metadata.token_usage = JaiResultTokenUsage(
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            reasoning_tokens=usage.get("completion_tokens_details", {}).get(
                "reasoning_tokens"
            ),
            total_tokens=usage.get("total_tokens"),
        )

    if not text:
        # Rejection?
        xlog(user, f"No result text: {groq_result!r}")
        track_stats("groq.rejected")
        return JaiResult(502, "Response blocked/empty.", metadata=metadata)

    track_stats("groq.succeeded")
    return JaiResult(200, text, metadata=metadata)
