from typing import Any

import httpx2

from .._globals import PROCESS_TIMEOUT
from ..http_client import http_client
from ..logging import xlog
from ..models import JaiMessage, JaiResult, JaiResultMetadata, JaiResultTokenUsage
from ..statistics import track_stats
from ..xuiduser import XUID


def openai_generate_content(
    user: XUID,
    api_key: str,
    model: str,
    messages: list[JaiMessage],
    settings: dict[str, Any] | None = None,
) -> JaiResult:
    """Wrapper around OpenAI's Chat Completions API.

    User paramater is only used for logging."""

    openai_request = {
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
            openai_request["temperature"] = value
        elif key == "max_tokens":
            openai_request["max_tokens"] = value
        elif key == "top_k":
            openai_request["top_k"] = value
        elif key == "top_p":
            openai_request["top_p"] = value
        elif key == "frequency_penalty":
            openai_request["frequency_penalty"] = value
        elif key == "repetition_penalty":
            openai_request["presence_penalty"] = value

    headers = {
        "Authorization": f"Bearer {api_key.removeprefix('openai/')}",
        "Content-Type": "application/json",
    }

    try:
        openai_response = http_client.post(
            "https://api.openai.com/v1/chat/completions",
            json=openai_request,
            headers=headers,
            timeout=PROCESS_TIMEOUT,
        )
        openai_response.raise_for_status()
        openai_result = openai_response.json()
    except httpx2.TimeoutException:
        track_stats("openai.time_out")
        return JaiResult(504, "Gateway Timeout")
    except httpx2.HTTPStatusError as e:
        message = "Error from OpenAI"

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
            track_stats("openai.failed.client")
        elif e.response.is_server_error:
            track_stats("openai.failed.server")
        else:
            track_stats("openai.failed.unknown")

        return JaiResult(e.response.status_code, message)
    except Exception as e:  # ruff: ignore[BLE001]
        xlog(user, repr(e))
        track_stats("openai.failed.exception")
        return JaiResult(502, "Unhanded exception from OpenAI.")

    try:
        text = str(openai_result["choices"][0]["message"]["content"] or "")
    except (KeyError, IndexError, TypeError):
        text = ""

    metadata = JaiResultMetadata()
    if usage := openai_result.get("usage"):
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
        xlog(user, f"No result text: {openai_result!r}")
        track_stats("openai.rejected")
        return JaiResult(502, "Response blocked/empty.", metadata=metadata)

    track_stats("openai.succeeded")
    return JaiResult(200, text, metadata=metadata)
