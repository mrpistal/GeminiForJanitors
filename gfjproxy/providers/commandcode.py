from typing import Any

import httpx2

from .._globals import PROCESS_TIMEOUT
from ..http_client import http_client
from ..logging import xlog
from ..models import JaiMessage, JaiResult, JaiResultMetadata, JaiResultTokenUsage
from ..statistics import track_stats
from ..xuiduser import XUID


def commandcode_generate_content(
    user: XUID,
    api_key: str,
    model: str,
    messages: list[JaiMessage],
    settings: dict[str, Any] | None = None,
) -> JaiResult:
    """Wrapper around CommandCode's Provider API (OpenAI Chat Completions).

    User paramater is only used for logging."""

    commandcode_request = {
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
            commandcode_request["temperature"] = value
        elif key == "max_tokens":
            commandcode_request["max_tokens"] = value
        elif key == "top_k":
            commandcode_request["top_k"] = value
        elif key == "top_p":
            commandcode_request["top_p"] = value
        elif key == "frequency_penalty":
            commandcode_request["frequency_penalty"] = value
        elif key == "repetition_penalty":
            commandcode_request["presence_penalty"] = value

    headers = {
        "Authorization": f"Bearer {api_key.removeprefix('commandcode/')}",
        "Content-Type": "application/json",
    }

    try:
        commandcode_response = http_client.post(
            "https://api.commandcode.ai/provider/v1/chat/completions",
            json=commandcode_request,
            headers=headers,
            timeout=PROCESS_TIMEOUT,
        )
        commandcode_response.raise_for_status()
        commandcode_result = commandcode_response.json()
    except httpx2.TimeoutException:
        track_stats("commandcode.time_out")
        return JaiResult(504, "Gateway Timeout")
    except httpx2.HTTPStatusError as e:
        message = "Error from CommandCode"

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
            track_stats("commandcode.failed.client")
        elif e.response.is_server_error:
            track_stats("commandcode.failed.server")
        else:
            track_stats("commandcode.failed.unknown")

        return JaiResult(e.response.status_code, message)
    except Exception as e:  # ruff: ignore[BLE001]
        xlog(user, repr(e))
        track_stats("commandcode.failed.exception")
        return JaiResult(502, "Unhanded exception from CommandCode.")

    try:
        text = str(commandcode_result["choices"][0]["message"]["content"] or "")
    except (KeyError, IndexError, TypeError):
        text = ""

    metadata = JaiResultMetadata()
    if usage := commandcode_result.get("usage"):
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
        xlog(user, f"No result text: {commandcode_result!r}")
        track_stats("commandcode.rejected")
        return JaiResult(502, "Response blocked/empty.", metadata=metadata)

    track_stats("commandcode.succeeded")
    return JaiResult(200, text, metadata=metadata)
