from typing import Any

import httpx2

from .._globals import PROCESS_TIMEOUT
from ..http_client import http_client
from ..logging import xlog
from ..models import JaiMessage, JaiResult, JaiResultMetadata, JaiResultTokenUsage
from ..statistics import track_stats
from ..xuiduser import XUID


def kenari_generate_content(
    user: XUID,
    api_key: str,
    model: str,
    messages: list[JaiMessage],
    settings: dict[str, Any] | None = None,
) -> JaiResult:
    """Wrapper around Kenari's Chat Completions API.

    User paramater is only used for logging."""

    kenari_request = {
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
            kenari_request["temperature"] = value
        elif key == "max_tokens":
            kenari_request["max_tokens"] = value
        elif key == "top_k":
            kenari_request["top_k"] = value
        elif key == "top_p":
            kenari_request["top_p"] = value
        elif key == "frequency_penalty":
            kenari_request["frequency_penalty"] = value
        elif key == "repetition_penalty":
            kenari_request["presence_penalty"] = value

    headers = {
        "Authorization": f"Bearer {api_key.removeprefix('kenari/')}",
        "Content-Type": "application/json",
    }

    try:
        kenari_response = http_client.post(
            "https://kenari.id/v1/chat/completions",
            json=kenari_request,
            headers=headers,
            timeout=PROCESS_TIMEOUT,
        )
        kenari_response.raise_for_status()
        kenari_result = kenari_response.json()
    except httpx2.TimeoutException:
        track_stats("kenari.time_out")
        return JaiResult(504, "Gateway Timeout")
    except httpx2.HTTPStatusError as e:
        message = "Error from Kenari"

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
            track_stats("kenari.failed.client")
        elif e.response.is_server_error:
            track_stats("kenari.failed.server")
        else:
            track_stats("kenari.failed.unknown")

        return JaiResult(e.response.status_code, message)
    except Exception as e:  # ruff: ignore[BLE001]
        xlog(user, repr(e))
        track_stats("kenari.failed.exception")
        return JaiResult(502, "Unhanded exception from Kenari.")

    try:
        text = str(kenari_result["choices"][0]["message"]["content"] or "")
    except (KeyError, IndexError, TypeError):
        text = ""

    metadata = JaiResultMetadata()
    if usage := kenari_result.get("usage"):
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
        xlog(user, f"No result text: {kenari_result!r}")
        track_stats("kenari.rejected")
        return JaiResult(502, "Response blocked/empty.", metadata=metadata)

    track_stats("kenari.succeeded")
    return JaiResult(200, text, metadata=metadata)
