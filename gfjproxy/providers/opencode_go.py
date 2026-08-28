from typing import Any

import httpx2

from .._globals import PROCESS_TIMEOUT
from ..http_client import http_client
from ..logging import xlog
from ..models import JaiMessage, JaiResult, JaiResultMetadata, JaiResultTokenUsage
from ..statistics import track_stats
from ..xuiduser import XUID


def opencode_go_generate_content(
    user: XUID,
    api_key: str,
    model: str,
    messages: list[JaiMessage],
    settings: dict[str, Any] | None = None,
) -> JaiResult:
    """Wrapper around OpenCode Go's Chat Completions API.

    OpenCode Go is a $10/month subscription for open coding models (Grok 4.6,
    GLM, Kimi, MiniMax, Qwen, DeepSeek, ...). API keys come from the OpenCode
    Zen dashboard after subscribing to Go.

    User paramater is only used for logging."""

    opencode_go_request = {
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
            opencode_go_request["temperature"] = value
        elif key == "max_tokens":
            opencode_go_request["max_tokens"] = value
        elif key == "top_k":
            opencode_go_request["top_k"] = value
        elif key == "top_p":
            opencode_go_request["top_p"] = value
        elif key == "frequency_penalty":
            opencode_go_request["frequency_penalty"] = value
        elif key == "repetition_penalty":
            opencode_go_request["presence_penalty"] = value

    headers = {
        "Authorization": f"Bearer {api_key.removeprefix('opencode_go/')}",
        "Content-Type": "application/json",
    }

    try:
        opencode_go_response = http_client.post(
            "https://opencode.ai/zen/go/v1/chat/completions",
            json=opencode_go_request,
            headers=headers,
            timeout=PROCESS_TIMEOUT,
        )
        opencode_go_response.raise_for_status()
        opencode_go_result = opencode_go_response.json()
    except httpx2.TimeoutException:
        track_stats("opencode_go.time_out")
        return JaiResult(504, "Gateway Timeout")
    except httpx2.HTTPStatusError as e:
        message = "Error from OpenCode Go"

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
            track_stats("opencode_go.failed.client")
        elif e.response.is_server_error:
            track_stats("opencode_go.failed.server")
        else:
            track_stats("opencode_go.failed.unknown")

        return JaiResult(e.response.status_code, message)
    except Exception as e:  # ruff: ignore[BLE001]
        xlog(user, repr(e))
        track_stats("opencode_go.failed.exception")
        return JaiResult(502, "Unhanded exception from OpenCode Go.")

    try:
        text = str(opencode_go_result["choices"][0]["message"]["content"] or "")
    except (KeyError, IndexError, TypeError):
        text = ""

    metadata = JaiResultMetadata()
    if usage := opencode_go_result.get("usage"):
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
        xlog(user, f"No result text: {opencode_go_result!r}")
        track_stats("opencode_go.rejected")
        return JaiResult(502, "Response blocked/empty.", metadata=metadata)

    track_stats("opencode_go.succeeded")
    return JaiResult(200, text, metadata=metadata)
