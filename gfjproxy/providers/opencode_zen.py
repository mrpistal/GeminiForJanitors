from typing import Any

import httpx2

from .._globals import PROCESS_TIMEOUT
from ..http_client import http_client
from ..logging import xlog
from ..models import JaiMessage, JaiResult, JaiResultMetadata, JaiResultTokenUsage
from ..statistics import track_stats
from ..xuiduser import XUID


def opencode_zen_generate_content(
    user: XUID,
    api_key: str,
    model: str,
    messages: list[JaiMessage],
    settings: dict[str, Any] | None = None,
) -> JaiResult:
    """Wrapper around OpenCode Zen's OpenAI-compatible Chat Completions endpoint.

    Only models served through `/zen/v1/chat/completions` work (DeepSeek, GLM,
    Kimi, MiniMax and free models). GPT/Grok (`/responses`), Claude/Qwen
    (`/messages`) and Gemini (`/models/<id>`) are not OpenAI-shaped and cannot
    be called through this proxy.

    User paramater is only used for logging."""

    opencode_zen_request = {
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
            opencode_zen_request["temperature"] = value
        elif key == "max_tokens":
            opencode_zen_request["max_tokens"] = value
        elif key == "top_k":
            opencode_zen_request["top_k"] = value
        elif key == "top_p":
            opencode_zen_request["top_p"] = value
        elif key == "frequency_penalty":
            opencode_zen_request["frequency_penalty"] = value
        elif key == "repetition_penalty":
            opencode_zen_request["presence_penalty"] = value

    headers = {
        "Authorization": f"Bearer {api_key.removeprefix('opencode_zen/')}",
        "Content-Type": "application/json",
    }

    try:
        opencode_zen_response = http_client.post(
            "https://opencode.ai/zen/v1/chat/completions",
            json=opencode_zen_request,
            headers=headers,
            timeout=PROCESS_TIMEOUT,
        )
        opencode_zen_response.raise_for_status()
        opencode_zen_result = opencode_zen_response.json()
    except httpx2.TimeoutException:
        track_stats("opencode_zen.time_out")
        return JaiResult(504, "Gateway Timeout")
    except httpx2.HTTPStatusError as e:
        message = "Error from OpenCode Zen"

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
            track_stats("opencode_zen.failed.client")
        elif e.response.is_server_error:
            track_stats("opencode_zen.failed.server")
        else:
            track_stats("opencode_zen.failed.unknown")

        return JaiResult(e.response.status_code, message)
    except Exception as e:  # ruff: ignore[BLE001]
        xlog(user, repr(e))
        track_stats("opencode_zen.failed.exception")
        return JaiResult(502, "Unhanded exception from OpenCode Zen.")

    try:
        text = str(opencode_zen_result["choices"][0]["message"]["content"] or "")
    except (KeyError, IndexError, TypeError):
        text = ""

    metadata = JaiResultMetadata()
    if usage := opencode_zen_result.get("usage"):
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
        xlog(user, f"No result text: {opencode_zen_result!r}")
        track_stats("opencode_zen.rejected")
        return JaiResult(502, "Response blocked/empty.", metadata=metadata)

    track_stats("opencode_zen.succeeded")
    return JaiResult(200, text, metadata=metadata)
