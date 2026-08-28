from typing import Any

import httpx2
from httpx2 import ReadTimeout
from pytest_mock import MockerFixture

from gfjproxy.models import JaiMessage
from gfjproxy.providers.opencode_zen import opencode_zen_generate_content
from gfjproxy.xuiduser import XUID, LocalUserStorage, UserSettings

################################################################################


def make_mock_response(
    text: str, usage: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": text}}],
        "usage": usage or {},
    }


def make_http_error(code: int, response_json: dict[str, Any]) -> httpx2.HTTPStatusError:
    req = httpx2.Request("POST", "https://opencode.ai/zen/v1/")
    resp = httpx2.Response(code, json=response_json, request=req)
    return httpx2.HTTPStatusError(f"HTTP {code}", request=req, response=resp)


def make_user() -> UserSettings:
    return UserSettings(LocalUserStorage(), XUID("john", "smith"))


def make_messages() -> list[JaiMessage]:
    return [JaiMessage(content="Hello", role="user")]


################################################################################


def test_success(mocker: MockerFixture):
    mock_post = mocker.patch("gfjproxy.providers.opencode_zen.http_client.post")
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = make_mock_response(
        "Hello there!",
        {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "completion_tokens_details": {"reasoning_tokens": 3},
        },
    )
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    result = opencode_zen_generate_content(
        make_user(), "oczen_test123", "deepseek-v4-flash", make_messages()
    )

    assert result.status == 200
    assert result.text == "Hello there!"
    assert result.metadata.token_usage is not None
    assert result.metadata.token_usage.prompt_tokens == 10
    assert result.metadata.token_usage.completion_tokens == 5
    assert result.metadata.token_usage.reasoning_tokens == 3
    assert result.metadata.token_usage.total_tokens == 15

    args, kwargs = mock_post.call_args
    assert args[0] == "https://opencode.ai/zen/v1/chat/completions"
    assert kwargs["headers"]["Authorization"] == "Bearer oczen_test123"
    assert kwargs["json"]["model"] == "deepseek-v4-flash"


def test_explicit_prefix_is_stripped(mocker: MockerFixture):
    mock_post = mocker.patch("gfjproxy.providers.opencode_zen.http_client.post")
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = make_mock_response("Hi")
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    opencode_zen_generate_content(
        make_user(), "opencode_zen/oczen_test123", "deepseek-v4-flash", make_messages()
    )

    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer oczen_test123"


def test_settings_mapping(mocker: MockerFixture):
    mock_post = mocker.patch("gfjproxy.providers.opencode_zen.http_client.post")
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = make_mock_response("Hi")
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    opencode_zen_generate_content(
        make_user(),
        "oczen_test123",
        "deepseek-v4-flash",
        make_messages(),
        settings={
            "temperature": 0.7,
            "max_tokens": 512,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "repetition_penalty": 1.2,
        },
    )

    _, kwargs = mock_post.call_args
    body = kwargs["json"]
    assert body["temperature"] == 0.7
    assert body["max_tokens"] == 512
    assert body["top_p"] == 0.9
    assert body["frequency_penalty"] == 0.1
    assert body["presence_penalty"] == 1.2


def test_timeout(mocker: MockerFixture):
    mock_post = mocker.patch("gfjproxy.providers.opencode_zen.http_client.post")
    mock_post.side_effect = ReadTimeout("")

    result = opencode_zen_generate_content(
        make_user(), "oczen_test123", "deepseek-v4-flash", make_messages()
    )

    assert result.status == 504
    assert result.error == "Gateway Timeout"


def test_http_error(mocker: MockerFixture):
    mock_post = mocker.patch("gfjproxy.providers.opencode_zen.http_client.post")
    mock_post.side_effect = make_http_error(
        401, {"error": {"code": "invalid_api_key", "message": "Bad key"}}
    )

    result = opencode_zen_generate_content(
        make_user(), "oczen_test123", "deepseek-v4-flash", make_messages()
    )

    assert result.status == 401
    assert "invalid_api_key" in result.error
    assert "Bad key" in result.error


def test_generic_exception(mocker: MockerFixture):
    mock_post = mocker.patch("gfjproxy.providers.opencode_zen.http_client.post")
    mock_post.side_effect = Exception("boom")

    result = opencode_zen_generate_content(
        make_user(), "oczen_test123", "deepseek-v4-flash", make_messages()
    )

    assert result.status == 502
    assert "Unhanded exception" in result.error


def test_empty_response(mocker: MockerFixture):
    mock_post = mocker.patch("gfjproxy.providers.opencode_zen.http_client.post")
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = make_mock_response("")
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    result = opencode_zen_generate_content(
        make_user(), "oczen_test123", "deepseek-v4-flash", make_messages()
    )

    assert result.status == 502
    assert result.error == "Response blocked/empty."
