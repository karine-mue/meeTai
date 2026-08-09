"""call_gpt() の HTTP error 変換と、/chat・/ask-all 両経路の診断保持の test。

httpx.MockTransport で OpenAI を差し替えるため、実 API へは一切出ない。
"""

import json

import httpx
import pytest

import app
from openai_gpt import REASONING_MIN_OUTPUT_TOKENS

SECRET_KEY = "sk-test-secret-value-must-not-leak"

UNSUPPORTED_TEMPERATURE = {
    "error": {
        "message": "Unsupported parameter: 'temperature' is not supported with this model.",
        "type": "invalid_request_error",
        "param": "temperature",
        "code": None,
    }
}


def completed_body(text: str = "回答本文") -> dict:
    return {
        "status": "completed",
        "output": [
            {"type": "reasoning", "summary": []},
            {"type": "message", "content": [{"type": "output_text", "text": text}]},
        ],
    }


@pytest.fixture
def gpt_env(monkeypatch):
    """Issue #23 の再現構成: OPENAI_MODEL のみ指定、他は未指定。"""
    monkeypatch.setenv("OPENAI_API_KEY", SECRET_KEY)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4")
    monkeypatch.delenv("GPT_MODEL", raising=False)
    monkeypatch.delenv("GPT_REASONING_EFFORT", raising=False)


@pytest.fixture
def mock_openai(monkeypatch):
    """status / body を指定して OpenAI を差し替える。送信された request も記録する。"""
    sent = []

    def install(status: int, body, *, text_body: str = None):
        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(request)
            if text_body is not None:
                return httpx.Response(status, text=text_body)
            return httpx.Response(status, json=body)

        def factory() -> httpx.AsyncClient:
            return httpx.AsyncClient(transport=httpx.MockTransport(handler))

        monkeypatch.setattr(app, "_gpt_http_client", factory)
        return sent

    return install


# ==========
# 送信 body（回帰: Issue #23 の構成）
# ==========
async def test_gpt_5_4_request_omits_temperature(gpt_env, mock_openai):
    sent = mock_openai(200, completed_body())
    await app.call_gpt("ping", "sys", 4096)

    body = json.loads(sent[0].content)
    assert body["model"] == "gpt-5.4"
    assert "temperature" not in body
    assert body["reasoning"] == {"effort": "high"}
    assert body["max_output_tokens"] == REASONING_MIN_OUTPUT_TOKENS


# ==========
# 成功経路 / incomplete 判定（既存挙動の維持）
# ==========
async def test_completed_response_returns_text(gpt_env, mock_openai):
    mock_openai(200, completed_body("  こんにちは  "))
    assert await app.call_gpt("ping", "sys", 4096) == "こんにちは"


async def test_reasoning_items_are_skipped_when_joining_output(gpt_env, mock_openai):
    body = {
        "status": "completed",
        "output": [
            {"type": "reasoning", "content": [{"type": "output_text", "text": "内部思考"}]},
            {"type": "message", "content": [
                {"type": "output_text", "text": "前半"},
                {"type": "refusal", "refusal": "無視される"},
                {"type": "output_text", "text": "後半"},
            ]},
        ],
    }
    mock_openai(200, body)
    assert await app.call_gpt("ping", "sys", 4096) == "前半後半"


async def test_incomplete_response_raises_with_reason(gpt_env, mock_openai):
    mock_openai(200, {
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [],
    })
    with pytest.raises(app.IncompleteLLMResponse) as excinfo:
        await app.call_gpt("ping", "sys", 4096)
    assert excinfo.value.finish_reason == "max_output_tokens"
    assert "max_output_tokens" in str(excinfo.value)


async def test_empty_output_raises_incomplete(gpt_env, mock_openai):
    mock_openai(200, {"status": "completed", "output": []})
    with pytest.raises(app.IncompleteLLMResponse):
        await app.call_gpt("ping", "sys", 4096)


# ==========
# HTTP error → 構造化
# ==========
async def test_400_is_converted_to_structured_error(gpt_env, mock_openai):
    mock_openai(400, UNSUPPORTED_TEMPERATURE)
    with pytest.raises(app.OpenAIRequestError) as excinfo:
        await app.call_gpt("ping", "sys", 4096)

    info = excinfo.value.info
    assert info.http_status == 400
    assert info.type == "invalid_request_error"
    assert info.param == "temperature"
    assert info.code is None
    assert info.message == UNSUPPORTED_TEMPERATURE["error"]["message"]


@pytest.mark.parametrize("status", [401, 404, 429, 500, 503])
async def test_other_statuses_are_converted_too(gpt_env, mock_openai, status):
    mock_openai(status, {"error": {"message": "boom", "type": "server_error"}})
    with pytest.raises(app.OpenAIRequestError) as excinfo:
        await app.call_gpt("ping", "sys", 4096)
    assert excinfo.value.info.http_status == status


@pytest.mark.parametrize("text_body", ["", "<html>Bad Gateway</html>", "not json"])
async def test_non_json_error_body_does_not_break(gpt_env, mock_openai, text_body):
    mock_openai(502, None, text_body=text_body)
    with pytest.raises(app.OpenAIRequestError) as excinfo:
        await app.call_gpt("ping", "sys", 4096)
    assert excinfo.value.info.http_status == 502
    assert "502" in str(excinfo.value)


async def test_error_message_replaces_bare_http_status_text(gpt_env, mock_openai):
    """従来は httpx の 'Client error ...' だけで param が見えなかった。"""
    mock_openai(400, UNSUPPORTED_TEMPERATURE)
    with pytest.raises(app.OpenAIRequestError) as excinfo:
        await app.call_gpt("ping", "sys", 4096)

    rendered = str(excinfo.value)
    for fragment in ("400", "invalid_request_error", "param=temperature", "not supported"):
        assert fragment in rendered


# ==========
# secret redaction
# ==========
async def test_diagnostics_never_contain_secrets_or_prompt(gpt_env, mock_openai):
    echoed = {
        "error": {"message": "Unsupported parameter: 'temperature'.", "type": "invalid_request_error"},
        "request": {"instructions": "SYSTEM-PROMPT-TEXT", "input": "USER-PROMPT-TEXT"},
    }
    mock_openai(400, echoed)
    with pytest.raises(app.OpenAIRequestError) as excinfo:
        await app.call_gpt("USER-PROMPT-TEXT", "SYSTEM-PROMPT-TEXT", 4096)

    surfaces = [str(excinfo.value), json.dumps(excinfo.value.info.as_dict())]
    surfaces.append(json.dumps(app._error_response("gpt", excinfo.value)))
    for surface in surfaces:
        assert SECRET_KEY not in surface
        assert "Bearer" not in surface
        assert "Authorization" not in surface
        assert "SYSTEM-PROMPT-TEXT" not in surface
        assert "USER-PROMPT-TEXT" not in surface


# ==========
# /ask-all 経路（_error_response が構造化を保持する）
# ==========
async def test_ask_all_error_response_carries_detail(gpt_env, mock_openai):
    mock_openai(400, UNSUPPORTED_TEMPERATURE)
    with pytest.raises(app.OpenAIRequestError) as excinfo:
        await app.call_gpt("ping", "sys", 4096)

    response = app._error_response("gpt", excinfo.value)
    assert response["ok"] is False
    assert response["error_detail"] == {
        "provider": "openai",
        "http_status": 400,
        "message": UNSUPPORTED_TEMPERATURE["error"]["message"],
        "type": "invalid_request_error",
        "param": "temperature",
        "code": None,
    }
    # content 側にも同じ診断が入る（frontend は content を表示する）
    for fragment in ("400", "invalid_request_error", "param=temperature"):
        assert fragment in response["content"]


def test_ask_all_error_response_without_structured_info():
    """provider 由来でない例外でも壊れない。"""
    response = app._error_response("gpt", RuntimeError("boom"))
    assert response["ok"] is False
    assert "error_detail" not in response
    assert response["content"] == "[error] boom"


# ==========
# /chat 経路（agent_node が同じ診断を message content へ載せる）
# ==========
async def test_chat_path_keeps_diagnostics_in_message(gpt_env, mock_openai):
    mock_openai(400, UNSUPPORTED_TEMPERATURE)
    state = {
        "next_agent": "gpt",
        "phase": "FREE",
        "messages": [{"role": "user", "content": "ping", "agent": "human"}],
    }
    result = await app.agent_node(state)

    content = result["messages"][0]["content"]
    assert result["need_human"] is True
    for fragment in ("400", "invalid_request_error", "param=temperature", "not supported"):
        assert fragment in content


async def test_chat_path_success_is_unaffected(gpt_env, mock_openai):
    mock_openai(200, completed_body("応答"))
    state = {
        "next_agent": "gpt",
        "phase": "FREE",
        "messages": [{"role": "user", "content": "ping", "agent": "human"}],
    }
    result = await app.agent_node(state)
    assert result["messages"][0]["content"] == "応答"
    assert result["need_human"] is False


# ==========
# 回帰: error が archive / commit されない
# ==========
async def test_error_content_is_recognised_on_both_paths(gpt_env, mock_openai):
    """_error_content() の prefix と、両経路が生成する文字列を対で縛る。"""
    mock_openai(400, UNSUPPORTED_TEMPERATURE)
    state = {
        "next_agent": "gpt",
        "phase": "FREE",
        "messages": [{"role": "user", "content": "ping", "agent": "human"}],
    }
    chat_message = (await app.agent_node(state))["messages"][0]

    with pytest.raises(app.OpenAIRequestError) as excinfo:
        await app.call_gpt("ping", "sys", 4096)
    fanout_response = app._error_response("gpt", excinfo.value)

    assert app._error_content(chat_message["content"])
    assert app._error_content(fanout_response["content"])

    # /chat: archive 対象から外れる
    assert app._archivable_message(chat_message) is False
    # /ask-all: commit 対象から外れる
    assert app._response_ok(fanout_response) is False


async def test_successful_response_is_still_archivable(gpt_env, mock_openai):
    mock_openai(200, completed_body("通常の発言"))
    state = {
        "next_agent": "gpt",
        "phase": "FREE",
        "messages": [{"role": "user", "content": "ping", "agent": "human"}],
    }
    message = (await app.agent_node(state))["messages"][0]

    assert app._error_content(message["content"]) is False
    assert app._archivable_message(message) is True
    assert app._response_ok({"agent": "gpt", "content": message["content"], "ok": True}) is True
