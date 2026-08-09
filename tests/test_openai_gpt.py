"""openai_gpt.py の request builder / capability profile / error 変換の test。

env は引数で渡すため、実行環境の .env に依存しない。
"""

import pytest

from openai_gpt import (
    DEFAULT_GPT_MODEL,
    REASONING_MIN_OUTPUT_TOKENS,
    UNKNOWN_MODEL_CAPABILITIES,
    OpenAIErrorInfo,
    build_gpt_request,
    format_openai_error,
    parse_openai_error,
    resolve_capabilities,
    resolve_gpt_model,
)

BIG_BUDGET = REASONING_MIN_OUTPUT_TOKENS * 2


def build(model: str, **env):
    return build_gpt_request(
        prompt="ping",
        instructions="sys",
        max_output_tokens=BIG_BUDGET,
        env={"OPENAI_MODEL": model, **env},
    )


# ==========
# model matrix
# ==========
def test_required_fields_always_present():
    body = build("gpt-5.4")
    assert body["model"] == "gpt-5.4"
    assert body["instructions"] == "sys"
    assert body["input"] == "ping"
    assert body["max_output_tokens"] == BIG_BUDGET


def test_gpt_5_4_sends_reasoning_and_omits_temperature():
    """Issue #23 の直接原因。gpt-5.4 は temperature を受け付けない。"""
    body = build("gpt-5.4")
    assert body["reasoning"] == {"effort": "high"}
    assert "temperature" not in body


def test_gpt_5_4_snapshot_uses_same_profile():
    snapshot = build("gpt-5.4-2026-03-05")
    base = build("gpt-5.4")
    assert snapshot["reasoning"] == base["reasoning"]
    assert "temperature" not in snapshot
    assert resolve_capabilities("gpt-5.4-2026-03-05") is resolve_capabilities("gpt-5.4")


def test_gpt_5_5_sends_only_profiled_parameters():
    body = build("gpt-5.5")
    assert body["reasoning"] == {"effort": "high"}
    assert "temperature" not in body
    assert set(body) == {"model", "instructions", "input", "max_output_tokens", "reasoning"}


@pytest.mark.parametrize("model", ["gpt-5.6", "gpt-5.45", "gpt-6", "gpt-nonexistent"])
def test_unknown_model_is_fail_closed(model):
    """`gpt-5` 始まりでも未登録なら optional parameter を一切送らない。"""
    body = build(model)
    assert "temperature" not in body
    assert "reasoning" not in body
    assert set(body) == {"model", "instructions", "input", "max_output_tokens"}
    assert resolve_capabilities(model) is UNKNOWN_MODEL_CAPABILITIES


def test_gpt_4o_is_non_reasoning_and_takes_temperature():
    body = build("gpt-4o")
    assert "reasoning" not in body
    assert body["temperature"] == 0


def test_gpt_4o_snapshot_and_mini_use_same_profile():
    for model in ("gpt-4o-2024-08-06", "gpt-4o-mini"):
        assert resolve_capabilities(model) is resolve_capabilities("gpt-4o")


def test_longest_prefix_wins_over_shorter_one():
    """gpt-5.4 が gpt-5 profile に吸われない。"""
    assert resolve_capabilities("gpt-5.4") is resolve_capabilities("gpt-5.5")
    assert resolve_capabilities("gpt-5") is not UNKNOWN_MODEL_CAPABILITIES
    assert "temperature" not in build("gpt-5")


# ==========
# max_output_tokens の下限
# ==========
def test_reasoning_model_raises_small_budget_to_floor():
    """reasoning token が budget を食い切って本文が空になるのを防ぐ。"""
    body = build_gpt_request(
        prompt="ping", instructions="sys", max_output_tokens=4096,
        env={"OPENAI_MODEL": "gpt-5.4"},
    )
    assert body["max_output_tokens"] == REASONING_MIN_OUTPUT_TOKENS


@pytest.mark.parametrize("configured", [1, 512, 4096, REASONING_MIN_OUTPUT_TOKENS - 1])
def test_floor_overrides_explicitly_configured_budget(configured):
    """明示設定を意図的に上書きする、という仕様をここで固定する。

    GPT_MAX_TOKENS / LLM_MAX_TOKENS で小さい値を指定しても（app.py の
    _agent_max_tokens() 経由で max_output_tokens に入る）、reasoning model
    では下限まで引き上げる。挙動を変えたい場合は
    openai_gpt.REASONING_MIN_OUTPUT_TOKENS を下げる。根拠は同定数のコメント。
    """
    body = build_gpt_request(
        prompt="ping", instructions="sys", max_output_tokens=configured,
        env={"OPENAI_MODEL": "gpt-5.4"},
    )
    assert body["max_output_tokens"] == REASONING_MIN_OUTPUT_TOKENS


def test_reasoning_model_keeps_budget_above_floor():
    body = build_gpt_request(
        prompt="ping", instructions="sys", max_output_tokens=BIG_BUDGET,
        env={"OPENAI_MODEL": "gpt-5.4"},
    )
    assert body["max_output_tokens"] == BIG_BUDGET


def test_non_reasoning_model_budget_is_untouched():
    body = build_gpt_request(
        prompt="ping", instructions="sys", max_output_tokens=4096,
        env={"OPENAI_MODEL": "gpt-4o"},
    )
    assert body["max_output_tokens"] == 4096


# ==========
# env precedence
# ==========
def test_gpt_model_beats_openai_model():
    assert resolve_gpt_model({"GPT_MODEL": "gpt-4o", "OPENAI_MODEL": "gpt-5.4"}) == "gpt-4o"


def test_openai_model_used_when_gpt_model_unset():
    assert resolve_gpt_model({"OPENAI_MODEL": "gpt-5.4"}) == "gpt-5.4"


def test_blank_values_fall_through_to_next_source():
    assert resolve_gpt_model({"GPT_MODEL": "   ", "OPENAI_MODEL": "gpt-4o"}) == "gpt-4o"
    assert resolve_gpt_model({"GPT_MODEL": "", "OPENAI_MODEL": ""}) == DEFAULT_GPT_MODEL


def test_default_model_when_nothing_set():
    assert resolve_gpt_model({}) == DEFAULT_GPT_MODEL


def test_reasoning_effort_defaults_to_high_when_unset():
    assert build("gpt-5.4")["reasoning"] == {"effort": "high"}


def test_reasoning_effort_env_is_honoured():
    body = build("gpt-5.4", GPT_REASONING_EFFORT="low")
    assert body["reasoning"] == {"effort": "low"}


@pytest.mark.parametrize("value", ["extreme", "HIGH ", ""])
def test_reasoning_effort_is_normalised_or_falls_back(value):
    """profile 外の値を素通しして 400 にしない。"""
    body = build("gpt-5.4", GPT_REASONING_EFFORT=value)
    assert body["reasoning"]["effort"] in {"high"}


def test_reasoning_effort_ignored_for_non_reasoning_model():
    body = build("gpt-4o", GPT_REASONING_EFFORT="high")
    assert "reasoning" not in body


# ==========
# error 変換
# ==========
UNSUPPORTED_TEMPERATURE = {
    "error": {
        "message": "Unsupported parameter: 'temperature' is not supported with this model.",
        "type": "invalid_request_error",
        "param": "temperature",
        "code": None,
    }
}


def test_parses_openai_json_error():
    import json

    info = parse_openai_error(400, json.dumps(UNSUPPORTED_TEMPERATURE))
    assert info.http_status == 400
    assert info.message == UNSUPPORTED_TEMPERATURE["error"]["message"]
    assert info.type == "invalid_request_error"
    assert info.param == "temperature"
    assert info.code is None
    assert info.provider == "openai"


def test_as_dict_exposes_only_whitelisted_keys():
    info = parse_openai_error(400, '{"error": {"message": "boom"}}')
    assert set(info.as_dict()) == {"provider", "http_status", "message", "type", "param", "code"}


@pytest.mark.parametrize(
    "status,body",
    [
        (400, ""),
        (401, "   "),
        (403, "<html>Forbidden</html>"),
        (429, "not json at all"),
        (500, "{}"),
        (502, '{"error": null}'),
        (503, '{"error": {}}'),
        (504, "null"),
        (500, "[1, 2, 3]"),
    ],
)
def test_non_json_and_missing_fields_do_not_break(status, body):
    info = parse_openai_error(status, body)
    assert info.http_status == status
    assert info.message  # 空にはならない
    assert info.type is None and info.param is None and info.code is None
    assert format_openai_error(info).startswith(f"openai HTTP {status}")


def test_long_message_is_truncated():
    import json

    info = parse_openai_error(400, json.dumps({"error": {"message": "x" * 5000}}))
    assert len(info.message) < 5000


def test_raw_body_is_never_echoed():
    """whitelist 外の key（request body の echo 等）を診断へ持ち込まない。"""
    import json

    body = json.dumps(
        {
            "error": {
                "message": "bad request",
                "type": "invalid_request_error",
                "extra_secret": "sk-should-never-appear",
            },
            "request": {"instructions": "SYSTEM PROMPT", "input": "USER PROMPT"},
        }
    )
    rendered = format_openai_error(parse_openai_error(400, body))
    assert "sk-should-never-appear" not in rendered
    assert "SYSTEM PROMPT" not in rendered
    assert "USER PROMPT" not in rendered


def test_format_includes_status_type_param_and_code():
    info = OpenAIErrorInfo(
        http_status=400, message="boom", type="invalid_request_error",
        param="temperature", code="some_code",
    )
    rendered = format_openai_error(info)
    for fragment in ("400", "invalid_request_error", "param=temperature", "code=some_code", "boom"):
        assert fragment in rendered


def test_format_omits_absent_optional_fields():
    rendered = format_openai_error(OpenAIErrorInfo(http_status=500, message="boom"))
    assert rendered == "openai HTTP 500: boom"
