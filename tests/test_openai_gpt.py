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


def test_gpt_5_takes_no_temperature():
    assert "temperature" not in build("gpt-5")
    assert resolve_capabilities("gpt-5") is not UNKNOWN_MODEL_CAPABILITIES


# ==========
# snapshot と variant の区別
# ==========
@pytest.mark.parametrize(
    "snapshot,base",
    [
        ("gpt-5.4-2026-03-05", "gpt-5.4"),
        ("gpt-5.4-pro-2026-03-05", "gpt-5.4-pro"),
        ("gpt-4o-2024-08-06", "gpt-4o"),
        ("gpt-4o-mini-2024-07-18", "gpt-4o-mini"),
    ],
)
def test_date_suffix_is_treated_as_snapshot_of_base(snapshot, base):
    assert resolve_capabilities(snapshot) is resolve_capabilities(base)


def test_gpt_5_4_snapshot_body_matches_base():
    snapshot = build("gpt-5.4-2026-03-05")
    base = build("gpt-5.4")
    assert snapshot["reasoning"] == base["reasoning"]
    assert "temperature" not in snapshot


def test_pro_variant_is_not_a_snapshot_of_base():
    """gpt-5.4-pro は gpt-5.4 と effort の許容値が違うので別 profile。"""
    pro = resolve_capabilities("gpt-5.4-pro")
    base = resolve_capabilities("gpt-5.4")
    assert pro is not base
    assert pro is not UNKNOWN_MODEL_CAPABILITIES
    assert pro.supported_reasoning_efforts != base.supported_reasoning_efforts


@pytest.mark.parametrize("model", ["gpt-5.4-pro-max", "gpt-5.4-turbo", "gpt-5.4-preview"])
def test_unregistered_variant_does_not_inherit_base(model):
    """日付以外の suffix は継承させない（prefix match だと吸われていた）。"""
    assert resolve_capabilities(model) is UNKNOWN_MODEL_CAPABILITIES
    body = build(model)
    assert "reasoning" not in body and "temperature" not in body


@pytest.mark.parametrize("model", ["gpt-5.4-2026-3-5", "gpt-5.4-202603-05", "gpt-5.4-v2"])
def test_malformed_date_suffix_is_not_a_snapshot(model):
    """日付形式でない suffix を snapshot と誤認しない。"""
    assert resolve_capabilities(model) is UNKNOWN_MODEL_CAPABILITIES


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


@pytest.mark.parametrize("effort", ["none", "low", "medium", "high", "xhigh"])
def test_gpt_5_4_forwards_every_supported_effort(effort):
    """gpt-5.4 の許容値は none / low / medium / high / xhigh。"""
    assert build("gpt-5.4", GPT_REASONING_EFFORT=effort)["reasoning"] == {"effort": effort}


@pytest.mark.parametrize("effort", ["minimal", "extreme", "reasoning", "0"])
def test_gpt_5_4_does_not_forward_unsupported_effort(effort):
    """minimal は gpt-5.4 では未対応。素通しすると 400 になる。"""
    body = build("gpt-5.4", GPT_REASONING_EFFORT=effort)
    assert body["reasoning"] == {"effort": "high"}
    assert body["reasoning"]["effort"] != effort


@pytest.mark.parametrize("effort", ["medium", "high", "xhigh"])
def test_gpt_5_4_pro_forwards_its_supported_efforts(effort):
    assert build("gpt-5.4-pro", GPT_REASONING_EFFORT=effort)["reasoning"] == {"effort": effort}


@pytest.mark.parametrize("effort", ["none", "low", "minimal"])
def test_gpt_5_4_pro_falls_back_on_efforts_it_lacks(effort):
    """pro は low 以下を持たない。base で有効な値でも送らない。"""
    body = build("gpt-5.4-pro", GPT_REASONING_EFFORT=effort)
    assert body["reasoning"] == {"effort": "high"}


def test_pro_and_base_differ_on_the_same_env_value():
    """同じ env でも profile ごとに結果が変わることを 1 本で示す。"""
    assert build("gpt-5.4", GPT_REASONING_EFFORT="low")["reasoning"] == {"effort": "low"}
    assert build("gpt-5.4-pro", GPT_REASONING_EFFORT="low")["reasoning"] == {"effort": "high"}


@pytest.mark.parametrize("value,expected", [("HIGH ", "high"), (" XHigh", "xhigh"), ("", "high")])
def test_reasoning_effort_is_normalised(value, expected):
    assert build("gpt-5.4", GPT_REASONING_EFFORT=value)["reasoning"] == {"effort": expected}


def test_unverified_profiles_keep_a_conservative_effort_set():
    """gpt-5 / gpt-5.5 は未検証のため none / xhigh を許可しない（400 側に倒さない）。"""
    for model in ("gpt-5", "gpt-5.5"):
        caps = resolve_capabilities(model)
        assert caps.supported_reasoning_efforts == frozenset({"low", "medium", "high"})
        for effort in ("none", "xhigh", "minimal"):
            assert build(model, GPT_REASONING_EFFORT=effort)["reasoning"] == {"effort": "high"}


def test_every_profile_default_is_within_its_own_supported_set():
    """default が許容集合の外にあると、退避先そのものが 400 になる。"""
    for model in ("gpt-5", "gpt-5.4", "gpt-5.4-pro", "gpt-5.5"):
        caps = resolve_capabilities(model)
        assert caps.default_reasoning_effort in caps.supported_reasoning_efforts


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
