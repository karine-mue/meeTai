"""profile が実仕様より広かった場合の capability fallback の test。

400 の error.param で名指しされた optional parameter を落として再送する経路と、
再送してはいけない error で再送しないことを検証する。
"""

import json
import logging

import httpx
import pytest

import app
import openai_gpt
from openai_gpt import MAX_CAPABILITY_FALLBACKS, droppable_param

SECRET_KEY = "sk-test-secret-value-must-not-leak"


def unsupported(param: str, *, message: str = None) -> dict:
    return {
        "error": {
            "message": message or f"Unsupported parameter: '{param}' is not supported with this model.",
            "type": "invalid_request_error",
            "param": param,
            "code": None,
        }
    }


def _sole_warning(caplog) -> str:
    records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(records) == 1, f"expected exactly 1 WARNING, got {len(records)}"
    return records[0].getMessage()


def completed_body(text: str = "回答本文") -> dict:
    return {
        "status": "completed",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
    }


@pytest.fixture
def gpt_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET_KEY)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4")
    monkeypatch.delenv("GPT_MODEL", raising=False)
    monkeypatch.delenv("GPT_REASONING_EFFORT", raising=False)


@pytest.fixture
def scripted_openai(monkeypatch):
    """(status, body) の列を順に返す。送信された body を全件記録する。"""
    sent: list[dict] = []

    def install(*responses):
        queue = list(responses)

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(json.loads(request.content))
            status, payload = queue.pop(0) if queue else (500, {"error": {"message": "no script left"}})
            return httpx.Response(status, json=payload)

        monkeypatch.setattr(
            app, "_gpt_http_client",
            lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        return sent

    return install


@pytest.fixture
def warnings(caplog):
    caplog.set_level(logging.WARNING, logger="openai_gpt")
    return caplog


@pytest.fixture
def overbroad_profile(monkeypatch):
    """profile が実仕様より広い状態を作る。

    reasoning と temperature の両方を「対応」と主張する profile を一時登録する。
    fallback が想定しているのはまさにこの状態（表が API より広い）で、
    gpt-5.4 のように temperature を最初から送らない profile では再現できない。
    """
    monkeypatch.setitem(
        openai_gpt._CAPABILITY_PROFILES,
        "gpt-overbroad",
        openai_gpt.OpenAIModelCapabilities(
            supports_reasoning=True,
            supports_temperature=True,
            supported_reasoning_efforts=frozenset({"high"}),
            default_reasoning_effort="high",
            min_output_tokens=openai_gpt.REASONING_MIN_OUTPUT_TOKENS,
        ),
    )
    monkeypatch.setenv("OPENAI_MODEL", "gpt-overbroad")
    return "gpt-overbroad"


# ==========
# 再送して成功する経路
# ==========
async def test_temperature_rejection_is_retried_without_temperature(
    gpt_env, overbroad_profile, scripted_openai, warnings
):
    """profile が temperature 対応と誤っていても応答を得られる。"""
    sent = scripted_openai((400, unsupported("temperature")), (200, completed_body("応答")))
    assert await app.call_gpt("ping", "sys", 4096) == "応答"

    assert len(sent) == 2
    assert sent[0]["temperature"] == 0
    assert "temperature" not in sent[1]
    # 落としたのは temperature だけ。必須 field と reasoning は残る
    assert sent[1]["model"] == overbroad_profile
    assert sent[1]["input"] == "ping"
    assert sent[1]["instructions"] == "sys"
    assert sent[1]["reasoning"] == {"effort": "high"}
    assert sent[1]["max_output_tokens"] == sent[0]["max_output_tokens"]


async def test_reasoning_effort_rejection_drops_whole_reasoning_object(gpt_env, scripted_openai, warnings):
    sent = scripted_openai((400, unsupported("reasoning.effort")), (200, completed_body("応答")))
    assert await app.call_gpt("ping", "sys", 4096) == "応答"

    assert len(sent) == 2
    assert "reasoning" not in sent[1]
    assert sent[1]["max_output_tokens"] == sent[0]["max_output_tokens"]


async def test_reasoning_rejection_is_retried(gpt_env, scripted_openai, warnings):
    sent = scripted_openai((400, unsupported("reasoning")), (200, completed_body("応答")))
    assert await app.call_gpt("ping", "sys", 4096) == "応答"
    assert "reasoning" not in sent[1]


async def test_two_rejections_in_sequence_are_both_dropped(
    gpt_env, overbroad_profile, scripted_openai, warnings
):
    """temperature と reasoning の両方を拒否されても上限内で回復する。"""
    sent = scripted_openai(
        (400, unsupported("temperature")),
        (400, unsupported("reasoning.effort")),
        (200, completed_body("応答")),
    )
    assert await app.call_gpt("ping", "sys", 4096) == "応答"

    assert len(sent) == MAX_CAPABILITY_FALLBACKS + 1
    assert "temperature" not in sent[2] and "reasoning" not in sent[2]
    # 必須 field は最後まで残る
    assert set(sent[2]) == {"model", "instructions", "input", "max_output_tokens"}
    assert len([r for r in warnings.records if r.levelno == logging.WARNING]) == 2


async def test_non_reasoning_model_temperature_rejection_is_retried(gpt_env, monkeypatch, scripted_openai, warnings):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    sent = scripted_openai((400, unsupported("temperature")), (200, completed_body("応答")))
    assert await app.call_gpt("ping", "sys", 4096) == "応答"
    assert sent[0]["temperature"] == 0
    assert "temperature" not in sent[1]


# ==========
# 再送しない経路
# ==========
@pytest.mark.parametrize("status", [401, 403, 404, 429, 500, 502, 503])
async def test_non_400_statuses_are_never_retried(gpt_env, overbroad_profile, scripted_openai, status):
    """429 / 5xx は推論コストが乗りうるので絶対に再送しない。

    body に temperature が乗る profile を使うこと。乗っていない profile だと
    「body に無いから落とせない」側で止まってしまい、status 判定を検証できない。
    """
    sent = scripted_openai((status, unsupported("temperature")))
    with pytest.raises(app.OpenAIRequestError) as excinfo:
        await app.call_gpt("ping", "sys", 4096)
    assert excinfo.value.info.http_status == status
    assert len(sent) == 1


@pytest.mark.parametrize("error_type", ["insufficient_quota", "server_error", "rate_limit_error", None])
async def test_400_with_other_error_type_is_not_retried(
    gpt_env, overbroad_profile, scripted_openai, error_type
):
    sent = scripted_openai((400, {
        "error": {"message": "nope", "type": error_type, "param": "temperature"},
    }))
    with pytest.raises(app.OpenAIRequestError):
        await app.call_gpt("ping", "sys", 4096)
    assert len(sent) == 1


@pytest.mark.parametrize("param", ["model", "input", "instructions", "max_output_tokens", "stream"])
async def test_required_and_unknown_params_are_not_dropped(
    gpt_env, overbroad_profile, scripted_openai, param
):
    """必須 field を落として再送しても意味がない。"""
    sent = scripted_openai((400, unsupported(param)))
    with pytest.raises(app.OpenAIRequestError) as excinfo:
        await app.call_gpt("ping", "sys", 4096)
    assert excinfo.value.info.param == param
    assert len(sent) == 1


async def test_param_absent_from_body_is_not_retried(gpt_env, scripted_openai):
    """gpt-5.4 は temperature を送らない。無いものを落とす再送はしない。"""
    sent = scripted_openai((400, unsupported("temperature")), (200, completed_body()))
    # gpt-5.4 の body に temperature は無いので再送されない
    with pytest.raises(app.OpenAIRequestError):
        await app.call_gpt("ping", "sys", 4096)
    assert len(sent) == 1


async def test_retry_budget_is_bounded(
    gpt_env, overbroad_profile, monkeypatch, scripted_openai, warnings
):
    """落とせる parameter が残っていても、上限を超えて再送しない。

    通常 body の droppable key は temperature / reasoning の 2 つで上限と一致する
    ため、上限そのものが効いているかを見るには droppable を 1 つ増やす必要がある。
    増やさないと「落とせるものが尽きた」側で止まり、上限を外しても同じ結果になる。
    """
    monkeypatch.setitem(openai_gpt._DROPPABLE_PARAMS, "max_output_tokens", "max_output_tokens")
    sent = scripted_openai(
        (400, unsupported("temperature")),
        (400, unsupported("reasoning")),
        (400, unsupported("max_output_tokens")),
        (200, completed_body("上限が効いていれば到達しない")),
    )
    with pytest.raises(app.OpenAIRequestError) as excinfo:
        await app.call_gpt("ping", "sys", 4096)

    assert len(sent) == MAX_CAPABILITY_FALLBACKS + 1
    assert excinfo.value.info.param == "max_output_tokens"


async def test_last_error_is_surfaced_after_fallback_fails(gpt_env, monkeypatch, scripted_openai, warnings):
    """再送後の error が診断として返る（最初の error で潰さない）。"""
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    scripted_openai(
        (400, unsupported("temperature")),
        (400, {"error": {"message": "second failure", "type": "invalid_request_error", "param": "model"}}),
    )
    with pytest.raises(app.OpenAIRequestError) as excinfo:
        await app.call_gpt("ping", "sys", 4096)
    assert excinfo.value.info.message == "second failure"
    assert excinfo.value.info.param == "model"


# ==========
# param が null で message にしか名前が無い場合
# ==========
async def test_param_extracted_from_message_when_field_is_null(gpt_env, monkeypatch, scripted_openai, warnings):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    body = {"error": {
        "message": "Unsupported parameter: 'temperature' is not supported with this model.",
        "type": "invalid_request_error",
        "param": None,
    }}
    sent = scripted_openai((400, body), (200, completed_body("応答")))
    assert await app.call_gpt("ping", "sys", 4096) == "応答"
    assert "temperature" not in sent[1]


async def test_unparseable_message_does_not_trigger_retry(gpt_env, monkeypatch, scripted_openai):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    body = {"error": {"message": "something went wrong", "type": "invalid_request_error", "param": None}}
    sent = scripted_openai((400, body))
    with pytest.raises(app.OpenAIRequestError):
        await app.call_gpt("ping", "sys", 4096)
    assert len(sent) == 1


async def test_message_naming_a_required_param_does_not_trigger_retry(gpt_env, scripted_openai):
    """message 由来の抽出でも whitelist の外は落とさない。"""
    body = {"error": {
        "message": "Unsupported parameter: 'instructions' is not supported with this model.",
        "type": "invalid_request_error",
        "param": None,
    }}
    sent = scripted_openai((400, body))
    with pytest.raises(app.OpenAIRequestError):
        await app.call_gpt("ping", "sys", 4096)
    assert len(sent) == 1


# ==========
# WARNING が必ず残る（profile 修正が必要だと分かる）
# ==========
async def test_fallback_always_logs_a_warning(gpt_env, monkeypatch, scripted_openai, warnings):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    scripted_openai((400, unsupported("temperature")), (200, completed_body()))
    await app.call_gpt("ping", "sys", 4096)

    records = [r for r in warnings.records if r.levelno == logging.WARNING]
    assert len(records) == 1
    rendered = records[0].getMessage()
    assert "temperature" in rendered
    assert "gpt-4o" in rendered
    # 後で profile を直す必要があることが読み取れる
    assert "_CAPABILITY_PROFILES" in rendered
    assert "openai_gpt.py" in rendered


async def test_warning_is_emitted_even_when_the_retry_fails(gpt_env, monkeypatch, scripted_openai, warnings):
    """再送が失敗しても「profile が誤っている」事実は記録に残る。"""
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    scripted_openai(
        (400, unsupported("temperature")),
        (500, {"error": {"message": "boom", "type": "server_error"}}),
    )
    with pytest.raises(app.OpenAIRequestError):
        await app.call_gpt("ping", "sys", 4096)
    assert any(r.levelno == logging.WARNING for r in warnings.records)


async def test_no_warning_when_nothing_is_dropped(gpt_env, scripted_openai, warnings):
    scripted_openai((200, completed_body()))
    await app.call_gpt("ping", "sys", 4096)
    assert [r for r in warnings.records if r.levelno == logging.WARNING] == []


async def test_warning_never_contains_secrets_or_prompt(gpt_env, monkeypatch, scripted_openai, warnings):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    scripted_openai((400, unsupported("temperature")), (200, completed_body()))
    await app.call_gpt("USER-PROMPT-TEXT", "SYSTEM-PROMPT-TEXT", 4096)

    for record in warnings.records:
        rendered = record.getMessage()
        for forbidden in (SECRET_KEY, "Bearer", "Authorization", "USER-PROMPT-TEXT", "SYSTEM-PROMPT-TEXT"):
            assert forbidden not in rendered


def test_warning_reaches_stderr_without_logging_setup():
    """logger 基盤が無くても WARNING は既定で出る（別 Issue を待たずに見える）。"""
    assert logging.getLogger("openai_gpt").getEffectiveLevel() <= logging.WARNING
    assert logging.lastResort is not None
    assert logging.lastResort.level <= logging.WARNING


# ==========
# profile は書き換えない
# ==========
async def test_profile_table_is_never_mutated(gpt_env, monkeypatch, scripted_openai, warnings):
    """自動書き換えはしない。ファイルと実行時状態が乖離しないことを保証する。"""
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    before = dict(openai_gpt._CAPABILITY_PROFILES)
    caps_before = openai_gpt.resolve_capabilities("gpt-4o")

    scripted_openai((400, unsupported("temperature")), (200, completed_body()))
    await app.call_gpt("ping", "sys", 4096)

    assert openai_gpt._CAPABILITY_PROFILES == before
    assert openai_gpt.resolve_capabilities("gpt-4o") is caps_before
    assert caps_before.supports_temperature is True


async def test_next_call_repeats_the_same_fallback(gpt_env, monkeypatch, scripted_openai, warnings):
    """profile を直すまで毎回 fallback と WARNING が繰り返されるのが正しい状態。"""
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    sent = scripted_openai(
        (400, unsupported("temperature")), (200, completed_body()),
        (400, unsupported("temperature")), (200, completed_body()),
    )
    await app.call_gpt("ping", "sys", 4096)
    await app.call_gpt("ping", "sys", 4096)

    assert len(sent) == 4
    assert sent[2]["temperature"] == 0  # 2 回目も profile 通りに temperature を送る
    assert len([r for r in warnings.records if r.levelno == logging.WARNING]) == 2


# ==========
# 述語の単体 test
# ==========
@pytest.mark.parametrize(
    "param,rejected,key",
    [
        ("temperature", "temperature", "temperature"),
        ("reasoning", "reasoning", "reasoning"),
        ("reasoning.effort", "reasoning.effort", "reasoning"),
        ("  temperature  ", "temperature", "temperature"),
    ],
)
def test_droppable_param_keeps_both_names(param, rejected, key):
    """provider の拒否名と実際に外す key の両方を保持する。"""
    info = openai_gpt.OpenAIErrorInfo(
        http_status=400, message="x", type="invalid_request_error", param=param,
    )
    body = {"model": "m", "temperature": 0, "reasoning": {"effort": "high"}}
    target = droppable_param(info, body)
    assert target.rejected == rejected
    assert target.key == key


@pytest.mark.parametrize("param", ["model", "max_output_tokens", ""])
def test_droppable_param_rejects_non_droppable(param):
    info = openai_gpt.OpenAIErrorInfo(
        http_status=400, message="x", type="invalid_request_error", param=param or None,
    )
    body = {"model": "m", "temperature": 0, "reasoning": {"effort": "high"}}
    assert droppable_param(info, body) is None


# ==========
# WARNING が profile の直し方を区別できる（Codex P3-2）
# ==========
async def test_effort_rejection_warning_names_both_params(
    gpt_env, overbroad_profile, scripted_openai, warnings
):
    """reasoning.effort 拒否時、拒否名と削除 key の両方が WARNING に出る。

    削除 key（reasoning）しか出ないと、運用者は supports_reasoning を切る方向へ
    誘導される。実際に必要なのは supported_reasoning_efforts を狭めること。
    """
    scripted_openai((400, unsupported("reasoning.effort")), (200, completed_body()))
    await app.call_gpt("ping", "sys", 4096)

    rendered = _sole_warning(warnings)
    assert "reasoning.effort" in rendered      # provider が拒否した parameter
    assert "'reasoning'" in rendered           # 実際に外した key
    assert "supported_reasoning_efforts" in rendered
    assert "supports_reasoning" in rendered


async def test_effort_rejection_warning_reports_the_value_that_was_sent(
    gpt_env, overbroad_profile, monkeypatch, scripted_openai, warnings
):
    """どの effort 値が拒否されたのか分からないと profile を直せない。"""
    monkeypatch.setitem(
        openai_gpt._CAPABILITY_PROFILES, overbroad_profile,
        openai_gpt.OpenAIModelCapabilities(
            supports_reasoning=True, supports_temperature=True,
            supported_reasoning_efforts=frozenset({"xhigh"}),
            default_reasoning_effort="xhigh",
            min_output_tokens=openai_gpt.REASONING_MIN_OUTPUT_TOKENS,
        ),
    )
    scripted_openai((400, unsupported("reasoning.effort")), (200, completed_body()))
    await app.call_gpt("ping", "sys", 4096)

    assert "xhigh" in _sole_warning(warnings)


async def test_direct_reasoning_rejection_does_not_suggest_narrowing_efforts(
    gpt_env, overbroad_profile, scripted_openai, warnings
):
    """reasoning 自体を拒否された場合は capability を切るのが正しい。"""
    scripted_openai((400, unsupported("reasoning")), (200, completed_body()))
    await app.call_gpt("ping", "sys", 4096)

    rendered = _sole_warning(warnings)
    assert "supported_reasoning_efforts" not in rendered
    assert "turn the corresponding capability off" in rendered


async def test_temperature_rejection_warning_is_the_simple_form(
    gpt_env, overbroad_profile, scripted_openai, warnings
):
    scripted_openai((400, unsupported("temperature")), (200, completed_body()))
    await app.call_gpt("ping", "sys", 4096)

    rendered = _sole_warning(warnings)
    assert "temperature" in rendered
    assert "supported_reasoning_efforts" not in rendered


async def test_effort_and_direct_rejections_produce_different_warnings(
    gpt_env, overbroad_profile, scripted_openai, warnings
):
    """2 つのケースが WARNING で判別できることを 1 本で示す。"""
    scripted_openai((400, unsupported("reasoning.effort")), (200, completed_body()))
    await app.call_gpt("ping", "sys", 4096)
    effort_warning = _sole_warning(warnings)

    warnings.clear()
    scripted_openai((400, unsupported("reasoning")), (200, completed_body()))
    await app.call_gpt("ping", "sys", 4096)
    direct_warning = _sole_warning(warnings)

    assert effort_warning != direct_warning
