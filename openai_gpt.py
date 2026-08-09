"""
OpenAI Responses API の model capability profile / request builder / error 変換

model によって送信できる optional parameter が変わる（reasoning model は
`temperature` を受け付けない等）ため、判定を profile 表 1 箇所に集約している。
app.py 側に regex を散らすと、片方だけ更新された結果 400 になる。

新しい model を追加するときに触るのは `_CAPABILITY_PROFILES` と
`tests/test_openai_gpt.py` の 2 箇所だけ。
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional

DEFAULT_GPT_MODEL = "gpt-5.4"

# reasoning model では reasoning token も max_output_tokens に含まれる。budget が
# 小さいと reasoning だけで使い切り、本文が 1 文字も返らないまま
# status=incomplete / reason=max_output_tokens になる。
#
# この 16384 は測定値ではなく heuristic。根拠として言えるのは次の 2 点だけ:
#   - 従来の既定 4096 は失敗域にある（effort=high の reasoning だけで到達する）
#   - meeTai の応答は Kernel/Diag/Residue 構造で短くないため、reasoning を賄った
#     うえで本文にも数千 token 残る必要がある
# 適正値は model・effort・prompt 長で動くので「これ以上なら安全」という閾値は
# 存在しない。「これ未満はほぼ確実に壊れる」側の下限として置いている。
#
# NOTE: GPT_MAX_TOKENS / LLM_MAX_TOKENS に明示指定した値も、reasoning model では
# この下限まで引き上げる（明示設定の上書き）。コスト実験などで意図的に小さい
# budget で走らせたい場合は、この定数を下げること。env 側に逃げ道は用意していない
# ――reasoning model に小さい budget を渡すのは「空応答を受け入れる」判断であり、
# 設定ミスと区別できないため、コードを触る手間を意図的に残している。
REASONING_MIN_OUTPUT_TOKENS = 16384

_REASONING_EFFORTS = frozenset({"minimal", "low", "medium", "high"})

# error.message は provider 由来のテキストなので表示可としつつ、
# 長大な本文がそのまま UI / log に流れないよう頭を切る。
_ERROR_MESSAGE_MAX_LEN = 500


# ==========
# Capability profile
# ==========
@dataclass(frozen=True)
class OpenAIModelCapabilities:
    supports_reasoning: bool = False
    supports_temperature: bool = False
    supported_reasoning_efforts: frozenset = frozenset()
    default_reasoning_effort: Optional[str] = None
    min_output_tokens: int = 1


# 未知の model は fail-closed: 必須 field だけ送り、optional parameter は付けない。
UNKNOWN_MODEL_CAPABILITIES = OpenAIModelCapabilities()

_GPT5_REASONING = OpenAIModelCapabilities(
    supports_reasoning=True,
    supports_temperature=False,
    supported_reasoning_efforts=_REASONING_EFFORTS,
    default_reasoning_effort="high",
    min_output_tokens=REASONING_MIN_OUTPUT_TOKENS,
)

_GPT4_CHAT = OpenAIModelCapabilities(
    supports_reasoning=False,
    supports_temperature=True,
)

_CAPABILITY_PROFILES: dict[str, OpenAIModelCapabilities] = {
    "gpt-5": _GPT5_REASONING,
    "gpt-5.4": _GPT5_REASONING,
    "gpt-5.5": _GPT5_REASONING,
    "gpt-4o": _GPT4_CHAT,
    "gpt-4.1": _GPT4_CHAT,
}


def resolve_capabilities(model: str) -> OpenAIModelCapabilities:
    """model 名から capability profile を引く。

    snapshot 名（`gpt-5.4-2026-03-05`）を base model と同じ profile に解決するため、
    prefix match は `-` 区切りの場合のみ許可する。この境界条件により、
    未登録の `gpt-5.6` は `gpt-5` に吸われず unknown 扱いになり、
    「名前が gpt-5 で始まる」だけの理由で optional parameter を送ることがない。
    """
    name = (model or "").strip()
    if not name:
        return UNKNOWN_MODEL_CAPABILITIES

    # 長い prefix を優先（gpt-5.4 が gpt-5 に負けないようにする）
    for prefix in sorted(_CAPABILITY_PROFILES, key=len, reverse=True):
        if name == prefix or name.startswith(f"{prefix}-"):
            return _CAPABILITY_PROFILES[prefix]
    return UNKNOWN_MODEL_CAPABILITIES


# ==========
# Request builder
# ==========
def _env(env: Optional[Mapping[str, str]]) -> Mapping[str, str]:
    return os.environ if env is None else env


def resolve_gpt_model(env: Optional[Mapping[str, str]] = None) -> str:
    """GPT_MODEL > OPENAI_MODEL > 既定値 の優先順で model 名を決める。"""
    source = _env(env)
    for key in ("GPT_MODEL", "OPENAI_MODEL"):
        value = (source.get(key) or "").strip()
        if value:
            return value
    return DEFAULT_GPT_MODEL


def resolve_reasoning_effort(
    caps: OpenAIModelCapabilities,
    env: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """profile が許可する effort だけを返す。未指定・不正値は profile 既定値へ退避。"""
    if not caps.supports_reasoning:
        return None
    requested = (_env(env).get("GPT_REASONING_EFFORT") or "").strip().lower()
    if requested and requested in caps.supported_reasoning_efforts:
        return requested
    return caps.default_reasoning_effort


def build_gpt_request(
    *,
    prompt: str,
    instructions: str,
    max_output_tokens: int,
    env: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """/v1/responses の request body を組み立てる。"""
    model = resolve_gpt_model(env)
    caps = resolve_capabilities(model)

    body: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": prompt,
        # 明示指定より profile の下限が優先される。理由は
        # REASONING_MIN_OUTPUT_TOKENS のコメント参照。
        "max_output_tokens": max(max_output_tokens, caps.min_output_tokens),
    }

    effort = resolve_reasoning_effort(caps, env)
    if effort is not None:
        body["reasoning"] = {"effort": effort}
    if caps.supports_temperature:
        body["temperature"] = 0
    return body


# ==========
# Error 変換
# ==========
@dataclass(frozen=True)
class OpenAIErrorInfo:
    http_status: int
    message: str
    type: Optional[str] = None
    param: Optional[str] = None
    code: Optional[str] = None
    provider: str = "openai"

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "http_status": self.http_status,
            "message": self.message,
            "type": self.type,
            "param": self.param,
            "code": self.code,
        }


def format_openai_error(info: OpenAIErrorInfo) -> str:
    """診断情報を 1 行へ整形する。

    LangGraph の Message は role/content/agent しか持たないため、/chat 経路では
    構造化 field を運べない。両経路で同じ情報を見せるための共通表現。
    """
    head = f"{info.provider} HTTP {info.http_status}"
    if info.type:
        head += f" {info.type}"

    tags = []
    if info.param:
        tags.append(f"param={info.param}")
    if info.code:
        tags.append(f"code={info.code}")
    if tags:
        head += f" ({', '.join(tags)})"
    return f"{head}: {info.message}"


def _clean(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    text = text.strip()
    return text or None


def parse_openai_error(status_code: int, body: str) -> OpenAIErrorInfo:
    """error response から whitelist した field だけを取り出す。

    API key / Authorization header / request body / prompt / instructions /
    raw response が診断へ混入しないよう、既知の key 以外は読まない。
    JSON でない body・空 body・field 欠落でも壊れず、HTTP status と
    固定文言へ退避する。
    """
    message = type_ = param = code = None

    try:
        payload = json.loads(body or "")
    except (TypeError, ValueError):
        payload = None

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = _clean(error.get("message"))
            type_ = _clean(error.get("type"))
            param = _clean(error.get("param"))
            code = _clean(error.get("code"))

    if message is None:
        message = "no error details in response"
    elif len(message) > _ERROR_MESSAGE_MAX_LEN:
        message = message[:_ERROR_MESSAGE_MAX_LEN] + "…"

    return OpenAIErrorInfo(
        http_status=status_code,
        message=message,
        type=type_,
        param=param,
        code=code,
    )


class OpenAIRequestError(RuntimeError):
    """OpenAI が返した 4xx/5xx を構造化して運ぶ例外。"""

    def __init__(self, info: OpenAIErrorInfo):
        super().__init__(format_openai_error(info))
        self.info = info
