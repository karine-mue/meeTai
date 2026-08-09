"""
OpenAI Responses API の model capability profile / request builder / error 変換

model によって送信できる optional parameter が変わる（reasoning model は
`temperature` を受け付けない等）ため、判定を profile 表 1 箇所に集約している。
app.py 側に regex を散らすと、片方だけ更新された結果 400 になる。

新しい model を追加するときに触るのは `_CAPABILITY_PROFILES` と
`tests/test_openai_gpt.py` の 2 箇所だけ。snapshot（`gpt-5.4-2026-03-05`）は
自動で base の profile を継承するが、variant（`gpt-5.4-pro` 等）は capability が
base と異なるため 1 行ずつ明示登録する。登録のない名前は fail-closed。
"""

import json
import logging
import os
import re
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

# reasoning effort の許容値は model ごとに違う。共有定数にすると 1 モデルの
# 実測値が他モデルへ黙って伝播するため、profile ごとに独立して定義する。
#
# whitelist の外し方は非対称で、広すぎる方が危険:
#   - 広すぎる → 未対応値をそのまま送って 400（この PR が直している不具合そのもの）
#   - 狭すぎる → default_reasoning_effort へ退避するだけ（400 にはならない）
# よって未検証の model には保守的な集合を当てる。
_EFFORTS_GPT_5_4 = frozenset({"none", "low", "medium", "high", "xhigh"})
_EFFORTS_GPT_5_4_PRO = frozenset({"medium", "high", "xhigh"})

# gpt-5 / gpt-5.5 の許容値は誰も現行仕様と突き合わせていない。low/medium/high は
# reasoning model でほぼ共通なのでこれだけを許可し、none / minimal / xhigh は
# 検証できるまで意図的に除外している（除外側の失敗は既定値への退避で済む）。
_EFFORTS_UNVERIFIED = frozenset({"low", "medium", "high"})

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

def _reasoning(efforts: frozenset, default: str = "high") -> OpenAIModelCapabilities:
    return OpenAIModelCapabilities(
        supports_reasoning=True,
        supports_temperature=False,
        supported_reasoning_efforts=efforts,
        default_reasoning_effort=default,
        min_output_tokens=REASONING_MIN_OUTPUT_TOKENS,
    )


_CHAT = OpenAIModelCapabilities(
    supports_reasoning=False,
    supports_temperature=True,
)

# variant（-pro, -mini 等）は base と capability が違うため、1 行ずつ明示登録する。
# 未登録の名前は継承せず fail-closed になる。
_CAPABILITY_PROFILES: dict[str, OpenAIModelCapabilities] = {
    # gpt-5.4 / gpt-5.4-pro の effort は現行仕様と突き合わせ済み。
    # pro は low 以下を持たないため base とは別 profile。
    "gpt-5.4": _reasoning(_EFFORTS_GPT_5_4),
    "gpt-5.4-pro": _reasoning(_EFFORTS_GPT_5_4_PRO),
    # 以下 2 つの effort は未検証（_EFFORTS_UNVERIFIED のコメント参照）。
    # temperature 非対応は commit 263456e の時点で判明していた挙動。
    "gpt-5": _reasoning(_EFFORTS_UNVERIFIED),
    "gpt-5.5": _reasoning(_EFFORTS_UNVERIFIED),
    # non-reasoning chat model。temperature を受け付ける。
    "gpt-4o": _CHAT,
    "gpt-4o-mini": _CHAT,
    "gpt-4.1": _CHAT,
    "gpt-4.1-mini": _CHAT,
}

# snapshot suffix は日付（gpt-5.4-2026-03-05）。variant 名（gpt-5.4-pro）と
# 区別するため、日付形式のときだけ base の profile を継承する。
_SNAPSHOT_NAME = re.compile(r"^(?P<base>.+)-\d{4}-\d{2}-\d{2}$")


def resolve_capabilities(model: str) -> OpenAIModelCapabilities:
    """model 名から capability profile を引く。

    解決は「完全一致 → 日付 suffix を落として完全一致 → unknown」の 3 段。
    prefix match を使わないのは、snapshot と variant を区別できないため:

        gpt-5.4-2026-03-05  → snapshot。gpt-5.4 と同じ capability
        gpt-5.4-pro         → 別 variant。effort の許容値が gpt-5.4 と異なる

    「gpt-5.4- で始まれば gpt-5.4」という規則ではこの 2 つが同じ profile に
    落ちてしまい、pro へ未対応の effort を送りうる。日付形式かどうかで
    判定することで、未登録の variant（gpt-5.6 や未知の -pro 系）は
    継承されず fail-closed になる。
    """
    name = (model or "").strip()
    if not name:
        return UNKNOWN_MODEL_CAPABILITIES

    profile = _CAPABILITY_PROFILES.get(name)
    if profile is not None:
        return profile

    snapshot = _SNAPSHOT_NAME.match(name)
    if snapshot:
        # variant の snapshot（gpt-5.4-pro-2026-03-05）も base 名で引ける
        return _CAPABILITY_PROFILES.get(snapshot.group("base"), UNKNOWN_MODEL_CAPABILITIES)

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


# ==========
# Capability fallback
# ==========
# profile が実際の model 仕様より広い場合、400 の error.param が「送ってはいけない
# parameter」を名指ししてくる。その parameter を落として再送することで、profile
# 未整備の model でも応答を得られるようにする。再送は MAX_CAPABILITY_FALLBACKS
# 回（= 落とせる key の数）まで。
#
# 400 は推論前に弾かれるので再送コストはほぼ無い（429 / 5xx では再送しない）。
#
# 落とすのは自分が付けた optional parameter だけ。model / input / instructions /
# max_output_tokens は必須なので対象外。
_DROPPABLE_PARAMS: dict[str, str] = {
    "temperature": "temperature",
    "reasoning": "reasoning",
    # effort の値が未対応でも reasoning object ごと落とす。reasoning: {} が有効か
    # 保証がないため、API 既定の effort に委ねる方が安全。
    "reasoning.effort": "reasoning",
}

# 落とせる body key は temperature / reasoning の 2 つなので、再送は最大 2 回。
MAX_CAPABILITY_FALLBACKS = 2

# error.param が null で message 側にしか parameter 名が無い場合の保険。
# 抽出した名前は _DROPPABLE_PARAMS に載っているものだけ採用するため、
# 誤検出しても temperature / reasoning 以外は落ちない。
_UNSUPPORTED_PARAM_IN_MESSAGE = re.compile(r"[Uu]nsupported parameter: '(?P<param>[^']+)'")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DroppableParam:
    """provider が拒否した parameter と、実際に body から外す top-level key。

    この 2 つは一致しないことがある（`reasoning.effort` を拒否されても外すのは
    `reasoning` object 全体）。profile の直し方が変わるため、両方を保持する。
    """

    rejected: str
    key: str


def droppable_param(info: OpenAIErrorInfo, body: Mapping[str, Any]) -> Optional[DroppableParam]:
    """400 が「未対応 optional parameter」由来なら、拒否名と落とす key の組を返す。

    再送してよい条件を全て満たさない限り None を返す（= 呼び出し側は素直に
    例外を上げる）。429 / 5xx / 認証エラーはここで弾かれる。
    """
    if info.http_status != 400 or info.type != "invalid_request_error":
        return None

    name = info.param
    if name is None:
        matched = _UNSUPPORTED_PARAM_IN_MESSAGE.search(info.message or "")
        name = matched.group("param") if matched else None
    if name is None:
        return None

    rejected = name.strip()
    key = _DROPPABLE_PARAMS.get(rejected)
    # body に無いものは落とせない（同じ 400 で無限に再送するのを防ぐ）
    if key is None or key not in body:
        return None
    return DroppableParam(rejected=rejected, key=key)


def _fallback_remedy(target: DroppableParam, sent: Optional[str]) -> str:
    """profile の直し方を WARNING 内で指示する。

    拒否名と削除 key が違う場合（reasoning.effort）、`reasoning` 全体を落とした
    ことだけを見て supports_reasoning を切ると過剰修正になる。実際に必要なのは
    supported_reasoning_efforts を狭めることなので、そこを明示する。
    """
    if target.rejected == target.key:
        return (
            f"the profile claims {target.key!r} is supported but the API rejected it; "
            f"turn the corresponding capability off for this model"
        )
    return (
        f"the API rejected {target.rejected!r} (sent {sent!r}), not {target.key!r} itself -- "
        f"remove {sent!r} from supported_reasoning_efforts rather than clearing "
        f"supports_reasoning. This retry ran without {target.key!r}, so the API default was used"
    )


def drop_unsupported_param(body: dict[str, Any], target: DroppableParam) -> None:
    """未対応 parameter を body から外し、profile 修正が必要なことを WARNING で残す。

    profile（_CAPABILITY_PROFILES）は書き換えない。実行時に書き換えると
    プロセスごとに状態が分岐し、ファイルの内容と一致しなくなるため。
    運用者が profile を直すまで、この WARNING が毎回出続けるのが正しい状態。
    """
    reasoning = body.get("reasoning")
    sent_effort = reasoning.get("effort") if isinstance(reasoning, dict) else None

    body.pop(target.key, None)
    logger.warning(
        "openai capability fallback: model %r rejected %r, dropped %r from the request "
        "and retried. Update _CAPABILITY_PROFILES in openai_gpt.py -- %s.",
        body.get("model"),
        target.rejected,
        target.key,
        _fallback_remedy(target, sent_effort),
    )
