"""
Multi-AI Meeting Server (LangGraph版) - Asynchronous & Context-Aware Patch
修正点：
- Reducer (operator.add) の導入によるコンテキストのサイレント破棄バグ修正
- ノード関数からの差分返却（純粋関数化）によるイミュータビリティ確保
- ainvoke(None) による適正な状態レジューム
- セッション台帳とMarkdown export APIの追加
"""

import os
import asyncio
import operator
import re
from typing import Literal, TypedDict, List, Dict, Optional, Annotated, Any
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from session_archive import (
    append_messages,
    create_session_record,
    get_session_record,
    init_archive,
    list_session_records,
    render_session_markdown,
)

# LangGraph
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# SDKs
import httpx

from google import genai
from google.genai import types as genai_types
from anthropic import AsyncAnthropic

load_dotenv()

# ---------- Gemini ----------
_gemini_client = None
if os.getenv("GOOGLE_API_KEY"):
    _gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# ---------- Claude ----------
anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))

# ---------- GPT ----------
GPT_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))

# ==========
# State schema
# ==========
class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str
    agent: Optional[str]

class MeetingState(TypedDict):
    # operator.addにより、差分返却時にリストが自動結合される
    messages: Annotated[List[Message], operator.add]
    phase: Literal["CONTEXT", "CRITIQUE", "SYNTHESIS", "FREE"]
    target_agents: List[str]
    next_agent: Optional[str]
    read_only: bool
    need_human: bool
    meta: Dict

DEFAULT_SYSTEM = "あなたはAIアシスタントです。"

DIRECT_ANSWER_RULE = (
    "ユーザーが具体案・候補・レシピ・プラン・比較結果・結論を求めている場合、"
    "「追加情報が必要」「次に提案する必要がある」と述べるだけで終えず、"
    "必ず現時点の情報と明示した仮定に基づいて具体案を提示してください。"
)

KDR_RULE = (
    "不足情報は Residue に回してよいが、Kernel/Diag/Residue のすべてを未確定事項だけで埋めないでください。"
    "Kernel にはその時点での暫定判断または提案を置いてください。"
    "Diag にはその根拠・比較・制約を置いてください。"
    "Residue には本当に未解決で次回へ送るものだけを置いてください。"
)

PHASE_PROMPTS: dict[str, str] = {
    "FREE": (
        "あなたはAIアシスタントです。"
        "ユーザーの入力に直接答えてください。"
        "会議形式・前提/仮説/反証/結論/次アクション形式は使わないでください。"
        "回答は薄くしすぎず、ユーザーがそのまま使える具体性を持たせてください。"
        "条件が示されている場合はそれを守ってください。"
        "入力にない議題・評価実験・次アクションを追加しないでください。"
    ),
    "CONTEXT": (
        # DIVERGE フェーズ: 他のメンバーの意見を見ずに独立して ∇(X) を出す
        "あなたはR&D会議の参加者です。"
        "お題に対して、他のメンバーの意見に影響されず、あなた独自の観点から応答してください。"
        "Kernel（核となる主張・判断）、Diag（分析・根拠・観察）、Residue（未解決・保留・追加で必要な情報）の観点で整理してください。"
        "結論を急がず、不明な点は仮定として明示してください。"
        "ユーザーが具体案を求めている場合は、前提確認だけで終えず、必ず具体案を含めてください。不明点は仮定として明示し、Residueへ送ってください。"
    ),
    "CRITIQUE": (
        # CROSS フェーズ: 共有された全応答を見て φ(responses) を出す
        "あなたはR&D会議の参加者です。"
        "ログに記録された他のメンバーの意見を比較し、共通点・差分・矛盾・補足を指摘してください。"
        "Kernel（収束できる合意点）、Diag（議論のポイント・論点の差分）、Residue（未解決・次に必要なこと）を示してください。"
        "形式的な構造埋めではなく、実際の差分や矛盾に絞って指摘してください。"
        "ユーザーが具体案を求めている場合は、比較だけで終えず、あなた自身の改訂案または推奨案を出してください。"
    ),
    "SYNTHESIS": (
        "あなたはR&D会議の参加者です。"
        "議論全体を踏まえ、Decision/Kernel（合意・決定できること）、Diag（判断根拠）、Residue（次の会議に持ち越す未解決事項）を整理してください。"
        "Residueは次の会議のテーマ候補として明示してください。"
        "全部解決しようとせず、持ち越すものは持ち越す判断をしてください。"
        "ユーザーが選択・提案・具体化を求めている場合は、Decision/Kernel に暫定結論または具体案を必ず置いてください。Residueだけを出して終わらないでください。"
    ),
}

# ==========
# Agent registry
# ==========
def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _agent_max_tokens(agent: str, default: int = 4096) -> int:
    shared = _env_int("LLM_MAX_TOKENS", default)
    return _env_int(f"{agent.upper()}_MAX_TOKENS", shared)


class AgentConfig(BaseModel):
    enabled: bool = True
    max_tokens: int = 4096
    system_prompt: Optional[str] = None

class Registry(BaseModel):
    gemini: AgentConfig = AgentConfig(
        enabled=bool(os.getenv("GOOGLE_API_KEY")),
        max_tokens=_agent_max_tokens("gemini"),
    )
    claude: AgentConfig = AgentConfig(
        enabled=bool(os.getenv("ANTHROPIC_API_KEY")),
        max_tokens=_agent_max_tokens("claude"),
    )
    gpt:    AgentConfig = AgentConfig(
        enabled=GPT_AVAILABLE,
        max_tokens=_agent_max_tokens("gpt"),
    )

REGISTRY = Registry()

# ==========
# Availability checks
# ==========
def available_agents_sync() -> List[str]:
    avail = []
    if REGISTRY.gemini.enabled and os.getenv("GOOGLE_API_KEY"):
        avail.append("gemini")
    if REGISTRY.claude.enabled and os.getenv("ANTHROPIC_API_KEY"):
        avail.append("claude")
    if REGISTRY.gpt.enabled and os.getenv("OPENAI_API_KEY"):
        avail.append("gpt")
    return avail

# ==========
# Context Builder
# ==========
def resolve_sys_prompt(agent: str, phase: str, cfg: AgentConfig) -> str:
    base = cfg.system_prompt or PHASE_PROMPTS.get(phase, PHASE_PROMPTS["FREE"])
    rules = DIRECT_ANSWER_RULE
    if phase != "FREE":
        rules += KDR_RULE
    return (
        f"{base}\n"
        f"あなたは現在 [{agent}] として発言しています。"
        f"会議ログ内の [{agent}] の発言はあなた自身のものです。\n"
        f"{rules}"
    )

def build_context_prompt(messages: List[Message], phase: str = "FREE") -> str:
    if phase in {"FREE", "CONTEXT"}:
        # ∇フェーズ (FREE/CONTEXT=DIVERGE): 同一ターン内の先行LLM応答を除外して盲目独立性を保つ
        last_user_idx = max(
            (i for i, m in enumerate(messages) if m["role"] == "user"),
            default=-1,
        )
        messages = messages[: last_user_idx + 1] if last_user_idx >= 0 else messages
    # φフェーズ (CRITIQUE=CROSS / SYNTHESIS): 全ログを渡す（比較・統合が目的）

    prompt = "【会議ログ（コンテキスト）】\n"
    has_user_input = False
    for m in messages:
        if m["role"] == "system":
            continue
        name = m.get("agent", "human")
        prompt += f"■ [{name}]\n{m['content']}\n\n"
        if m["role"] == "user":
            has_user_input = True

    if not has_user_input:
        return ""

    prompt += "上記の会議ログを踏まえ、あなたの役割とシステムプロンプトの制約に従って発言を生成してください。"
    return prompt

# ==========
# LLM callers
# ==========
class IncompleteLLMResponse(RuntimeError):
    def __init__(self, message: str, *, finish_reason: Optional[str] = None, partial: str = ""):
        super().__init__(message)
        self.finish_reason = finish_reason
        self.partial = partial


def _finish_reason_name(reason: Any) -> Optional[str]:
    if reason is None:
        return None
    name = getattr(reason, "name", None)
    if name:
        return str(name)
    value = getattr(reason, "value", None)
    if isinstance(value, str):
        return value
    text = str(reason)
    return text.rsplit(".", 1)[-1]


def _gemini_stop_reason(reason: Any) -> bool:
    finish_reason_enum = getattr(genai_types, "FinishReason", None)
    stop_reason = getattr(finish_reason_enum, "STOP", None) if finish_reason_enum else None
    if stop_reason is not None and reason == stop_reason:
        return True

    name = _finish_reason_name(reason)
    return name is not None and name.upper() in {"STOP", "FINISH_REASON_STOP"}


def _gemini_candidate(resp: Any) -> Any:
    candidates = getattr(resp, "candidates", None) or []
    return candidates[0] if candidates else None


def _gemini_parts_text(candidate: Any) -> str:
    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None) or []
    texts = []
    for part in parts:
        text = getattr(part, "text", None)
        if text:
            texts.append(text)
    return "".join(texts).strip()


def _gemini_text(resp: Any, candidate: Any) -> str:
    try:
        return (resp.text or "").strip()
    except Exception:
        # google-genai may reject resp.text access when the candidate is not a complete text response.
        return _gemini_parts_text(candidate)


def _is_gemini_complete(reason: Any, text: str) -> bool:
    if not text:
        return False
    if reason is None:
        # Older SDKs or unusual response shapes may not expose finish_reason.
        # Treat a non-empty text as complete instead of failing every Gemini call.
        return True
    return _gemini_stop_reason(reason)


def _gemini_thinking_budget(model: str) -> Optional[int]:
    raw = os.getenv("GEMINI_THINKING_BUDGET")
    if raw is None:
        return None

    value = raw.strip().lower()
    if value in {"", "none", "default"}:
        return None
    if value in {"auto", "dynamic"}:
        return -1
    try:
        return int(value)
    except ValueError:
        return None


def _gemini_thinking_config(model: str) -> Any:
    budget = _gemini_thinking_budget(model)
    thinking_config_type = getattr(genai_types, "ThinkingConfig", None)
    if budget is None or thinking_config_type is None:
        return None
    return thinking_config_type(thinking_budget=budget)


async def call_gemini(prompt: str, sys: str, max_tokens: int) -> str:
    if _gemini_client is None:
        raise RuntimeError("Gemini client not initialized")

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    config_kwargs = {
        "system_instruction": sys,
        "max_output_tokens": max_tokens,
    }
    thinking_config = _gemini_thinking_config(model)
    if thinking_config is not None:
        config_kwargs["thinking_config"] = thinking_config

    last_error: Optional[IncompleteLLMResponse] = None
    for _ in range(2):
        resp = await _gemini_client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(**config_kwargs)
        )
        candidate = _gemini_candidate(resp)
        finish_reason = getattr(candidate, "finish_reason", None)
        finish_reason_name = _finish_reason_name(finish_reason)
        text = _gemini_text(resp, candidate)
        if _is_gemini_complete(finish_reason, text):
            return text

        partial = text[:200]
        last_error = IncompleteLLMResponse(
            f"Gemini incomplete response: finish_reason={finish_reason_name or 'unknown'}, partial={partial!r}",
            finish_reason=finish_reason_name,
            partial=text,
        )

    raise last_error or IncompleteLLMResponse("Gemini empty response", finish_reason=None, partial="")

async def call_claude(prompt: str, sys: str, max_tokens: int) -> str:
    resp = await anthropic_client.messages.create(
        model=os.getenv("CLAUDE_MODEL", "claude-opus-4-6"),
        max_tokens=max_tokens,
        system=sys,
        messages=[{"role": "user", "content": prompt}]
    )
    text = "".join(getattr(block, "text", "") for block in resp.content).strip()
    stop_reason = getattr(resp, "stop_reason", None)

    if text and stop_reason in {None, "end_turn", "stop_sequence"}:
        return text

    raise IncompleteLLMResponse(
        f"Claude incomplete response: stop_reason={stop_reason or 'unknown'}, partial={text[:200]!r}",
        finish_reason=stop_reason,
        partial=text,
    )

_GPT_REASONING = re.compile(r"^gpt-5")
_GPT_NO_TEMPERATURE = re.compile(r"^gpt-5\.5")

async def call_gpt(prompt: str, sys: str, max_tokens: int) -> str:
    model = os.getenv("GPT_MODEL") or os.getenv("OPENAI_MODEL", "gpt-5.4")
    reasoning_effort = os.getenv("GPT_REASONING_EFFORT", "high")
    body: dict = {
        "model": model,
        "instructions": sys,
        "input": prompt,
        "max_output_tokens": max_tokens,
    }
    if _GPT_REASONING.match(model):
        body["reasoning"] = {"effort": reasoning_effort}
    if not _GPT_NO_TEMPERATURE.match(model):
        body["temperature"] = 0
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=120.0,
        )
        r.raise_for_status()
    data = r.json()
    status = data.get("status")
    content = "".join(
        part["text"]
        for item in data.get("output", [])
        if item.get("type") == "message"
        for part in item.get("content", [])
        if part.get("type") == "output_text" and part.get("text")
    )
    if content and status == "completed":
        return content.strip()
    incomplete_reason = (data.get("incomplete_details") or {}).get("reason")
    raise IncompleteLLMResponse(
        f"GPT incomplete response: status={status or 'unknown'}, reason={incomplete_reason or 'unknown'}, partial={content[:200]!r}",
        finish_reason=incomplete_reason or status,
        partial=content,
    )

CALLERS = {
    "gemini": call_gemini,
    "claude": call_claude,
    "gpt":    call_gpt,
}

# ==========
# Completion / archive guards
# ==========
def _error_content(content: str) -> bool:
    return content.startswith("[error]") or re.match(r"^\[[A-Za-z0-9_-]+\] error:", content) is not None


def _archivable_message(message: dict[str, Any]) -> bool:
    role = message.get("role")
    content = message.get("content") or ""
    if role == "user":
        return bool(content)
    if role == "assistant":
        return bool(content) and not _error_content(content)
    return False


def _response_ok(response: dict[str, Any]) -> bool:
    content = response.get("content") or ""
    return response.get("ok", True) is not False and bool(content) and not _error_content(content)


def _error_response(agent: str, exc: Exception) -> dict[str, Any]:
    finish_reason = getattr(exc, "finish_reason", None)
    message = str(exc)
    return {
        "agent": agent,
        "content": f"[error] {message}",
        "ok": False,
        "finish_reason": finish_reason,
        "error": message,
    }

# ==========
# Supervisor node
# ==========
def pick_next_agent(state: MeetingState) -> dict:
    dynamic = set(state.get("target_agents") or [])
    avail = set(available_agents_sync())

    candidates = list(dynamic.intersection(avail)) if dynamic else list(avail)

    if state.get("read_only", False):
        return {"next_agent": None, "need_human": False}

    if not candidates:
        return {"next_agent": None, "need_human": True}

    phase = state.get("phase", "FREE")
    priority = {
        "CRITIQUE":  ["claude", "gemini", "gpt"],
        "SYNTHESIS": ["gemini", "gpt", "claude"],
        "CONTEXT":   ["gemini", "claude", "gpt"],
        "FREE":      ["gemini", "claude", "gpt"],
    }
    for a in priority[phase]:
        if a in candidates:
            return {"next_agent": a, "need_human": False}

    return {"next_agent": None, "need_human": True}

# ==========
# Agent node
# ==========
async def agent_node(state: MeetingState) -> dict:
    agent = state.get("next_agent")
    if not agent:
        return {}

    cfg: AgentConfig = getattr(REGISTRY, agent)
    caller = CALLERS.get(agent)

    if caller is None:
        return {
            "messages": [{"role": "assistant", "content": f"[{agent}] unknown agent", "agent": agent}],
            "need_human": True
        }

    phase = state.get("phase", "FREE")
    sys_prompt = resolve_sys_prompt(agent, phase, cfg)
    prompt = build_context_prompt(state.get("messages", []), phase)
    if not prompt:
        return {"next_agent": None, "need_human": True}

    try:
        text = await caller(prompt, sys_prompt, cfg.max_tokens)
        return {
            "messages": [{"role": "assistant", "content": text, "agent": agent}],
            "next_agent": None,
            "need_human": False
        }
    except Exception as e:
        return {
            "messages": [{"role": "assistant", "content": f"[{agent}] error: {e}", "agent": agent}],
            "next_agent": None,
            "need_human": True
        }

# ==========
# Build graph
# ==========
graph = StateGraph(MeetingState)
graph.add_node("supervisor", pick_next_agent)
graph.add_node("agent", agent_node)

graph.set_entry_point("supervisor")
graph.add_edge("supervisor", "agent")
graph.add_edge("agent", END)

db_path = os.getenv("CHECKPOINT_DB", "checkpoints.sqlite")
archive_db_path = os.getenv("APP_DB", "meetai_app.sqlite")

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_archive(archive_db_path)
    app.state.archive_db_path = archive_db_path
    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        app.state.app_graph = graph.compile(checkpointer=checkpointer)
        yield

# ==========
# FastAPI
# ==========
app = FastAPI(title="Multi-AI Meeting", lifespan=lifespan)

# ==========
# Archive access scope
# ==========
def _normalize_email(email: Optional[str]) -> Optional[str]:
    if not email:
        return None
    normalized = email.strip().lower()
    return normalized or None


def _archive_scope() -> str:
    scope = os.getenv("SESSION_ARCHIVE_SCOPE", "user").strip().lower()
    return scope if scope in {"user", "shared", "all"} else "user"


def _shared_archive_emails() -> set[str]:
    raw = os.getenv("SESSION_ARCHIVE_SHARED_EMAILS", "")
    return {
        e.strip().lower()
        for e in raw.split(",")
        if e.strip()
    }


def _allowed_archive_emails(requester_email: Optional[str]) -> Optional[list[str]]:
    scope = _archive_scope()
    if scope == "all":
        return None

    requester = _normalize_email(requester_email)
    if requester is None:
        return []

    if scope == "shared":
        shared = _shared_archive_emails()
        if requester in shared:
            return sorted(shared)

    return [requester]


def _assert_archive_visible(record: dict, requester_email: Optional[str]) -> None:
    allowed = _allowed_archive_emails(requester_email)
    if allowed is None:
        return
    owner = _normalize_email(record.get("user_email"))
    if owner not in allowed:
        raise HTTPException(status_code=404, detail="session not found")


class StartPayload(BaseModel):
    session_id: str = Field(..., description="会議ID")
    participants: List[str] = Field(default_factory=lambda: ["gemini", "claude", "gpt"])
    phase: Literal["CONTEXT", "CRITIQUE", "SYNTHESIS", "FREE"] = "FREE"
    title: Optional[str] = None
    user_email: Optional[str] = None

@app.post("/session")
async def start_session(p: StartPayload):
    state = {
        "messages": [{"role": "system", "content": f"session:{p.session_id}", "agent": "system"}],
        "phase": p.phase,
        "target_agents": p.participants,
        "next_agent": None,
        "read_only": False,
        "need_human": False,
        "meta": {}
    }
    await app.state.app_graph.aupdate_state(
        config={"configurable": {"thread_id": p.session_id}},
        values=state,
        as_node="__start__"
    )
    create_session_record(
        app.state.archive_db_path,
        session_id=p.session_id,
        title=p.title,
        user_email=_normalize_email(p.user_email),
        phase=p.phase,
        agents=p.participants,
    )
    return {"ok": True, "session_id": p.session_id}

class ChatPayload(BaseModel):
    session_id: str
    text: str
    phase: Optional[Literal["CONTEXT", "CRITIQUE", "SYNTHESIS", "FREE"]] = None
    read_only: bool = False

@app.post("/chat")
async def send_chat(cp: ChatPayload):
    cfg = {"configurable": {"thread_id": cp.session_id}}
    before = await app.state.app_graph.aget_state(config=cfg)
    before_len = len(before.values.get("messages", [])) if before and before.values else 0
    phase = cp.phase or "FREE"

    await app.state.app_graph.aupdate_state(
        config=cfg,
        values={
            "messages": [{"role": "user", "content": cp.text, "agent": "human"}],
            "phase": phase,
            "read_only": cp.read_only
        },
        as_node="__start__"
    )
    # 適切なレジュームトリガー(None)を送信
    await app.state.app_graph.ainvoke(None, config=cfg)
    current = (await app.state.app_graph.aget_state(config=cfg)).values
    all_messages = current.get("messages", [])
    new_messages = [m for m in all_messages[before_len:] if m.get("role") != "system"]
    archived_messages = [m for m in new_messages if _archivable_message(m)]
    if archived_messages:
        append_messages(app.state.archive_db_path, cp.session_id, archived_messages, phase=phase)

    return {
        "messages": all_messages[-5:],
        "phase": current.get("phase"),
        "need_human": current.get("need_human"),
        "next_agent": current.get("next_agent")
    }

class AskAllPayload(BaseModel):
    session_id: str
    text: str
    participants: Optional[List[str]] = None
    phase: Optional[Literal["CONTEXT", "CRITIQUE", "SYNTHESIS", "FREE"]] = None

@app.post("/ask-all")
async def ask_all(p: AskAllPayload):
    cfg_state = {"configurable": {"thread_id": p.session_id}}
    current = await app.state.app_graph.aget_state(config=cfg_state)
    state_values = current.values if current and current.values else {}
    messages: List[Message] = state_values.get("messages", [])
    phase: str = p.phase or state_values.get("phase", "FREE")

    # 共有コンテキスト + 今回のユーザー入力でプロンプトを生成
    # 各エージェントには同一コンテキストを渡す（先行応答はstateに未commit）
    user_msg: Message = {"role": "user", "content": p.text, "agent": "human"}
    context_prompt = build_context_prompt(messages + [user_msg], phase)

    # セッション開始時の参加者を優先、なければ利用可能な全エージェント
    session_targets: List[str] = state_values.get("target_agents") or []
    default_agents = session_targets or available_agents_sync()
    participants = [a for a in (p.participants or default_agents) if hasattr(REGISTRY, a)]

    async def call_one(agent: str) -> Optional[dict[str, Any]]:
        cfg: AgentConfig = getattr(REGISTRY, agent)
        if not cfg.enabled:
            return None
        caller = CALLERS.get(agent)
        if caller is None:
            return None
        sys_prompt = resolve_sys_prompt(agent, phase, cfg)
        try:
            text = await caller(context_prompt, sys_prompt, cfg.max_tokens)
            return {
                "agent": agent,
                "content": text,
                "ok": True,
                "finish_reason": None,
                "error": None,
            }
        except Exception as e:
            return _error_response(agent, e)

    results = await asyncio.gather(*[call_one(a) for a in participants])
    responses = [r for r in results if r is not None]

    return {"responses": responses, "phase": phase}

class CommitPayload(BaseModel):
    session_id: str
    human_text: str
    responses: List[Dict[str, Any]]
    phase: Optional[Literal["CONTEXT", "CRITIQUE", "SYNTHESIS", "FREE"]] = None

@app.post("/commit")
async def commit_fanout(p: CommitPayload):
    cfg = {"configurable": {"thread_id": p.session_id}}
    phase = p.phase or "FREE"

    messages_to_add: List[Message] = [
        {"role": "user", "content": p.human_text, "agent": "human"},
        *[
            {"role": "assistant", "content": str(r["content"]), "agent": str(r["agent"])}
            for r in p.responses
            if _response_ok(r)
        ],
    ]

    await app.state.app_graph.aupdate_state(
        config=cfg,
        values={"messages": messages_to_add, "phase": phase},
        as_node="__start__"
    )

    append_messages(app.state.archive_db_path, p.session_id, messages_to_add, phase=phase)

    return {"ok": True, "committed": len(messages_to_add)}


class ConfigPayload(BaseModel):
    session_id: str
    participants: Optional[List[str]] = None
    system_prompts: Optional[Dict[str, str]] = None
    enable: Optional[Dict[str, bool]] = None

@app.post("/config")
async def reconfigure(cp: ConfigPayload):
    if cp.participants is not None:
        await app.state.app_graph.aupdate_state(
            config={"configurable": {"thread_id": cp.session_id}},
            values={"target_agents": cp.participants},
            as_node="__start__"
        )
    if cp.system_prompts:
        for k, v in cp.system_prompts.items():
            if hasattr(REGISTRY, k):
                getattr(REGISTRY, k).system_prompt = v
    if cp.enable:
        for k, v in cp.enable.items():
            if hasattr(REGISTRY, k):
                getattr(REGISTRY, k).enabled = bool(v)
    return {"ok": True}

@app.get("/sessions")
async def list_sessions(
    range_key: Optional[str] = Query(default="7d", alias="range"),
    month: Optional[str] = None,
    user_email: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=500),
):
    try:
        sessions = list_session_records(
            app.state.archive_db_path,
            range_key=range_key,
            month=month,
            limit=limit,
            allowed_user_emails=_allowed_archive_emails(user_email),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return {"sessions": sessions}

@app.get("/sessions/{session_id}")
async def get_session(session_id: str, user_email: Optional[str] = None):
    record = get_session_record(app.state.archive_db_path, session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="session not found")
    _assert_archive_visible(record, user_email)
    return record

@app.get("/sessions/{session_id}/export")
async def export_session(
    session_id: str,
    format: str = Query(default="markdown"),
    user_email: Optional[str] = None,
):
    record = get_session_record(app.state.archive_db_path, session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="session not found")
    _assert_archive_visible(record, user_email)

    if format == "json":
        return JSONResponse(record)
    if format == "markdown":
        markdown = render_session_markdown(record)
        return PlainTextResponse(markdown, media_type="text/markdown; charset=utf-8")

    raise HTTPException(status_code=400, detail="format must be markdown or json")

@app.get("/health")
async def health():
    return {
        "gemini": REGISTRY.gemini.enabled and bool(os.getenv("GOOGLE_API_KEY")),
        "claude": REGISTRY.claude.enabled and bool(os.getenv("ANTHROPIC_API_KEY")),
        "gpt":    GPT_AVAILABLE,
    }

# Run: uvicorn app:app --host 127.0.0.1 --port 8008
