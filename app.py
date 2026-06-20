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
from typing import Literal, TypedDict, List, Dict, Optional, Annotated
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
from google import genai
from google.genai import types as genai_types
from anthropic import AsyncAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

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
        "あなたはR&D会議の参加者です。"
        "議論の背景・制約・既知情報を整理し、判断材料を増やしてください。"
        "結論を急がず、不明な点は仮定として明示してください。"
    ),
    "CRITIQUE": (
        "あなたはR&D会議の参加者です。"
        "議論の弱点・リスク・抜けを指摘し、代替案を提示してください。"
        "形式的な構造埋めは避けてください。"
    ),
    "SYNTHESIS": (
        "あなたはR&D会議の参加者です。"
        "議論を統合し、残すべき判断材料を整理してください。"
        "次アクションは必要な場合のみ提示してください。"
    ),
}

# ==========
# Agent registry
# ==========
class AgentConfig(BaseModel):
    enabled: bool = True
    max_tokens: int = 2000
    system_prompt: Optional[str] = None

class Registry(BaseModel):
    gemini: AgentConfig = AgentConfig(enabled=bool(os.getenv("GOOGLE_API_KEY")))
    claude: AgentConfig = AgentConfig(enabled=bool(os.getenv("ANTHROPIC_API_KEY")))
    gpt:    AgentConfig = AgentConfig(enabled=GPT_AVAILABLE)

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
    return (
        f"{base}\n"
        f"あなたは現在 [{agent}] として発言しています。"
        f"会議ログ内の [{agent}] の発言はあなた自身のものです。"
    )

def build_context_prompt(messages: List[Message], phase: str = "FREE") -> str:
    if phase == "FREE":
        # 同一ターン内で先行LLMの応答がコンテキストに混入しないよう、
        # 最後のユーザー入力より後のassistantメッセージを除外する
        last_user_idx = max(
            (i for i, m in enumerate(messages) if m["role"] == "user"),
            default=-1,
        )
        messages = messages[: last_user_idx + 1] if last_user_idx >= 0 else messages

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
async def call_gemini(prompt: str, sys: str, max_tokens: int) -> str:
    if _gemini_client is None:
        raise RuntimeError("Gemini client not initialized")
    resp = await _gemini_client.aio.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            system_instruction=sys,
            max_output_tokens=max_tokens,
        )
    )
    return (resp.text or "").strip()

async def call_claude(prompt: str, sys: str, max_tokens: int) -> str:
    resp = await anthropic_client.messages.create(
        model=os.getenv("CLAUDE_MODEL", "claude-opus-4-6"),
        max_tokens=max_tokens,
        system=sys,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.content[0].text.strip()

async def call_gpt(prompt: str, sys: str, max_tokens: int) -> str:
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        max_tokens=max_tokens,
        temperature=0.5
    )
    msgs = [SystemMessage(content=sys), HumanMessage(content=prompt)]
    out = await llm.ainvoke(msgs)
    return out.content.strip()

CALLERS = {
    "gemini": call_gemini,
    "claude": call_claude,
    "gpt":    call_gpt,
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
    if new_messages:
        append_messages(app.state.archive_db_path, cp.session_id, new_messages, phase=phase)

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

    async def call_one(agent: str):
        cfg: AgentConfig = getattr(REGISTRY, agent)
        if not cfg.enabled:
            return agent, None
        caller = CALLERS.get(agent)
        if caller is None:
            return agent, None
        sys_prompt = resolve_sys_prompt(agent, phase, cfg)
        try:
            text = await caller(context_prompt, sys_prompt, cfg.max_tokens)
            return agent, text
        except Exception as e:
            return agent, f"[error] {e}"

    results = await asyncio.gather(*[call_one(a) for a in participants])
    responses = [{"agent": a, "content": c} for a, c in results if c is not None]

    return {"responses": responses, "phase": phase}

class CommitPayload(BaseModel):
    session_id: str
    human_text: str
    responses: List[Dict[str, str]]
    phase: Optional[Literal["CONTEXT", "CRITIQUE", "SYNTHESIS", "FREE"]] = None

@app.post("/commit")
async def commit_fanout(p: CommitPayload):
    cfg = {"configurable": {"thread_id": p.session_id}}
    phase = p.phase or "FREE"

    messages_to_add: List[Message] = [
        {"role": "user", "content": p.human_text, "agent": "human"},
        *[
            {"role": "assistant", "content": r["content"], "agent": r["agent"]}
            for r in p.responses
            if r.get("content") and not r["content"].startswith("[error]")
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
