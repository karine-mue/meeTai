"""
Multi-AI Meeting Server (LangGraph版) - Asynchronous & Context-Aware Patch
修正点：
- Reducer (operator.add) の導入によるコンテキストのサイレント破棄バグ修正
- ノード関数からの差分返却（純粋関数化）によるイミュータビリティ確保
- ainvoke(None) による適正な状態レジューム
- セッション台帳とMarkdown export APIの追加
"""

import os
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

DEFAULT_SYSTEM = (
    "あなたはR&D会議の参加者です。発話は簡潔（最大600字）。"
    "フェーズに従い、前提→仮説→反証→結論→次アクションの順で述べる。"
)

# ==========
# Agent registry
# ==========
class AgentConfig(BaseModel):
    enabled: bool = True
    max_tokens: int = 2000
    system_prompt: str = DEFAULT_SYSTEM

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
def build_context_prompt(messages: List[Message]) -> str:
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

    prompt = build_context_prompt(state.get("messages", []))
    if not prompt:
        return {"next_agent": None, "need_human": True}

    cfg: AgentConfig = getattr(REGISTRY, agent)
    caller = CALLERS.get(agent)

    if caller is None:
        return {
            "messages": [{"role": "assistant", "content": f"[{agent}] unknown agent", "agent": agent}],
            "need_human": True
        }

    try:
        text = await caller(prompt, cfg.system_prompt, cfg.max_tokens)
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

class StartPayload(BaseModel):
    session_id: str = Field(..., description="会議ID")
    participants: List[str] = Field(default_factory=lambda: ["gemini", "claude", "gpt"])
    phase: Literal["CONTEXT", "CRITIQUE", "SYNTHESIS", "FREE"] = "FREE"
    title: Optional[str] = None
    user_email: Optional[str] = None

@app.post("/session")
async def start_session(p: StartPayload):
    state = {
        "messages": [{"role": "system", "content": DEFAULT_SYSTEM, "agent": "system"}],
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
        user_email=p.user_email,
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
    limit: int = Query(default=100, ge=1, le=500),
):
    return {
        "sessions": list_session_records(
            app.state.archive_db_path,
            range_key=range_key,
            month=month,
            limit=limit,
        )
    }

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    record = get_session_record(app.state.archive_db_path, session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="session not found")
    return record

@app.get("/sessions/{session_id}/export")
async def export_session(session_id: str, format: str = Query(default="markdown")):
    record = get_session_record(app.state.archive_db_path, session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="session not found")

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
