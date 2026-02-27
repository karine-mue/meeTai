"""
Multi-AI Meeting Server (LangGraph版) - Asynchronous & Context-Aware Patch
修正点：
- コンテキスト喪失バグの修正（全メッセージ履歴を文字列化して渡す構造へ変更）
- 完全非同期I/O化（AsyncAnthropic, generate_content_async）
- langgraph 0.3系対応（SqliteSaverのimport）
- QWEN_BASE_URL をollama互換に修正（11434/v1）
- GPTノードを休眠状態で保持（APIキー未定義で自動パージ）
"""

import os
import asyncio
from typing import Literal, TypedDict, List, Dict, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

# LangGraph
from langgraph.graph import StateGraph, END

# SqliteSaver: 0.3系では langgraph-checkpoint-sqlite が別パッケージ
# pip install langgraph-checkpoint-sqlite
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:
    from langgraph_checkpoint_sqlite import SqliteSaver

# SDKs
import google.generativeai as genai
from anthropic import AsyncAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# ---------- Gemini ----------
if os.getenv("GOOGLE_API_KEY"):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ---------- Claude ----------
anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))

# ---------- Qwen (ollama) ----------
# ollamaのデフォルトポートは11434
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "http://localhost:11434/v1")
QWEN_API_KEY  = os.getenv("QWEN_API_KEY", "ollama")   # ollamaはダミーキーでOK
QWEN_MODEL    = os.getenv("QWEN_MODEL", "qwen2.5:14b")

# ---------- GPT (optional) ----------
GPT_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))

# ==========
# State schema
# ==========
class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str
    agent: Optional[str]

class MeetingState(TypedDict):
    messages: List[Message]
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
    max_tokens: int = 600
    system_prompt: str = DEFAULT_SYSTEM

class Registry(BaseModel):
    gemini: AgentConfig = AgentConfig(enabled=bool(os.getenv("GOOGLE_API_KEY")))
    claude: AgentConfig = AgentConfig(enabled=bool(os.getenv("ANTHROPIC_API_KEY")))
    qwen:   AgentConfig = AgentConfig(enabled=True)
    gpt:    AgentConfig = AgentConfig(enabled=GPT_AVAILABLE)

REGISTRY = Registry()

# ==========
# Availability checks
# ==========
async def is_qwen_up(timeout: float = 2.0) -> bool:
    """ollama互換エンドポイントの死活確認"""
    url = QWEN_BASE_URL.rstrip("/")
    # /v1/models または /api/tags で確認
    check_url = url + "/models" if url.endswith("/v1") else url + "/v1/models"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(
                check_url,
                headers={"Authorization": f"Bearer {QWEN_API_KEY}"}
            )
            return r.status_code == 200
    except Exception:
        return False

def available_agents_sync() -> List[str]:
    avail = []
    if REGISTRY.gemini.enabled and os.getenv("GOOGLE_API_KEY"):
        avail.append("gemini")
    if REGISTRY.claude.enabled and os.getenv("ANTHROPIC_API_KEY"):
        avail.append("claude")
    if REGISTRY.gpt.enabled and os.getenv("OPENAI_API_KEY"):
        avail.append("gpt")
    # qwenは非同期ヘルスチェックのためここでは保留
    return avail

# ==========
# Context Builder
# ==========
def build_context_prompt(messages: List[Message]) -> str:
    """全メッセージ履歴を単一のコンテキストテキストへ変換し、APIの厳密なRole制約を回避する"""
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
# LLM callers (完全非同期化)
async def call_gemini(prompt: str, sys: str, max_tokens: int) -> str:
    model = genai.GenerativeModel(
        model_name=os.getenv("GEMINI_MODEL", "gemini-1.5-pro"),
        system_instruction=sys
    )
    resp = await model.generate_content_async(
        prompt,
        generation_config=genai.types.GenerationConfig(max_output_tokens=max_tokens)
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

async def call_qwen(prompt: str, sys: str, max_tokens: int) -> str:
    if not await is_qwen_up():
        raise RuntimeError("Qwen (ollama) server not available")
    llm = ChatOpenAI(
        base_url=QWEN_BASE_URL,
        api_key=QWEN_API_KEY,
        model=QWEN_MODEL,
        max_tokens=max_tokens,
        temperature=0.6
    )
    msgs = [SystemMessage(content=sys), HumanMessage(content=prompt)]
    out = await llm.ainvoke(msgs)
    return out.content.strip()

async def call_gpt(prompt: str, sys: str, max_tokens: int) -> str:
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        max_tokens=max_tokens,
        temperature=0.5
    )
    msgs = [SystemMessage(content=sys), HumanMessage(content=prompt)]
    out = await llm.ainvoke(msgs)
    return out.content.strip()

# callerのディスパッチテーブル（CALLERSの構文エラーをここで修正）
CALLERS = {
    "gemini": call_gemini,
    "claude": call_claude,
    "qwen":   call_qwen,
    "gpt":    call_gpt,
}

# ==========
# Supervisor node
# ==========
def pick_next_agent(state: MeetingState) -> MeetingState:
    dynamic = set(state.get("target_agents") or [])
    avail = set(available_agents_sync())

    # qwenはユーザが指定していれば候補に含める（実呼び出し時にヘルスチェック）
    if "qwen" in dynamic:
        avail.add("qwen")

    candidates = list(dynamic.intersection(avail)) if dynamic else list(avail)

    if state.get("read_only", False):
        state["next_agent"] = None
        state["need_human"] = False
        return state

    if not candidates:
        state["next_agent"] = None
        state["need_human"] = True
        return state

    phase = state.get("phase", "FREE")
    priority = {
        "CRITIQUE":  ["claude", "gemini", "qwen", "gpt"],
        "SYNTHESIS": ["qwen", "gemini", "gpt", "claude"],
        "CONTEXT":   ["gemini", "claude", "qwen", "gpt"],
        "FREE":      ["gemini", "claude", "qwen", "gpt"],
    }
    for a in priority[phase]:
        if a in candidates:
            state["next_agent"] = a
            state["need_human"] = False
            return state

    state["next_agent"] = None
    state["need_human"] = True
    return state

# ==========
# Agent node
# ==========
async def agent_node(state: MeetingState) -> MeetingState:
    agent = state.get("next_agent")
    if not agent:
        return state

    prompt = build_context_prompt(state["messages"])
    if not prompt:
        state["next_agent"] = None
        state["need_human"] = True
        return state

    cfg: AgentConfig = getattr(REGISTRY, agent)
    caller = CALLERS.get(agent)

    if caller is None:
        state["messages"].append({
            "role": "assistant",
            "content": f"[{agent}] unknown agent",
            "agent": agent
        })
        state["need_human"] = True
        return state

    try:
        text = await caller(prompt, cfg.system_prompt, cfg.max_tokens)
        state["messages"].append({
            "role": "assistant",
            "content": text,
            "agent": agent
        })
        state["next_agent"] = None
        state["need_human"] = False
    except Exception as e:
        state["messages"].append({
            "role": "assistant",
            "content": f"[{agent}] error: {e}",
            "agent": agent
        })
        state["next_agent"] = None
        state["need_human"] = True

    return state

# ==========
# Build graph
# ==========
graph = StateGraph(MeetingState)
graph.add_node("supervisor", pick_next_agent)
graph.add_node("agent", agent_node)

graph.set_entry_point("supervisor")
graph.add_edge("supervisor", "agent")
graph.add_edge("agent", END)

#db_path = os.getenv("CHECKPOINT_DB", "checkpoints.sqlite")
#checkpointer = SqliteSaver.from_conn_string(db_path)
#app_graph = graph.compile(checkpointer=checkpointer)

import sqlite3
db_path = os.getenv("CHECKPOINT_DB", "checkpoints.sqlite")
conn = sqlite3.connect(db_path, check_same_thread=False)
checkpointer = SqliteSaver(conn)
app_graph = graph.compile(checkpointer=checkpointer)

# ==========
# FastAPI
# ==========
app = FastAPI(title="Multi-AI Meeting")

class StartPayload(BaseModel):
    session_id: str = Field(..., description="会議ID")
    participants: List[str] = Field(default_factory=lambda: ["gemini", "claude", "qwen"])
    phase: Literal["CONTEXT", "CRITIQUE", "SYNTHESIS", "FREE"] = "FREE"

@app.post("/session")
async def start_session(p: StartPayload):
    state: MeetingState = {
        "messages": [{"role": "system", "content": DEFAULT_SYSTEM, "agent": "system"}],
        "phase": p.phase,
        "target_agents": p.participants,
        "next_agent": None,
        "read_only": False,
        "need_human": False,
        "meta": {}
    }
    app_graph.update_state(
        config={"configurable": {"thread_id": p.session_id}},
        values=state
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
    app_graph.update_state(
        config=cfg,
        values={
            "messages": [{"role": "user", "content": cp.text, "agent": "human"}],
            "phase": cp.phase or "FREE",
            "read_only": cp.read_only
        }
    )
    out = await app_graph.ainvoke({}, config=cfg)
    current = app_graph.get_state(config=cfg).values
    return {
        "messages": current["messages"][-5:],
        "phase": current["phase"],
        "need_human": current["need_human"],
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
        app_graph.update_state(
            config={"configurable": {"thread_id": cp.session_id}},
            values={"target_agents": cp.participants}
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

@app.get("/health")
async def health():
    """各エージェントの現在の可用性を返す"""
    qwen_up = await is_qwen_up()
    return {
        "gemini": REGISTRY.gemini.enabled and bool(os.getenv("GOOGLE_API_KEY")),
        "claude": REGISTRY.claude.enabled and bool(os.getenv("ANTHROPIC_API_KEY")),
        "qwen":   qwen_up,
        "gpt":    GPT_AVAILABLE,
    }

# Run: uvicorn app:app --reload --host 127.0.0.1 --port 8008
