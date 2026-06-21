"""Multi-AI Meeting Frontend (Streamlit)."""

import html
import json
import os
import re
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
import streamlit as st

BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8008")
AGENTS = ["gemini", "claude", "gpt"]
PHASES = ["FREE", "CONTEXT", "CRITIQUE", "SYNTHESIS"]
VIEWS = [a.upper() for a in AGENTS] + ["MINUTES"]
DEFAULT_TIMEZONE = "Asia/Tokyo"
ALLOWED_EMAILS = {e.strip() for e in os.getenv("ALLOWED_EMAILS", "").split(",") if e.strip()}
AGENT_COLOR = {"gemini": "#4285F4", "claude": "#D4A853", "gpt": "#10A37F", "human": "#888888", "system": "#444444"}

st.set_page_config(page_title="AI Meeting", page_icon="⬡", layout="centered", initial_sidebar_state="collapsed")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@600;700&display=swap');
html, body, [class*="css"] { font-family:'DM Mono', monospace; background:#0d0d0d; color:#e8e8e8; }
h1,h2,h3 { font-family:'Syne', sans-serif; letter-spacing:-0.02em; }
div.stButton > button { background:#1a1a1a; color:#888; border:1px solid #2a2a2a; border-radius:2px; font-family:'DM Mono', monospace; font-size:11px; padding:4px 10px; width:100%; transition:all .15s; }
div.stButton > button:hover { border-color:#555; color:#e8e8e8; }
div[data-phase-active="true"] > div.stButton > button { background:#e8e8e8; color:#0d0d0d; border-color:#e8e8e8; }
textarea { background:#111 !important; border:1px solid #2a2a2a !important; border-radius:2px !important; color:#e8e8e8 !important; font-family:'DM Mono', monospace !important; font-size:13px !important; }
label[data-baseweb="checkbox"] span, label[data-baseweb="radio"] span { font-family:'DM Mono', monospace; font-size:12px; color:#888; }
.msg-block { border-left:2px solid #2a2a2a; padding:8px 12px; margin:6px 0; font-size:13px; line-height:1.7; background:#111; border-radius:0 2px 2px 0; max-height:70vh; overflow-y:auto; }
.msg-label { font-size:10px; letter-spacing:.1em; margin-bottom:4px; opacity:.85; }
.msg-content, .msg-content * { color:#e8e8e8 !important; }
.msg-content { white-space:pre-wrap; overflow-wrap:anywhere; }
.latest-turn { border:1px solid #222; background:#101010; padding:8px 12px; margin-bottom:10px; max-height:40vh; overflow-y:auto; white-space:pre-wrap; overflow-wrap:anywhere; color:#e8e8e8; font-size:12px; line-height:1.7; }
.empty-message { color:#333; font-size:12px; padding:16px 0; }
hr { border-color:#1e1e1e; }
.status-bar { font-size:10px; color:#444; letter-spacing:.08em; padding:4px 0; border-top:1px solid #1a1a1a; }
.archive-row { font-size:11px; color:#777; line-height:1.7; }
.need-human { background:#1a0a00; border:1px solid #4a2000; color:#ff8c42; font-size:12px; padding:8px 12px; border-radius:2px; margin:6px 0; }
</style>
""", unsafe_allow_html=True)

if not st.user.is_logged_in:
    st.markdown("<h2 style='margin:0;padding:12px 0 4px'>⬡ meeTai</h2>", unsafe_allow_html=True)
    if st.button("Google でログイン", use_container_width=False):
        st.login("google")
    st.stop()
if not ALLOWED_EMAILS:
    st.error("サーバ設定エラー: ALLOWED_EMAILS が未設定です（管理者に連絡してください）")
    st.stop()
if st.user.email not in ALLOWED_EMAILS:
    st.error("アクセスが許可されていません")
    if st.button("ログアウト"):
        st.logout()
    st.stop()


def init_state():
    defaults = {
        "session_id": None,
        "phase": "FREE",
        "logs": {a: [] for a in AGENTS},
        "minutes": [],
        "need_human": False,
        "sending": False,
        "clear_count": 0,
        "archive_range": "7d",
        "archive_month": "",
        "session_participants": [],
        "response_view": "MINUTES",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def app_timezone():
    name = os.getenv("APP_TIMEZONE", DEFAULT_TIMEZONE).strip() or DEFAULT_TIMEZONE
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        return ZoneInfo(DEFAULT_TIMEZONE)


def current_month_placeholder():
    return datetime.now(app_timezone()).strftime("%Y-%m")


def post(path, payload, timeout):
    r = httpx.post(f"{BACKEND}{path}", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def get(path, params=None, timeout=10.0):
    r = httpx.get(f"{BACKEND}{path}", params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r


def start_session(participants, title):
    sid = f"mtg_{uuid.uuid4().hex[:8]}"
    post("/session", {"session_id": sid, "participants": participants, "phase": st.session_state.phase, "title": title, "user_email": st.user.email}, 10.0)
    return sid


def send_chat(text, phase, read_only):
    return post("/chat", {"session_id": st.session_state.session_id, "text": text, "phase": phase, "read_only": read_only}, 120.0)


def ask_all(text):
    return post("/ask-all", {"session_id": st.session_state.session_id, "text": text, "phase": st.session_state.phase, "participants": st.session_state.session_participants or None}, 180.0)


def commit_fanout(human_text, responses, phase):
    return post("/commit", {"session_id": st.session_state.session_id, "human_text": human_text, "responses": responses, "phase": phase}, 30.0)


def get_health():
    try:
        return get("/health", timeout=3.0).json()
    except Exception:
        return {}


def safe_filename(title, session_id, ext):
    base = re.sub(r"[^0-9A-Za-zぁ-んァ-ン一-龥ー_-]+", "_", title).strip("_") or session_id
    return f"{base}_{session_id}.{ext}"


def session_label(item):
    title = item.get("title") or "untitled"
    updated = (item.get("updated_at") or "").replace("T", " ")[:16]
    try:
        agents = ",".join(json.loads(item.get("agents") or "[]"))
    except Exception:
        agents = item.get("agents") or "[]"
    return f"{updated} · {title} · {item.get('turns_used') or 0} turns · {item.get('llm_calls_used') or 0} calls · {agents}"


def card(message):
    agent = message.get("agent", "?")
    color = AGENT_COLOR.get(agent, "#555")
    st.markdown(f"""
    <div class='msg-block' style='border-left-color:{color}'>
        <div class='msg-label' style='color:{color}'>{html.escape(agent.upper())}</div>
        <div class='msg-content'>{html.escape(message.get('content', ''))}</div>
    </div>
    """, unsafe_allow_html=True)


def minutes_text(messages):
    return "".join(f"[{m.get('agent', '?').upper()}]\n{m.get('content', '')}\n\n" for m in messages)


def latest_turn(messages):
    last_user_idx = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
    return [] if last_user_idx < 0 else messages[last_user_idx:]


def render_archive():
    with st.expander("ARCHIVE / past sessions", expanded=False):
        labels = {"7d": "直近7日", "30d": "直近30日", "90d": "直近90日", "month": "年月指定"}
        range_choice = st.selectbox("期間", ["7d", "30d", "90d", "month"], format_func=lambda x: labels[x], key="archive_range_choice")
        month = ""
        if range_choice == "month":
            month = st.text_input("年月", value=st.session_state.archive_month, placeholder=current_month_placeholder(), key="archive_month_input").strip()
            st.session_state.archive_month = month
            if not month:
                st.markdown("<div style='color:#555;font-size:12px;padding:8px 0'>年月を YYYY-MM で入力</div>", unsafe_allow_html=True)
                return
        try:
            params = {"limit": 100, "user_email": st.user.email, "month" if month else "range": month or range_choice}
            sessions = get("/sessions", params=params).json().get("sessions", [])
        except Exception as e:
            st.error(f"archive取得失敗: {e}")
            return
        if not sessions:
            st.markdown("<div style='color:#333;font-size:12px;padding:8px 0'>no archived sessions</div>", unsafe_allow_html=True)
            return
        selected = sessions[st.selectbox("SESSION", options=list(range(len(sessions))), format_func=lambda i: session_label(sessions[i]), key="archive_session_select")]
        sid = selected["session_id"]
        title = selected.get("title") or "untitled"
        st.markdown(f"<div class='archive-row'>selected: {html.escape(title)} · {html.escape(sid)}</div>", unsafe_allow_html=True)
        try:
            md = get(f"/sessions/{sid}/export", params={"format": "markdown", "user_email": st.user.email}, timeout=15.0).text
            js = json.dumps(get(f"/sessions/{sid}/export", params={"format": "json", "user_email": st.user.email}, timeout=15.0).json(), ensure_ascii=False, indent=2)
        except Exception as e:
            st.error(f"export取得失敗: {e}")
            return
        st.download_button("↓ DOWNLOAD MARKDOWN", data=md, file_name=safe_filename(title, sid, "md"), mime="text/markdown", use_container_width=True, key=f"archive_md_{sid}")
        st.download_button("↓ DOWNLOAD JSON", data=js, file_name=safe_filename(title, sid, "json"), mime="application/json", use_container_width=True, key=f"archive_json_{sid}")
        with st.expander("preview", expanded=False):
            st.text_area("", value=md, height=260, label_visibility="collapsed", key=f"archive_preview_{sid}")


init_state()
st.markdown("<h2 style='margin:0;padding:12px 0 4px'>⬡ meeTai</h2>", unsafe_allow_html=True)
health = get_health()
render_archive()

if not st.session_state.session_id:
    st.markdown("<div style='color:#555;font-size:12px;margin-bottom:12px'>参加エージェントを選択してセッションを開始</div>", unsafe_allow_html=True)
    title = st.text_input("SESSION TITLE", placeholder="例: API利用上限の設計", label_visibility="visible")
    selected = []
    for i, agent in enumerate(AGENTS):
        with st.columns(3)[i]:
            if st.checkbox(f"{'●' if health.get(agent) else '○'} {agent}", value=health.get(agent, False), key=f"init_{agent}"):
                selected.append(agent)
    if st.button("▶  START SESSION", use_container_width=True):
        if not selected:
            st.warning("1つ以上選択")
        else:
            try:
                st.session_state.session_id = start_session(selected, title)
                st.session_state.session_participants = selected
                st.session_state.logs = {a: [] for a in AGENTS}
                st.session_state.minutes = []
                st.session_state.need_human = False
                st.session_state.response_view = "MINUTES"
                st.rerun()
            except Exception as e:
                st.error(f"接続失敗: {e}")
    st.stop()

st.markdown(f"<div class='status-bar'>SESSION {st.session_state.session_id[-8:]} · PHASE {st.session_state.phase}</div>", unsafe_allow_html=True)
st.markdown("<div style='margin:8px 0 4px;font-size:10px;color:#444;letter-spacing:.1em'>PHASE</div>", unsafe_allow_html=True)
for i, phase in enumerate(PHASES):
    with st.columns(4)[i]:
        if st.button(f"{'▸ ' if phase == st.session_state.phase else ''}{phase}", key=f"phase_{phase}", use_container_width=True):
            st.session_state.phase = phase
            st.rerun()

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='font-size:10px;color:#444;letter-spacing:.1em;margin-bottom:4px'>SEND TO</div>", unsafe_allow_html=True)
agent_options = [f"{'●' if health.get(a) else '○'} {a}" for a in AGENTS]
target = st.radio("", agent_options, horizontal=True, label_visibility="collapsed").split(" ")[-1]
read_only = st.toggle("read-only（文脈共有のみ、応答なし）", value=False)
user_input = st.text_area("", placeholder="入力...", height=100, label_visibility="collapsed", key=f"input_{st.session_state.clear_count}")

send_col, fanout_col, clear_col = st.columns([2, 2, 1])
with send_col:
    send_btn = st.button("SEND ▶", use_container_width=True, type="primary", disabled=st.session_state.sending)
with fanout_col:
    fanout_btn = st.button("全員に聞く ▶▶", use_container_width=True, disabled=st.session_state.sending)
with clear_col:
    if st.button("✕", use_container_width=True):
        st.session_state.clear_count += 1
        st.rerun()

if send_btn and user_input.strip() and not st.session_state.sending:
    st.session_state.sending = True
    with st.spinner("dispatching..."):
        st.session_state.minutes.append({"role": "user", "content": user_input, "agent": "human"})
        if read_only:
            try:
                send_chat(user_input, st.session_state.phase, True)
                st.session_state.response_view = "MINUTES"
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            try:
                post("/config", {"session_id": st.session_state.session_id, "participants": [target]}, 5.0)
                result = send_chat(user_input, st.session_state.phase, False)
                for m in result.get("messages", []):
                    if m.get("role") == "assistant" and m.get("agent") == target:
                        st.session_state.logs[target].append(m)
                        st.session_state.minutes.append(m)
                st.session_state.response_view = target.upper()
                if result.get("need_human"):
                    st.session_state.need_human = True
            except Exception as e:
                st.session_state.logs[target].append({"role": "assistant", "content": f"[error] {e}", "agent": target})
                st.session_state.response_view = target.upper()
    st.session_state.sending = False
    st.session_state.clear_count += 1
    st.rerun()

if fanout_btn and user_input.strip() and not st.session_state.sending:
    st.session_state.sending = True
    with st.spinner("fan-out dispatching..."):
        st.session_state.minutes.append({"role": "user", "content": user_input, "agent": "human"})
        try:
            result = ask_all(user_input)
            responses = result.get("responses", [])
            commit_fanout(user_input, responses, result.get("phase", st.session_state.phase))
            first_agent = None
            for r in responses:
                agent, content = r.get("agent"), r.get("content", "")
                if agent in st.session_state.logs:
                    msg = {"role": "assistant", "content": content, "agent": agent}
                    st.session_state.logs[agent].append(msg)
                    st.session_state.minutes.append(msg)
                    first_agent = first_agent or agent
            st.session_state.response_view = first_agent.upper() if first_agent else "MINUTES"
        except Exception as e:
            st.error(f"fan-out error: {e}")
    st.session_state.sending = False
    st.session_state.clear_count += 1
    st.rerun()

if st.session_state.need_human:
    st.markdown("<div class='need-human'>⚠ No agent available — check API keys or agent selection</div>", unsafe_allow_html=True)
    if st.button("dismiss", key="dismiss_nh"):
        st.session_state.need_human = False

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='font-size:10px;color:#444;letter-spacing:.1em;margin-bottom:8px'>RESPONSES</div>", unsafe_allow_html=True)
if st.session_state.response_view not in VIEWS:
    st.session_state.response_view = "MINUTES"
st.radio("", VIEWS, horizontal=True, key="response_view", label_visibility="collapsed")

view = st.session_state.response_view
if view in [a.upper() for a in AGENTS]:
    msgs = st.session_state.logs[view.lower()]
    if not msgs:
        st.markdown("<div class='empty-message'>no messages</div>", unsafe_allow_html=True)
    for m in msgs[-20:]:
        card(m)
else:
    st.markdown("<div style='font-size:10px;color:#444;letter-spacing:.1em;margin-bottom:8px'>LATEST TURN</div>", unsafe_allow_html=True)
    latest = minutes_text(latest_turn(st.session_state.minutes))
    if latest:
        st.markdown(f"<div class='latest-turn'>{html.escape(latest)}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='empty-message'>no latest turn yet</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:10px;color:#444;letter-spacing:.1em;margin-bottom:8px'>MINUTES（全ログ）</div>", unsafe_allow_html=True)
    full = minutes_text(st.session_state.minutes)
    if full:
        st.text_area("", value=full, height=300, label_visibility="collapsed")
        st.download_button("↓ EXPORT MINUTES", data=full, file_name=f"minutes_{st.session_state.session_id}.txt", mime="text/plain", use_container_width=True)
    else:
        st.markdown("<div class='empty-message'>no minutes yet</div>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
if st.button("■  END SESSION", use_container_width=True):
    for key in ["session_id", "phase", "logs", "minutes", "need_human", "response_view"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# Run: streamlit run frontend.py --server.port 8501
