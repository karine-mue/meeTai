"""
Multi-AI Meeting Frontend (Streamlit)
- フェーズボタン（FREE/CONTEXT/CRITIQUE/SYNTHESIS）
- 送信先チェックボックス（複数選択）
- 各AI応答タブ表示
- スマホ縦持ち対応
- FastAPI (app.py :8008) をバックエンドとして使用
"""

import json
import os
import re
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
import streamlit as st

# ---------- 設定 ----------
BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8008")
AGENTS  = ["gemini", "claude", "gpt"]
ALLOWED_EMAILS = {e.strip() for e in os.getenv("ALLOWED_EMAILS", "").split(",") if e.strip()}
PHASES  = ["FREE", "CONTEXT", "CRITIQUE", "SYNTHESIS"]
DEFAULT_TIMEZONE = "Asia/Tokyo"

AGENT_COLOR = {
    "gemini": "#4285F4",
    "claude": "#D4A853",
    "gpt":    "#10A37F",
    "human":  "#888888",
    "system": "#444444",
}

# ---------- ページ設定 ----------
st.set_page_config(
    page_title="AI Meeting",
    page_icon="⬡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------- スタイル ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #0d0d0d;
    color: #e8e8e8;
}

h1, h2, h3 { font-family: 'Syne', sans-serif; letter-spacing: -0.02em; }

/* フェーズボタン */
div.stButton > button {
    background: #1a1a1a;
    color: #888;
    border: 1px solid #2a2a2a;
    border-radius: 2px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    padding: 4px 10px;
    width: 100%;
    transition: all 0.15s;
}
div.stButton > button:hover {
    border-color: #555;
    color: #e8e8e8;
}

/* アクティブフェーズ */
div[data-phase-active="true"] > div.stButton > button {
    background: #e8e8e8;
    color: #0d0d0d;
    border-color: #e8e8e8;
}

/* テキストエリア */
textarea {
    background: #111 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 2px !important;
    color: #e8e8e8 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
}

/* タブ */
button[data-baseweb="tab"] {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: #555;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #e8e8e8;
    border-bottom-color: #e8e8e8 !important;
}

/* チェックボックス */
label[data-baseweb="checkbox"] span {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: #888;
}

/* メッセージブロック */
.msg-block {
    border-left: 2px solid #2a2a2a;
    padding: 8px 12px;
    margin: 6px 0;
    font-size: 13px;
    line-height: 1.7;
    background: #111;
    border-radius: 0 2px 2px 0;
}
.msg-label {
    font-size: 10px;
    letter-spacing: 0.1em;
    margin-bottom: 4px;
    opacity: 0.6;
}

/* セパレータ */
hr { border-color: #1e1e1e; }

/* ステータスバー */
.status-bar {
    font-size: 10px;
    color: #444;
    letter-spacing: 0.08em;
    padding: 4px 0;
    border-top: 1px solid #1a1a1a;
}

/* archive */
.archive-row {
    font-size: 11px;
    color: #777;
    line-height: 1.7;
}

/* read-only badge */
.badge-ro {
    background: #1a1a1a;
    border: 1px solid #333;
    color: #666;
    font-size: 10px;
    padding: 1px 6px;
    border-radius: 2px;
}

/* need_human警告 */
.need-human {
    background: #1a0a00;
    border: 1px solid #4a2000;
    color: #ff8c42;
    font-size: 12px;
    padding: 8px 12px;
    border-radius: 2px;
    margin: 6px 0;
}
</style>
""", unsafe_allow_html=True)

# ---------- 認証 ----------
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

# ---------- セッション初期化 ----------
def init_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "phase" not in st.session_state:
        st.session_state.phase = "FREE"
    if "logs" not in st.session_state:
        # {agent: [{"role", "content", "agent"}]}
        st.session_state.logs = {a: [] for a in AGENTS}
    if "minutes" not in st.session_state:
        st.session_state.minutes = []
    if "need_human" not in st.session_state:
        st.session_state.need_human = False
    if "sending" not in st.session_state:
        st.session_state.sending = False
    if "clear_count" not in st.session_state:
        st.session_state.clear_count = 0
    if "archive_range" not in st.session_state:
        st.session_state.archive_range = "7d"
    if "archive_month" not in st.session_state:
        st.session_state.archive_month = ""

init_state()

# ---------- バックエンド接続 ----------
def app_timezone():
    name = os.getenv("APP_TIMEZONE", DEFAULT_TIMEZONE).strip() or DEFAULT_TIMEZONE
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        return ZoneInfo(DEFAULT_TIMEZONE)


def current_month_placeholder():
    return datetime.now(app_timezone()).strftime("%Y-%m")


def start_session(participants, title):
    sid = f"mtg_{uuid.uuid4().hex[:8]}"
    r = httpx.post(f"{BACKEND}/session", json={
        "session_id": sid,
        "participants": participants,
        "phase": st.session_state.phase,
        "title": title,
        "user_email": st.user.email,
    }, timeout=10.0)
    r.raise_for_status()
    return sid

def send_chat(text, phase, read_only):
    r = httpx.post(f"{BACKEND}/chat", json={
        "session_id": st.session_state.session_id,
        "text": text,
        "phase": phase,
        "read_only": read_only,
    }, timeout=120.0)
    r.raise_for_status()
    return r.json()

def get_health():
    try:
        r = httpx.get(f"{BACKEND}/health", timeout=3.0)
        return r.json()
    except Exception:
        return {}

def list_archived_sessions(range_key, month):
    params = {"limit": 100, "user_email": st.user.email}
    if month:
        params["month"] = month
    else:
        params["range"] = range_key
    r = httpx.get(f"{BACKEND}/sessions", params=params, timeout=10.0)
    r.raise_for_status()
    return r.json().get("sessions", [])

def export_archived_session(session_id, fmt):
    r = httpx.get(
        f"{BACKEND}/sessions/{session_id}/export",
        params={"format": fmt, "user_email": st.user.email},
        timeout=15.0,
    )
    r.raise_for_status()
    if fmt == "json":
        return json.dumps(r.json(), ensure_ascii=False, indent=2)
    return r.text

def safe_filename(title, session_id, ext):
    base = re.sub(r"[^0-9A-Za-zぁ-んァ-ン一-龥ー_-]+", "_", title).strip("_")
    if not base:
        base = session_id
    return f"{base}_{session_id}.{ext}"

def format_session_label(item):
    title = item.get("title") or "untitled"
    updated = (item.get("updated_at") or "").replace("T", " ")[:16]
    turns = item.get("turns_used") or 0
    calls = item.get("llm_calls_used") or 0
    agents = item.get("agents") or "[]"
    try:
        agents_text = ",".join(json.loads(agents))
    except Exception:
        agents_text = agents
    return f"{updated} · {title} · {turns} turns · {calls} calls · {agents_text}"

def render_archive():
    with st.expander("ARCHIVE / past sessions", expanded=False):
        range_labels = {
            "7d": "直近7日",
            "30d": "直近30日",
            "90d": "直近90日",
            "month": "年月指定",
        }
        range_choice = st.selectbox(
            "期間",
            ["7d", "30d", "90d", "month"],
            format_func=lambda x: range_labels[x],
            key="archive_range_choice",
        )

        month = ""
        if range_choice == "month":
            month = st.text_input(
                "年月",
                value=st.session_state.archive_month,
                placeholder=current_month_placeholder(),
                key="archive_month_input",
            ).strip()
            st.session_state.archive_month = month
            if not month:
                st.markdown("<div style='color:#555;font-size:12px;padding:8px 0'>年月を YYYY-MM で入力</div>", unsafe_allow_html=True)
                return

        try:
            sessions = list_archived_sessions(
                range_key=range_choice if range_choice != "month" else "7d",
                month=month if range_choice == "month" else "",
            )
        except Exception as e:
            st.error(f"archive取得失敗: {e}")
            return

        if not sessions:
            st.markdown("<div style='color:#333;font-size:12px;padding:8px 0'>no archived sessions</div>", unsafe_allow_html=True)
            return

        selected_idx = st.selectbox(
            "SESSION",
            options=list(range(len(sessions))),
            format_func=lambda i: format_session_label(sessions[i]),
            key="archive_session_select",
        )
        selected = sessions[selected_idx]
        sid = selected["session_id"]
        title = selected.get("title") or "untitled"

        st.markdown(
            f"<div class='archive-row'>selected: {title} · {sid}</div>",
            unsafe_allow_html=True,
        )

        try:
            markdown_text = export_archived_session(sid, "markdown")
            json_text = export_archived_session(sid, "json")
        except Exception as e:
            st.error(f"export取得失敗: {e}")
            return

        st.download_button(
            "↓ DOWNLOAD MARKDOWN",
            data=markdown_text,
            file_name=safe_filename(title, sid, "md"),
            mime="text/markdown",
            use_container_width=True,
            key=f"archive_md_{sid}",
        )
        st.download_button(
            "↓ DOWNLOAD JSON",
            data=json_text,
            file_name=safe_filename(title, sid, "json"),
            mime="application/json",
            use_container_width=True,
            key=f"archive_json_{sid}",
        )

        with st.expander("preview", expanded=False):
            st.text_area(
                "",
                value=markdown_text,
                height=260,
                label_visibility="collapsed",
                key=f"archive_preview_{sid}",
            )

# ---------- UI ----------
# ヘッダー
st.markdown("<h2 style='margin:0;padding:12px 0 4px'>⬡ meeTai</h2>", unsafe_allow_html=True)

# ヘルスチェック＆セッション開始
health = get_health()

render_archive()

if not st.session_state.session_id:
    st.markdown("<div style='color:#555;font-size:12px;margin-bottom:12px'>参加エージェントを選択してセッションを開始</div>", unsafe_allow_html=True)

    session_title = st.text_input(
        "SESSION TITLE",
        placeholder="例: API利用上限の設計",
        label_visibility="visible",
    )

    # 参加者選択
    selected = []
    cols = st.columns(3)
    for i, agent in enumerate(AGENTS):
        status = "●" if health.get(agent) else "○"
        with cols[i]:
            if st.checkbox(f"{status} {agent}", value=health.get(agent, False), key=f"init_{agent}"):
                selected.append(agent)

    if st.button("▶  START SESSION", use_container_width=True):
        if not selected:
            st.warning("1つ以上選択")
        else:
            try:
                sid = start_session(selected, session_title)
                st.session_state.session_id = sid
                st.session_state.logs = {a: [] for a in AGENTS}
                st.session_state.minutes = []
                st.session_state.need_human = False
                st.rerun()
            except Exception as e:
                st.error(f"接続失敗: {e}")
    st.stop()

# ---------- セッション中 ----------
sid_short = st.session_state.session_id[-8:]
st.markdown(f"<div class='status-bar'>SESSION {sid_short} · PHASE {st.session_state.phase}</div>", unsafe_allow_html=True)

# フェーズボタン
st.markdown("<div style='margin:8px 0 4px;font-size:10px;color:#444;letter-spacing:0.1em'>PHASE</div>", unsafe_allow_html=True)
pcols = st.columns(4)
for i, ph in enumerate(PHASES):
    with pcols[i]:
        label = f"{'▸ ' if ph == st.session_state.phase else ''}{ph}"
        if st.button(label, key=f"phase_{ph}", use_container_width=True):
            st.session_state.phase = ph
            st.rerun()

st.markdown("<hr>", unsafe_allow_html=True)

# 送信先選択（Radio：1ターン1エージェント、コンテキスト汚染を防ぐ）
st.markdown("<div style='font-size:10px;color:#444;letter-spacing:0.1em;margin-bottom:4px'>SEND TO</div>", unsafe_allow_html=True)
agent_options = [f"{'●' if health.get(a) else '○'} {a}" for a in AGENTS]
selected_label = st.radio(
    label="",
    options=agent_options,
    horizontal=True,
    label_visibility="collapsed",
)
target = selected_label.split(" ")[-1]  # "● gemini" → "gemini"

# read-only トグル
read_only = st.toggle("read-only（文脈共有のみ、応答なし）", value=False)

# 入力欄（clear_countをkeyに含めることでCLEARボタン押下時に強制リセット）
user_input = st.text_area(
    label="",
    placeholder="入力...",
    height=100,
    label_visibility="collapsed",
    key=f"input_{st.session_state.clear_count}",
)

# 送信 / クリア
send_col, clear_col, _ = st.columns([2, 1, 2])
with send_col:
    send_btn = st.button(
        "SEND ▶",
        use_container_width=True,
        type="primary",
        disabled=st.session_state.sending  # 送信中は非活性化
    )
with clear_col:
    if st.button("✕", use_container_width=True):
        st.session_state.clear_count += 1
        st.rerun()

if send_btn and user_input.strip() and not st.session_state.sending:
    st.session_state.sending = True
    with st.spinner("dispatching..."):
        # humanメッセージを先に議事録へ（時系列の正順を保持）
        st.session_state.minutes.append({
            "role": "user", "content": user_input, "agent": "human"
        })
        if read_only:
            try:
                send_chat(user_input, st.session_state.phase, True)
                # read-onlyはDBに積むだけ、応答なし
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            # 1ターン1エージェント（シングルターン仕様に適合）
            try:
                httpx.post(f"{BACKEND}/config", json={
                    "session_id": st.session_state.session_id,
                    "participants": [target],
                }, timeout=5.0)
                result = send_chat(user_input, st.session_state.phase, False)
                msgs = result.get("messages", [])
                for m in msgs:
                    if m.get("role") == "assistant" and m.get("agent") == target:
                        st.session_state.logs[target].append(m)
                        st.session_state.minutes.append(m)
                if result.get("need_human"):
                    st.session_state.need_human = True
            except Exception as e:
                st.session_state.logs[target].append({
                    "role": "assistant",
                    "content": f"[error] {e}",
                    "agent": target
                })
    st.session_state.sending = False
    st.session_state.clear_count += 1  # 送信成功後に入力欄をクリア
    st.rerun()

# need_human警告
if st.session_state.need_human:
    st.markdown("<div class='need-human'>⚠ No agent available — check API keys or agent selection</div>", unsafe_allow_html=True)
    if st.button("dismiss", key="dismiss_nh"):
        st.session_state.need_human = False

st.markdown("<hr>", unsafe_allow_html=True)

# ---------- 応答表示（タブ） ----------
st.markdown("<div style='font-size:10px;color:#444;letter-spacing:0.1em;margin-bottom:8px'>RESPONSES</div>", unsafe_allow_html=True)

tabs = st.tabs([a.upper() for a in AGENTS] + ["MINUTES"])

for i, agent in enumerate(AGENTS):
    with tabs[i]:
        msgs = st.session_state.logs[agent]
        if not msgs:
            st.markdown("<div style='color:#333;font-size:12px;padding:16px 0'>no messages</div>", unsafe_allow_html=True)
        else:
            for m in msgs[-20:]:
                color = AGENT_COLOR.get(m.get("agent", ""), "#555")
                st.markdown(f"""
                <div class='msg-block' style='border-left-color:{color}'>
                    <div class='msg-label' style='color:{color}'>{m.get("agent","?").upper()}</div>
                    <div>{m.get("content","")}</div>
                </div>
                """, unsafe_allow_html=True)

# MINUTESタブ
with tabs[-1]:
    st.markdown("<div style='font-size:10px;color:#444;letter-spacing:0.1em;margin-bottom:8px'>MINUTES（全ログ）</div>", unsafe_allow_html=True)
    minutes_text = ""
    for m in st.session_state.minutes:
        agent = m.get("agent", "?")
        content = m.get("content", "")
        minutes_text += f"[{agent.upper()}]\n{content}\n\n"

    if minutes_text:
        st.text_area("", value=minutes_text, height=300, label_visibility="collapsed")
        st.download_button(
            "↓ EXPORT MINUTES",
            data=minutes_text,
            file_name=f"minutes_{st.session_state.session_id}.txt",
            mime="text/plain",
            use_container_width=True,
        )
    else:
        st.markdown("<div style='color:#333;font-size:12px;padding:16px 0'>no minutes yet</div>", unsafe_allow_html=True)

# ---------- セッションリセット ----------
st.markdown("<hr>", unsafe_allow_html=True)
if st.button("■  END SESSION", use_container_width=True):
    for key in ["session_id", "phase", "logs", "minutes", "need_human"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# Run: streamlit run frontend.py --server.port 8501
