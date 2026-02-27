"""
Multi-AI Meeting Frontend (Streamlit)
- フェーズボタン（FREE/CONTEXT/CRITIQUE/SYNTHESIS）
- 送信先チェックボックス（複数選択）
- 各AI応答タブ表示
- スマホ縦持ち対応
- FastAPI (app.py :8008) をバックエンドとして使用
"""

import uuid
import httpx
import streamlit as st

# ---------- 設定 ----------
BACKEND = "http://127.0.0.1:8008"
AGENTS  = ["gemini", "claude", "qwen"]
PHASES  = ["FREE", "CONTEXT", "CRITIQUE", "SYNTHESIS"]

AGENT_COLOR = {
    "gemini": "#4285F4",
    "claude": "#D4A853",
    "qwen":   "#7B61FF",
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

init_state()

# ---------- バックエンド接続 ----------
def start_session(participants):
    sid = f"mtg_{uuid.uuid4().hex[:8]}"
    r = httpx.post(f"{BACKEND}/session", json={
        "session_id": sid,
        "participants": participants,
        "phase": st.session_state.phase,
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

# ---------- UI ----------
# ヘッダー
st.markdown("<h2 style='margin:0;padding:12px 0 4px'>⬡ meeTai</h2>", unsafe_allow_html=True)

# ヘルスチェック＆セッション開始
health = get_health()

if not st.session_state.session_id:
    st.markdown("<div style='color:#555;font-size:12px;margin-bottom:12px'>参加エージェントを選択してセッションを開始</div>", unsafe_allow_html=True)

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
                sid = start_session(selected)
                st.session_state.session_id = sid
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

# 入力欄
user_input = st.text_area(
    label="",
    placeholder="入力...",
    height=100,
    label_visibility="collapsed",
)

# 送信
send_col, _ = st.columns([1, 3])
with send_col:
    send_btn = st.button("SEND ▶", use_container_width=True, type="primary")

if send_btn and user_input.strip():
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
    st.rerun()

# need_human警告
if st.session_state.need_human:
    st.markdown("<div class='need-human'>⚠ Qwen unavailable — human moderation required</div>", unsafe_allow_html=True)
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
