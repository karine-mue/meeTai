import sqlite3
import json
import os
from datetime import datetime

try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False
    print("[Warning] msgpack not installed. Run: pip install msgpack")

DB_PATH = "checkpoints.sqlite"
EXPORT_DIR = "exports"

def decode_blob(blob, type_hint):
    """type列に応じてBLOBをデコードする"""
    if blob is None:
        return {}
    if type_hint == "msgpack" and HAS_MSGPACK:
        return msgpack.unpackb(blob, raw=False, strict_map_key=False)
    try:
        return json.loads(blob.decode("utf-8") if isinstance(blob, bytes) else blob)
    except Exception:
        return {}

def get_latest_checkpoints():
    if not os.path.exists(DB_PATH):
        print(f"[Error] Database not found: {DB_PATH}")
        return {}

    conn = sqlite3.connect(DB_PATH)
    # WALをフラッシュして最新データを読み込む
    conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
    cur = conn.cursor()

    try:
        cur.execute("""
            SELECT thread_id, type, checkpoint
            FROM checkpoints
            ORDER BY checkpoint_id DESC
        """)
    except sqlite3.OperationalError as e:
        print(f"[Error] {e}")
        conn.close()
        return {}

    rows = cur.fetchall()
    conn.close()

    sessions = {}
    for thread_id, type_hint, checkpoint_blob in rows:
        if thread_id in sessions:
            continue
        try:
            chk_data = decode_blob(checkpoint_blob, type_hint)
            # LangGraph v0.3のmsgpack構造: channel_values.messages
            channel_values = chk_data.get("channel_values", {})
            messages = channel_values.get("messages", [])
            if messages:
                sessions[thread_id] = messages
        except Exception as e:
            pass

    return sessions

def format_title(messages):
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            title = content[:40].replace("\n", " ")
            return title + ("..." if len(content) > 40 else "")
    return "(no user input)"

def export_session(thread_id, messages):
    os.makedirs(EXPORT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{EXPORT_DIR}/session_{thread_id}_{timestamp}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Session ID : {thread_id}\n")
        f.write(f"Export Time: {timestamp}\n")
        f.write("=" * 50 + "\n\n")

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "unknown")
            if role == "system":
                continue
            agent = msg.get("agent", role)
            content = msg.get("content", "")
            f.write(f"[{agent.upper()}]\n{content}\n")
            f.write("-" * 40 + "\n\n")

    print(f"[Success] Exported: {filename}")

def main():
    if not HAS_MSGPACK:
        print("[Error] msgpack required. Run: pip install msgpack")
        return

    print("[System] Scanning SQLite checkpointer...")
    sessions = get_latest_checkpoints()

    if not sessions:
        print("[System] No valid session history found.")
        # デバッグ用：DBに何件あるか確認
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*), COUNT(DISTINCT thread_id) FROM checkpoints")
        total, threads = cur.fetchone()
        conn.close()
        print(f"         (DB has {total} checkpoints across {threads} threads)")
        return

    session_list = list(sessions.items())
    print(f"\n[Available Sessions: {len(session_list)}]")
    print("-" * 50)
    for i, (thread_id, messages) in enumerate(session_list):
        title = format_title(messages)
        msg_count = len([m for m in messages if m.get("role") != "system"])
        print(f"  [{i}] {thread_id}  ({msg_count} msgs)  {title}")
    print("-" * 50)

    print("\nEnter number to export, 'a' for all, 'q' to quit:")
    while True:
        try:
            choice = input("> ").strip()
            if choice.lower() in ["q", "quit", "exit"]:
                print("[System] Exiting.")
                break
            elif choice.lower() == "a":
                for thread_id, messages in session_list:
                    export_session(thread_id, messages)
                break
            elif choice.isdigit():
                idx = int(choice)
                if 0 <= idx < len(session_list):
                    thread_id, messages = session_list[idx]
                    export_session(thread_id, messages)
                    # 続けて別セッションも出力できるようにループ継続
                    print("Export another? (number / 'a' / 'q')")
                else:
                    print(f"[Error] 0 〜 {len(session_list)-1} の範囲で入力")
            else:
                print("[Error] Invalid input.")
        except KeyboardInterrupt:
            print("\n[System] Interrupted.")
            break

if __name__ == "__main__":
    main()