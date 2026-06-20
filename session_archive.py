"""SQLite-backed session archive for meeTai.

This module is intentionally small and dependency-free. LangGraph checkpoints
remain the source of runtime state; this database is the human-facing ledger for
listing, exporting, and later usage tracking.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_dict(row: sqlite3.Row | None) -> Optional[dict[str, Any]]:
    return dict(row) if row is not None else None


def init_archive(db_path: str) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
              session_id TEXT PRIMARY KEY,
              title TEXT NOT NULL,
              topic TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              user_email TEXT,
              phase TEXT,
              agents TEXT,
              turns_used INTEGER DEFAULT 0,
              llm_calls_used INTEGER DEFAULT 0,
              input_tokens INTEGER DEFAULT 0,
              output_tokens INTEGER DEFAULT 0,
              total_tokens INTEGER DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session_messages (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              session_id TEXT NOT NULL,
              created_at TEXT NOT NULL,
              role TEXT NOT NULL,
              agent TEXT,
              phase TEXT,
              content TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_session_messages_session_id ON session_messages(session_id)"
        )


def fallback_title(text: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return "untitled"
    return compact[:40]


def create_session_record(
    db_path: str,
    *,
    session_id: str,
    title: Optional[str],
    user_email: Optional[str],
    phase: str,
    agents: list[str],
) -> None:
    now = _now()
    clean_title = (title or "").strip() or "untitled"
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO sessions (
              session_id, title, topic, created_at, updated_at,
              user_email, phase, agents,
              turns_used, llm_calls_used, input_tokens, output_tokens, total_tokens
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0, 0, 0)
            """,
            (
                session_id,
                clean_title,
                clean_title if clean_title != "untitled" else None,
                now,
                now,
                user_email,
                phase,
                json.dumps(agents, ensure_ascii=False),
            ),
        )


def append_messages(
    db_path: str,
    session_id: str,
    messages: list[dict[str, Any]],
    *,
    phase: str,
) -> None:
    if not messages:
        return

    now = _now()
    with _connect(db_path) as conn:
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content") or ""
            if role not in {"user", "assistant"} or not content:
                continue
            conn.execute(
                """
                INSERT INTO session_messages (
                  session_id, created_at, role, agent, phase, content
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    now,
                    role,
                    msg.get("agent"),
                    phase,
                    content,
                ),
            )

        counts = conn.execute(
            """
            SELECT
              SUM(CASE WHEN role = 'user' THEN 1 ELSE 0 END) AS turns_used,
              SUM(CASE WHEN role = 'assistant' THEN 1 ELSE 0 END) AS llm_calls_used
            FROM session_messages
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()

        title_row = conn.execute(
            "SELECT title FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()

        title = title_row["title"] if title_row else "untitled"
        first_user = next(
            (m.get("content", "") for m in messages if m.get("role") == "user"),
            "",
        )
        new_title = fallback_title(first_user) if title == "untitled" and first_user else title
        topic = new_title if new_title != "untitled" else None

        conn.execute(
            """
            UPDATE sessions
            SET updated_at = ?,
                phase = ?,
                title = ?,
                topic = COALESCE(topic, ?),
                turns_used = COALESCE(?, 0),
                llm_calls_used = COALESCE(?, 0)
            WHERE session_id = ?
            """,
            (
                now,
                phase,
                new_title,
                topic,
                counts["turns_used"],
                counts["llm_calls_used"],
                session_id,
            ),
        )


def _range_start(range_key: Optional[str]) -> str:
    now = datetime.now(timezone.utc)
    lookup = {
        "7d": timedelta(days=7),
        "30d": timedelta(days=30),
        "90d": timedelta(days=90),
    }
    delta = lookup.get(range_key or "7d", timedelta(days=7))
    return (now - delta).isoformat(timespec="seconds")


def _month_bounds(month: str) -> tuple[str, str]:
    if not re.fullmatch(r"\d{4}-\d{2}", month):
        raise ValueError("month must be YYYY-MM")
    year, month_number = map(int, month.split("-"))
    start = datetime(year, month_number, 1, tzinfo=timezone.utc)
    if month_number == 12:
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(year, month_number + 1, 1, tzinfo=timezone.utc)
    return start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds")


def list_session_records(
    db_path: str,
    *,
    range_key: Optional[str] = "7d",
    month: Optional[str] = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        if month:
            start, end = _month_bounds(month)
            rows = conn.execute(
                """
                SELECT *
                FROM sessions
                WHERE created_at >= ? AND created_at < ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (start, end, limit),
            ).fetchall()
        else:
            start = _range_start(range_key)
            rows = conn.execute(
                """
                SELECT *
                FROM sessions
                WHERE updated_at >= ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (start, limit),
            ).fetchall()

    return [dict(row) for row in rows]


def get_session_record(db_path: str, session_id: str) -> Optional[dict[str, Any]]:
    with _connect(db_path) as conn:
        session = _row_to_dict(
            conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        )
        if session is None:
            return None

        rows = conn.execute(
            """
            SELECT id, created_at, role, agent, phase, content
            FROM session_messages
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()

    session["messages"] = [dict(row) for row in rows]
    return session


def _agents_text(raw: str | None) -> str:
    if not raw:
        return ""
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return ", ".join(str(x) for x in data)
    except Exception:
        pass
    return raw


def render_session_markdown(session: dict[str, Any]) -> str:
    title = session.get("title") or "untitled"
    lines = [
        f"# {title}",
        "",
        f"- session_id: {session.get('session_id')}",
        f"- created_at: {session.get('created_at')}",
        f"- updated_at: {session.get('updated_at')}",
        f"- phase: {session.get('phase') or ''}",
        f"- agents: {_agents_text(session.get('agents'))}",
        f"- turns: {session.get('turns_used') or 0}",
        f"- llm_calls: {session.get('llm_calls_used') or 0}",
        f"- total_tokens: {session.get('total_tokens') or 0}",
        "",
        "## Minutes",
        "",
    ]

    for msg in session.get("messages", []):
        agent = (msg.get("agent") or msg.get("role") or "?").upper()
        content = msg.get("content") or ""
        lines.append(f"[{agent}]")
        lines.append(content)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
