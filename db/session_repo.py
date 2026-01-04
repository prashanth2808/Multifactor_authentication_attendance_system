# db/session_repo.py
"""Session repository backed by Supabase Postgres.

This version is *schema-minimal*:
- `sessions` table is assumed to contain ONLY:
  user_id, date, login_time, logout_time, duration_minutes, status

We intentionally do NOT store `name` or `email` in the sessions table.
Reports join against the `users` table to display name/email.

Expected tables:
- public.users(id, name, email, ...)
- public.sessions(user_id, date, login_time, logout_time, duration_minutes, status)

Recommended constraint:
- unique(user_id, date)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

from rich.console import Console

from db.supabase_client import supabase

console = Console()


def _now() -> datetime:
    return datetime.now()


def _parse_iso(ts: Any) -> datetime | None:
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            return None
    return None


def mark_session(user_id: str, name: str | None = None, email: str | None = None) -> Tuple[str, str]:
    """Mark session for today.

    `name`/`email` are accepted for compatibility with callers but are not stored
    in the sessions table.
    """

    today_str = _now().date().isoformat()
    now = _now()

    resp = (
        supabase.table("sessions")
        .select("*")
        .eq("user_id", user_id)
        .eq("date", today_str)
        .limit(1)
        .execute()
    )
    session = (resp.data[0] if resp.data else None)

    if not session:
        supabase.table("sessions").insert(
            {
                "user_id": user_id,
                "date": today_str,
                "login_time": now.isoformat(),
                "logout_time": None,
                "duration_minutes": None,
                "status": "active",
            }
        ).execute()
        login_time = now.strftime("%I:%M %p")
        display = name or "User"
        console.print(f"[bold green]LOGIN 8 {display} at {login_time}[/bold green]")
        return "LOGIN", f"Logged in at {login_time}"

    # If already logged out => malpractice
    if session.get("logout_time") is not None:
        if session.get("status") == "absent_fault":
            return "MALPRACTICE", "Already marked ABSENT today (forgot to logout)"
        return "MALPRACTICE", "Already marked PRESENT today  Malpractice detected"

    login_time_dt = _parse_iso(session.get("login_time")) or now
    hours_passed = (now - login_time_dt).total_seconds() / 3600

    if hours_passed >= 9:
        auto_logout_time = login_time_dt + timedelta(hours=9)
        supabase.table("sessions").update(
            {
                "logout_time": auto_logout_time.isoformat(),
                "duration_minutes": 540,
                "status": "absent_fault",
            }
        ).eq("user_id", user_id).eq("date", today_str).execute()
        display = name or "User"
        console.print(f"[bold red]AUTO ABSENT 8 {display} (no logout for 9+ hours)[/bold red]")
        return "ABSENT_AUTO", "Marked ABSENT  forgot to logout (9+ hours)"

    duration_min = int(hours_passed * 60)
    supabase.table("sessions").update(
        {"logout_time": now.isoformat(), "duration_minutes": duration_min, "status": "present"}
    ).eq("user_id", user_id).eq("date", today_str).execute()

    display = name or "User"
    console.print(f"[bold magenta]LOGOUT 8 {display} after {duration_min} min[/bold magenta]")
    return "LOGOUT", "Logged out  Present today"


def get_today_status(user_id: str) -> Dict[str, Any]:
    today_str = _now().date().isoformat()
    now = _now()

    resp = (
        supabase.table("sessions")
        .select("*")
        .eq("user_id", user_id)
        .eq("date", today_str)
        .limit(1)
        .execute()
    )
    session = (resp.data[0] if resp.data else None)

    if not session:
        return {"status": "absent", "reason": "No login"}

    if session.get("logout_time"):
        if session.get("status") == "absent_fault":
            return {"status": "absent", "reason": "Forgot logout (9+ hours)"}
        return {"status": "present", "reason": "Proper session"}

    # Active session
    login_time_dt = _parse_iso(session.get("login_time")) or now
    hours = (now - login_time_dt).total_seconds() / 3600

    if hours >= 9:
        # Trigger auto-absent
        mark_session(user_id)
        return {"status": "absent", "reason": "Auto-absent (9+ hours)"}

    return {"status": "present", "reason": "Active session"}


def _fetch_users_map(user_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    if not user_ids:
        return {}
    try:
        # Supabase expects something like .in_("id", [...])
        resp = supabase.table("users").select("id,name,email").in_("id", user_ids).execute()
        rows = resp.data or []
        return {str(r.get("id")): r for r in rows}
    except Exception as e:
        console.print(f"[red]Failed to fetch users for report: {e}[/red]")
        return {}


def get_report(date_str: str) -> List[Dict[str, Any]]:
    resp = supabase.table("sessions").select("*").eq("date", date_str).execute()
    sessions = resp.data or []

    user_ids = [str(s.get("user_id")) for s in sessions if s.get("user_id")]
    users_map = _fetch_users_map(list(sorted(set(user_ids))))

    def _fmt(ts: Any) -> str:
        dt = _parse_iso(ts)
        return dt.strftime("%I:%M %p") if dt else ""

    result: List[Dict[str, Any]] = []
    for s in sessions:
        uid = str(s.get("user_id"))
        u = users_map.get(uid, {})

        login_raw = s.get("login_time")
        logout_raw = s.get("logout_time")

        login = _fmt(login_raw)
        logout = _fmt(logout_raw)
        duration = (
            f"{s.get('duration_minutes', '')} min" if s.get("duration_minutes") is not None else ""
        )

        if s.get("logout_time"):
            if s.get("status") == "absent_fault":
                status = "[bold red]Absent[/bold red] (Forgot Logout)"
            else:
                status = "[bold green]Present[/bold green]"
        else:
            status = "[yellow]Active[/yellow]"

        result.append(
            {
                "name": u.get("name", ""),
                "email": u.get("email", ""),
                "login": login,
                "logout": logout,
                "duration": duration,
                "status": status,
                "user_id": uid,
                # raw fields
                "login_time": login_raw,
                "logout_time": logout_raw,
                "duration_minutes": s.get("duration_minutes"),
            }
        )

    return result
