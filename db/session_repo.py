# db/session_repo.py
"""
FINAL SESSION REPOSITORY - STRICT 9-HOUR FAULT + MALPRACTICE PROTECTION
Rules:
- Login → No logout → <9 hours  → Present (grace)
- Login → No logout → ≥9 hours  → Absent (user fault)
- Already marked today → "Malpractice detected"
- Proper login/logout → Present
"""

from pymongo.collection import Collection
from pymongo.results import UpdateResult
from db.client import get_db
from rich.console import Console
from typing import Tuple, Dict, List, Any
from datetime import datetime, timedelta

console = Console()

def get_sessions_collection() -> Collection | None:
    db = get_db()
    if db is None:
        return None
    return db.sessions

def mark_session(user_id: str, name: str, email: str) -> Tuple[str, str]:
    """
    Main function called during biometric scan
    Returns: (action, message)
    Possible actions: LOGIN, LOGOUT, MALPRACTICE, ABSENT_AUTO, ERROR
    """
    collection = get_sessions_collection()
    if collection is None:
        return "ERROR", "Database not available"

    # Use system local time (not UTC) so displayed timings match the machine clock
    today = datetime.now().date()
    today_str = today.isoformat()
    now = datetime.now()

    # Find today's session record
    session = collection.find_one({
        "user_id": user_id,
        "date": today_str
    })

    # CASE 1: First appearance today → LOGIN
    if not session:
        record = {
            "user_id": user_id,
            "name": name,
            "email": email,
            "login_time": now,
            "logout_time": None,
            "duration_minutes": None,
            "status": "active",  # temporary
            "date": today_str
        }
        collection.insert_one(record)
        login_time = now.strftime("%I:%M %p")
        console.print(f"[bold green]LOGIN → {name} at {login_time}[/bold green]")
        return "LOGIN", f"Logged in at {login_time}"

    # CASE 2: Already has a completed record (logged out or auto-absent)
    if session.get("logout_time") is not None:
        if session.get("status") == "absent_fault":
            return "MALPRACTICE", "Already marked ABSENT today (forgot to logout)"
        return "MALPRACTICE", "Already marked PRESENT today — Malpractice detected"

    # CASE 3: Active session (logged in, no logout yet)
    login_time = session["login_time"]
    hours_passed = (now - login_time).total_seconds() / 3600

    # SUBCASE 3A: 9+ hours passed → AUTO MARK AS ABSENT (user fault)
    if hours_passed >= 9:
        auto_logout_time = login_time + timedelta(hours=9)
        duration_min = 540  # 9 hours
        collection.update_one(
            {"_id": session["_id"]},
            {"$set": {
                "logout_time": auto_logout_time,
                "duration_minutes": duration_min,
                "status": "absent_fault"
            }}
        )
        console.print(f"[bold red]AUTO ABSENT → {name} (no logout for 9+ hours)[/bold red]")
        return "ABSENT_AUTO", "Marked ABSENT — forgot to logout (9+ hours)"

    # SUBCASE 3B: Still within 9 hours → NORMAL LOGOUT
    duration_min = int(hours_passed * 60)
    collection.update_one(
        {"_id": session["_id"]},
        {"$set": {
            "logout_time": now,
            "duration_minutes": duration_min,
            "status": "present"
        }}
    )
    logout_time = now.strftime("%I:%M %p")
    console.print(f"[bold magenta]LOGOUT → {name} after {duration_min} min[/bold magenta]")
    return "LOGOUT", f"Logged out — Present today"

def get_today_status(user_id: str) -> Dict[str, Any]:
    """Helper for reports — returns final status"""
    collection = get_sessions_collection()
    if collection is None:
        return {"status": "absent"}

    today_str = datetime.now().date().isoformat()
    session = collection.find_one({"user_id": user_id, "date": today_str})

    if not session:
        return {"status": "absent", "reason": "No login"}

    if session.get("logout_time"):
        if session.get("status") == "absent_fault":
            return {"status": "absent", "reason": "Forgot logout (9+ hours)"}
        return {"status": "present", "reason": "Proper session"}

    hours = (datetime.now() - session["login_time"]).total_seconds() / 3600
    if hours >= 9:
        # Trigger auto-absent
        dummy_name = session.get("name", "User")
        dummy_email = session.get("email", "")
        mark_session(user_id, dummy_name, dummy_email)
        return {"status": "absent", "reason": "Auto-absent (9+ hours)"}

    return {"status": "present", "reason": "Active session"}

def get_report(date_str: str) -> List[Dict]:
    """Enhanced report with correct Present/Absent"""
    collection = get_sessions_collection()
    if collection is None:
        return []

    sessions = list(collection.find({"date": date_str}).sort("login_time", 1))
    result = []

    for s in sessions:
        login = s["login_time"].strftime("%I:%M %p") if s.get("login_time") else "—"
        logout = s["logout_time"].strftime("%I:%M %p") if s.get("logout_time") else "—"
        duration = f"{s.get('duration_minutes', '—')} min" if s.get('duration_minutes') else "—"

        if s.get("logout_time"):
            if s.get("status") == "absent_fault":
                status = "[bold red]Absent[/bold red] (Forgot Logout)"
            else:
                status = "[bold green]Present[/bold green]"
        else:
            hours = (datetime.now() - s["login_time"]).total_seconds() / 3600
            if hours >= 9:
                status = "[bold red]Absent[/bold red] (Auto)"
            else:
                status = "[yellow]Active[/yellow]"

        result.append({
            "name": s["name"],
            "email": s["email"],
            "login": login,
            "logout": logout,
            "duration": duration,
            "status": status
        })

    return result

# Keep other functions (create_session, end_session, etc.) as they are if you use them
# Or remove if not needed