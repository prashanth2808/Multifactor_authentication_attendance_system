# db/client.py
"""Deprecated MongoDB client.

This project has been migrated from MongoDB to Supabase Postgres.

Some older CLI code imported `get_db()` to talk to Mongo. That should no longer
be used. We keep a stub here to avoid import crashes and to provide a helpful
message if any code path still calls it.
"""

from __future__ import annotations

from rich.console import Console

console = Console()


def get_db():
    console.print(
        "[bold yellow]MongoDB has been removed. Configure DATABASE_URL and use the Postgres repos (db/user_repo.py, db/session_repo.py).[/bold yellow]"
    )
    return None


def close_connection():
    return None
