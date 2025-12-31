# db/client.py
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConnectionFailure
from config.settings import settings
from rich.console import Console

console = Console()

# Global variables
_client: MongoClient | None = None
_db: Database | None = None

def get_client() -> MongoClient | None:
    """Create and return MongoClient (singleton)"""
    global _client
    if _client is None:
        try:
            _client = MongoClient(
                settings.mongodb_uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                maxPoolSize=10,
            )
            # Test connection
            _client.admin.command('ping')
            console.print("[green]Connected to MongoDB[/green]")
        except Exception as e:
            console.print(f"[bold red]MongoDB connection failed: {e}[/bold red]")
            _client = None
    return _client

def get_db() -> Database | None:
    """Return database object (singleton)"""
    global _db
    if _db is not None:
        return _db  # Already initialized

    client = get_client()
    if client is None:
        console.print("[bold red]No MongoDB client available[/bold red]")
        return None

    try:
        _db = client[settings.db_name]
        console.print(f"[cyan]Using database: {settings.db_name}[/cyan]")
        _create_indexes(_db)
        return _db
    except Exception as e:
        console.print(f"[bold red]Failed to access database '{settings.db_name}': {e}[/bold red]")
        _db = None
        return None

def _create_indexes(db: Database):
    """Create necessary indexes"""
    try:
        db.users.create_index("email", unique=True)
        db.attendance.create_index([("user_id", 1), ("timestamp", -1)])
        db.attendance.create_index("timestamp")
        console.print("[cyan]Database indexes created[/cyan]")
    except Exception as e:
        console.print(f"[yellow]Index warning: {e}[/yellow]")

def close_connection():
    """Close MongoDB connection"""
    global _client, _db
    if _client:
        _client.close()
        console.print("[bold]MongoDB connection closed[/bold]")
    _client = None
    _db = None