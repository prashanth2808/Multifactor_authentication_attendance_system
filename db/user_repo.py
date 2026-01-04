# db/user_repo.py - Supabase version
"""
User repository - CRUD operations for users
Table: users (in Supabase)
"""

from db.supabase_client import supabase
from rich.console import Console
from typing import Dict, Any, List, Optional

console = Console()

def save_user(user_data: Dict[str, Any]) -> Optional[Dict]:
    """
    Save new user with face + voice embeddings
    Returns inserted row or None
    """
    try:
        # === FACE EMBEDDINGS ===
        if "face_embeddings" in user_data:
            user_data["face_embeddings"] = [
                emb if isinstance(emb, list) else emb.tolist()
                for emb in user_data["face_embeddings"]
            ]

        # === VOICE EMBEDDING ===
        if "voice_embedding" in user_data:
            emb = user_data["voice_embedding"]
            if not isinstance(emb, list):
                emb = emb.tolist()
            user_data["voice_embedding"] = emb

        response = supabase.table("users").insert(user_data).execute()

        if response.data:
            inserted = response.data[0]
            console.print(f"[green]User saved with ID: {inserted['id']}[/green]")
            return inserted
        else:
            console.print(f"[red]Failed to save user: {response.error}[/red]")
            return None

    except Exception as e:
        console.print(f"[red]Failed to save user: {e}[/red]")
        return None

def find_user_by_email(email: str) -> Optional[Dict]:
    """Find user by email.

    IMPORTANT: Avoid `.single()` here.
    Supabase/PostgREST raises PGRST116 when 0 rows are returned, which is normal
    for a "does this user exist?" check.
    """
    try:
        resp = (
            supabase.table("users")
            .select("*")
            .eq("email", email)
            .limit(1)
            .execute()
        )
        return resp.data[0] if resp.data else None
    except Exception as e:
        console.print(f"[red]Query error (find_user_by_email): {e}[/red]")
        return None

def get_all_users() -> List[Dict]:
    """Get all registered users with face + voice embeddings"""
    try:
        # NOTE: `voice_audio_path` is optional; not all Supabase schemas include it.
        # Avoid selecting columns that may not exist to prevent 42703 errors.
        response = supabase.table("users").select(
            "id, name, email, photo_count, voice_clips, registered_at, "
            "face_embeddings, voice_embedding"
        ).execute()

        users = response.data or []
        console.print(f"[cyan]Loaded {len(users)} users[/cyan]")
        return users
    except Exception as e:
        console.print(f"[red]Failed to load users: {e}[/red]")
        return []

def get_user_embeddings(user_id: str) -> Optional[Dict[str, Any]]:
    """Get face + voice embeddings for a user"""
    try:
        response = supabase.table("users").select("face_embeddings, voice_embedding").eq("id", user_id).single().execute()
        if response.data:
            user = response.data
            return {
                "face_embeddings": user.get("face_embeddings", []),
                "voice_embedding": user.get("voice_embedding")
            }
        return None
    except Exception as e:
        console.print(f"[red]Failed to get embeddings: {e}[/red]")
        return None

def create_user(user_data: Dict[str, Any]) -> Optional[str]:
    """Create a new user and return user ID"""
    try:
        response = supabase.table("users").insert(user_data).execute()
        if response.data:
            user_id = response.data[0]["id"]
            console.print(f"[green]User created with ID: {user_id}[/green]")
            return str(user_id)
        return None
    except Exception as e:
        console.print(f"[red]Failed to create user: {e}[/red]")
        return None

def get_user_by_id(user_id: str) -> Optional[Dict]:
    """Get user by UUID"""
    try:
        response = supabase.table("users").select("*").eq("id", user_id).single().execute()
        return response.data if response.data else None
    except Exception as e:
        console.print(f"[red]Failed to get user by ID: {e}[/red]")
        return None

def update_user_face_data(user_id: str, face_embeddings: List, photo_count: int) -> bool:
    """Update user's face embeddings and photo count"""
    try:
        response = supabase.table("users").update({
            "face_embeddings": face_embeddings,
            "photo_count": photo_count
        }).eq("id", user_id).execute()

        success = len(response.data) > 0
        if success:
            console.print(f"[green]Updated face data for user {user_id}[/green]")
        return success
    except Exception as e:
        console.print(f"[red]Failed to update face data: {e}[/red]")
        return False

def update_user_voice_data(user_id: str, voice_embedding: List, voice_clips: int) -> bool:
    """Update user's voice embedding and voice clips count"""
    try:
        response = supabase.table("users").update({
            "voice_embedding": voice_embedding,
            "voice_clips": voice_clips
        }).eq("id", user_id).execute()

        success = len(response.data) > 0
        if success:
            console.print(f"[green]Updated voice data for user {user_id}[/green]")
        return success
    except Exception as e:
        console.print(f"[red]Failed to update voice data: {e}[/red]")
        return False

def update_user_registration_status(user_id: str, registration_complete: bool) -> bool:
    """Mark user registration as complete"""
    try:
        response = supabase.table("users").update({
            "registration_complete": registration_complete
        }).eq("id", user_id).execute()

        success = len(response.data) > 0
        if success:
            console.print(f"[green]Registration completed for user {user_id}[/green]")
        return success
    except Exception as e:
        console.print(f"[red]Failed to update registration status: {e}[/red]")
        return False

def search_users_by_name_email(query: str) -> List[Dict]:
    """Search users by name or email containing query string (case-insensitive)"""
    try:
        response = supabase.table("users").select("*").or_(
            f"name.ilike.%{query}%,email.ilike.%{query}%"
        ).limit(20).execute()

        users = response.data or []
        console.print(f"[cyan]Found {len(users)} users matching '{query}'[/cyan]")
        return users
    except Exception as e:
        console.print(f"[red]Error searching users: {e}[/red]")
        return []

def delete_user_by_id(user_id: str) -> bool:
    """Delete user by UUID"""
    try:
        response = supabase.table("users").delete().eq("id", user_id).execute()
        success = response.count > 0
        if success:
            console.print(f"[green]User {user_id} deleted successfully[/green]")
        else:
            console.print(f"[yellow]User {user_id} not found for deletion[/yellow]")
        return success
    except Exception as e:
        console.print(f"[red]Failed to delete user: {e}[/red]")
        return False