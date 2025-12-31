# db/user_repo.py
"""
User repository - CRUD operations for users
Collection: users
"""

from pymongo.collection import Collection
from pymongo.results import InsertOneResult
from db.client import get_db
from rich.console import Console
from typing import Dict, Any, List

console = Console()

def get_users_collection() -> Collection | None:
    """Get users collection"""
    db = get_db()
    if db is None:
        return None
    return db.users

def save_user(user_data: Dict[str, Any]) -> InsertOneResult | None:
    """
    Save new user with face + voice embeddings + voice_audio_path
    Returns InsertOneResult or None
    """
    collection = get_users_collection()
    if collection is None:
        console.print("[red]Database not available[/red]")
        return None

    try:
        # === FACE EMBEDDINGS (unchanged) ===
        if "face_embeddings" in user_data:
            user_data["face_embeddings"] = [
                emb if isinstance(emb, list) else emb.tolist()
                for emb in user_data["face_embeddings"]
            ]

        # === VOICE EMBEDDING (unchanged) ===
        if "voice_embedding" in user_data:
            emb = user_data["voice_embedding"]
            if not isinstance(emb, list):
                emb = emb.tolist()
            user_data["voice_embedding"] = emb  # 256D list

        # === NEW: Save voice audio file path (for backup) ===
        if "voice_audio_path" in user_data:
            console.print(f"[green]Voice backup saved: {user_data['voice_audio_path']}[/green]")

        result = collection.insert_one(user_data)
        console.print(f"[green]User saved with ID: {result.inserted_id}[/green]")
        return result

    except Exception as e:
        console.print(f"[red]Failed to save user: {e}[/red]")
        return None

def find_user_by_email(email: str) -> Dict | None:
    """Find user by email"""
    collection = get_users_collection()
    if collection is None:
        return None

    try:
        user = collection.find_one({"email": email})
        return user
    except Exception as e:
        console.print(f"[red]Query error: {e}[/red]")
        return None

def get_all_users() -> List[Dict] | None:
    """Get all registered users WITH face + voice embeddings + voice audio path"""
    collection = get_users_collection()
    if collection is None:
        return None

    try:
        users = list(collection.find({}, {
            "name": 1,
            "email": 1,
            "photo_count": 1,
            "voice_clips": 1,          # ← NEW
            "registered_at": 1,
            "face_embeddings": 1,      # ← Renamed
            "voice_embedding": 1,      # ← NEW
            "voice_audio_path": 1,     # ← NEW: shows .wav path in admin
            "_id": 1
        }))
        console.print(f"[cyan]Loaded {len(users)} users[/cyan]")
        return users
    except Exception as e:
        console.print(f"[red]Failed to load users: {e}[/red]")
        return None

def get_user_embeddings(user_id: str) -> Dict[str, Any] | None:
    """Get face + voice embeddings for a user"""
    collection = get_users_collection()
    if collection is None:
        return None

    try:
        user = collection.find_one(
            {"_id": user_id},
            {
                "face_embeddings": 1,
                "voice_embedding": 1
            }
        )
        if user:
            return {
                "face_embeddings": user.get("face_embeddings", []),
                "voice_embedding": user.get("voice_embedding")
            }
        return None
    except Exception as e:
        console.print(f"[red]Failed to get embeddings: {e}[/red]")
        return None

def create_user(user_data: Dict[str, Any]) -> str | None:
    """Create a new user and return user ID"""
    collection = get_users_collection()
    if collection is None:
        console.print("[red]Database not available[/red]")
        return None

    try:
        result = collection.insert_one(user_data)
        console.print(f"[green]User created with ID: {result.inserted_id}[/green]")
        return str(result.inserted_id)
    except Exception as e:
        console.print(f"[red]Failed to create user: {e}[/red]")
        return None

def get_user_by_id(user_id: str) -> Dict | None:
    """Get user by ObjectId"""
    collection = get_users_collection()
    if collection is None:
        return None

    try:
        from bson import ObjectId
        user = collection.find_one({"_id": ObjectId(user_id)})
        return user
    except Exception as e:
        console.print(f"[red]Failed to get user by ID: {e}[/red]")
        return None

def update_user_face_data(user_id: str, face_embeddings: List, photo_count: int) -> bool:
    """Update user's face embeddings and photo count"""
    collection = get_users_collection()
    if collection is None:
        return False

    try:
        from bson import ObjectId
        result = collection.update_one(
            {"_id": ObjectId(user_id)},
            {
                "$set": {
                    "face_embeddings": face_embeddings,
                    "photo_count": photo_count
                }
            }
        )
        success = result.modified_count > 0
        if success:
            console.print(f"[green]Updated face data for user {user_id}[/green]")
        return success
    except Exception as e:
        console.print(f"[red]Failed to update face data: {e}[/red]")
        return False

def update_user_voice_data(user_id: str, voice_embedding: List, voice_clips: int) -> bool:
    """Update user's voice embedding and voice clips count"""
    collection = get_users_collection()
    if collection is None:
        return False

    try:
        from bson import ObjectId
        result = collection.update_one(
            {"_id": ObjectId(user_id)},
            {
                "$set": {
                    "voice_embedding": voice_embedding,
                    "voice_clips": voice_clips
                }
            }
        )
        success = result.modified_count > 0
        if success:
            console.print(f"[green]Updated voice data for user {user_id}[/green]")
        return success
    except Exception as e:
        console.print(f"[red]Failed to update voice data: {e}[/red]")
        return False

def update_user_registration_status(user_id: str, registration_complete: bool) -> bool:
    """Mark user registration as complete"""
    collection = get_users_collection()
    if collection is None:
        return False

    try:
        from bson import ObjectId
        result = collection.update_one(
            {"_id": ObjectId(user_id)},
            {
                "$set": {
                    "registration_complete": registration_complete
                }
            }
        )
        success = result.modified_count > 0
        if success:
            console.print(f"[green]Registration completed for user {user_id}[/green]")
        return success
    except Exception as e:
        console.print(f"[red]Failed to update registration status: {e}[/red]")
        return False

def search_users_by_name_email(query: str) -> List[Dict]:
    """Search users by name or email containing query string"""
    collection = get_users_collection()
    if collection is None:
        return []
    
    try:
        # Create regex pattern for case-insensitive search
        regex_pattern = {"$regex": query, "$options": "i"}
        
        # Search in both name and email fields
        cursor = collection.find({
            "$or": [
                {"name": regex_pattern},
                {"email": regex_pattern}
            ]
        }).limit(20)  # Limit results to prevent too many matches
        
        users = list(cursor)
        console.print(f"[cyan]Found {len(users)} users matching '{query}'[/cyan]")
        return users
        
    except Exception as e:
        console.print(f"[red]Error searching users: {e}[/red]")
        return []

def delete_user_by_id(user_id: str) -> bool:
    """Delete user by ObjectId"""
    collection = get_users_collection()
    if collection is None:
        return False

    try:
        from bson import ObjectId
        result = collection.delete_one({"_id": ObjectId(user_id)})
        success = result.deleted_count > 0
        if success:
            console.print(f"[green]User {user_id} deleted successfully[/green]")
        else:
            console.print(f"[yellow]User {user_id} not found for deletion[/yellow]")
        return success
    except Exception as e:
        console.print(f"[red]Failed to delete user: {e}[/red]")
        return False