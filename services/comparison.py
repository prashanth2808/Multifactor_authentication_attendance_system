# services/comparison.py
"""
Face comparison using cosine similarity
Supports face_embeddings + voice_embedding
"""

import numpy as np
from typing import Optional, Tuple
from db.user_repo import get_all_users
from config.settings import settings
from rich.console import Console
from rich.table import Table

console = Console()

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return np.dot(a, b) / (a_norm * b_norm)

def find_best_match(
    query_embedding: np.ndarray
) -> Optional[Tuple[str, str, float, str, list]]:
    users = get_all_users()
    if not users:
        console.print("[red]No users in database[/red]")
        return None

    best_match = None
    best_score = -1.0

    for user in users:
        user_id = str(user["_id"])
        name = user["name"]
        email = user["email"]
        
        # SUPPORT BOTH OLD AND NEW SCHEMA
        embeddings = user.get("face_embeddings") or user.get("embeddings", [])
        voice_emb = user.get("voice_embedding")
        
        if not embeddings:
            continue

        user_embs = [np.array(emb, dtype=np.float32) for emb in embeddings]
        scores = [cosine_similarity(query_embedding, emb) for emb in user_embs]
        max_score = max(scores)

        if max_score > best_score:
            best_score = max_score
            best_match = (user_id, name, max_score, email, voice_emb)

    if best_match and best_score >= settings.similarity_threshold:
        return best_match
    return None

def verify_match(
    query_embedding: np.ndarray,
    threshold: float = None
) -> dict:
    if threshold is None:
        threshold = settings.similarity_threshold

    result = {
        "matched": False,
        "user_id": None,
        "name": None,
        "email": None,
        "confidence": 0.0,
        "voice_embedding": None,
        "message": "No face detected"
    }

    match = find_best_match(query_embedding)
    if match:
        user_id, name, score, email, voice_emb = match
        if score >= threshold:
            result.update({
                "matched": True,
                "user_id": user_id,
                "name": name,
                "email": email,
                "confidence": round(score, 4),
                "voice_embedding": voice_emb,
                "message": f"Welcome, {name}!"
            })
        else:
            result["message"] = f"Face recognized but below threshold ({score:.3f})"
    else:
        result["message"] = "Unknown user"

    return result

def debug_similarity_table(query_embedding: np.ndarray):
    users = get_all_users()
    if not users:
        return

    table = Table(title="Similarity Scores")
    table.add_column("Name")
    table.add_column("Email")
    table.add_column("Max Score")
    table.add_column("Voice?")
    table.add_column("Match?")

    for user in users:
        name = user["name"]
        email = user["email"]
        embeddings = user.get("face_embeddings") or user.get("embeddings", [])
        voice_emb = user.get("voice_embedding")
        if not embeddings:
            continue

        user_embs = [np.array(emb) for emb in embeddings]
        scores = [cosine_similarity(query_embedding, emb) for emb in user_embs]
        max_score = max(scores)

        voice_status = "Yes" if voice_emb else "No"
        is_match = "Yes" if max_score >= settings.similarity_threshold else "No"
        table.add_row(name, email, f"{max_score:.4f}", voice_status, is_match)

    console.print(table)