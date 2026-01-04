"""api_server.py
Cloud backend for Face+Voice verification.

IMPORTANT:
- This server must NEVER access webcam/microphone hardware.
- The browser captures image/audio and uploads them.

Render start command (recommended):
  gunicorn api_server:app
"""

from __future__ import annotations

import os
from typing import Any
from datetime import datetime

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

from config.settings import settings
from db.supabase_client import supabase
from db.session_repo import mark_session
from db.user_repo import save_user, find_user_by_email
from services.embedding import get_face_embedding
from services.comparison import verify_match
from services.face_detection import get_cropped_face
from services.io_helpers import decode_image_bytes
from services.voice_embedding import (
    verify_voice_from_audio_bytes_detailed,
    get_voice_embedding_from_audio_bytes,
)

import traceback


# === PRE-LOAD LIGHTWEIGHT INSIGHTFACE MODEL AT STARTUP ===
# This prevents timeout/OOM on first request
# We use 'antelopev2' — much smaller (~100MB) than buffalo_l (~280MB), fits in Render free tier 512MB RAM
print("Pre-loading lightweight InsightFace model (antelopev2)...")

from insightface.app import FaceAnalysis

# Force lightweight model
_insightface_app = FaceAnalysis(
    name='antelopev2',  # ← Lightweight, accurate, fits in 512MB RAM
    providers=['CPUExecutionProvider'],
    root=os.path.expanduser('~/.insightface/models')  # Default cache dir
)
_insightface_app.prepare(ctx_id=0)  # CPU context

print("InsightFace antelopev2 model loaded and ready")


def _json_safe(obj: Any):
    """Convert numpy/scalar objects into JSON-safe types."""
    try:
        import numpy as np

        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
    except Exception:
        pass

    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    # === GLOBAL ERROR HANDLER ===
    @app.errorhandler(Exception)
    def handle_exception(e):
        if isinstance(e, HTTPException):
            return e
        print(traceback.format_exc())
        return {"error": "Internal Server Error"}, 500

    # === FAVICON ===
    @app.route("/favicon.ico")
    def favicon():
        return "", 204

    # === HEALTH CHECK ===
    @app.get("/api/health")
    def health():
        try:
            resp = supabase.table("users").select("id").limit(1).execute()
            return jsonify({"ok": True, "db": True, "users_visible": bool(resp.data)})
        except Exception as e:
            return jsonify({"ok": True, "db": False, "error": str(e)}), 200

    # === REGISTRATION ===
    @app.post("/api/register")
    def register():
        name = (request.form.get("name") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        face_file = request.files.get("face")
        voice_file = request.files.get("voice")

        if not name or not email:
            return jsonify({"ok": False, "error": "name and email are required"}), 400
        if not face_file or not voice_file:
            return jsonify({"ok": False, "error": "face and voice are required"}), 400

        existing = find_user_by_email(email)
        if existing:
            return jsonify({
                "ok": False,
                "error": "user already exists",
                "user": {
                    "id": existing.get("id"),
                    "email": existing.get("email"),
                    "name": existing.get("name")
                }
            }), 409

        img = decode_image_bytes(face_file.read())
        if img is None:
            return jsonify({"ok": False, "error": "invalid image"}), 400

        cropped = get_cropped_face(img)
        if cropped is None:
            return jsonify({"ok": False, "error": "no_face_detected"}), 400

        face_emb = get_face_embedding(cropped)
        if face_emb is None:
            return jsonify({"ok": False, "error": "face_embedding_failed"}), 400

        voice_bytes = voice_file.read()
        voice_emb = get_voice_embedding_from_audio_bytes(voice_bytes)
        if voice_emb is None:
            return jsonify({"ok": False, "error": "voice_embedding_failed"}), 400

        user_data = {
            "name": name,
            "email": email,
            "face_embeddings": [face_emb.tolist()],
            "voice_embedding": voice_emb.tolist(),
            "photo_count": 1,
            "voice_clips": 1,
            "registered_at": datetime.utcnow().isoformat(),
        }

        inserted = save_user(user_data)
        if not inserted:
            return jsonify({"ok": False, "error": "failed_to_save_user"}), 500

        return jsonify({
            "ok": True,
            "user": {
                "id": inserted.get("id"),
                "name": inserted.get("name"),
                "email": inserted.get("email")
            }
        })

    # === SESSION (AUTO) ===
    @app.post("/api/session/auto")
    def session_auto():
        face_file = request.files.get("face")
        voice_file = request.files.get("voice")
        if not face_file or not voice_file:
            return jsonify({"ok": False, "error": "face and voice are required"}), 400

        img = decode_image_bytes(face_file.read())
        if img is None:
            return jsonify({"ok": False, "error": "invalid image"}), 400

        cropped = get_cropped_face(img)
        if cropped is None:
            return jsonify({"ok": True, "verified": False, "reason": "no_face_detected"}), 200

        face_emb = get_face_embedding(cropped)
        if face_emb is None:
            return jsonify({"ok": True, "verified": False, "reason": "face_embedding_failed"}), 200

        face_result = verify_match(face_emb)
        stored_voice = face_result.get("voice_embedding")

        if isinstance(face_result, dict):
            face_result.pop("voice_embedding", None)

        if not face_result.get("matched"):
            return jsonify({"ok": True, "verified": False, "reason": "face_not_matched", "face": _json_safe(face_result)}), 200

        if not stored_voice:
            return jsonify({"ok": True, "verified": False, "reason": "no_voice_enrolled", "face": _json_safe(face_result)}), 200

        score, passed, voice_reason = verify_voice_from_audio_bytes_detailed(
            stored_emb=stored_voice,
            audio_bytes=voice_file.read(),
            threshold=settings.voice_threshold,
        )

        response = {
            "ok": True,
            "verified": bool(passed),
            "reason": "ok" if passed else voice_reason,
            "face": _json_safe(face_result),
            "voice": {"score": float(score), "passed": bool(passed)},
        }

        if passed:
            user_id = face_result.get("user_id")
            name = face_result.get("name")
            email = face_result.get("email")
            action, message = mark_session(user_id, name, email)
            response["session"] = {"action": action, "message": message}

        return jsonify(response)

    # === STATIC UI ===
    WEB_DIR = os.path.join(os.path.dirname(__file__), "web")

    @app.get("/")
    def index():
        return send_from_directory(WEB_DIR, "verify.html")

    @app.get("/register")
    def register_page():
        register_path = os.path.join(WEB_DIR, "register.html")
        if not os.path.exists(register_path):
            return (
                f"register.html not found at: {register_path}. "
                f"Check you restarted the server and are running the correct project folder.",
                404,
            )
        return send_from_directory(WEB_DIR, "register.html")

    @app.get("/web/<path:filename>")
    def web_static(filename: str):
        return send_from_directory(WEB_DIR, filename)

    @app.get("/web/register.js")
    def register_js():
        return send_from_directory(WEB_DIR, "register.js")

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
