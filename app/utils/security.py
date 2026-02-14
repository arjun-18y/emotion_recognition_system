import re
from functools import wraps
from flask import request, jsonify
from flask_jwt_extended import get_jwt_identity
from ..extensions import db
from ..models.user_model import User


def sanitize_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = re.sub(r"<[^>]*>", "", text)  # strip HTML tags
    text = re.sub(r"[\r\n\t]", " ", text)
    text = text.strip()
    return text

ALLOWED_TEXT_EXTENSIONS = {"txt", "csv"}
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

def _has_allowed_extension(filename: str, allowed: set) -> bool:
    return "." in filename and filename.rsplit('.', 1)[1].lower() in allowed


def allowed_text_file(filename: str) -> bool:
    return _has_allowed_extension(filename, ALLOWED_TEXT_EXTENSIONS)


def allowed_image_file(filename: str) -> bool:
    return _has_allowed_extension(filename, ALLOWED_IMAGE_EXTENSIONS)


def allowed_file(filename: str) -> bool:
    return allowed_text_file(filename)


def get_client_ip():
    return request.headers.get('X-Forwarded-For', request.remote_addr)


def role_required(role_name):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            identity = get_jwt_identity()
            try:
                user_id = int(identity)
            except (TypeError, ValueError):
                return jsonify({"error": "Unauthorized"}), 403
            user = db.session.get(User, user_id)
            if not user or user.role != role_name:
                return jsonify({"error": "Unauthorized"}), 403
            return fn(*args, **kwargs)
        return wrapper
    return decorator
