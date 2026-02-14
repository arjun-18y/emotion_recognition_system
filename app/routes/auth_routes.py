from datetime import datetime
from hashlib import sha256
from flask import Blueprint, jsonify, render_template, request
from flask_jwt_extended import create_access_token
from ..extensions import bcrypt, db, limiter
from ..models.user_model import LoginLog, PasswordResetToken, User
from ..services.email_service import generate_reset_token, send_password_reset_email
from ..utils.security import get_client_ip

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")


def _validate_email(email):
    return isinstance(email, str) and "@" in email and "." in email


def _get_request_data():
    if request.is_json:
        return request.get_json(silent=True) or {}
    return request.form


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")

    data = _get_request_data()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not _validate_email(email) or len(password) < 8:
        return jsonify({"error": "Invalid input"}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already registered"}), 400

    hashed_pw = bcrypt.generate_password_hash(password).decode("utf-8")
    user = User(email=email, password=hashed_pw)
    db.session.add(user)
    db.session.commit()
    return jsonify({"message": "User registered successfully"})


@auth_bp.route("/login", methods=["GET", "POST"])
@limiter.limit("5 per minute")
def login():
    if request.method == "GET":
        return render_template("login.html")

    data = _get_request_data()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    user = User.query.filter_by(email=email).first()

    if user and bcrypt.check_password_hash(user.password, password):
        token = create_access_token(identity=str(user.id))
        db.session.add(
            LoginLog(
                user_id=user.id,
                email=email,
                success=True,
                ip_address=get_client_ip(),
            )
        )
        db.session.commit()
        return jsonify(access_token=token, role=user.role)

    db.session.add(LoginLog(email=email, success=False, ip_address=get_client_ip()))
    db.session.commit()
    return jsonify({"error": "Invalid credentials"}), 401


@auth_bp.route("/forgot", methods=["GET", "POST"])
@limiter.limit("3 per minute")
def forgot_password():
    if request.method == "GET":
        return render_template("forgot_password.html")

    data = _get_request_data()
    email = data.get("email", "").strip().lower()
    user = User.query.filter_by(email=email).first()

    if not user:
        return jsonify({"message": "If the email exists, a reset link will be sent."})

    raw = generate_reset_token(user.id)
    send_password_reset_email(user, raw)
    return jsonify({"message": "Reset link sent"})


@auth_bp.route("/reset", methods=["GET", "POST"])
def reset_password():
    if request.method == "GET":
        token = request.args.get("token", "")
        return render_template("reset_password.html", token=token)

    data = _get_request_data()
    token = data.get("token", "").strip()
    new_pw = data.get("password", "")

    if not token:
        return jsonify({"error": "Missing token"}), 400
    if not new_pw or len(new_pw) < 8:
        return jsonify({"error": "Invalid password"}), 400

    token_hash = sha256(token.encode("utf-8")).hexdigest()
    entry = PasswordResetToken.query.filter_by(token_hash=token_hash, used=False).first()
    if not entry or entry.expires_at < datetime.utcnow():
        return jsonify({"error": "Invalid or expired token"}), 400

    user = db.session.get(User, entry.user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    user.password = bcrypt.generate_password_hash(new_pw).decode("utf-8")
    entry.used = True
    db.session.commit()
    return jsonify({"message": "Password reset successful"})


@auth_bp.route("/test", methods=["GET"])
def test():
    return {"message": "Auth working"}
