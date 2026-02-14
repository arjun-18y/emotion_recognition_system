import hashlib
from datetime import datetime, timedelta
from flask import url_for
from ..extensions import mail, db
from ..models.user_model import PasswordResetToken
from flask_mail import Message
import secrets

RESET_TOKEN_MINUTES = 30


def generate_reset_token(user_id):
    raw = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw.encode('utf-8')).hexdigest()
    expires = datetime.utcnow() + timedelta(minutes=RESET_TOKEN_MINUTES)
    entry = PasswordResetToken(user_id=user_id, token_hash=token_hash, expires_at=expires)
    db.session.add(entry)
    db.session.commit()
    return raw


def send_password_reset_email(user, raw_token):
    reset_link = url_for('auth.reset_password', token=raw_token, _external=True)
    msg = Message(
        subject='Password Reset Request',
        recipients=[user.email],
        body=f"Hello,\n\nClick the link below to reset your password (valid for {RESET_TOKEN_MINUTES} minutes):\n{reset_link}\n\nIf you did not request this, please ignore this email."
    )
    try:
        mail.send(msg)
    except Exception as e:
        # Fallback: log the link to console when SMTP is not configured
        print(f"[WARN] Failed to send email: {e}. Reset link: {reset_link}")
