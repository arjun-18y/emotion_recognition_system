import os
import atexit
from flask import Flask, jsonify, render_template, redirect, request, url_for
from sqlalchemy import text
from .config import Config
from .extensions import db, jwt, bcrypt, mail, mongo, limiter


def _apply_sqlite_compat_migrations(app):
    db_uri = app.config.get("SQLALCHEMY_DATABASE_URI", "")
    if not db_uri.startswith("sqlite:///"):
        return

    with db.engine.connect() as conn:
        table_exists = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='user'")
        ).first()
        if not table_exists:
            return

        columns = {
            row[1] for row in conn.execute(text("PRAGMA table_info('user')")).fetchall()
        }

        if "role" not in columns:
            conn.execute(
                text("ALTER TABLE user ADD COLUMN role VARCHAR(20) DEFAULT 'user'")
            )
        if "active" not in columns:
            conn.execute(
                text("ALTER TABLE user ADD COLUMN active BOOLEAN DEFAULT 1")
            )
        if "created_at" not in columns:
            conn.execute(text("ALTER TABLE user ADD COLUMN created_at DATETIME"))
        if "updated_at" not in columns:
            conn.execute(text("ALTER TABLE user ADD COLUMN updated_at DATETIME"))
        conn.commit()


def _ensure_default_admin():
    from .models.user_model import User

    admin_email = os.environ.get("DEFAULT_ADMIN_EMAIL", "admin@gmail.com")
    admin_password = os.environ.get("DEFAULT_ADMIN_PASSWORD", "admin123")

    admin = User.query.filter_by(email=admin_email).first()
    hashed_pw = bcrypt.generate_password_hash(admin_password).decode("utf-8")

    if not admin:
        admin = User(email=admin_email, password=hashed_pw, role="admin", active=True)
        db.session.add(admin)
    else:
        admin.password = hashed_pw
        admin.role = "admin"
        admin.active = True
    db.session.commit()


def create_app():
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.config.from_object(Config)

    db.init_app(app)
    jwt.init_app(app)
    bcrypt.init_app(app)
    mail.init_app(app)
    mongo.init_app(app)
    limiter.init_app(app)

    with app.app_context():
        # Ensure model metadata is loaded before create_all.
        from .models import user_model  # noqa: F401

        db.create_all()
        _apply_sqlite_compat_migrations(app)
        _ensure_default_admin()

    from .routes.auth_routes import auth_bp
    from .routes.prediction_routes import prediction_bp
    from .routes.admin_routes import admin_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(admin_bp)

    @app.route('/')
    def home():
        return redirect(url_for('auth.login'))

    # Convenience aliases for common paths
    @app.route('/login')
    def login_alias():
        return redirect(url_for('auth.login'))

    @app.route('/register')
    def register_alias():
        return redirect(url_for('auth.register'))

    @app.route('/admin')
    def admin_alias():
        return redirect(url_for('admin.admin_home'))

    @app.route('/predict')
    def predict_alias():
        return redirect(url_for('prediction.predict_page'))

    # IDE preview compatibility: provide a placeholder for Vite client to avoid JS parse error
    @app.route('/@vite/client')
    def vite_client_placeholder():
        return "console.debug('Vite client placeholder');", 200, {"Content-Type": "application/javascript"}

    @app.errorhandler(429)
    def ratelimit_handler(e):
        if request.path.startswith("/auth/") or request.is_json:
            return jsonify({"error": "Too many requests. Please wait and try again."}), 429
        return e

    def _close_mongo_client_on_exit():
        # Close once at process exit (not per-request), to avoid breaking
        # Mongo access during normal request handling.
        try:
            if getattr(mongo, "cx", None):
                mongo.cx.close()
        except Exception:
            pass

    atexit.register(_close_mongo_client_on_exit)

    return app
