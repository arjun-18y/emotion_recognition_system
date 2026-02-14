import csv
import io
import os
from flask import Blueprint, jsonify, render_template, request, current_app
from flask_jwt_extended import jwt_required
from ..extensions import mongo
from ..utils.security import allowed_file, role_required

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_DIR = os.path.join(BASE_DIR, "data")


def _normalize_key(key):
    return str(key or "").replace("\ufeff", "").strip().strip('"').strip("'").lower()


def _row_value(row, name):
    target = _normalize_key(name)
    for k, v in (row or {}).items():
        if _normalize_key(k) == target:
            return v
    return ""


@admin_bp.route("/dashboard", methods=["GET"])
def admin_home():
    return render_template("admin_dashboard.html")


@admin_bp.route("/bootstrap", methods=["GET"])
@jwt_required()
@role_required("admin")
def admin_bootstrap():
    models = []
    mongo_error = None
    try:
        raw_models = list(mongo.db.models.find().sort("created_at", -1))
        for m in raw_models:
            models.append(
                {
                    "version": m.get("version"),
                    "status": m.get("status"),
                    "accuracy": (m.get("metrics") or {}).get("accuracy"),
                }
            )
    except Exception:
        mongo_error = "MongoDB is unavailable. Admin features are limited."

    return jsonify({"models": models, "mongo_error": mongo_error})


@admin_bp.route("/dataset", methods=["POST"])
@jwt_required()
@role_required("admin")
def upload_dataset():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    files = [f for f in request.files.getlist("file") if f and f.filename]
    if not files:
        return jsonify({"error": "No file"}), 400

    all_records = []
    processed_files = 0
    invalid_files = []

    try:
        for f in files:
            if not allowed_file(f.filename):
                invalid_files.append(f.filename)
                continue

            content = f.read().decode("utf-8", errors="ignore")
            reader = csv.DictReader(io.StringIO(content))
            file_records = []
            for row in reader:
                text = str(_row_value(row, "text")).strip()
                label = str(_row_value(row, "label")).strip()
                if text and label:
                    file_records.append({"text": text, "label": label})

            if file_records:
                all_records.extend(file_records)
                processed_files += 1

        if not all_records:
            return jsonify({"error": "No valid rows found. Expected CSV columns: text,label"}), 400

        fallback_msg = ""
        try:
            mongo.db.datasets.insert_many(all_records)
        except Exception:
            os.makedirs(DATA_DIR, exist_ok=True)
            fallback_csv = os.path.join(DATA_DIR, "uploaded_dataset_fallback.csv")
            write_header = not os.path.exists(fallback_csv)
            with open(fallback_csv, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["text", "label"])
                if write_header:
                    writer.writeheader()
                writer.writerows(all_records)
            fallback_msg = " MongoDB unavailable, saved locally to data/uploaded_dataset_fallback.csv."

        msg = f"Dataset uploaded ({len(all_records)} rows from {processed_files} file(s)).{fallback_msg}"
        if invalid_files:
            msg += f". Skipped invalid file(s): {', '.join(invalid_files)}"
        return jsonify({"message": msg})
    except Exception:
        return jsonify({"error": "Failed to store dataset. MongoDB may be unavailable."}), 503


@admin_bp.route("/retrain", methods=["POST"])
@jwt_required()
@role_required("admin")
def retrain():
    from ..services.training_service import train_from_mongo

    try:
        result = train_from_mongo()
    except Exception as exc:
        if current_app.debug:
            return jsonify({"error": f"Retraining failed: {exc}"}), 500
        return jsonify({"error": "Retraining failed due to a server error"}), 500
    status = 200 if "error" not in result else 400
    return jsonify(result), status
