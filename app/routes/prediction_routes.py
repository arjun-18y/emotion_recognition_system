from flask import Blueprint, request, jsonify, render_template
from flask_jwt_extended import jwt_required
from ..services.model_service import predict_emotion
from ..services.ocr_service import extract_text_from_image
from ..utils.security import sanitize_text, allowed_text_file, allowed_image_file

prediction_bp = Blueprint("prediction", __name__, url_prefix="/predict")


@prediction_bp.route("/", methods=["GET"])
def predict_page():
    return render_template("predict.html")


@prediction_bp.route("/", methods=["POST"])
@jwt_required()
def predict():
    text = None
    if request.is_json:
        data = request.get_json()
        text = sanitize_text(data.get("text", ""))
    elif "file" in request.files:
        f = request.files["file"]
        if f and allowed_text_file(f.filename):
            text = f.read().decode("utf-8", errors="ignore")
            text = sanitize_text(text)
        elif f and allowed_image_file(f.filename):
            extracted, err = extract_text_from_image(f)
            if err:
                return jsonify({"error": err}), 400
            text = sanitize_text(extracted)
        else:
            return jsonify(
                {
                    "error": "Unsupported file type. Use .txt or image files (.png/.jpg/.jpeg/.webp)"
                }
            ), 400
    else:
        text = sanitize_text(request.form.get("text", ""))
    if not text:
        return jsonify({"error": "Empty input"}), 400
    emotion, confidence = predict_emotion(text)
    return jsonify(
        {
            "predicted_emotion": emotion,
            "confidence_scores": confidence,
            "chars": len(text),
            "long_text_mode": len(text) > 900,
        }
    )
