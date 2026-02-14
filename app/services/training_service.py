import os
import joblib
import csv
import re
from datetime import datetime
from .nlp_pipeline import preprocess_text
from ..extensions import mongo

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
ML_DIR = os.path.join(BASE_DIR, "ml")
DATA_DIR = os.path.join(BASE_DIR, "data")
MAX_TRAIN_CHUNK_CHARS = 450
MAX_CHUNKS_PER_SAMPLE = 12


def _ensure_dirs():
    if not os.path.exists(ML_DIR):
        os.makedirs(ML_DIR, exist_ok=True)
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)


def _normalize_key(key):
    return str(key or "").replace("\ufeff", "").strip().strip('"').strip("'").lower()


def _row_value(row, name):
    target = _normalize_key(name)
    for k, v in (row or {}).items():
        if _normalize_key(k) == target:
            return v
    return ""


def _split_for_training(text, target_chunk_chars=MAX_TRAIN_CHUNK_CHARS):
    text = str(text or "").strip()
    if not text:
        return []
    if len(text) <= target_chunk_chars:
        return [text]

    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", text) if p.strip()]
    if not parts:
        return [text[:target_chunk_chars]]

    chunks = []
    current = ""
    for part in parts:
        if not current:
            current = part
            continue
        if len(current) + 1 + len(part) <= target_chunk_chars:
            current = f"{current} {part}"
        else:
            chunks.append(current)
            current = part
            if len(chunks) >= MAX_CHUNKS_PER_SAMPLE:
                break
    if current and len(chunks) < MAX_CHUNKS_PER_SAMPLE:
        chunks.append(current)
    return chunks


def train_from_mongo():
    _ensure_dirs()
    mongo_available = True
    try:
        datasets = list(mongo.db.datasets.find())
    except Exception:
        mongo_available = False
        datasets = []

    if not datasets:
        # fallback to local CSV datasets if Mongo is unavailable/empty
        texts = []
        labels = []
        csv_files = []
        for name in os.listdir(DATA_DIR):
            if name.lower().endswith(".csv"):
                csv_files.append(os.path.join(DATA_DIR, name))

        for csv_path in csv_files:
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    t = str(_row_value(row, "text")).strip()
                    y = str(_row_value(row, "label")).strip() or "Neutral"
                    if t:
                        texts.append(t)
                        labels.append(y)

        if not texts:
            return {"error": "No dataset available"}
    else:
        texts = [d.get("text", "") for d in datasets]
        labels = [d.get("label", "Neutral") for d in datasets]

    expanded_texts = []
    expanded_labels = []
    for t, y in zip(texts, labels):
        for chunk in _split_for_training(t):
            clean = preprocess_text(chunk)
            if clean:
                expanded_texts.append(clean)
                expanded_labels.append(y)

    texts = expanded_texts
    labels = expanded_labels
    if not texts:
        return {"error": "No valid text content to train on"}

    # Lazy import scikit-learn to make it optional in environments without wheels
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    except Exception:
        return {"error": "Training unavailable: scikit-learn not installed in current environment"}

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(texts)
    y = labels

    label_counts = {}
    for label in y:
        label_counts[label] = label_counts.get(label, 0) + 1
    min_count = min(label_counts.values()) if label_counts else 0
    use_stratify = len(set(y)) > 1 and min_count >= 2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if use_stratify else None
    )
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro', zero_division=0
    )

    version = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    model_path = os.path.join(ML_DIR, f"emotion_model_{version}.pkl")
    vec_path = os.path.join(ML_DIR, f"vectorizer_{version}.pkl")
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)
    # Keep default paths updated so prediction can work even without Mongo metadata.
    joblib.dump(model, os.path.join(ML_DIR, "emotion_model.pkl"))
    joblib.dump(vectorizer, os.path.join(ML_DIR, "vectorizer.pkl"))

    # store metadata in MongoDB
    if mongo_available:
        try:
            models_col = mongo.db.models
            # archive previously active models correctly
            models_col.update_many({"status": "active"}, {"$set": {"status": "archived"}})
            models_col.insert_one({
                "version": version,
                "model_path": model_path,
                "vectorizer_path": vec_path,
                "metrics": {
                    "accuracy": acc,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1
                },
                "status": "active",
                "created_at": datetime.utcnow(),
                "dataset_count": len(texts),
                "chunked_training": True,
                "max_train_chunk_chars": MAX_TRAIN_CHUNK_CHARS,
            })
        except Exception:
            mongo_available = False

    result = {
        "version": version,
        "metrics": {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
    }
    if not mongo_available:
        result["warning"] = "Trained with local CSV fallback; MongoDB metadata unavailable"
    return result
