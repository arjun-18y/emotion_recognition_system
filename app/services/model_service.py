import os
import re
import joblib
from datetime import datetime
from .nlp_pipeline import preprocess_text
from ..extensions import mongo

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
ML_DIR = os.path.join(BASE_DIR, "ml")

_model = None
_vectorizer = None
_active_version = None
_fallback = True

EMOTION_LABELS = [
    "Admiration",
    "Amusement",
    "Anger",
    "Annoyance",
    "Approval",
    "Caring",
    "Confusion",
    "Curiosity",
    "Desire",
    "Disappointment",
    "Disapproval",
    "Disgust",
    "Embarrassment",
    "Excitement",
    "Fear",
    "Gratitude",
    "Grief",
    "Joy",
    "Love",
    "Nervousness",
    "Optimism",
    "Pride",
    "Realization",
    "Relief",
    "Remorse",
    "Sadness",
    "Surprise",
    "Neutral",
    "Crisis",
]

EMOTION_KEYWORDS = {
    "Admiration": ["admire", "respect", "inspired", "amazing", "impressive"],
    "Amusement": ["funny", "hilarious", "laugh", "lol", "lmao"],
    "Anger": ["angry", "furious", "rage", "hate", "outraged"],
    "Annoyance": ["annoyed", "irritated", "bothered", "frustrated"],
    "Approval": ["agree", "approve", "good job", "well done"],
    "Caring": ["care", "support", "help", "concerned", "protect"],
    "Confusion": ["confused", "unclear", "lost", "dont understand"],
    "Curiosity": ["curious", "wonder", "interested", "how", "why"],
    "Desire": ["desire", "i really want", "i strongly want", "need badly", "craving"],
    "Disappointment": ["disappointed", "let down", "upset with", "expected more", "forgotten", "no one wished", "no one remembered"],
    "Disapproval": ["disapprove", "wrong", "bad idea", "not okay"],
    "Disgust": ["disgusting", "gross", "nasty", "revolting"],
    "Embarrassment": ["embarrassed", "awkward", "ashamed", "cringe"],
    "Excitement": ["excited", "thrilled", "pumped", "cant wait"],
    "Fear": ["scared", "afraid", "fear", "terrified", "panic"],
    "Gratitude": ["thank", "grateful", "appreciate", "thanks"],
    "Grief": ["grief", "mourning", "loss", "heartbroken", "goodbye", "worst day of my life"],
    "Joy": ["happy", "joy", "glad", "delighted", "awesome", ":)", ":d"],
    "Love": ["love", "adore", "cherish", "affection"],
    "Nervousness": ["nervous", "anxious", "anxiety", "worried", "tense", "overthinking", "fear of abandonment"],
    "Optimism": ["hopeful", "optimistic", "positive", "it will work"],
    "Pride": ["proud", "accomplished", "achievement", "earned it"],
    "Realization": ["realized", "figured out", "now i see", "it hit me"],
    "Relief": ["relieved", "finally", "thank god", "what a relief"],
    "Remorse": ["sorry", "regret", "my fault", "guilty"],
    "Sadness": ["sad", "down", "unhappy", "depressed", "hurt", "alone", "forgotten", "pain", "silence", "distance", ":("],
    "Surprise": ["surprised", "shocked", "wow", "unexpected"],
    "Neutral": ["normal", "okay", "fine", "neutral", "alright"],
    "Crisis": [
        "kill myself",
        "suicide",
        "end my life",
        "want to die",
        "self harm",
        "harm myself",
        "die today",
        "no reason to live",
    ],
}

LEGACY_TO_EXPANDED = {
    "Happy": "Joy",
    "Sad": "Sadness",
    "Angry": "Anger",
    "Fear": "Fear",
    "Neutral": "Neutral",
}

CONTRAST_CUES = [
    "but",
    "however",
    "cant stand",
    "cannot stand",
    "leave me",
    "stay away",
]

NEGATIVE_BOOST = ["Anger", "Annoyance", "Disapproval", "Disappointment", "Sadness"]


def _load_active_model():
    global _model, _vectorizer, _active_version, _fallback
    if _model is not None and _vectorizer is not None:
        return
    models_col = mongo.db.models
    active = None
    try:
        active = models_col.find_one({"status": "active"})
    except Exception:
        active = None
    if active:
        model_path = active.get("model_path")
        vec_path = active.get("vectorizer_path")
        _active_version = active.get("version")
    else:
        model_path = os.path.join(ML_DIR, "emotion_model.pkl")
        vec_path = os.path.join(ML_DIR, "vectorizer.pkl")
        _active_version = "default"
    try:
        _model = joblib.load(model_path)
        _vectorizer = joblib.load(vec_path)
        _fallback = False
    except Exception:
        # Fallback to keyword-based classifier if model can't be loaded
        _model = None
        _vectorizer = None
        _fallback = True


def _contains_crisis_language(raw_text, clean_text):
    raw = (raw_text or "").lower()
    clean = (clean_text or "").lower()

    raw_keywords = EMOTION_KEYWORDS.get("Crisis", [])
    normalized_keywords = [
        "want kill",
        "kill",
        "suicide",
        "end life",
        "self harm",
        "harm",
        "want die",
        "die",
        "no reason live",
    ]
    return any(keyword in raw for keyword in raw_keywords) or any(
        keyword in clean for keyword in normalized_keywords
    )


def _apply_context_rules(raw_text, clean_text, scores):
    raw = (raw_text or "").lower()
    clean = (clean_text or "").lower()
    joined = f"{raw} {clean}"

    if any(cue in joined for cue in CONTRAST_CUES):
        for label in NEGATIVE_BOOST:
            scores[label] += 2
        if scores.get("Love", 0) > 0:
            scores["Love"] = max(0, scores["Love"] - 1)
        if scores.get("Joy", 0) > 0:
            scores["Joy"] = max(0, scores["Joy"] - 1)

    # In emotionally heavy long messages, generic "want/wish" clauses should not
    # dominate over clear distress markers.
    distress_markers = ["hurt", "alone", "forgotten", "anxiety", "overthinking", "goodbye", "heartbroken"]
    if any(marker in joined for marker in distress_markers):
        scores["Sadness"] += 2
        scores["Nervousness"] += 1
        if scores.get("Desire", 0) > 0:
            scores["Desire"] = max(0, scores["Desire"] - 2)

    return scores


def _predict_fallback(raw_text, clean_text):
    scores = {label: 0 for label in EMOTION_LABELS}

    if _contains_crisis_language(raw_text, clean_text):
        scores["Crisis"] = 1
        probs = [1.0 if label == "Crisis" else 0.0 for label in EMOTION_LABELS]
        return "Crisis", probs

    for emotion, keywords in EMOTION_KEYWORDS.items():
        for kw in keywords:
            if kw in clean_text:
                scores[emotion] += 1
    scores = _apply_context_rules(raw_text, clean_text, scores)
    # Choose max score, but default to Neutral if nothing matched.
    max_score = max(scores.values())
    if max_score == 0:
        pred = "Neutral"
    else:
        pred = max(scores.items(), key=lambda x: x[1])[0]
    probs = [scores[label] for label in EMOTION_LABELS]
    total = sum(probs)
    if total > 0:
        probs = [s / total for s in probs]
    else:
        probs = [1.0 if label == "Neutral" else 0.0 for label in EMOTION_LABELS]
    return pred, probs


def _fallback_scores(raw_text, clean_text):
    scores = {label: 0 for label in EMOTION_LABELS}
    if _contains_crisis_language(raw_text, clean_text):
        scores["Crisis"] = 1
        return scores
    for emotion, keywords in EMOTION_KEYWORDS.items():
        for kw in keywords:
            if kw in clean_text:
                scores[emotion] += 1
    return _apply_context_rules(raw_text, clean_text, scores)


def _predict_single(text):
    _load_active_model()
    clean = preprocess_text(text)
    if not _fallback:
        vec = _vectorizer.transform([clean])
        model_pred = _model.predict(vec)[0]
        pred = LEGACY_TO_EXPANDED.get(model_pred, model_pred)
        probs = []
        model_conf = 0.0
        try:
            probs = _model.predict_proba(vec)[0].tolist()
            if probs:
                model_conf = max(probs)
        except Exception:
            probs = []

        # Hybrid behavior: use keyword signal when model is uncertain.
        fallback_pred, fallback_probs = _predict_fallback(text, clean)
        fallback_signal = _fallback_scores(text, clean).get(fallback_pred, 0)
        if fallback_pred == "Crisis":
            pred = "Crisis"
            probs = fallback_probs
        elif pred == "Neutral" and fallback_pred != "Neutral":
            pred = fallback_pred
            probs = fallback_probs
        elif fallback_pred != "Neutral" and fallback_signal > 0 and model_conf < 0.60:
            pred = fallback_pred
            probs = fallback_probs
        elif fallback_pred == "Neutral" and model_conf < 0.60:
            pred = "Neutral"
            probs = [1.0 if label == "Neutral" else 0.0 for label in EMOTION_LABELS]

        if _contains_crisis_language(text, clean):
            pred = "Crisis"
            probs = [1.0 if label == "Crisis" else 0.0 for label in EMOTION_LABELS]
    else:
        pred, probs = _predict_fallback(text, clean)
    # Log prediction
    try:
        mongo.db.predictions.insert_one({
            "text": text,
            "clean_text": clean,
            "predicted": pred,
            "probs": probs,
            "model_version": _active_version,
            "created_at": datetime.utcnow()
        })
    except Exception:
        pass
    return pred, probs


def _split_long_text(text, target_chunk_chars=450):
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", text) if p.strip()]
    if not parts:
        return [text]

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
    if current:
        chunks.append(current)
    return chunks


def _aggregate_chunk_predictions(predictions):
    if not predictions:
        return "Neutral", [1.0 if label == "Neutral" else 0.0 for label in EMOTION_LABELS]

    score_by_label = {label: 0.0 for label in EMOTION_LABELS}
    for pred, probs, weight in predictions:
        if probs and len(probs) == len(EMOTION_LABELS):
            for i, label in enumerate(EMOTION_LABELS):
                score_by_label[label] += float(probs[i]) * weight
        else:
            score_by_label[pred] += 1.0 * weight

    best = max(score_by_label.items(), key=lambda x: x[1])[0]
    total = sum(score_by_label.values())
    if total > 0:
        probs = [score_by_label[label] / total for label in EMOTION_LABELS]
    else:
        probs = [1.0 if label == "Neutral" else 0.0 for label in EMOTION_LABELS]
    return best, probs


def predict_emotion(text):
    text = (text or "").strip()
    if not text:
        return "Neutral", [1.0 if label == "Neutral" else 0.0 for label in EMOTION_LABELS]

    if len(text) <= 900:
        return _predict_single(text)

    chunks = _split_long_text(text, target_chunk_chars=450)
    # Keep bounded work for very large inputs.
    chunks = chunks[:120]
    chunk_predictions = []
    for chunk in chunks:
        pred, probs = _predict_single(chunk)
        chunk_predictions.append((pred, probs, max(len(chunk), 1)))
    return _aggregate_chunk_predictions(chunk_predictions)
