import re

try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = None
except ImportError:
    nlp = None

EMOJI_MAP = {
    ":)": "happy",
    ":(": "sad",
    ":D": "happy",
    ":P": "playful",
}


def preprocess_text(text):
    if not text:
        return ""
    text = text.lower()
    for k, v in EMOJI_MAP.items():
        text = text.replace(k, f" {v} ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if nlp is None:
        return text
    doc = nlp(text)
    tokens = [t.lemma_ for t in doc if not t.is_stop and not t.is_punct]
    return " ".join(tokens)
