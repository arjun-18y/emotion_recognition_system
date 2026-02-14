import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

texts = [
    "I am very happy today",
    "I am feeling sad",
    "This makes me angry",
    "I am scared",
    "I feel normal"
]

labels = ["Happy", "Sad", "Angry", "Fear", "Neutral"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)

joblib.dump(model, "emotion_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
