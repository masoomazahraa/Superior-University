from flask import Flask, request, jsonify
import numpy as np
import joblib
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download("stopwords")
nltk.download("wordnet")
app=Flask(__name__)
model=load_model("sentiment_model_tfidf.h5")
vectorizer=joblib.load("tfidf_vectorizer.pkl")
lemmatizer=WordNetLemmatizer()
stop_words=set(stopwords.words("english"))-{"not","no","nor","never"}
inverse_label_map={0: "negative", 1: "neutral", 2: "positive"}
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)
def predict_sentiment(text):
    cleaned_text = preprocess_text(text)
    vectorized = vectorizer.transform([cleaned_text]).toarray()

    if vectorized.shape[1] != 5000:
        vectorized = np.pad(vectorized, [(0, 0), (0, 5000 - vectorized.shape[1])], mode="constant")
    vectorized = vectorized.reshape(1, 5000, 1)
    prediction = model.predict(vectorized)
    predicted_class = np.argmax(prediction)
    confidence_score = np.max(prediction)
    return inverse_label_map[predicted_class], round(float(confidence_score), 3)
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Welcome to the Sentiment Classifier API. Use POST /classify with a JSON like {'text': 'your input here'}."
    })
@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400
    predicted_class, confidence_score = predict_sentiment(data["text"])
    return jsonify({
        "prediction": predicted_class,
        "confidence_score": confidence_score
    })
if __name__ == "__main__":
    app.run(debug=True)
