import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import joblib
import matplotlib.pyplot as plt

nltk.download("stopwords")
nltk.download("wordnet")
df = pd.read_csv("dataset.csv")
df.drop(columns="ID",axis=1,inplace=True)
lemmatizer=WordNetLemmatizer()
stopwords=set(stopwords.words("english"))-{"not","no","nor","never"}

def preprocess(text):
    text=text.lower()
    text=text.translate(str.maketrans("","",string.punctuation))
    words=text.split()
    words=[lemmatizer.lemmatize(word) for word in words if word not in stopwords]
    return " ".join(words)
df["cleaned_text"]=df["Sentence"].apply(preprocess)
vectorizer =TfidfVectorizer(max_features=5000)
X=vectorizer.fit_transform(df["cleaned_text"]).toarray()
if X.shape[1]<5000:
    X=np.pad(X,[(0,0),(0,5000-X.shape[1])],mode="constant")
elif X.shape[1]>5000:
    print("Lower max_features to 5000")
X=X.reshape(X.shape[0],5000,1)
label_map={"negative": 0, "neutral":1,"positive":2}
inverse_label_map={0:"negative",1:"neutral",2:"positive"}
y=to_categorical(df["Sentiment"].map(label_map))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=Sequential([
    Conv1D(128,3,activation="relu",input_shape=(5000,1)),
    MaxPooling1D(2),
    Conv1D(64,3,activation="relu"),
    MaxPooling1D(2),
    Flatten(),
    Dense(64,activation="relu"),
    Dropout(0.5),
    Dense(3,activation="softmax"),
])

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
callbacks=[
    EarlyStopping(patience=3,restore_best_weights=True),
    ModelCheckpoint("best_model_tfidf.h5",save_best_only=True),
]
history=model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
)
print("\nTest Evaluation:")
loss,accuracy=model.evaluate(X_test,y_test)
print(f"test accuracy: {accuracy:.4f}")
model.save("sentiment_model_tfidf.h5")
joblib.dump(vectorizer,"tfidf_vectorizer.pkl")
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"],label="Train Accuracy",marker="o")
plt.plot(history.history["val_accuracy"],label="Val Accuracy",marker="o")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.subplot(1, 2,2)
plt.plot(history.history["loss"],label="Train Loss",marker="o")
plt.plot(history.history["val_loss"],label="Val Loss",marker="o")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
def predict_sentiment(text,model,vectorizer):
    cleaned_text=preprocess(text)
    vectorized=vectorizer.transform([cleaned_text]).toarray()
    if vectorized.shape[1]<5000:
        vectorized=np.pad(vectorized,[(0,0),(0,5000-vectorized.shape[1])],mode="constant")
    vectorized=vectorized.reshape(1,5000,1)
    prediction=model.predict(vectorized)
    return inverse_label_map[np.argmax(prediction)],float(np.max(prediction))
print("\nSample Predictions:")
model=load_model("sentiment_model_tfidf.h5")
vectorizer=joblib.load("tfidf_vectorizer.pkl")
test_samples=[
    "This product is absolutely amazing and works perfectly!",
    "Terrible experience, would not recommend to anyone.",
    "The service was average, nothing special.",
]
for text in test_samples:
    label, confidence=predict_sentiment(text, model, vectorizer)
    print(f"Text: {text}")
    print(f"Prediction: {label} (Confidence: {confidence:.2f})\n")
