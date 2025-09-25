import streamlit as st
import pickle
import nltk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

# Load your models
svm_model = pickle.load(open("svm_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
cnn_lstm_model = load_model("cnn_lstm_model.h5")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

# Config
max_len = 250

# Title
st.title("🎬 IMDB Sentiment Analysis")
st.write("Compare **SVM (TF-IDF)** and **CNN-LSTM (Deep Learning)** predictions on movie reviews!")
st.write("Min length of review should be at least 10 words.")

# User input
review = st.text_area("Enter your movie review:")

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review first.")
    elif len(review.split()) < 10:
        st.warning("Review is too short. Please enter at least 10 words.")
    else:
        # SVM Prediction
        review_tfidf = vectorizer.transform([review])
        pred_svm = svm_model.predict(review_tfidf)[0]
        label_svm = "Positive 😀" if pred_svm == 1 else "Negative 😞"

        # CNN-LSTM Prediction
        review_seq = tokenizer.texts_to_sequences([review])
        review_pad = pad_sequences(review_seq, maxlen=max_len, padding="post")
        pred_dl = (cnn_lstm_model.predict(review_pad) > 0.5).astype("int32")[0][0]
        label_dl = "Positive 😀" if pred_dl == 1 else "Negative 😞"

        # Results
        st.subheader("Results:")
        st.write(f"**SVM Prediction:** {label_svm}")
        st.write(f"**CNN-LSTM Prediction:** {label_dl}")
