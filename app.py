import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Config
max_len = 250

# Cache models
@st.cache_resource
def load_svm_model():
    return pickle.load(open("svm_model.pkl", "rb"))

@st.cache_resource
def load_vectorizer():
    return pickle.load(open("tfidf_vectorizer.pkl", "rb"))

@st.cache_resource
def load_cnn_lstm_model():
    return load_model("cnn_lstm_model.h5")

@st.cache_resource
def load_tokenizer():
    return pickle.load(open("tokenizer.pkl", "rb"))

# Load models
svm_model = load_svm_model()
vectorizer = load_vectorizer()
cnn_lstm_model = load_cnn_lstm_model()
tokenizer = load_tokenizer()

# Streamlit UI
st.title("🎬 IMDB Sentiment Analysis")
st.write("Compare **SVM (TF-IDF)** and **CNN-LSTM (Deep Learning)** predictions on movie reviews!")
st.write("Min length of review should be at least 10 words.")

review = st.text_area("Enter your movie review:")

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review first.")
    elif len(review.split()) < 10:
        st.warning("Review is too short. Please enter at least 10 words.")
    else:
        st.write("No. of words in review:", len(review.split()))

        # SVM Prediction
        review_tfidf = vectorizer.transform([review])
        pred_svm = svm_model.predict(review_tfidf)[0]
        label_svm = "Positive 😀" if pred_svm == 1 else "Negative 😞"

        # CNN-LSTM Prediction
        with st.spinner("Predicting with CNN-LSTM..."):
            review_seq = tokenizer.texts_to_sequences([review])
            review_pad = pad_sequences(review_seq, maxlen=max_len, padding="post")
            pred_dl = (cnn_lstm_model.predict(review_pad)[0] > 0.5).astype("int32")
            label_dl = "Positive 😀" if pred_dl == 1 else "Negative 😞"

        # Results
        st.subheader("Results:")
        st.write(f"**SVM Prediction:** {label_svm}")
        st.write(f"**CNN-LSTM Prediction:** {label_dl}")
