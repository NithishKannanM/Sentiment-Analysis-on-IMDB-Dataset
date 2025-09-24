🎬 IMDB Sentiment Analysis: SVM vs. CNN-LSTM
Project Overview

This project performs binary sentiment analysis (positive/negative) on the IMDB movie reviews dataset. The goal is to compare the performance of a classical machine learning model (SVM) with a deep learning model (CNN-LSTM).

The workflow includes:

Text preprocessing (cleaning, lemmatization, stopword removal)

Feature engineering (TF-IDF for SVM, tokenization + padding for CNN-LSTM)

Model training with hyperparameter tuning

Evaluation using accuracy, precision, recall, F1-score, and confusion matrices

Performance comparison using visualization

1. Data Cleaning & Preprocessing

Steps applied to raw IMDB reviews:

Convert text to lowercase

Remove HTML tags, URLs, numbers, punctuation

Expand contractions (e.g., "can't" → "cannot")

Tokenize sentences into words

Remove stopwords

Lemmatize words to their dictionary base form

2. Feature Engineering
For SVM (classical ML)

TF-IDF Vectorizer: transforms text into weighted numerical features

Limited vocabulary (max_features = 5000)

Stopwords removed automatically

For CNN-LSTM (deep learning)

Tokenizer: converts text to sequences of integers

Vocabulary size: max_words = 10000

Sequences padded to length max_len = 250

Unknown words replaced with <OOV>

3. Model Architectures
Support Vector Machine (SVM)

LinearSVC classifier

Input: TF-IDF features

Output: Binary classification (positive/negative)

CNN-LSTM

Embedding Layer: 64-dim word embeddings

Conv1D + MaxPooling1D: learns local n-gram patterns

LSTM Layer: captures sequential dependencies

Dropout + L2 Regularization: prevents overfitting

Optimizer: Adam (learning rate = 1e-4)

Callbacks: ReduceLROnPlateau + EarlyStopping

4. Evaluation Metrics

The models were evaluated using:

Accuracy

Precision

Recall

F1-score

Confusion Matrix for error analysis

5. Results
Model	Accuracy	Precision	Recall	F1-score
SVM	0.88	0.85	0.90	0.87
CNN-LSTM	0.92	0.91	0.93	0.92

Observation: CNN-LSTM consistently outperforms SVM, as it captures both local n-grams and long-range dependencies in reviews.

 6. Conclusion

SVM with TF-IDF provides a strong baseline but ignores word order and context.

CNN-LSTM leverages embeddings + sequential modeling, achieving higher accuracy and F1-score.

Proper preprocessing, regularization, and early stopping greatly improved the deep learning model’s performance.

 7. Future Work

Integrate pretrained embeddings (e.g., GloVe, FastText)

Experiment with Bidirectional LSTMs or Transformer models (BERT, DistilBERT)

Perform hyperparameter tuning (batch size, learning rate, architecture depth)

Explore data augmentation for handling imbalanced datasets

Deploy as a Streamlit app for real-time sentiment prediction
