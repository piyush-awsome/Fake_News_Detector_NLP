import streamlit as st
import pickle
import nltk
import re
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary nltk data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ✅ Use absolute path to load model & vectorizer
model_path = r"C:\Piyush Tech\Python\NLP_PROJECTS\fake_news_model.pkl"
vectorizer_path = r"C:\Piyush Tech\Python\NLP_PROJECTS\vectorizer.pkl"

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Model or vectorizer file not found. Please check the path.")
    st.stop()

# Preprocessing setup
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = word_tokenize(text)
    new_words = []
    for word in tokens:
        if word not in stop_words and word.isalnum():
            new_words.append(lemmatizer.lemmatize(word))
    return " ".join(new_words)

# 🧠 Streamlit UI
st.title("📰 Fake News Detector (Absolute Path Version)")
input_text = st.text_input("Enter news headline:")

if st.button("Check"):
    cleaned = clean(input_text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)

    if prediction[0] == 1:
        st.error("❌ FAKE NEWS!")
    else:
        st.success("✅ REAL NEWS!")

''' To run this file we will use this one
'''