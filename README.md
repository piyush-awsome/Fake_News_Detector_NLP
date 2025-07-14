# Fake_News_Detector_NLP

**Full Project Explanation: Fake News Detector using NLP (Jupyter Notebook)
This notebook walks through the complete end-to-end pipeline for building a Fake News Detection system using Natural Language Processing (NLP) and Logistic Regression. The model is trained on a labeled dataset and saved for use in a web app later.**

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
from nltk.tokenize import word_tokenize
import os

✅ Purpose: Load all the necessary libraries for:

Data manipulation (pandas, numpy)

Text preprocessing (nltk, re)

Model training & evaluation (scikit-learn)

Saving model (pickle)

Step 2: Load Dataset

new_df = pd.read_csv('train.csv')
new_df.head()
new_df.shape

📘 Dataset: train.csv with 20800 rows and 5 columns

🔹 Step 3: Handle Missing Values

new_df.isnull().sum()
new_df = new_df.fillna(' ')

Missing values are filled with blank spaces instead of being dropped.

✅ Rationale: Data is not too large, so we retain all rows and preserve context.

🔹 Step 4: Feature Engineering

new_df['content'] = new_df['author'] + " " + new_df['title']

📌 Created a new column: content, which is a combination of author and title — useful for semantic understanding.

🔹 Step 5: Preprocessing Text

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

Defined a stopword list and lemmatizer for preprocessing.

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = word_tokenize(text)
    cleaned = []
    for word in tokens:
        if word not in stop_words and word.isalnum():
            lemma = lemmatizer.lemmatize(word, pos='v')
            cleaned.append(lemma)
    return " ".join(cleaned)

🔍 What this function does:

Lowercases all text

Removes non-alphabet characters

Tokenizes using word_tokenize

Removes stopwords

Applies lemmatization (converts words to their base form)

new_df['cleaned_content'] = new_df['content'].apply(preprocess)

✅ Cleaned text is stored in a new column cleaned_content

🔹 Step 6: Vectorization using TF-IDF

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(new_df['cleaned_content']).toarray()
y = new_df['label']

Transformed cleaned text into numerical format using TF-IDF

Limited to top 5000 features to reduce dimensionality

X → Features (vectors), y → Target labels (0: Real, 1: Fake)

🔹 Step 7: Split Data into Train & Test Sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

80% for training, 20% for testing

Set random_state=42 for reproducibility

🔹 Step 8: Train Logistic Regression Model

model = LogisticRegression()
model.fit(X_train, y_train)

✅ Trained a simple yet effective binary classification model using Logistic Regression

🔹 Step 9: Model Evaluation

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {acc:.4f}")

Made predictions on test data

Calculated accuracy as the evaluation metric

✅ Printed final performance (e.g., 0.95 → 95% accuracy)

🔹 Step 10: Save Model & Vectorizer

with open('fake_news_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

📦 Saved:

fake_news_model.pkl: the trained model

vectorizer.pkl: the TF-IDF vectorizer
✅ These files are reused in the Streamlit app (app.py) for live predictions

🎯 Summary
✅ Collected & cleaned raw text data

✅ Applied NLP preprocessing: stopword removal, lemmatization, TF-IDF

✅ Built a Logistic Regression model

✅ Achieved high accuracy

✅ Created serialized .pkl files for deployment

