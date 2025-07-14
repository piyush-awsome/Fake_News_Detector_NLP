📰 Fake News Detector using NLP and Logistic Regression
Complete Project with Code and Explanation – Line by Line

This project detects fake news using a Logistic Regression model trained on TF-IDF vectorized text. It includes full preprocessing using NLTK and is deployment-ready via a Streamlit app. Below is the complete .ipynb code with detailed explanations for every step.

🔹 Import Required Libraries
python
Copy
Edit
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
numpy, pandas: For data manipulation.

re: Regular expressions for text cleaning.

nltk: To handle stopwords, tokenization, and lemmatization.

sklearn: For vectorization, model training, splitting, and accuracy.

pickle: To save model and vectorizer for deployment.

🔹 Load and Explore the Dataset
python
Copy
Edit
new_df = pd.read_csv('train.csv')
new_df.head()
Loads the dataset train.csv and displays the top 5 rows.

python
Copy
Edit
new_df.shape  # 20800 rows and 5 columns
Shows the dataset contains 20800 records and 5 columns.

python
Copy
Edit
new_df.isnull().sum()
Checks for missing/null values in each column.

🔹 Handle Missing Values
python
Copy
Edit
new_df = new_df.fillna(' ')
new_df.isnull().sum()
Fills all missing values with blank strings.

This retains the original dataset size rather than dropping rows.

🔹 Combine Author and Title into One Column
python
Copy
Edit
new_df['content'] = new_df['author'] + " " + new_df['title']
new_df.head()
Combines the author and title columns into a single text column named content.

This is the text we'll later clean and use to predict fake or real news.

🔹 Setup Preprocessing Tools
python
Copy
Edit
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
Loads a set of English stopwords and initializes the lemmatizer.

🔹 Define Text Preprocessing Function
python
Copy
Edit
def preprocess(text):
    text = text.lower()  # Lowercase conversion
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove special characters and digits
    tokens = word_tokenize(text)  # Tokenize the text
    cleaned = []
    for word in tokens:
        if word not in stop_words and word.isalnum():
            lemma = lemmatizer.lemmatize(word, pos='v')  # Lemmatize the word
            cleaned.append(lemma)
    return " ".join(cleaned)
This function:

Converts text to lowercase

Removes unwanted characters

Tokenizes the sentence

Removes stopwords and punctuation

Lemmatizes each word to its base form

🔹 Apply Preprocessing to Dataset
python
Copy
Edit
new_df['cleaned_content'] = new_df['content'].apply(preprocess)
print(new_df['cleaned_content'][1])
Applies the preprocess() function to every row in the content column.

The cleaned result is stored in a new column called cleaned_content.

🔹 Convert Text to Vectors Using TF-IDF
python
Copy
Edit
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(new_df['cleaned_content']).toarray()
y = new_df['label']
Transforms the cleaned text into numerical features using TF-IDF.

Limits the vocabulary to the top 5000 words.

X: Features for the model, y: Labels (0 = Real, 1 = Fake)

🔹 Split the Data into Training and Test Sets
python
Copy
Edit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Splits the dataset into 80% training and 20% testing data.

🔹 Train the Logistic Regression Model
python
Copy
Edit
model = LogisticRegression()
model.fit(X_train, y_train)
Creates and trains a logistic regression model using the training data.

🔹 Evaluate the Model
python
Copy
Edit
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {acc:.4f}")
Predicts on the test set and prints the model’s accuracy.

🔹 Save the Model and Vectorizer
python
Copy
Edit
with open('fake_news_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("🎉 Model and vectorizer saved successfully.")
Saves both the model and vectorizer into .pkl files using pickle.

These files will be used later for prediction in the frontend app.

✅ Summary of Pipeline
Loaded and cleaned the dataset

Combined text columns into one

Preprocessed and lemmatized the text

Vectorized the text using TF-IDF

Trained a logistic regression model

Evaluated and saved the model for deployment

You can now use this trained model and vectorizer in your Streamlit web app (app.py) to create a real-time Fake News Detector!
