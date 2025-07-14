**📰 Fake News Detector using NLP & Machine Learning
📌 Project Overview
This project is a machine learning solution to classify whether a news headline is fake or real using Natural Language Processing (NLP) techniques. The model is trained on a real-world dataset and deployed using Streamlit for live interaction.**

**🎯 Objective**
To build a text classification model that can detect fake news headlines.

To apply NLP preprocessing such as stopword removal and lemmatization.

To transform text into numerical data using TF-IDF Vectorization.

To deploy the model into an interactive web app using Streamlit.

**📂 Dataset Information**
The dataset was taken from Kaggle's Fake News Challenge.

It includes headline, author, and label columns.

The label 1 represents Fake, while 0 represents Real news.

For this project, only author and title columns were merged and used for training.

**🧠 Project Workflow**
🔹 1. Data Cleaning
All missing values were filled with a space " " instead of dropping rows.

Two text-based columns author and title were merged to create a new column called content.

**🔹 2. Text Preprocessing**
Converted all text to lowercase.

Removed special characters and digits using regular expressions.

Applied tokenization to split text into words.

Removed stopwords (common words that do not add meaning).

Used lemmatization to reduce words to their base form (e.g., "running" → "run").

**🔹 3. Feature Extraction**
Used TF-IDF Vectorization to convert the cleaned text into numerical features.

Limited to the top 5000 features for performance.

**🔹 4. Model Building**
Split the data into 80% training and 20% testing sets.

Trained a Logistic Regression model using the training set.

Evaluated the model on the test set and achieved high accuracy.

**🔹 5. Model Saving**
The trained model and vectorizer were saved as .pkl files using pickle for future use or deployment.

**🔹 6. Web App Deployment**
Built an interactive web app using Streamlit where users can input any news headline.

On clicking “Check,” the app preprocesses the text, vectorizes it, and then classifies it as either:

✅ Real News

❌ Fake News

**🌟 Key Features**
Full end-to-end project from data preprocessing to deployment.

User-friendly Streamlit interface to test model predictions.

Robust NLP techniques for clean and meaningful data preparation.

Light-weight model using Logistic Regression for fast execution.

Easily scalable and extendable for future improvements.

**⚠️ Challenges Faced**
Cleaning the text effectively while maintaining meaning.

Ensuring that NLTK stopwords and lemmatizer libraries are properly downloaded.

Managing paths when saving/loading .pkl files across environments.

Streamlit throwing errors on invalid inputs or incorrect model path.

**🚀 Future Enhancements**
Replace Logistic Regression with Transformer-based models like BERT for better accuracy.

Use full article text (not just titles and authors) for better context.

Add visualization dashboards showing common fake news words or trends.

Deploy the app using Hugging Face Spaces, Render, or AWS Lambda for wider access.

**💡 What I Learned**
Hands-on experience with NLP techniques such as preprocessing, TF-IDF, and lemmatization.

Solidified understanding of the machine learning pipeline.

Learned how to deploy a real-time ML app using Streamlit.

Understood the importance of handling data imbalance, cleaning, and model evaluation.

**🙌 Final Thoughts**
This project not only helped me understand how fake news can be automatically detected using machine learning and NLP but also gave me the confidence to take a project from scratch to production. It reflects the ability to build scalable AI applications and is a valuable addition to my machine learning portfolio.
