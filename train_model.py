import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import urllib.request

# Configuration
DATA_URL = "https://raw.githubusercontent.com/sahanaramesh09/SMS-Spam-Classification/master/SMSSpamCollection.csv"
DATA_PATH = "sms_spam_app/data/spam.csv"
MODEL_PATH = "sms_spam_app/models/spam_classifier.pkl"

def download_data():
    if not os.path.exists(DATA_PATH):
        print(f"Downloading data from {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
        print("Download complete.")
    else:
        print("Data already exists.")

def train_model():
    # Load data
    # The dataset typically has no headers or specific columns.
    # We'll try loading it and inspecting.
    # Inspecting the raw file from URL suggests it might have columns like 'Class', 'sms' or v1, v2.
    # Let's try reading it.
    try:
        df = pd.read_csv(DATA_PATH, encoding='latin-1')
        print("Columns found:", df.columns)
        
        # Adjust column names if needed based on common versions of this dataset
        if 'v1' in df.columns and 'v2' in df.columns:
            df = df.rename(columns={'v1': 'label', 'v2': 'message'})
        elif 'Class' in df.columns and 'sms' in df.columns:
            df = df.rename(columns={'Class': 'label', 'sms': 'message'})
        
        # Ensure we have label and message
        if 'label' not in df.columns or 'message' not in df.columns:
            # Fallback: maybe no header?
            df = pd.read_csv(DATA_PATH, header=None, names=['label', 'message'], encoding='latin-1')
        
        print(f"Dataset shape: {df.shape}")
        print(df.head())

        X = df['message']
        y = df['label']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Build pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('clf', MultinomialNB())
        ])

        # Train
        print("Training model...")
        pipeline.fit(X_train, y_train)

        # Evaluate
        predictions = pipeline.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, predictions))
        print(classification_report(y_test, predictions))

        # Save
        joblib.dump(pipeline, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
        
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    download_data()
    train_model()
