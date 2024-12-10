import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

# Load the datasets
cleaned_tweets_path = '/Users/aryankarki/Desktop/misc project/cleaned_nepse_tweets.csv'
features_tweets_path = '/Users/aryankarki/Desktop/misc project/features_processed_tweets.csv'


cleaned_tweets = pd.read_csv(cleaned_tweets_path)
features_tweets = pd.read_csv(features_tweets_path)

# Simulate sentiment labels for demonstration purposes
# Replace this with actual sentiment labels for real-world use
cleaned_tweets['sentiment'] = np.random.choice(['positive', 'neutral', 'negative'], size=len(cleaned_tweets))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    cleaned_tweets['cleaned_text'], cleaned_tweets['sentiment'], test_size=0.2, random_state=42
)

# Create a TF-IDF vectorizer and Random Forest classifier pipeline
pipeline = make_pipeline(
    TfidfVectorizer(max_features=5000),  # Limit to 5000 features
    RandomForestClassifier(n_estimators=100, random_state=42)
)

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
report = classification_report(y_test, y_pred)

# Output classification report
print("Classification Report:\n", report)

# Optional: Save the trained model for future use
import joblib
joblib.dump(pipeline, 'sentiment_analysis_model.pkl')

# Load model if needed later
# pipeline = joblib.load('sentiment_analysis_model.pkl')
