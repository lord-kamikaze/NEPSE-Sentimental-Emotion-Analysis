import joblib

# Load the saved model
loaded_model = joblib.load('sentiment_analysis_model.pkl')

# Check if the model loaded successfully
print("Model loaded successfully!")
# Example text for sentiment prediction
example_tweets = [
    "I love the stock market today!",
    "The market crash was devastating.",
    "Not sure about the market trends."
]

# Predict sentiment
predictions = loaded_model.predict(example_tweets)
print("Predicted Sentiments:", predictions)
