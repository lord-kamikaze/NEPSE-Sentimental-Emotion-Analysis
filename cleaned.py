import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Make sure you have downloaded the stopwords if you haven't
nltk.download('stopwords')

# Load the cleaned data (assuming 'tweet_text' is the column containing raw tweets)
cleaned_df = pd.read_csv('cleaned_nepse_tweets.csv')

# Define the preprocessing function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove URLs (http, https, www)
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove any non-alphabetic characters (e.g., punctuation, numbers)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize the text
    tokens = text.split()
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Join the tokens back into a string
    cleaned_text = " ".join(tokens)
    return cleaned_text

# Check if 'tweet_text' exists and apply the preprocessing function
if 'tweet_text' in cleaned_df.columns:
    cleaned_df['cleaned_text'] = cleaned_df['tweet_text'].apply(preprocess_text)
else:
    print("Column 'tweet_text' not found!")

# Save the cleaned dataframe with the 'cleaned_text' column
cleaned_df.to_csv('cleaned_nepse_tweets.csv', index=False)

# Print the first few rows to verify
print(cleaned_df.head())
