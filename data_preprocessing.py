import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK resources (run this once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
nepse_tweets = pd.read_csv("nepse_tweets.csv")  # Replace with your dataset's path

# Initialize the lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function for preprocessing
def preprocess_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # Remove mentions and hashtags
    text = re.sub(r"@\w+|#\w+", '', text)
    # Remove numbers and special characters
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # Join tokens back into a string
    return ' '.join(tokens)

# Apply preprocessing to the text column
nepse_tweets['cleaned_text'] = nepse_tweets['text'].apply(preprocess_text)

# Save the cleaned dataset (optional)
nepse_tweets.to_csv("cleaned_nepse_tweets.csv", index=False)

# Display a preview of the cleaned data
print(nepse_tweets[['text', 'cleaned_text']].head())
