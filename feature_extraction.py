import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from transformers import pipeline
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv("processed_tweets.csv")

# Ensure 'created_at' is parsed as datetime if it's present in your dataset
if 'created_at' in df.columns:
    df['created_at'] = pd.to_datetime(df['created_at'])

### TEXT PREPROCESSING ###

# Preprocess the tweet text to create the cleaned_text column
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize and remove stopwords
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Join tokens back into a string
    return " ".join(tokens)

# Assuming the raw tweet text is in the 'text' or 'tweet_text' column
if 'text' in df.columns:
    df['cleaned_text'] = df['text'].apply(preprocess_text)
elif 'tweet_text' in df.columns:
    df['cleaned_text'] = df['tweet_text'].apply(preprocess_text)
else:
    print("Column 'text' or 'tweet_text' not found in the dataset!")

### TEXT FEATURES ###

# 1. TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=100)  # Adjust max_features as needed
tfidf_features = tfidf_vectorizer.fit_transform(df['cleaned_text']).toarray()

# Convert to DataFrame for better handling
tfidf_df = pd.DataFrame(tfidf_features, columns=tfidf_vectorizer.get_feature_names_out())

# 2. Embedding using BERT (Optional for semantic understanding)
# Load a BERT model pipeline for feature extraction
bert_pipeline = pipeline('feature-extraction', model='bert-base-uncased', tokenizer='bert-base-uncased')

# Generate embeddings (optional, comment if not needed for now)
# df['bert_embeddings'] = df['cleaned_text'].apply(lambda x: bert_pipeline(x)[0][0])  # [CLS] token embeddings

### METADATA FEATURES ###

# Normalize engagement metrics (assuming columns like 'favorite_count', 'retweet_count', 'reply_count' exist)
engagement_columns = ['favorite_count', 'retweet_count', 'reply_count']  # Adjust to your actual column names
if all(col in df.columns for col in engagement_columns):
    scaler = MinMaxScaler()
    engagement_features = scaler.fit_transform(df[engagement_columns])
    engagement_df = pd.DataFrame(engagement_features, columns=[f'normalized_{col}' for col in engagement_columns])
else:
    engagement_df = pd.DataFrame()

# Time-based patterns
if 'created_at' in df.columns:
    df['hour'] = df['created_at'].dt.hour  # Extract hour
    df['day_of_week'] = df['created_at'].dt.dayofweek  # Extract day of the week

### FINAL DATAFRAME ###

# Combine all features
final_df = pd.concat(
    [
        tfidf_df, 
        engagement_df, 
        df[['hour', 'day_of_week']] if 'created_at' in df.columns else pd.DataFrame()
    ],
    axis=1
)

# Save the processed features to a new CSV
output_file = "features_processed_tweets.csv"
final_df.to_csv(output_file, index=False)
print(f"Features saved successfully to {output_file}")
