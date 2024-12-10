

import tweepy

# Example of API setup
client = tweepy.Client(bearer_token="AAAAAAAAAAAAAAAAAAAAALKLxQEAAAAAG6VAQS7AOSxE0d3%2BsA8MJSdTAGQ%3DGgv23zwJNhVUMbV3DGsQwpjNzlDgJwqas0SFmqrIBrngdafhGf")

# Fetch tweets
query = "#NEPSE -is:retweet lang:en"
response = client.search_recent_tweets(query=query, max_results=100, tweet_fields=["created_at"])

# Now `tweets` can refer to the `response.data`
tweets = response.data

# Print the tweets or process them
for tweet in tweets:
    print(tweet.text)
