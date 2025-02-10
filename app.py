import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
import tweepy

# Download stopwords once, using Streamlit's caching
@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    return stopwords.words('english')

# Load model and vectorizer once
@st.cache_resource
def load_model_and_vectorizer():
    with open('lr_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# Define sentiment prediction function
def predict_sentiment(text, model, vectorizer, stop_words):
    # Preprocess text
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)
    text = [text]
    text = vectorizer.transform(text)

    # Predict sentiment
    sentiment = model.predict(text)
    return "Negative" if sentiment == 0 else "Positive"

# Initialize Tweepy client
@st.cache_resource
def initialize_tweepy():
    # Replace these with your actual API keys and tokens
    api_key = "o161nrPYCRwlBHahBoaoZAWQy"
    api_key_secret = "LCM8OFFvXWFe8VVRxG6sT4d3OMWlVvWb4BezWCf5rrQrMMzG9K"
    bearer_token = "AAAAAAAAAAAAAAAAAAAAAMgMzAEAAAAAYXW32%2Fn9APVWhnRBhtB0QjgOxeo%3DEGEXSGmNrgBczHFh8DKec5KKUuyJcQQ1b9PrhtNxwCb5Kw9qiA"
    access_token = "1472765649034219523-PfvhEBxPOBMGF9tT0n5wWwtqIr2L8n"
    access_token_secret = "CzYIo9wIYlPh69Sken7E9p2ZOJjnVWSE4p3TvOBdZQqon"

    # Authenticate with Tweepy
    client = tweepy.Client(bearer_token=bearer_token,
                           consumer_key=api_key,
                           consumer_secret=api_key_secret,
                           access_token=access_token,
                           access_token_secret=access_token_secret)
    return client

# Function to create a colored card
def create_card(tweet_text, sentiment):
    color = "green" if sentiment == "Positive" else "red"
    card_html = f"""
    <div style="background-color: {color}; padding: 10px; border-radius: 5px; margin: 10px 0;">
        <h5 style="color: white;">{sentiment} Sentiment</h5>
        <p style="color: white;">{tweet_text}</p>
    </div>
    """
    return card_html

# Main app logic
def main():
    st.title("Twitter Sentiment Analysis")

    # Load stopwords, model, vectorizer, and Tweepy client only once
    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()
    client = initialize_tweepy()

    # User input: either text input or Twitter username
    option = st.selectbox("Choose an option", ["Input text", "Get tweets from user"])

    if option == "Input text":
        text_input = st.text_area("Enter text to analyze sentiment")
        if st.button("Analyze"):
            sentiment = predict_sentiment(text_input, model, vectorizer, stop_words)
            st.write(f"Sentiment: {sentiment}")

    elif option == "Get tweets from user":
        username = st.text_input("Enter Twitter username")
        if st.button("Fetch Tweets"):
            try:
                # Fetch user ID from username
                user = client.get_user(username=username)
                user_id = user.data.id

                # Fetch tweets from the user
                tweets = client.get_users_tweets(id=user_id, max_results=5)

                if tweets.data:
                    for tweet in tweets.data:
                        tweet_text = tweet.text
                        sentiment = predict_sentiment(tweet_text, model, vectorizer, stop_words)

                        # Create and display the colored card for the tweet
                        card_html = create_card(tweet_text, sentiment)
                        st.markdown(card_html, unsafe_allow_html=True)
                else:
                    st.write("No tweets found for this user.")
            except tweepy.TweepyException as e:
                st.write(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
