# Twitter/X Sentiment Analysis Model
Project Overview:
This project aims to build a sentiment analysis model for Twitter (X) using machine learning techniques. The model is capable of analyzing tweets fetched either by a Twitter username or by direct input and classifying them into sentiment categories. The final model was implemented using Logistic Regression.

Technologies Used:
Pandas (import pandas as pd):
Used for data manipulation and handling the dataset of 162,980 tweets.

Natural Language Toolkit (nltk) (import nltk):
Utilized for text preprocessing, including tokenization, stemming, and removing stopwords.

Regular Expressions (re):
Applied for text cleaning, such as removing special characters and unwanted symbols from the tweet data.

Stopwords Removal and Stemming:

Stopwords (from nltk.corpus import stopwords): Words that are filtered out during the text preprocessing step (like "the", "is", etc.).
Stemming (from nltk.stem.porter import PorterStemmer): Simplifying words to their root form for better sentiment classification.
TF-IDF Vectorizer (from sklearn.feature_extraction.text import TfidfVectorizer):
This method converts text into numerical representations by considering the importance of a word in a document relative to its occurrence in the entire dataset.

Logistic Regression Model (from sklearn.linear_model import LogisticRegression):
This model was chosen as the final classifier for the project, providing high accuracy for sentiment classification.

XGBoost Classifier (from xgboost import XGBClassifier):
Used as an alternative model to Logistic Regression during experimentation to compare results.

Train-Test Split (from sklearn.model_selection import train_test_split):
The dataset was divided into training and testing sets to validate the model's performance.

Model Evaluation:

Accuracy Score (from sklearn.metrics import accuracy_score): Used to measure the model's performance.
Classification Report (from sklearn.metrics import classification_report): Generated detailed insights on the precision, recall, and F1-score of the model.
Final Model:
After experimentation with different models, Logistic Regression was chosen for its balance between accuracy and performance. The model effectively classifies the sentiment of tweets into categories (positive, negative, or neutral).

Web Application:
The model is integrated into a web application that provides users with two functionalities:

Fetching tweets by username: The app allows users to input a Twitter username, and it will fetch the latest tweets from that user.
Typing tweets for analysis: Users can also type in their own text, which will be analyzed by the model for sentiment
