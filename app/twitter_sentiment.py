"""
Twitter sentiment analysis for bitcoin with tweepy.
Run with: python3 app/twitter_sentiment.py
"""

import os
import asyncio
import tweepy
import matplotlib.pyplot as plt
import pandas as pd

consumer_key = os.getenv("CONSUMER_KEY")
consumer_secret = os.getenv("CONSUMER_SECRET")


auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)
api = tweepy.API(auth, wait_on_rate_limit= True, wait_on_rate_limit_notify = True)

# handle pagination
def limited_handled(cursor):
    while True:
        try:
            yield cursor.next()
        except tweepy.RateLimitError:
            print("Reached rate limit, waiting for 15 minutes")
            time.sleep(15 * 61)
        except StopIteration:
            break

query = "#bitcoin" + " -filter:retweets"
count = 10

search = limit_handled(tweepy.Cursor(api.search, q=query, tweet_mode='extended', lang='en', result_type='recent').items(count))


tweets = [results.full_text for results in search]

model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
hf_token = os.getenv("HF_TOKEN")

API_URL = f"https://api-inference.huggingface.co/models/{model}"
headers = {"Authorization": f"Bearer {hf_token}"}

def analaysis(data):
    payload = dict(inputs=data, options=dict(wait_for_model=True))
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

tweets_analysis = []
for tweet in tweets:
    try:
        sentiment_result = analysis(tweet)[0]
        top_sentiment = max(sentiment_result, key=lambda x: x['score'])
        tweets_analysis.append({'tweet': tweet, 'sentiment': top_sentiment['label']})
 
    except Exception as e:
        print(e)

# Load the data in a dataframe
pd.set_option('max_colwidth', None)
pd.set_option('display.width', 3000)
df = pd.DataFrame(tweets_analysis)
 
# Show a tweet for each sentiment
display(df[df["sentiment"] == 'Positive'].head(1))
display(df[df["sentiment"] == 'Neutral'].head(1))
display(df[df["sentiment"] == 'Negative'].head(1))

sentiment_counts = df.groupby(['sentiment']).size()
print(sentiment_counts)

fig = plt.figure(figsize=(6,6), dpi=100)
ax = plt.subplot(111)
sentiment_counts.plot.pie(ax=ax, autopct='%1.1f%%', startangle=270, fontsize=12, label="")