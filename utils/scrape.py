import snscrape.modules.twitter as sntwitter
import pandas as pd

def scrape_tweets(query, max_tweets=500):
    tweets = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScrapper(query).get_items()):
        if i>= max_tweets:
            break
        tweets.append([tweet.date, tweet.id, tweet.content, tweet.user.username])
    return pd.DataFrame(tweets, columns=["date", "id", "content", "user"]
                        )
df = scrape_tweets('"bendera one piece" lang:id since:2024-07-01 until:2024-08-01', max_tweets=500)
df.to_csv("data/raw_tweets.csv", index=False)
