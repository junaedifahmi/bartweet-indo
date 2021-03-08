"""
Twitter data scraper. This script is used to scrap the most trending topic on twitter that posted in indonesia language.

Usage: scrap_twitter_hashtag.py OUTDIR [options]

Options:
    --max_trending=N  Maximum trending topic that want to be scraped [default: 5]
    --max_tweets=N  Maximum tweets want to get per trending [default: 100]

"""
import os
import re
from datetime import datetime
from pathlib import Path

import tweepy
import emoji
from dotenv import load_dotenv
from docopt import docopt
from tqdm import tqdm

load_dotenv(verbose=True)
WOEID = 23424846


def normalize_emoji(text):
    text = emoji.demojize(text)
    for x in re.findall(r':[a-zAz_-]+:', text):
        text = text.replace(x, f' {x} ')
    return text


def get_api():
    access_token = os.getenv('access_token')
    access_token_secret = os.getenv('access_token_secret')
    auth = tweepy.AppAuthHandler(access_token, access_token_secret)
    api = tweepy.API(auth)
    return api


def get_tweets(api, query, max=50):
    tweets = tweepy.Cursor(api.search, q=f"{query} -filter:retweets", lang='id', tweet_mode='extended').items(max)
    text = []
    for tweet in tweets:
        tweet = tweet.full_text
        tweet = tweet.replace('\n', ' ')                            # simple text clean
        tweet = re.sub('@[a-zA-Z_]+\s?', ' [USER] ', tweet)                   # Mask username
        tweet = re.sub('http(s)?:\/\/\w+.\w+(\/\w+)?', ' [URL] ', tweet)    # Mask url
        tweet = normalize_emoji(tweet)                              # Normalize emoji
        tweet = re.sub(r'\s+', ' ', tweet)
        tweet = tweet.strip()
        text.append(tweet)
    return text


def get_trends(api, items=10):
    res = api.trends_place(WOEID)[0]
    as_of = datetime.strptime(res['as_of'], '%Y-%m-%dT%H:%M:%SZ')
    trends = []
    for i in range(items):
        trends.append(res['trends'][i]['name'])

    return trends, as_of


def get_filename(root, as_of, trend):
    time = as_of.strftime('%Y%m%d_%H%M')
    trend = '_'.join(trend.split())
    return root.joinpath(f"{time}_{trend}.txt")


if __name__ == '__main__':
    args = docopt(__doc__)
    max_trending = int(args['--max_trending'])
    max_tweets = int(args['--max_tweets'])
    root = Path(args['OUTDIR']).absolute()
    root.mkdir(exist_ok=True, parents=True)

    api = get_api()
    trends, time = get_trends(api, max_trending)
    for trend in tqdm(trends, desc="Getting tweets for hashtag"):
        tweets = get_tweets(api, trend, max_tweets)
        with open(get_filename(root, time, trend), 'w', encoding='utf-8') as f:
            for tweet in tweets:
                print(f"{tweet}", file=f)
