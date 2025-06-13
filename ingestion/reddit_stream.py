# src/ingestion/reddit_stream.py

import praw
from dotenv import load_dotenv
import os
import json
import time

def connect_to_reddit():
    """Connects to Reddit API using credentials from .env"""
    load_dotenv()  # Load environment variables

    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT")
    )
    return reddit

def stream_reddit_posts(subreddits=["worldnews", "technology", "science"], limit=5):
    """
    Streams recent posts from specified subreddits.
    
    Args:
        subreddits (list): List of subreddit names
        limit (int): Number of posts to fetch per batch
    
    Yields:
        dict: A dictionary containing post metadata and text
    """
    reddit = connect_to_reddit()

    for subreddit_name in subreddits:
        subreddit = reddit.subreddit(subreddit_name)

        print(f"Fetching latest {limit} posts from r/{subreddit_name}")
        for submission in subreddit.new(limit=limit):
            post_data = {
                "id": submission.id,
                "title": submission.title,
                "text": submission.selftext,
                "url": submission.url,
                "subreddit": subreddit_name,
                "timestamp": submission.created_utc,
                "source": "reddit"
            }
            yield post_data
            time.sleep(0.5)  # Rate limiting

if __name__ == "__main__":
    for post in stream_reddit_posts():
        print(json.dumps(post, indent=2))