# src/ingestion/newsapi_fetch.py

import time
from newsapi import NewsApiClient
from dotenv import load_dotenv
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to Python path if needed
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Now import local modules
from src.kafka.kafka_producer import KafkaProducer

class NewsAPIFetcher:
    def __init__(self, api_key):
        """
        Initializes NewsAPI client for fetching news articles.
        
        Args:
            api_key (str): NewsAPI API key
        """
        self.newsapi = NewsApiClient(api_key=api_key)

    def fetch_latest_news(self, sources=None, categories=None, limit=5):
        """
        Fetches latest news articles from specified sources or categories.
        
        Args:
            sources (list): List of news source IDs (e.g., ['bbc-news', 'cnn'])
            categories (list): List of news categories (e.g., ['business', 'technology'])
            limit (int): Maximum number of articles to fetch
            
        Returns:
            list: List of news articles
        """
        try:
            if sources:
                articles = self.newsapi.get_top_headlines(sources=",".join(sources), page_size=limit)["articles"]
            elif categories:
                articles = []
                for category in categories:
                    articles += self.newsapi.get_top_headlines(category=category, page_size=limit // len(categories))["articles"]
            else:
                articles = self.newsapi.get_top_headlines(page_size=limit)["articles"]

            # Filter out duplicates and format articles
            seen_titles = set()
            formatted_articles = []
            for article in articles:
                title = article.get("title", "")
                if title not in seen_titles:
                    seen_titles.add(title)
                    formatted_articles.append({
                        "id": article["url"],
                        "title": article["title"],
                        "text": article["description"] or "",
                        "url": article["url"],
                        "source": article["source"]["name"],
                        "timestamp": int(time.time()),
                        "source_type": "newsapi"
                    })
                    if len(formatted_articles) >= limit:
                        break

            return formatted_articles

        except Exception as e:
            logger.error(f"Failed to fetch news: {e}")
            return []

def fetch_and_send_news(producer, topics=['raw_news'], limit=5):
    """
    Fetches news articles and sends them to Kafka.
    
    Args:
        producer (KafkaProducer): Kafka producer instance
        topics (list): List of Kafka topics to send to
        limit (int): Number of articles to fetch
    """
    load_dotenv()
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        raise ValueError("NEWSAPI_KEY not found in .env")

    news_fetcher = NewsAPIFetcher(api_key)
    articles = news_fetcher.fetch_latest_news(categories=["technology", "science"], limit=limit)

    for article in articles:
        print(f"[PRODUCER] Sending {article['id']} to topic '{topics[0]}'")
        producer.send_message(topic=topics[0], key=article["id"], value=article)

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Initialize Kafka producer
    bootstrap_servers = os.getenv("BOOTSTRAP_SERVERS", "localhost:9092")
    producer = KafkaProducer(bootstrap_servers=bootstrap_servers)

    # Fetch and send news
    fetch_and_send_news(producer, topics=['raw_news'], limit=5)

    # Flush and exit
    producer.flush()
