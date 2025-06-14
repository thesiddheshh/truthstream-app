# src/ingestion/simulate_stream.py

import json
import time
import random
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_sample_data(filepath="data/raw/sample_posts.json"):
    """
    Loads sample posts from a local file.
    
    Args:
        filepath (str): Relative path to the sample JSON file
        
    Returns:
        list: List of sample articles/posts
    """
    try:
        full_path = Path(__file__).parent.parent.parent / filepath
        with open(full_path, "r") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} sample posts")
        return data
    except Exception as e:
        logger.error(f"Failed to load sample data: {e}")
        return []

def stream_simulated_data(sample_data, interval=2, limit=5):
    """
    Yields simulated data at intervals.
    
    Args:
        sample_data (list): List of sample posts
        interval (float): Seconds between each message
        limit (int): Max number of messages to yield
    
    Yields:
        dict: A single post
    """
    count = 0
    while count < limit:
        item = random.choice(sample_data)
        logger.info(f"[SIMULATED] Sending: {item['title']}")
        yield item
        time.sleep(interval)
        count += 1

if __name__ == "__main__":
    # Load sample data
    sample_data = load_sample_data()

    if not sample_data:
        logger.warning("No sample data loaded. Exiting.")
    else:
        # Simulate streaming
        for post in stream_simulated_data(sample_data, interval=3, limit=5):
            print(json.dumps(post, indent=2))