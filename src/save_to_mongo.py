# src/storage/save_to_mongo.py

from pymongo import MongoClient
from dotenv import load_dotenv
import os
import logging
from datetime import datetime, timezone  # Add timezone importe

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MongoStorage:
    def __init__(self):
        """Initializes MongoDB connection using environment variables"""
        load_dotenv()

        self.mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        self.db_name = os.getenv("MONGO_DB", "truthstream_db")
        self.collection_name = os.getenv("MONGO_COLLECTION", "articles")

        try:
            self.client = MongoClient(self.mongo_uri)
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            logger.info(f"Connected to MongoDB: {self.mongo_uri}/{self.db_name}/{self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    def store_article(self, article_data):
        """
        Stores an article with metadata and model prediction.
        
        Args:
            article_data (dict): Dictionary containing:
                - id (str)
                - title (str)
                - text (str)
                - url (str)
                - subreddit (str)
                - timestamp (int)
                - source (str)
                - prediction (dict) from predict.py
                - verified (bool) from Wikidata check
        Returns:
            ObjectId: ID of inserted document
        """
        # Add timestamp for record-keeping
        article_data["stored_at"] = datetime.now(timezone.utc)

        try:
            result = self.collection.insert_one(article_data)
            logger.info(f"Stored article {result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            logger.error(f"Error storing article: {e}")
            return None

if __name__ == "__main__":
    # Test storage with sample data
    storage = MongoStorage()

    test_article = {
        "id": "abc123",
        "title": "New Climate Report Shows Record Melting Rates",
        "text": "A new study published today shows...",
        "url": "https://reddit.com/r/science/comments/abc123", 
        "subreddit": "science",
        "timestamp": 1698765432,
        "source": "reddit",
        "prediction": {
            "label": "real",
            "confidence": 0.96
        },
        "verified": True
    }

    inserted_id = storage.store_article(test_article)
    print("Inserted Document ID:", inserted_id)