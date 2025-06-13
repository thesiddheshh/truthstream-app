# src/integrate_pipeline.py

import time
import json
import logging
from threading import Thread

# Import modules
from ingestion.reddit_stream import stream_reddit_posts
from kafka import KafkaProducer, KafkaConsumer
from preprocessing.clean_text import clean_text
from model.predict import MisinformationPredictor
from verification.entity_linking import EntityLinker
from verification.verify_with_wikidata import WikidataFactChecker
from storage.save_to_mongo import MongoStorage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global setup
BOOTSTRAP_SERVERS = 'localhost:9092'
KAFKA_TOPIC_RAW = 'raw_news'

# Initialize modules
def init_producer():
    return KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        key_serializer=lambda k: json.dumps(k).encode('utf-8')
    )

def init_consumer():
    return KafkaConsumer(
        KAFKA_TOPIC_RAW,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        key_deserializer=lambda m: json.loads(m.decode("utf-8")) if m else None,
        auto_offset_reset='earliest',
        enable_auto_commit=False,
        group_id='truthstream-group',
        consumer_timeout_ms=10000,
        api_version=(2, 5, 0)
    )

producer = init_producer()
consumer = None
predictor = MisinformationPredictor()
linker = EntityLinker()
verifier = WikidataFactChecker()
storage = MongoStorage()

def start_producer():
    """Runs the Reddit stream and sends data to Kafka"""
    print("[PIPELINE] Starting Reddit stream and Kafka producer...")
    for post in stream_reddit_posts(subreddits=["worldnews", "technology", "science"], limit=5):
        print(f"[PRODUCER] Sending {post['id']} to topic '{KAFKA_TOPIC_RAW}'")
        try:
            producer.send(KAFKA_TOPIC_RAW, key=post["id"], value=post)
        except Exception as e:
            logger.error(f"[PRODUCER] Failed to send message: {e}")
    producer.flush()

def process_message(raw_post):
    """Process one message through the pipeline"""
    try:
        print(f"\n[PROCESSING] Article ID: {raw_post['id']}")
        
        # Step 1: Clean Text
        raw_text = raw_post.get("text", "")
        title = raw_post.get("title", "")
        full_text = title + " " + raw_text
        cleaned_text = clean_text(full_text)
        print("[CLEANED TEXT]", cleaned_text[:100] + "...")

        # Step 2: Predict Fake/Real
        prediction = predictor.predict(cleaned_text)
        print("[PREDICTION]", prediction)

        # Step 3: Link Entities
        entities = linker.extract_entities(cleaned_text)
        linked_entities = {
            entity: linker.link_entity_to_wikidata(entity)
            for entity in entities
        }
        print("[LINKED ENTITIES]", linked_entities)

        # Step 4: Fact Check (Optional: Add custom claims later)
        verified_claims = []
        if prediction["label"] == "fake":
            try:
                # Example hardcoded check: "Barack Obama" -> occupation -> "President"
                verified_claims = verifier.verify_article_entities(linked_entities)
                print("[VERIFIED CLAIMS]", verified_claims)
            except Exception as e:
                print("[ERROR] Wikidata verification failed:", str(e))

        # Step 5: Build final record
        processed_data = {
            **raw_post,
            "cleaned_text": cleaned_text,
            "prediction": prediction,
            "entities": linked_entities,
            "verified_claims": verified_claims
        }

        # Step 6: Save to MongoDB
        stored_id = storage.store_article(processed_data)
        print(f"[STORED] Document ID: {stored_id}")

        return processed_data

    except Exception as e:
        print("[ERROR] Processing failed:", str(e))
        return None

def start_consumer():
    """Consumes messages from Kafka and runs full pipeline"""
    global consumer
    print("[PIPELINE] Starting Kafka consumer and processing pipeline...")

    retry_attempts = 5
    retry_delay = 3  # seconds

    for attempt in range(1, retry_attempts + 1):
        try:
            if consumer is None:
                logger.info(f"Initializing Kafka consumer (Attempt {attempt}/{retry_attempts})")
                consumer = init_consumer()
            
            print("[CONSUMER] Waiting for messages...")
            while True:
                msg = consumer.poll(timeout_ms=1000)
                if msg is None:
                    continue
                if msg.topic() != KAFKA_TOPIC_RAW:
                    logger.warning(f"[CONSUMER] Unexpected topic: {msg.topic()}")
                    continue

                raw_post = msg.value
                logger.info(f"[CONSUMER] Received message with key: {msg.key}")
                process_message(raw_post)

            break  # Exit loop if successful
        except Exception as e:
            logger.error(f"[CONSUMER] Error: {e}")
            logger.info(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            consumer = None

if __name__ == "__main__":
    print("[PIPELINE] Starting TruthStream End-to-End Pipeline...\n")

    # Run producer in background thread
    producer_thread = Thread(target=start_producer)
    producer_thread.start()

    # Small delay to ensure messages arrive
    time.sleep(5)

    # Run consumer in main thread
    start_consumer()

    print("\n[PIPELINE] Pipeline execution completed.")