# src/kafka/kafka_producer.py

from confluent_kafka import Producer
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KafkaProducer:
    def __init__(self, bootstrap_servers='localhost:9092'):
        """
        Initializes Kafka producer with given bootstrap servers.
        
        Args:
            bootstrap_servers (str): Comma-separated list of Kafka brokers
        """
        self.conf = {
            'bootstrap.servers': bootstrap_servers,
        }
        self.producer = Producer(self.conf)

    def delivery_report(self, err, msg):
        """Callback for Kafka message delivery status"""
        if err:
            logger.error(f'Message delivery failed: {err}')
        else:
            logger.info(f'Message delivered to {msg.topic()} [{msg.partition()}]')

    def send_message(self, topic, key, value):
        """
        Sends a JSON message to the specified Kafka topic.
        
        Args:
            topic (str): Kafka topic name
            key (str): Message key (e.g., post ID)
            value (dict): Message body (as dictionary)
        """
        try:
            self.producer.produce(
                topic,
                key=key.encode('utf-8'),
                value=json.dumps(value).encode('utf-8'),
                callback=self.delivery_report
            )
            self.producer.poll(0)
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    def flush(self):
        """Waits for all messages to be delivered"""
        self.producer.flush()

if __name__ == "__main__":
    # Test producer by sending a sample Reddit post
    test_post = {
        "id": "abc123",
        "title": "New Climate Report Shows Record Melting Rates",
        "text": "A new study published today shows...",
        "url": "https://reddit.com/r/science/comments/abc123", 
        "subreddit": "science",
        "timestamp": 1698765432,
        "source": "reddit"
    }

    producer = KafkaProducer()
    producer.send_message(topic="raw_news", key=test_post["id"], value=test_post)
    producer.flush()