# src/kafka/kafka_consumer.py

from confluent_kafka import Consumer, KafkaException
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KafkaNewsConsumer:
    def __init__(self, bootstrap_servers='localhost:9092', group_id='truthstream-group'):
        """
        Initializes Kafka consumer for reading news/reddit posts.
        
        Args:
            bootstrap_servers (str): Comma-separated list of Kafka brokers
            group_id (str): Consumer group ID
        """
        self.conf = {
            'bootstrap.servers': bootstrap_servers,
            'group.id': group_id,
            'auto.offset.reset': 'earliest'
        }
        self.consumer = Consumer(self.conf)

    def consume_messages(self, topics=["raw_news"]):
        """
        Subscribes to topics and yields parsed messages.
        
        Args:
            topics (list): List of Kafka topics to consume from
        """
        self.consumer.subscribe(topics)

        try:
            while True:
                msg = self.consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    if "PARTITION_EOF" in msg.error().str():
                        logger.info(f"Reached end of partition: {msg.partition()}")
                    else:
                        raise KafkaException(msg.error())
                else:
                    # Parse message
                    key = msg.key().decode('utf-8') if msg.key() else None
                    value = json.loads(msg.value().decode('utf-8')) if msg.value() else None
                    yield {
                        "key": key,
                        "value": value,
                        "topic": msg.topic(),
                        "partition": msg.partition(),
                        "offset": msg.offset()
                    }

        except KeyboardInterrupt:
            logger.info("Stopping consumer...")
        finally:
            self.consumer.close()

if __name__ == "__main__":
    consumer = KafkaNewsConsumer()
    logger.info("Starting Kafka consumer...")

    for message in consumer.consume_messages():
        logger.info(f"Received message from {message['topic']} | Key: {message['key']}")
        print(json.dumps(message["value"], indent=2))