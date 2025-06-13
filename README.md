````markdown
# TruthStream: Real-Time Fake News Detection Pipeline

---

## Overview

TruthStream is a comprehensive, real-time fake news detection pipeline that ingests streaming data from Reddit and NewsAPI, processes and classifies the content using transformer-based NLP models, verifies claims with knowledge graph integration, stores results in MongoDB, and presents insights through an interactive Streamlit dashboard. The system is designed for scalability, modularity, and extensibility, leveraging Kafka for reliable message streaming and supporting research and production use cases.

---

## Features

- **Real-time ingestion** from Reddit (PRAW) and NewsAPI
- **Apache Kafka** for distributed, fault-tolerant streaming
- **Robust preprocessing**: text cleaning, tokenization, feature extraction
- **Transformer-based classification** using BERT/DeBERTa for fake news detection
- **Knowledge graph verification** with Wikidata entity linking and SPARQL queries
- **MongoDB persistence** for flexible, scalable storage and retrieval
- **Streamlit dashboard** for visualization, filtering, and interactive analysis
- **Extensible modular architecture** facilitating experimentation and deployment
- Jupyter notebooks for exploratory data analysis and model training

---

## Project Structure

```plaintext
truthstream/
│
├── data/                          # Data repository
│   ├── raw/                       # Raw Reddit and NewsAPI JSON dumps
│   ├── labeled/                   # Cleaned & labeled datasets for training
│   └── sources.md                 # Documentation of data sources & schemas
│
├── models/                        # Serialized ML/NLP models
│   └── bert_fake_news_classifier.pkl
│
├── src/                           # Core source code
│   ├── ingestion/                 # Scripts for data ingestion and streaming
│   │   ├── reddit_stream.py       # Live Reddit streaming via PRAW API
│   │   ├── newsapi_fetch.py       # Real-time NewsAPI headline fetching
│   │   └── simulate_stream.py     # Simulated stream from local datasets
│   │
│   ├── kafka/                     # Kafka message producers and consumers
│   │   ├── kafka_producer.py      # Publishes ingested data to Kafka topics
│   │   └── kafka_consumer.py      # Consumes Kafka messages and runs pipeline
│   │
│   ├── preprocessing/             # Text cleaning and feature engineering
│   │   └── clean_text.py
│   │
│   ├── model/                     # Model training and inference
│   │   ├── train_model.py         # Fine-tune BERT/DeBERTa classifiers
│   │   └── predict.py             # Model inference interface
│   │
│   ├── verification/              # Knowledge graph-based verification
│   │   ├── verify_with_wikidata.py  # SPARQL queries for fact-checking
│   │   └── entity_linking.py      # Named entity recognition & linking
│   │
│   ├── storage/                   # Database interaction layer
│   │   ├── save_to_mongo.py       # MongoDB CRUD operations
│   │   └── schema_example.json    # Example MongoDB document schema
│   │
│   └── dashboard/                 # Streamlit app frontend
│       └── app.py                 # Dashboard UI and logic
│
├── notebooks/                     # Exploratory and training notebooks
│   ├── data_exploration.ipynb
│   └── model_training.ipynb
│
├── config/                        # Configuration files
│   └── config.yaml                # API keys, paths, Kafka/MongoDB settings
│
├── tests/                         # Unit and integration tests
│   ├── test_cleaning.py
│   ├── test_model.py
│   └── test_pipeline.py
│
├── requirements.txt               # Python dependencies
├── README.md                      # This documentation
├── architecture.png               # Pipeline architecture diagram
└── .env                           # Environment variables (excluded from VCS)
````

---

## Installation & Setup

### Prerequisites

* Python 3.8 or later
* Apache Kafka (local or managed cluster)
* MongoDB (local instance or cloud hosted)
* Reddit API credentials (via PRAW)
* NewsAPI key

### Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/yourusername/truthstream.git
cd truthstream
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Install required Python packages:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory and add your API keys and connection strings:

```dotenv
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=your_reddit_user_agent
NEWSAPI_KEY=your_newsapi_key
MONGODB_URI=your_mongodb_connection_string
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
```

5. Customize `config/config.yaml` if necessary to update paths or parameters.

---

## Usage

### Data Ingestion

* To stream Reddit posts live:

```bash
python src/ingestion/reddit_stream.py
```

* To fetch latest news headlines from NewsAPI:

```bash
python src/ingestion/newsapi_fetch.py
```

* To simulate a data stream from offline datasets:

```bash
python src/ingestion/simulate_stream.py
```

### Kafka Messaging

* Start the Kafka producer (publishes messages):

```bash
python src/kafka/kafka_producer.py
```

* Start the Kafka consumer (runs processing pipeline):

```bash
python src/kafka/kafka_consumer.py
```

### Model Training and Prediction

* Train or fine-tune the fake news classification model:

```bash
python src/model/train_model.py
```

* Perform prediction on new text inputs:

```bash
python src/model/predict.py --input "News article or headline text here"
```

### Knowledge Graph Verification

* Verify news items against Wikidata:

```bash
python src/verification/verify_with_wikidata.py --text "Claim or statement text"
```

### Dashboard

* Launch the interactive Streamlit dashboard:

```bash
streamlit run src/dashboard/app.py
```

Use the dashboard to monitor incoming news streams, view classification results, filter content, and inspect verification metadata.

---

## Testing

Run automated tests for pipeline components:

```bash
pytest tests/
```

---

## Architecture

![TruthStream Architecture](architecture.png)

**Description:**

1. **Data Sources:**

   * Reddit via PRAW streaming API
   * NewsAPI for latest news headlines

2. **Kafka Messaging:**

   * Producers publish raw JSON messages to Kafka topics
   * Consumers subscribe and apply pipeline processing

3. **Preprocessing:**

   * Text cleaning and normalization
   * Feature extraction for downstream classification

4. **Classification:**

   * BERT/DeBERTa-based transformer model detects fake vs real news

5. **Verification:**

   * Named entity recognition and linking
   * Wikidata SPARQL queries validate factual claims

6. **Storage:**

   * Classified and verified results stored in MongoDB

7. **Visualization:**

   * Streamlit dashboard visualizes streaming data and insights

---

## Technologies

* **Programming:** Python 3.8+
* **Streaming:** Apache Kafka
* **Database:** MongoDB
* **APIs:** PRAW (Reddit), NewsAPI
* **NLP Models:** HuggingFace Transformers (BERT, DeBERTa)
* **Verification:** Wikidata (SPARQL)
* **Dashboard:** Streamlit
* **Testing:** Pytest

---

## Contributing

Contributions are welcome. Please fork the repository and submit pull requests with clear descriptions. Open issues for bugs or feature requests.

---

## License

Distributed under the MIT License. See `LICENSE` for details.

---

## Contact

Your Name – [your.email@example.com](mailto:your.email@example.com)
GitHub: [https://github.com/yourusername/truthstream](https://github.com/yourusername/truthstream)

```
```
