# TruthStream: Real-Time Fake News Detection Pipeline

TruthStream is a production-ready, real-time fake news detection pipeline that integrates live data ingestion, preprocessing, classification using BERT-based models, verification through Wikidata, and visualization via a Streamlit dashboard. Built for robustness and scalability, the system combines Kafka, MongoDB, and RESTful APIs to deliver accurate, explainable insights into media credibility.

---

## 📁 Project Structure

```
truthstream/
│
├── data/
│   ├── raw/                     # Raw JSON from Reddit and NewsAPI
│   ├── labeled/                 # Cleaned & labeled training data
│   └── sources.md              # Documentation of data origins
│
├── models/
│   └── bert_fake_news_classifier.pkl
│
├── src/
│   ├── ingestion/
│   │   ├── reddit_stream.py       # Live Reddit posts via PRAW
│   │   ├── newsapi_fetch.py       # Fetch headlines via NewsAPI
│   │   └── simulate_stream.py     # Local stream simulation
│   │
│   ├── kafka/
│   │   ├── kafka_producer.py      # Streams data to Kafka topic
│   │   └── kafka_consumer.py      # Applies NLP pipeline on stream
│   │
│   ├── preprocessing/
│   │   └── clean_text.py          # Tokenization, stopwords, lemmatization
│   │
│   ├── model/
│   │   ├── train_model.py         # Trains and saves BERT classifier
│   │   └── predict.py             # Loads model and predicts
│   │
│   ├── verification/
│   │   ├── verify_with_wikidata.py
│   │   └── entity_linking.py      # (Optional) Named entity matching
│   │
│   ├── storage/
│   │   ├── save_to_mongo.py       # Inserts verified output to MongoDB
│   │   └── schema_example.json
│   │
│   └── dashboard/
│       └── app.py                 # Streamlit frontend
│
├── notebooks/
│   ├── data_exploration.ipynb
│   └── model_training.ipynb
│
├── config/
│   └── config.yaml                # API keys, Kafka topics, model paths
│
├── tests/
│   ├── test_cleaning.py
│   ├── test_model.py
│   └── test_pipeline.py
│
├── requirements.txt
├── architecture.png               # Pipeline diagram
├── README.md
└── .env                           # API keys (excluded from git)
```

---

## 🚀 Features

- Real-time ingestion from Reddit and NewsAPI
- Modular NLP pipeline: cleaning, classification, verification
- Kafka streaming with producer-consumer architecture
- BERT-based fake news classifier with 90%+ test accuracy
- Optional entity linking with Wikidata for fact-checking
- Streamlit dashboard for user interaction and analytics
- MongoDB integration for storing verified results
- Notebooks for EDA and model experimentation
- Unit tests for pipeline stability

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/truthstream.git
cd truthstream
```

### 2. Set Up Python Environment

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file in the root directory:

```env
REDDIT_CLIENT_ID=your_id
REDDIT_SECRET=your_secret
NEWSAPI_KEY=your_key
MONGO_URI=mongodb://localhost:27017
```

Edit `config/config.yaml` to match your local paths and settings.

---

## 🧠 Model Training (Optional)

Use the provided notebook or script to train:

```bash
python src/model/train_model.py --data data/labeled/fake_news.csv --model_out models/bert_fake_news_classifier.pkl
```

---

## 🛰️ Running the Pipeline

### 1. Start Kafka and Zookeeper

Make sure Kafka and Zookeeper are running locally.

```bash
# Start Zookeeper
bin/zookeeper-server-start.sh config/zookeeper.properties

# Start Kafka
bin/kafka-server-start.sh config/server.properties
```

### 2. Run Data Producers

```bash
python src/ingestion/reddit_stream.py
python src/ingestion/newsapi_fetch.py
```

### 3. Start Kafka Consumer + Classifier

```bash
python src/kafka/kafka_consumer.py
```

This script applies the NLP pipeline and pushes the output to MongoDB.

---

## 📊 Run the Dashboard

```bash
streamlit run src/dashboard/app.py
```

---

## 🧪 Running Tests

```bash
pytest tests/
```

---

## 🧱 Tech Stack

- **Data Ingestion:** PRAW (Reddit), NewsAPI
- **Streaming:** Apache Kafka
- **Preprocessing:** NLTK, spaCy
- **Modeling:** BERT (Hugging Face Transformers)
- **Verification:** Wikidata SPARQL queries
- **Database:** MongoDB
- **Frontend:** Streamlit
- **Orchestration:** YAML, CLI scripts
- **Testing:** Pytest

---

## 📌 Pipeline Architecture

![Architecture](architecture.png)

1. Live articles/comments ingested from Reddit & NewsAPI
2. Data streamed via Kafka topics
3. Kafka Consumer runs:
   - Text cleaning
   - BERT-based fake/real prediction
   - Optional fact-checking via Wikidata
4. Final output stored in MongoDB
5. Streamlit dashboard queries and displays results

---

## 📂 MongoDB Schema

```json
{
  "source": "reddit" | "newsapi",
  "timestamp": "2025-06-13T15:30:00Z",
  "text": "...",
  "cleaned_text": "...",
  "prediction": "FAKE" | "REAL",
  "confidence": 0.92,
  "verified_entities": [...],
  "wikidata_verification": true
}
```

---

## 📎 Future Enhancements

- Multilingual support (IndicBERT, XLM-R)
- Source trustworthiness scoring
- SHAP-based model explainability
- FastAPI backend integration
- Live alerts on misinformation trends

---

## 📜 License

MIT License © 2025 Siddheshwar Wagawad

---

## 👤 Contact

For questions or contributions, contact:  
**siddhwagawad@gmail.com**  
LinkedIn | GitHub | Portfolio

