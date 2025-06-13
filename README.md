================================================================================
                               TruthStream
        Real-time Fake News Detection Pipeline using Kafka, MongoDB,
        BERT, and Knowledge-based Verification with Streamlit Dashboard
================================================================================

🧠 ABOUT THE PROJECT
---------------------
TruthStream is an end-to-end, real-time fake news detection pipeline that:
  • Streams live data from Reddit and NewsAPI
  • Cleans and classifies content using BERT-based NLP models
  • Verifies facts via external sources like Wikidata
  • Stores results in MongoDB
  • Presents analysis via an interactive Streamlit dashboard

Designed for robust misinformation detection in dynamic online environments.

📁 PROJECT STRUCTURE
---------------------

truthstream/
│
├── data/                            # Collected and labeled datasets
│   ├── raw/                         # Raw Reddit & NewsAPI JSON
│   ├── labeled/                     # Cleaned, labeled data for training
│   └── sources.md                   # Documentation of data sources
│
├── models/                          # Trained BERT model
│   └── bert_fake_news_classifier.pkl
│
├── src/                             # Core pipeline modules
│   ├── ingestion/                   # Stream and fetch data
│   │   ├── reddit_stream.py         # Live Reddit data via PRAW
│   │   ├── newsapi_fetch.py         # NewsAPI integration
│   │   └── simulate_stream.py       # Local dataset stream simulator
│
│   ├── kafka/                       # Kafka producers & consumers
│   │   ├── kafka_producer.py
│   │   └── kafka_consumer.py
│
│   ├── preprocessing/               # Cleaning and text processing
│   │   └── clean_text.py
│
│   ├── model/                       # ML model training & inference
│   │   ├── train_model.py
│   │   └── predict.py
│
│   ├── verification/                # Knowledge-based verification
│   │   ├── verify_with_wikidata.py
│   │   └── entity_linking.py
│
│   ├── storage/                     # MongoDB interactions
│   │   ├── save_to_mongo.py
│   │   └── schema_example.json
│
│   └── dashboard/                   # Streamlit-based UI
│       └── app.py
│
├── notebooks/                       # EDA and experimentation
│   ├── data_exploration.ipynb
│   └── model_training.ipynb
│
├── config/
│   └── config.yaml                  # API keys, paths, credentials
│
├── tests/                           # Unit tests
│   ├── test_cleaning.py
│   ├── test_model.py
│   └── test_pipeline.py
│
├── architecture.png                 # Full pipeline architecture diagram
├── requirements.txt
├── README.md
└── .env                             # Environment variables (not committed)


🛠️ TECH STACK
--------------
- Python 3.10+
- Kafka (Apache)
- MongoDB
- HuggingFace Transformers (BERT/DeBERTa)
- PRAW (Reddit API)
- NewsAPI
- Streamlit
- Wikidata SPARQL
- FAISS (optional: RAG)
- Docker (optional for containerization)


📈 ARCHITECTURE OVERVIEW
-------------------------

[ External APIs ] --> [ Ingestion (Reddit/NewsAPI) ] --> [ Kafka Producer ]
                                                ↓
                                    [ Kafka Topic (news_stream) ]
                                                ↓
                      [ Kafka Consumer ] --> [ Preprocessing ] --> [ BERT Model ]
                                                ↓                        ↓
                                [ Knowledge Verification ]        [ Label: REAL/FAKE ]
                                                ↓                        ↓
                                      [ MongoDB Storage ] <-- [ Metadata + Results ]
                                                ↓
                                     [ Streamlit Dashboard (app.py) ]


🚀 SETUP INSTRUCTIONS
----------------------

1. 📦 Install dependencies:
