# tests/test_pipeline.py

import os
import tempfile
import json
from src.preprocessing.clean_text import clean_text
from src.model.predict import MisinformationPredictor
from src.verification.verify_with_wikidata import WikidataFactChecker
from src.storage.save_to_mongo import MongoStorage

def test_full_pipeline_on_sample_data():
    """Runs a single article through all stages"""
    sample_article = {
        "id": "test123",
        "title": "NASA Confirms Climate Change Is Real",
        "text": "Scientists say the planet is warming due to human activity.",
        "source": "reddit"
    }

    # Clean text
    cleaned = clean_text(sample_article["text"])
    assert "http" not in cleaned
    assert "  " not in cleaned

    # Predict
    predictor = MisinformationPredictor(model_path="bert-base-uncased")
    prediction = predictor.predict(cleaned)
    assert prediction["label"] in ["real", "fake"]

    # Verify with Wikidata (optional)
    verifier = WikidataFactChecker()
    verified = verifier.verify_claim("Q2784", "P31", "Q35696")  # NASA → instance of → Government Agency
    assert isinstance(verified, bool)

    # Store in MongoDB
    storage = MongoStorage()
    stored_id = storage.store_article({
        **sample_article,
        "cleaned_text": cleaned,
        "prediction": prediction,
        "verified": verified
    })
    assert stored_id is not None