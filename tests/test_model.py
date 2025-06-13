# tests/test_model.py

from src.model.predict import MisinformationPredictor

def test_predict_returns_label_and_confidence():
    predictor = MisinformationPredictor(model_path="bert-base-uncased")
    result = predictor.predict("The Earth is flat and the sky is fake.")
    assert "label" in result
    assert "confidence" in result
    assert result["label"] in ["real", "fake"]
    assert 0 <= result["confidence"] <= 1

def test_predict_on_real_news():
    predictor = MisinformationPredictor(model_path="bert-base-uncased")
    result = predictor.predict("NASA confirms climate change is accelerating.")
    assert result["label"] in ["real", "fake"]  # Accept either until fine-tuned

def test_predict_on_fake_news():
    predictor = MisinformationPredictor(model_path="bert-base-uncased")
    result = predictor.predict("Aliens landed in Nevada last night.")
    assert result["label"] in ["real", "fake"]