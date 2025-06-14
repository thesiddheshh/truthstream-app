# src/model/predict.py

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MisinformationPredictor:
    def __init__(self, model_path="bert-base-uncased", device=None):
        """
        Initializes a predictor using a BERT model.
        
        Args:
            model_path (str): Path or name of HuggingFace model
            device (str): 'cuda' or 'cpu'
        """
        self.model_path = model_path
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model from '{model_path}' on {self.device}")

        try:
            # Load tokenizer and model
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model = BertForSequenceClassification.from_pretrained(model_path).to(self.device)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def predict(self, text, threshold=0.5):
        """
        Predicts whether the given text is real or fake news.
        
        Args:
            text (str): Input text (title + content)
            threshold (float): Confidence threshold for class decision
            
        Returns:
            dict: {"label": "real/fake", "confidence": float}
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
            add_special_tokens=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            fake_prob = probs[0][1].item()
            label = "fake" if fake_prob >= threshold else "real"

        return {
            "label": label,
            "confidence": round(fake_prob, 4)
        }

if __name__ == "__main__":
    predictor = MisinformationPredictor()

    test_text = (
        "New Study Shows Climate Change Is Not Real "
        "Scientists from around the world have agreed that global warming is a hoax."
    )
    result = predictor.predict(test_text)
    print("Prediction:", result)