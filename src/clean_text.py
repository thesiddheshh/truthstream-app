# src/preprocessing/clean_text.py

import re
import regex
from html import unescape
from nltk.corpus import stopwords
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load English stopwords once
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    import nltk
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words("english"))

def decode_html_entities(text):
    """Unescapes HTML entities like &amp; → &"""
    return unescape(text)

def remove_urls(text):
    """Removes URLs from text"""
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def remove_emails(text):
    """Removes email addresses"""
    return re.sub(r'\S+@\S+', '', text)

def remove_emojis(text):
    """Removes emojis using regex with proper Unicode ranges"""
    emoji_pattern = regex.compile(
        r"["
        r"\U0001F600-\U0001F64F"  # Emoticons
        r"\U0001F300-\U0001F5FF"  # Symbols & pictographs
        r"\U0001F680-\U0001F6FF"  # Transport & map symbols
        r"\U0001F1E0-\U0001F1FF"  # Flags (iOS)
        r"\U00002702-\U000027B0"  # Dingbats
        r"\U000024C2-\U0001F251"  # Miscellaneous Symbols
        r"\U0001F926-\U0001F937"  # Supplemental Symbols
        r"\U00002600-\U00002B55"  # Misc symbols
        r"\U000023CF"             # Eject
        r"\U0001F1F2-\U0001F1F4"  # Flags
        r"\U0001F1E6-\U0001F1FF"  # More flags
        r"\U0001F600-\U0001F64F"  # Emoticons again
        r"\U0001F680-\U0001F6FF"  # Transport again
        r"\U0000FE0F"             # Variation selector
        r"]+",
        flags=regex.UNICODE
    )
    return emoji_pattern.sub('', text)

def remove_special_characters(text):
    """Keeps only alphanumeric and basic punctuation"""
    return re.sub(r'[^a-zA-Z0-9\s.!,?@#$%&*+=\\/\-]', '', text)

def to_lowercase(text):
    """Convert text to lowercase"""
    return text.lower()

def normalize_whitespace(text):
    """Replaces multiple spaces/newlines with a single space"""
    return re.sub(r'\s+', ' ', text).strip()

def remove_stopwords(text):
    """Remove English stopwords"""
    tokens = text.split()
    return ' '.join([word for word in tokens if word not in STOPWORDS])

def clean_text(text):
    """
    Full cleaning pipeline for noisy user-generated or web-scraped text.
    
    Args:
        text (str): Raw input text
        
    Returns:
        str: Cleaned text ready for NLP
    """
    if not isinstance(text, str):
        logger.warning("Received non-string input. Converting to empty string.")
        return ""
        
    text = decode_html_entities(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = remove_emojis(text)
    text = remove_special_characters(text)
    text = to_lowercase(text)
    text = normalize_whitespace(text)
    text = remove_stopwords(text)
    
    return text.strip()

if __name__ == "__main__":
    sample_text = "Check out this site: https://example.com    😄 Visit us at info@example.org today! It’s absolutely AMAZING."
    cleaned = clean_text(sample_text)
    print("Original:", sample_text)
    print("Cleaned: ", cleaned)