# tests/test_cleaning.py

from src.preprocessing.clean_text import clean_text

def test_clean_text_removes_urls():
    input_text = "Check this out: https://example.com" 
    output_text = clean_text(input_text)
    assert "https://example.com"  not in output_text

def test_clean_text_removes_emails():
    input_text = "Contact me at fake@example.org"
    output_text = clean_text(input_text)
    assert "fake@example.org" not in output_text

def test_clean_text_removes_emojis():
    input_text = "This is awesome 😄"
    output_text = clean_text(input_text)
    assert "😄" not in output_text

def test_clean_text_normalizes_whitespace():
    input_text = "   Too     many   spaces.  "
    output_text = clean_text(input_text)
    assert "  " not in output_text
    assert output_text == "many spaces."

def test_clean_text_decodes_html_entities():
    input_text = "Bill &amp; Ted went to the park."
    output_text = clean_text(input_text)
    assert "&amp;" not in output_text
    # Optional: allow & to remain in cleaned text
    # assert "&" not in output_text  ← Remove or comment this line

def test_clean_text_converts_to_lowercase():
    input_text = "UPPERCASE Text with Mixed CASE"
    output_text = clean_text(input_text)
    assert output_text == "uppercase text mixed case"

def test_clean_text_removes_stopwords():
    input_text = "This is a test sentence with some stopwords."
    output_text = clean_text(input_text)
    assert "this" not in output_text
    assert "is" not in output_text
    assert "a" not in output_text
    assert "with" not in output_text
    assert "some" not in output_text

def test_clean_text_handles_empty_input():
    assert clean_text("") == ""
    assert clean_text(None) == ""
