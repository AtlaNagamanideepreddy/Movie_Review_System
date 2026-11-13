import re

def clean_text(text: str) -> str:
    """Removes HTML tags, URLs, special characters, and normalizes whitespace."""
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text
