"""
Text preprocessing utilities for customer feedback sentiment classification.
"""

import html
import re

# Precompiled regex patterns for efficiency
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_TAG_PATTERN = re.compile(r"<.*?>")
WHITESPACE_PATTERN = re.compile(r"\s+")
SPECIAL_CHAR_PATTERN = re.compile(r"[^\w\s\.,!?'-]")


def clean_text(text: str) -> str:
    """
    Sanitize and normalize review text before vectorization.
    Preserves punctuation that conveys sentiment (!, ?, .) while removing noise.
    """
    if not text or not isinstance(text, str):
        return ""

    # Unescape HTML entities
    text = html.unescape(text)

    # Strip URLs
    text = URL_PATTERN.sub(" ", text)

    # Strip HTML tags
    text = HTML_TAG_PATTERN.sub(" ", text)

    # Remove non-standard control or unprintable characters
    text = SPECIAL_CHAR_PATTERN.sub(" ", text)

    # Normalize whitespace
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    return text
