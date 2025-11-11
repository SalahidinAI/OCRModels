"""
Text processing utilities for OCR results.
"""

import re
from typing import List, Tuple


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison:
    - Remove special characters (except letters, numbers, spaces)
    - Preserve Cyrillic and Chinese characters
    - Convert to lowercase
    - Remove extra whitespace
    """
    if not text:
        return ""
    
    # Keep letters, numbers, spaces, Cyrillic (U+0400-U+04FF), Chinese (U+4E00-U+9FFF)
    normalized = re.sub(
        r'[^\w\s\u0400-\u04FF\u4e00-\u9fff]',
        '',
        text
    )
    
    # Remove extra whitespace and convert to lowercase
    normalized = re.sub(r'\s+', ' ', normalized).strip().lower()
    
    return normalized


def clean_text(text: str) -> str:
    """
    Clean text for display:
    - Remove excessive whitespace
    - Fix common OCR errors
    - Preserve original case
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    cleaned = re.sub(r'\s+', ' ', text).strip()
    
    # Fix common OCR errors (can be extended)
    # Example: fix common character confusions
    # cleaned = cleaned.replace('0', 'O')  # Uncomment if needed
    
    return cleaned


def split_into_lines(text: str) -> List[str]:
    """Split text into lines, removing empty lines."""
    lines = [line.strip() for line in text.splitlines()]
    return [line for line in lines if line]


def merge_texts(texts: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Merge duplicate texts from different sources.
    Returns list of (source, text) tuples without duplicates.
    """
    seen = set()
    merged = []
    
    for source, text in texts:
        normalized = normalize_text(text)
        if normalized and normalized not in seen:
            seen.add(normalized)
            merged.append((source, text))
    
    return merged


def extract_words(text: str) -> List[str]:
    """Extract words from text (handles multiple languages)."""
    # Pattern for words: letters, numbers, and Unicode word characters
    words = re.findall(r'\b\w+\b', text, re.UNICODE)
    return words


def calculate_text_length(text: str) -> int:
    """Calculate text length (characters)."""
    return len(text) if text else 0


def calculate_word_count(text: str) -> int:
    """Calculate word count."""
    words = extract_words(text)
    return len(words)

