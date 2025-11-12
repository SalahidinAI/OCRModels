"""
Text processing utilities for OCR results.
"""

import re
from typing import List, Tuple, Dict, Any


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


def align_texts_char_by_char(texts: List[str]) -> List[List[Tuple[str, int]]]:
    """
    Align multiple texts character by character.
    Returns list of aligned characters with match count.
    Each element is (char, match_count) where match_count is how many texts have this char at this position.
    
    Args:
        texts: List of text strings to align
        
    Returns:
        List of (character, match_count) tuples representing aligned text
    """
    if not texts:
        return []
    
    # Remove empty texts
    texts = [t for t in texts if t]
    if not texts:
        return []
    
    # If only one text, return it with match_count = 1
    if len(texts) == 1:
        return [(char, 1) for char in texts[0]]
    
    # Use longest text as base for alignment
    max_len = max(len(t) for t in texts)
    total_engines = len(texts)
    
    # Create aligned result
    aligned = []
    
    # For each position, check how many texts have the same character
    for pos in range(max_len):
        chars_at_pos = []
        for text in texts:
            if pos < len(text):
                chars_at_pos.append(text[pos])
            else:
                chars_at_pos.append(None)  # Text ended
        
        # Count matches - check if all non-None chars are the same
        non_none_chars = [c for c in chars_at_pos if c is not None]
        if not non_none_chars:
            continue
        
        # Count how many have the same character as the first non-None
        first_char = non_none_chars[0]
        match_count = sum(1 for c in chars_at_pos if c == first_char)
        
        # Use the most common character at this position
        if match_count == 0:
            # All different, use first available
            char_to_use = first_char
            match_count = 1
        else:
            char_to_use = first_char
        
        aligned.append((char_to_use, match_count))
    
    return aligned


def compare_texts_word_by_word(texts: List[str]) -> List[Dict[str, Any]]:
    """
    Compare texts word by word, matching words by content not position.
    Returns word mapping for each engine with match information.
    
    Args:
        texts: List of text strings from different engines
        
    Returns:
        List of dictionaries with word mappings for each engine
    """
    if not texts or len(texts) < 2:
        return []
    
    # Split into words
    word_lists = []
    for text in texts:
        words = text.split()
        word_lists.append(words)
    
    total_engines = len(texts)
    
    # Create normalized word sets for each engine
    normalized_sets = []
    for words in word_lists:
        norm_set = {}
        for word in words:
            norm = normalize_text(word)
            if norm not in norm_set:
                norm_set[norm] = []
            norm_set[norm].append(word)
        normalized_sets.append(norm_set)
    
    # For each engine, create word mapping with match info
    result = []
    for engine_idx, words in enumerate(word_lists):
        engine_result = []
        for word in words:
            norm_word = normalize_text(word)
            
            # Count how many engines have this normalized word
            match_count = 0
            for other_set in normalized_sets:
                if norm_word in other_set:
                    match_count += 1
            
            # Determine color
            if match_count == total_engines:
                color = "green"
            elif match_count >= 2:
                color = "yellow"
            else:
                color = "red"
            
            engine_result.append({
                'word': word,
                'normalized': norm_word,
                'match_count': match_count,
                'total_engines': total_engines,
                'color': color
            })
        
        result.append(engine_result)
    
    return result

