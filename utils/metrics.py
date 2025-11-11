"""
Metrics for comparing OCR results.
Includes similarity, accuracy, and performance metrics.
"""

from typing import List, Dict, Tuple, Any, Optional
from rapidfuzz import fuzz
from .text_utils import normalize_text, extract_words


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts (0-100).
    Uses multiple methods and returns average.
    """
    if not text1 and not text2:
        return 100.0
    if not text1 or not text2:
        return 0.0
    
    # Normalize texts
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    
    if norm1 == norm2:
        return 100.0
    
    # Calculate multiple similarity metrics
    ratios = []
    
    # 1. Token Sort Ratio (handles word order differences)
    ratios.append(fuzz.token_sort_ratio(norm1, norm2))
    
    # 2. Partial Ratio (handles substring matches)
    ratios.append(fuzz.partial_ratio(norm1, norm2))
    
    # 3. Token Set Ratio (handles duplicate words)
    ratios.append(fuzz.token_set_ratio(norm1, norm2))
    
    # 4. Simple Ratio
    ratios.append(fuzz.ratio(norm1, norm2))
    
    return sum(ratios) / len(ratios)


def calculate_character_error_rate(ref_text: str, hyp_text: str) -> float:
    """
    Calculate Character Error Rate (CER).
    Lower is better (0 = perfect match).
    """
    if not ref_text:
        return 1.0 if hyp_text else 0.0
    
    ref_chars = list(ref_text)
    hyp_chars = list(hyp_text)
    
    # Simple Levenshtein distance
    n = len(ref_chars)
    m = len(hyp_chars)
    
    if n == 0:
        return 1.0 if m > 0 else 0.0
    
    # Dynamic programming for edit distance
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_chars[i-1] == hyp_chars[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # deletion
                    dp[i][j-1] + 1,      # insertion
                    dp[i-1][j-1] + 1     # substitution
                )
    
    errors = dp[n][m]
    return errors / n


def calculate_word_error_rate(ref_text: str, hyp_text: str) -> float:
    """
    Calculate Word Error Rate (WER).
    Lower is better (0 = perfect match).
    """
    ref_words = extract_words(ref_text)
    hyp_words = extract_words(hyp_text)
    
    if not ref_words:
        return 1.0 if hyp_words else 0.0
    
    n = len(ref_words)
    m = len(hyp_words)
    
    if n == 0:
        return 1.0 if m > 0 else 0.0
    
    # Simple word-level edit distance
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i-1].lower() == hyp_words[j-1].lower():
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,
                    dp[i][j-1] + 1,
                    dp[i-1][j-1] + 1
                )
    
    errors = dp[n][m]
    return errors / n


def calculate_metrics(results: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for OCR results.
    
    Args:
        results: Dictionary {engine_name: [text_lines]}
        
    Returns:
        Dictionary with metrics
    """
    metrics = {
        'engines': list(results.keys()),
        'similarity_matrix': {},
        'average_similarity': {},
        'processing_times': {},
        'text_lengths': {},
        'line_counts': {},
    }
    
    # Calculate pairwise similarities
    engines = list(results.keys())
    for i, engine1 in enumerate(engines):
        text1 = "\n".join(results[engine1])
        metrics['text_lengths'][engine1] = len(text1)
        metrics['line_counts'][engine1] = len(results[engine1])
        
        similarities = []
        for engine2 in engines:
            if engine1 == engine2:
                similarity = 100.0
            else:
                text2 = "\n".join(results[engine2])
                similarity = calculate_similarity(text1, text2)
                similarities.append(similarity)
            
            key = f"{engine1} ↔ {engine2}"
            metrics['similarity_matrix'][key] = similarity
        
        if similarities:
            metrics['average_similarity'][engine1] = sum(similarities) / len(similarities)
    
    return metrics


def compare_results(
    results: Dict[str, Dict[str, Any]],
    reference_engine: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compare OCR results from multiple engines.
    
    Args:
        results: Dictionary {engine_name: {'text': [lines], 'time': float}}
        reference_engine: Engine to use as reference (default: first one)
        
    Returns:
        Comparison dictionary with metrics and analysis
    """
    if not results:
        return {}
    
    # Extract text from results
    texts = {name: data.get('text', []) for name, data in results.items()}
    
    # Use first engine as reference if not specified
    if reference_engine is None:
        reference_engine = list(results.keys())[0]
    
    if reference_engine not in texts:
        reference_engine = list(texts.keys())[0]
    
    ref_text = "\n".join(texts[reference_engine])
    
    # Calculate metrics
    comparison = {
        'reference_engine': reference_engine,
        'engines': list(results.keys()),
        'similarities': {},
        'error_rates': {},
        'processing_times': {name: data.get('time', 0.0) for name, data in results.items()},
        'text_lengths': {name: len("\n".join(text)) for name, text in texts.items()},
        'line_counts': {name: len(text) for name, text in texts.items()},
    }
    
    # Compare each engine to reference
    for engine, text_lines in texts.items():
        if engine == reference_engine:
            comparison['similarities'][engine] = 100.0
            comparison['error_rates'][engine] = {'cer': 0.0, 'wer': 0.0}
        else:
            text = "\n".join(text_lines)
            similarity = calculate_similarity(ref_text, text)
            cer = calculate_character_error_rate(ref_text, text)
            wer = calculate_word_error_rate(ref_text, text)
            
            comparison['similarities'][engine] = similarity
            comparison['error_rates'][engine] = {'cer': cer, 'wer': wer}
    
    # Calculate pairwise similarities
    comparison['pairwise_similarities'] = {}
    engines = list(texts.keys())
    for i, engine1 in enumerate(engines):
        for engine2 in engines[i+1:]:
            text1 = "\n".join(texts[engine1])
            text2 = "\n".join(texts[engine2])
            similarity = calculate_similarity(text1, text2)
            key = f"{engine1} ↔ {engine2}"
            comparison['pairwise_similarities'][key] = similarity
    
    return comparison

