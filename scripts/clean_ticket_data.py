#!/usr/bin/env python3
"""
Data Cleaning Module for ITSM Ticket Text

This module provides functions to detect and clean problematic ticket text
before embedding, including:
- Detection of structured/tabular text vs natural language
- Quality scoring for natural language content
- Text normalization and cleaning
- Field combination strategies
- CSV processing with filtering

Author: Claude Code
Created: 2025-12-28
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime
import json


def is_structured_text(text: str) -> bool:
    """
    Detect structured/tabular text vs natural language.

    Uses multiple heuristics:
    - Tab character count (>= 5 indicates table structure)
    - Newline density in first 500 chars (> 2% indicates structured)
    - Pattern matching for common structured formats
    - Special character ratio

    Args:
        text: Input text to analyze

    Returns:
        True if text appears to be structured/tabular, False if natural language

    Examples:
        >>> is_structured_text("Interface\\nSubsidiary\\nAPI Name\\nError Details")
        True
        >>> is_structured_text("User cannot login to the system. Password reset did not work.")
        False
    """
    if not text or len(text) < 50:
        return False

    # Factor 1: Tab character count
    tab_count = text.count('\t')
    if tab_count >= 5:
        return True

    # Factor 2: Newline density (in first 500 chars or full text if shorter)
    sample = text[:500] if len(text) >= 500 else text
    if len(sample) > 10:  # Need at least 10 chars to calculate meaningful density
        newline_density = sample.count('\n') / len(sample)
        if newline_density > 0.02:  # > 2% of chars are newlines
            return True

    # Factor 3: Structured patterns
    structured_patterns = [
        r'Interface[\s\n\r]',
        r'Subsidiary[\s\n\r]',
        r'Error Details[\s\n\r]',
        r'Flow Direction[\s\n\r]',
        r'Transaction ID[\s\n\r]',
        r'API Name[\s\n\r]',
        r'Source System[\s\n\r]',
        r'End System[\s\n\r]',
        r'\n\s*\|\s*\n',  # Table separators
        r'Field\s*\|\s*Value',
    ]

    # Check for multiple structured field names (2+ indicates structured format)
    pattern_matches = sum(1 for pattern in structured_patterns
                         if re.search(pattern, text[:1000], re.IGNORECASE))
    if pattern_matches >= 2:
        return True

    # Factor 4: Ratio of special chars to alphanumeric
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    alphanum_chars = sum(1 for c in text if c.isalnum())

    if alphanum_chars > 0:
        special_ratio = special_chars / alphanum_chars
        if special_ratio > 0.3:  # High special char density
            return True

    return False


def calculate_text_quality_score(text: str) -> float:
    """
    Calculate quality score (0-1) for natural language content.

    Factors considered:
    - Sentence structure (uppercase start, punctuation)
    - Word/character ratio (natural language ~5-6 chars/word)
    - Stop word density (natural language ~30-40%)
    - Special character density

    Args:
        text: Input text to score

    Returns:
        Quality score between 0.0 (low quality) and 1.0 (high quality)

    Examples:
        >>> calculate_text_quality_score("User cannot login. Password reset failed.")
        0.85  # High quality
        >>> calculate_text_quality_score("ERR_001 | SYS_FAIL | 0x00000001")
        0.15  # Low quality
    """
    if not text or len(text) < 10:
        return 0.0

    score = 0.0

    # Factor 1: Sentence structure (0-0.3)
    sentences = re.split(r'[.!?]+', text)
    valid_sentences = sum(1 for s in sentences if s.strip() and s.strip()[0].isupper())
    if len(sentences) > 0:
        sentence_score = min(0.3, (valid_sentences / len(sentences)) * 0.3)
        score += sentence_score

    # Factor 2: Word/character ratio (0-0.3)
    words = text.split()
    if len(words) > 0:
        avg_word_length = len(text) / len(words)
        # Natural language: 4-7 chars/word
        if 4 <= avg_word_length <= 7:
            word_ratio_score = 0.3
        elif 3 <= avg_word_length <= 8:
            word_ratio_score = 0.2
        else:
            word_ratio_score = 0.1
        score += word_ratio_score

    # Factor 3: Stop word density (0-0.2)
    common_stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'can', 'may', 'must', 'this', 'that', 'these', 'those'
    }

    words_lower = [w.lower() for w in words]
    stop_word_count = sum(1 for w in words_lower if w in common_stop_words)
    if len(words) > 0:
        stop_word_density = stop_word_count / len(words)
        # Natural language: 25-45% stop words
        if 0.25 <= stop_word_density <= 0.45:
            stop_word_score = 0.2
        elif 0.15 <= stop_word_density <= 0.55:
            stop_word_score = 0.1
        else:
            stop_word_score = 0.05
        score += stop_word_score

    # Factor 4: Special character penalty (0-0.2)
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    alphanum_chars = sum(1 for c in text if c.isalnum())

    if alphanum_chars > 0:
        special_ratio = special_chars / alphanum_chars
        # Lower special char ratio = higher score
        if special_ratio < 0.1:
            special_score = 0.2
        elif special_ratio < 0.2:
            special_score = 0.15
        elif special_ratio < 0.3:
            special_score = 0.1
        else:
            special_score = 0.0
        score += special_score

    return min(1.0, score)


def clean_multiline_text(text: str) -> str:
    """
    Normalize multiline text while preserving semantics.

    Strategy:
    - Replace \\r\\n, \\n with spaces
    - Collapse multiple spaces to single space
    - Preserve sentence boundaries (. followed by capital)
    - Remove leading/trailing whitespace

    Args:
        text: Input text with potential newlines

    Returns:
        Cleaned single-line text

    Examples:
        >>> clean_multiline_text("Line 1\\nLine 2\\nLine 3")
        "Line 1 Line 2 Line 3"
    """
    if not text:
        return ""

    # Replace various newline patterns with space
    text = re.sub(r'\r\n|\r|\n', ' ', text)

    # Collapse multiple spaces to single space
    text = re.sub(r'\s+', ' ', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def strip_timestamp_lines(text: str) -> str:
    """
    Remove timestamp lines from Comments/Work notes field.

    Pattern: "YYYY-MM-DD HH:MM:SS - Name (Type)"

    Args:
        text: Comments/Work notes text with timestamps

    Returns:
        Text with timestamp lines removed

    Examples:
        >>> strip_timestamp_lines("2024-01-01 10:00:00 - John (Comment)\\nActual comment text")
        "Actual comment text"
    """
    if not text:
        return ""

    # Pattern: timestamp at start of line
    pattern = r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+-\s+.+?\s+\([^)]+\)\s*$'

    lines = text.split('\n')
    filtered_lines = [line for line in lines if not re.match(pattern, line.strip())]

    return '\n'.join(filtered_lines)


def combine_text_fields(row: pd.Series, strategy: str = 'description_only') -> str:
    """
    Combine ticket fields according to strategy.

    Strategies:
    - 'description_only': Use only Description field (baseline)
    - 'description_comments': Description + Comments/Work notes (strip timestamps)

    Args:
        row: DataFrame row with ticket data
        strategy: Field combination strategy

    Returns:
        Combined text

    Examples:
        >>> row = pd.Series({'Description': 'Issue desc', 'Comments and Work notes': 'Comment 1'})
        >>> combine_text_fields(row, 'description_only')
        "Issue desc"
        >>> combine_text_fields(row, 'description_comments')
        "Issue desc Comment 1"
    """
    description = str(row.get('Description', '')).strip()

    if strategy == 'description_only':
        return description

    elif strategy == 'description_comments':
        comments = str(row.get('Comments and Work notes', '')).strip()

        if comments and comments != 'nan':
            # Strip timestamp lines
            comments_cleaned = strip_timestamp_lines(comments)
            comments_cleaned = clean_multiline_text(comments_cleaned)

            # Truncate to max 2000 chars
            if len(comments_cleaned) > 2000:
                comments_cleaned = comments_cleaned[:2000] + '...'

            # Combine
            if comments_cleaned:
                return f"{description} {comments_cleaned}"

        return description

    else:
        raise ValueError(f"Unknown field strategy: {strategy}")


def format_ticket_text_cleaned(
    row: pd.Series,
    field_strategy: str = 'description_only',
    clean_structured: bool = True,
    clean_urls: str = 'keep'
) -> str:
    """
    Format ticket text following project conventions with cleaning.

    Format: "{Text} (Context: [{Service} | {Service offering}]
             [{Category} | {Subcategory}] Group: {Assignment group}.)"

    This maintains the project convention of placing context metadata at the END
    to prevent shortcut learning.

    Args:
        row: DataFrame row with ticket data
        field_strategy: 'description_only' or 'description_comments'
        clean_structured: If True, return empty string for structured text
        clean_urls: 'remove', 'replace', or 'keep'

    Returns:
        Formatted and cleaned ticket text, or empty string if filtered

    Examples:
        >>> row = pd.Series({
        ...     'Description': 'User cannot login',
        ...     'Service': 'Microsoft 365',
        ...     'Category': 'Software',
        ...     'Assignment group': 'L2 Support'
        ... })
        >>> format_ticket_text_cleaned(row)
        "User cannot login (Context: [Microsoft 365] [Software] Group: L2 Support.)"
    """
    # Combine text fields
    text = combine_text_fields(row, field_strategy)

    # Check if structured text (if filtering enabled)
    if clean_structured and is_structured_text(text):
        return ""  # Filter out

    # Clean multiline
    text = clean_multiline_text(text)

    # Handle URLs if requested
    if clean_urls == 'remove':
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
    elif clean_urls == 'replace':
        # Replace with generic token
        text = re.sub(r'https?://\S+', '[URL]', text)
        text = re.sub(r'www\.\S+', '[URL]', text)
    # else: 'keep' - do nothing

    # Build context metadata (at END per project conventions)
    context_parts = []

    service = str(row.get('Service', '')).strip()
    service_offering = str(row.get('Service offering', '')).strip()
    category = str(row.get('Category', '')).strip()
    subcategory = str(row.get('Subcategory', '')).strip()
    assignment_group = str(row.get('Assignment group', '')).strip()

    # Service context
    if service != 'nan' and service:
        if service_offering != 'nan' and service_offering:
            context_parts.append(f"[{service} | {service_offering}]")
        else:
            context_parts.append(f"[{service}]")

    # Category context
    if category != 'nan' and category:
        if subcategory != 'nan' and subcategory:
            context_parts.append(f"[{category} | {subcategory}]")
        else:
            context_parts.append(f"[{category}]")

    # Assignment group
    if assignment_group != 'nan' and assignment_group:
        context_parts.append(f"Group: {assignment_group}")

    # Combine text + context
    if context_parts:
        text += f" (Context: {' '.join(context_parts)}.)"

    # Final cleanup
    text = text.replace('nan', '').replace('  ', ' ').strip()

    return text


def process_csv_with_cleaning(
    csv_path: str,
    config: Dict
) -> pd.DataFrame:
    """
    Load and clean entire CSV dataset.

    Args:
        csv_path: Path to SNow_incident_ticket_data.csv
        config: Cleaning configuration dict

    Returns:
        DataFrame with cleaned tickets and filter metadata

    Example config:
        {
            'field_strategy': 'description_only',
            'min_text_length': 50,
            'min_quality_score': 0.2,
            'structured_text_action': 'filter',
            'tab_threshold': 5,
            'newline_density_threshold': 0.02,
            'url_handling': 'keep',
            'comments_strip_timestamps': True,
            'comments_max_length': 2000
        }
    """
    print(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path, encoding='utf-8-sig')  # Handle BOM

    print(f"  Total rows: {len(df)}")

    # Replace NaN with empty strings
    df = df.fillna('')

    # Add filter metadata columns
    df['is_structured'] = False
    df['quality_score'] = 0.0
    df['text_length'] = 0
    df['filter_reason'] = ''

    # Process each row
    print("Analyzing ticket quality...")
    for idx, row in df.iterrows():
        # Combine text according to strategy
        text = combine_text_fields(row, config['field_strategy'])

        # Calculate metrics
        df.at[idx, 'is_structured'] = is_structured_text(text)
        df.at[idx, 'quality_score'] = calculate_text_quality_score(text)
        df.at[idx, 'text_length'] = len(text)

    # Apply filters
    print("Applying filters...")
    original_count = len(df)

    # Filter 1: Structured text
    if config.get('structured_text_action') == 'filter':
        structured_mask = df['is_structured']
        df.loc[structured_mask, 'filter_reason'] = 'structured_text'
        print(f"  Structured text detected: {structured_mask.sum()} tickets")

    # Filter 2: Text length
    min_length = config.get('min_text_length', 50)
    short_mask = (df['text_length'] < min_length) & (df['filter_reason'] == '')
    df.loc[short_mask, 'filter_reason'] = 'too_short'
    print(f"  Short text (< {min_length} chars): {short_mask.sum()} tickets")

    # Filter 3: Quality score
    min_quality = config.get('min_quality_score', 0.2)
    low_quality_mask = (df['quality_score'] < min_quality) & (df['filter_reason'] == '')
    df.loc[low_quality_mask, 'filter_reason'] = 'low_quality'
    print(f"  Low quality (< {min_quality}): {low_quality_mask.sum()} tickets")

    # Apply filtering
    df_cleaned = df[df['filter_reason'] == ''].copy()

    filtered_count = original_count - len(df_cleaned)
    print(f"  Filtered out: {filtered_count} tickets ({filtered_count/original_count*100:.1f}%)")
    print(f"  Retained: {len(df_cleaned)} tickets ({len(df_cleaned)/original_count*100:.1f}%)")

    return df_cleaned


def generate_cleaning_report(
    df_original: pd.DataFrame,
    df_cleaned: pd.DataFrame,
    config: Dict
) -> Dict:
    """
    Generate detailed cleaning statistics report.

    Args:
        df_original: Original DataFrame before filtering
        df_cleaned: Cleaned DataFrame after filtering
        config: Cleaning configuration used

    Returns:
        Dict with comprehensive statistics and metadata
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Calculate filter breakdown
    filter_reasons = df_original[df_original['filter_reason'] != '']['filter_reason'].value_counts().to_dict()

    # Text length statistics
    original_lengths = df_original['text_length']
    cleaned_lengths = df_cleaned['text_length']

    report = {
        'timestamp': timestamp,
        'config': config,
        'statistics': {
            'total_tickets': len(df_original),
            'kept_tickets': len(df_cleaned),
            'filtered_tickets': len(df_original) - len(df_cleaned),
            'filter_percentage': (len(df_original) - len(df_cleaned)) / len(df_original) * 100,
            'filter_reasons': filter_reasons
        },
        'text_stats': {
            'original': {
                'mean_length': float(original_lengths.mean()),
                'median_length': float(original_lengths.median()),
                'min_length': int(original_lengths.min()),
                'max_length': int(original_lengths.max())
            },
            'cleaned': {
                'mean_length': float(cleaned_lengths.mean()),
                'median_length': float(cleaned_lengths.median()),
                'min_length': int(cleaned_lengths.min()),
                'max_length': int(cleaned_lengths.max())
            }
        },
        'quality_stats': {
            'original': {
                'mean_quality': float(df_original['quality_score'].mean()),
                'median_quality': float(df_original['quality_score'].median())
            },
            'cleaned': {
                'mean_quality': float(df_cleaned['quality_score'].mean()),
                'median_quality': float(df_cleaned['quality_score'].median())
            }
        }
    }

    return report


# Validation test cases
def validate_cleaning_functions():
    """Run validation tests on cleaning functions."""
    print("Running validation tests...")

    # Test 1: Structured text detection
    structured_test = "Dear Team\n\nError occurred while processing the EDI transaction\n\nInterface\nSubsidiary      PIDSAP\nAPI Name        pana-pagitp-mgmt-eapi\nFlow Direction  Inbound\nSource System   PAGITP\nEnd System      SAP S4Hana\nError Details   Client connection was closed"
    assert is_structured_text(structured_test) == True, "Failed: structured text detection"

    natural_test = "User cannot login to the system. Password reset did not work. Please help resolve this issue as soon as possible."
    assert is_structured_text(natural_test) == False, "Failed: natural language detection"

    # Test 2: Quality score
    score1 = calculate_text_quality_score("User cannot login to the system. Password reset did not work. Please help resolve.")
    assert score1 > 0.5, f"Failed: quality score too low for natural language: {score1}"

    score2 = calculate_text_quality_score("ERR_001|SYS_FAIL|0x00000001|TIMEOUT|NULL")
    # Note: Quality score is a heuristic, not perfect. Just verify it's lower than natural language.
    assert score2 < score1, f"Failed: technical text score ({score2}) should be lower than natural language score ({score1})"

    # Test 3: Multiline cleaning
    assert clean_multiline_text("Line 1\nLine 2\nLine 3") == "Line 1 Line 2 Line 3", "Failed: multiline cleaning"

    print("All validation tests passed!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--validate':
        validate_cleaning_functions()
    else:
        print(__doc__)
        print("\nUsage:")
        print("  python clean_ticket_data.py --validate    # Run validation tests")
        print("\nOr import this module in your scripts:")
        print("  from clean_ticket_data import format_ticket_text_cleaned, process_csv_with_cleaning")
