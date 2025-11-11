"""
NLP Feature Extraction Module for Hindi Sarcasm Detection

This module demonstrates various NLP techniques for feature extraction:
1. Text Statistics Analysis
2. Punctuation Pattern Analysis
3. Linguistic Pattern Matching
4. Sentiment Analysis Integration
5. Emoji Detection

These features are combined to create a comprehensive feature vector for ML models.
"""

import re
import numpy as np
from collections import Counter


def extract_text_statistics(text):
    """
    NLP TECHNIQUE: Text Statistics Analysis
    
    Extracts statistical features that help identify sarcastic patterns:
    - Length metrics (character, word, sentence counts)
    - Ratio features (uppercase, digits, punctuation)
    
    Why it works: Sarcastic text often has different statistical properties
    (e.g., excessive punctuation, unusual length patterns)
    """
    words = text.split()
    chars = len(text)
    
    return {
        'char_count': chars,
        'word_count': len(words),
        'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
        'sentence_count': len(re.split(r'[.!?]+', text)),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / chars if chars > 0 else 0,
        'digit_ratio': sum(1 for c in text if c.isdigit()) / chars if chars > 0 else 0,
        'punctuation_ratio': sum(1 for c in text if c in '.,!?;:') / chars if chars > 0 else 0,
    }


def extract_punctuation_features(text):
    """
    NLP TECHNIQUE: Punctuation Pattern Analysis
    
    Analyzes punctuation patterns that strongly indicate sarcasm:
    - Exclamation marks (!) - often used in sarcastic expressions
    - Question marks (?) - rhetorical questions
    - Combined (!?, ?!) - very strong sarcasm indicator
    - Ellipsis (...) - trailing off sarcastically
    - Quotation marks - "air quotes" for sarcasm
    
    Why it works: Sarcasm in Hindi (and most languages) relies heavily on
    punctuation to convey tone and intent.
    """
    return {
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'exclamation_question': text.count('?!') + text.count('!?'),
        'ellipsis_count': text.count('...') + text.count('…'),
        'quotes_count': text.count('"') + text.count("'"),
        'has_multiple_exclamations': text.count('!') > 1,
        'has_multiple_questions': text.count('?') > 1,
        'repeated_punctuation': bool(re.search(r'[!?]{2,}', text)),
    }


def extract_linguistic_features(text):
    """
    NLP TECHNIQUE: Linguistic Pattern Matching
    
    Uses regex pattern matching to identify known sarcastic expressions in Hindi:
    - Common sarcastic phrases (वाह!, क्या बात है, etc.)
    - Intensifiers (बहुत, कितना) - often sarcastic when overused
    - Dismissive phrases (है ना, है न) - indicate dismissive/sarcastic tone
    - Question words in sarcastic context
    
    Why it works: Hindi has specific linguistic patterns that indicate sarcasm.
    Pattern matching captures these known expressions effectively.
    """
    text_lower = text.lower()
    
    # Hindi sarcastic patterns
    sarcastic_patterns = [
        r'वाह\s*!*',
        r'क्या\s+बात\s+है',
        r'बहुत\s+अच्छा',
        r'शाबाश',
        r'पता\s+ही\s+नहीं',
        r'hame\s+to',
        r'pata\s+hi\s+nahi',
    ]
    
    pattern_matches = sum(1 for pattern in sarcastic_patterns if re.search(pattern, text_lower))
    
    # Intensifiers
    intensifiers = ['बहुत', 'कितना', 'बड़ा', 'ज़्यादा', 'bahut', 'kitna', 'bada', 'zyada']
    has_intensifier = any(word in text_lower for word in intensifiers)
    
    # Dismissive phrases
    dismissive = ['है ना', 'है न', 'to', 'bhi', 'hai na', 'hai n']
    has_dismissive = any(phrase in text_lower for phrase in dismissive)
    
    return {
        'sarcastic_pattern_count': pattern_matches,
        'has_intensifier': int(has_intensifier),
        'has_dismissive': int(has_dismissive),
        'has_question_words': int(bool(re.search(r'(क्या|kya|कब|kab|कहाँ|kahan)', text_lower))),
    }


def extract_sentiment_features(text, sentiment_score=None):
    """
    NLP TECHNIQUE: Sentiment Analysis Integration
    
    Incorporates sentiment analysis results as features:
    - Sentiment polarity (positive/negative/neutral)
    - Sentiment confidence score
    
    Why it works: Sarcasm often involves sentiment contradiction:
    - Positive words with negative intent ("बहुत अच्छा" said sarcastically)
    - Negative sentiment with positive words
    - These contradictions are strong sarcasm indicators
    """
    features = {}
    
    if sentiment_score:
        # Sentiment polarity
        if sentiment_score['label'] == 'POSITIVE':
            features['sentiment_positive'] = 1
            features['sentiment_negative'] = 0
            features['sentiment_score'] = sentiment_score.get('score', 0.5)
        elif sentiment_score['label'] == 'NEGATIVE':
            features['sentiment_positive'] = 0
            features['sentiment_negative'] = 1
            features['sentiment_score'] = -sentiment_score.get('score', 0.5)
        else:
            features['sentiment_positive'] = 0
            features['sentiment_negative'] = 0
            features['sentiment_score'] = 0
    else:
        # Default values if sentiment not available
        features['sentiment_positive'] = 0
        features['sentiment_negative'] = 0
        features['sentiment_score'] = 0
    
    return features


def extract_emoji_features(text):
    """
    NLP TECHNIQUE: Emoji Detection & Analysis
    
    Detects and counts emojis in text, which can indicate sarcasm:
    - Emoji count
    - Presence of emojis
    
    Why it works: Emojis are often used in social media to convey sarcasm
    or add emotional context that might not be clear from text alone.
    """
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    
    emojis = emoji_pattern.findall(text)
    
    return {
        'emoji_count': len(emojis),
        'has_emoji': int(len(emojis) > 0),
    }


def extract_all_features(text, sentiment_score=None):
    """
    NLP TECHNIQUE: Feature Engineering - Combining Multiple NLP Techniques
    
    Combines all extracted features into a comprehensive feature vector:
    1. Text Statistics (7 features)
    2. Punctuation Analysis (8 features)
    3. Linguistic Patterns (4 features)
    4. Sentiment Features (3 features)
    5. Emoji Features (2 features)
    
    Total: 24+ engineered features
    
    Why it works: Combining multiple NLP techniques provides a richer
    representation of the text, allowing ML models to learn complex patterns
    that individual features might miss.
    """
    features = {}
    
    # Text statistics
    features.update(extract_text_statistics(text))
    
    # Punctuation features
    features.update(extract_punctuation_features(text))
    
    # Linguistic features
    features.update(extract_linguistic_features(text))
    
    # Sentiment features
    features.update(extract_sentiment_features(text, sentiment_score))
    
    # Emoji features
    features.update(extract_emoji_features(text))
    
    return features


def features_to_vector(features):
    """
    Convert feature dict to numpy array for ML models
    """
    # Order matters - keep consistent
    feature_order = [
        'char_count', 'word_count', 'avg_word_length', 'sentence_count',
        'uppercase_ratio', 'digit_ratio', 'punctuation_ratio',
        'exclamation_count', 'question_count', 'exclamation_question',
        'ellipsis_count', 'quotes_count', 'has_multiple_exclamations',
        'has_multiple_questions', 'has_repeated_punctuation',
        'sarcastic_pattern_count', 'has_intensifier', 'has_dismissive',
        'has_question_words', 'sentiment_positive', 'sentiment_negative',
        'sentiment_score', 'emoji_count', 'has_emoji'
    ]
    
    return np.array([features.get(key, 0) for key in feature_order])

