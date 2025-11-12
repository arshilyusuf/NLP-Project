import re
import numpy as np
from collections import Counter


def extract_text_statistics(text):
    # extract statistical features
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
    # analyze punctuation patterns
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
    # identify known sarcastic expressions
    text_lower = text.lower()
    
    sarcastic_patterns = [
        r'वाह\s*!',
        r'क्या\s+बात\s+है',
        r'बहुत\s+अच्छा',
        r'शाबाश',
        r'पता\s+ही\s+नहीं',
        r'hame\s+to',
        r'pata\s+hi\s+nahi',
    ]
    
    pattern_matches = sum(1 for pattern in sarcastic_patterns if re.search(pattern, text_lower))
    
    intensifiers = ['बहुत', 'कितना', 'बड़ा', 'ज़्यादा', 'bahut', 'kitna', 'bada', 'zyada']
    has_intensifier = any(word in text_lower for word in intensifiers)
    
    dismissive = ['है ना', 'है न', 'to', 'bhi', 'hai na', 'hai n']
    has_dismissive = any(phrase in text_lower for phrase in dismissive)
    
    return {
        'sarcastic_pattern_count': pattern_matches,
        'has_intensifier': int(has_intensifier),
        'has_dismissive': int(has_dismissive),
        'has_question_words': int(bool(re.search(r'(क्या|kya|कब|kab|कहाँ|kahan)', text_lower))),
    }


def extract_sentiment_features(text, sentiment_score=None):
    # incorporate sentiment analysis results as features
    features = {}
    
    if sentiment_score:
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
        features['sentiment_positive'] = 0
        features['sentiment_negative'] = 0
        features['sentiment_score'] = 0
    
    return features


def extract_emoji_features(text):
    # detect and count emojis in text
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    
    emojis = emoji_pattern.findall(text)
    
    return {
        'emoji_count': len(emojis),
        'has_emoji': int(len(emojis) > 0),
    }


def extract_all_features(text, sentiment_score=None):
    # combines all extracted features
    features = {}
    
    features.update(extract_text_statistics(text))
    features.update(extract_punctuation_features(text))
    features.update(extract_linguistic_features(text))
    features.update(extract_sentiment_features(text, sentiment_score))
    features.update(extract_emoji_features(text))
    
    return features


def features_to_vector(features):
    # convert feature dict to numpy array
    feature_order = [
        'char_count', 'word_count', 'avg_word_length', 'sentence_count',
        'uppercase_ratio', 'digit_ratio', 'punctuation_ratio',
        'exclamation_count', 'question_count', 'exclamation_question',
        'ellipsis_count', 'quotes_count', 'has_multiple_exclamations',
        'has_multiple_questions', 'repeated_punctuation',
        'sarcastic_pattern_count', 'has_intensifier', 'has_dismissive',
        'has_question_words', 'sentiment_positive', 'sentiment_negative',
        'sentiment_score', 'emoji_count', 'has_emoji'
    ]
    
    return np.array([features.get(key, 0) for key in feature_order])

