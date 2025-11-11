"""
Text Preprocessing Module for Hindi Sarcasm Detection
Handles both Devanagari script and Romanized Hindi
"""

import re
import unicodedata

# Try to import transliteration library (optional)
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    HAS_TRANSLITERATION = True
except ImportError:
    HAS_TRANSLITERATION = False
    print("Note: indic-transliteration not available. Romanized to Devanagari conversion limited.")


def detect_script(text):
    """
    Detect if text is in Devanagari script or Romanized
    Returns: 'devanagari', 'romanized', or 'mixed'
    """
    devanagari_pattern = re.compile(r'[\u0900-\u097F]')
    has_devanagari = bool(devanagari_pattern.search(text))
    
    # Check for romanized Hindi (common patterns)
    romanized_pattern = re.compile(r'[a-zA-Z]+')
    has_romanized = bool(romanized_pattern.search(text))
    
    if has_devanagari and has_romanized:
        return 'mixed'
    elif has_devanagari:
        return 'devanagari'
    elif has_romanized:
        return 'romanized'
    else:
        return 'unknown'


def normalize_text(text):
    """
    Normalize text by removing extra spaces and special characters
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep Hindi characters
    text = re.sub(r'[^\w\s\u0900-\u097F]', '', text)
    return text.strip()


def romanized_to_devanagari(text):
    """
    Convert Romanized Hindi to Devanagari script
    """
    if not HAS_TRANSLITERATION:
        # Return original if transliteration library not available
        return text
    
    try:
        # Convert romanized to devanagari
        devanagari_text = transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
        return devanagari_text
    except Exception as e:
        # If conversion fails, return original text
        return text


def preprocess_hindi_text(text):
    """
    Main preprocessing function for Hindi text
    Handles both Devanagari and Romanized Hindi
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Normalize text
    text = normalize_text(text)
    
    # Detect script
    script_type = detect_script(text)
    
    # Convert romanized to devanagari if needed
    if script_type == 'romanized':
        text = romanized_to_devanagari(text)
    elif script_type == 'mixed':
        # For mixed, try to convert romanized parts
        # This is a simplified approach
        text = romanized_to_devanagari(text)
    
    return text


def extract_features(text):
    """
    Extract linguistic features that might indicate sarcasm
    """
    features = {
        'length': len(text),
        'word_count': len(text.split()),
        'has_question_mark': '?' in text,
        'has_exclamation': '!' in text,
        'has_ellipsis': '...' in text or '…' in text,
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
        'has_quotes': '"' in text or "'" in text,
    }
    
    # Hindi-specific patterns
    hindi_sarcasm_indicators = [
        'बहुत', 'कितना', 'क्या', 'वाह', 'शाबाश','तो', 'बड़ा','भी','ही'
    ]
    
    features['sarcasm_indicators'] = sum(1 for word in hindi_sarcasm_indicators if word in text)
    
    return features

