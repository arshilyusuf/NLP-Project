import re
import unicodedata

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    HAS_TRANSLITERATION = True
except ImportError:
    HAS_TRANSLITERATION = False
    print("Note: indic-transliteration not available. Romanized to Devanagari conversion limited.")


def detect_script(text):
    # detect if text is in devanagari or romanized script
    devanagari_pattern = re.compile(r'[\u0900-\u097F]')
    has_devanagari = bool(devanagari_pattern.search(text))
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
    # normalize text by removing extra spaces and special characters
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\u0900-\u097F]', '', text)
    return text.strip()


def romanized_to_devanagari(text):
    # convert romanized hindi to devanagari script
    if not HAS_TRANSLITERATION:
        return text
    try:
        devanagari_text = transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
        return devanagari_text
    except Exception as e:
        return text


def preprocess_hindi_text(text):
    # main preprocessing function for hindi text
    if not text or not isinstance(text, str):
        return ""
    text = normalize_text(text)
    script_type = detect_script(text)
    if script_type == 'romanized':
        text = romanized_to_devanagari(text)
    elif script_type == 'mixed':
        text = romanized_to_devanagari(text)
    return text


def extract_features(text):
    # extract linguistic features that might indicate sarcasm
    features = {
        'length': len(text),
        'word_count': len(text.split()),
        'has_question_mark': '?' in text,
        'has_exclamation': '!' in text,
        'has_ellipsis': '...' in text or '…' in text,
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
        'has_quotes': '"' in text or "'" in text,
    }
    hindi_sarcasm_indicators = [
        'बहुत', 'कितना', 'क्या', 'वाह', 'शाबाश','तो', 'बड़ा','भी','ही'
    ]
    features['sarcasm_indicators'] = sum(1 for word in hindi_sarcasm_indicators if word in text)
    return features

