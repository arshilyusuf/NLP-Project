"""
Sarcasm Detection Model for Hindi Text
Uses a trained machine learning model (TF-IDF + Classifier)
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import re
import os
import joblib


class HindiSarcasmDetector:
    """
    Main class for Hindi Sarcasm Detection
    Handles both Devanagari and Romanized Hindi
    """
    
    def __init__(self, model_name='ai4bharat/indic-bert', use_sentiment=True, use_trained_model=True):
        """
        Initialize the sarcasm detector
        
        Args:
            model_name: HuggingFace model name for Hindi text
            use_sentiment: Whether to use sentiment analysis as a feature
            use_trained_model: Whether to use trained ML model (if available)
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_sentiment = use_sentiment
        self.use_trained_model = use_trained_model
        
        self.ml_model = None
        self.vectorizer = None
        self.transformer_model = None
        self.tokenizer = None
        
        # Try to load trained ML model (primary method)
        if use_trained_model:
            try:
                model_path = os.path.join('models', 'sarcasm_model.pkl')
                vectorizer_path = os.path.join('models', 'sarcasm_vectorizer.pkl')
                
                if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                    self.ml_model = joblib.load(model_path)
                    self.vectorizer = joblib.load(vectorizer_path)
                    # Model loaded silently
                else:
                    # Model not found - will use rule-based fallback
                    pass
            except Exception as e:
                print(f"Could not load trained model: {e}")
                print("Using rule-based approach as fallback.")
        
        # Try to load transformer model (optional)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception:
            self.transformer_model = None
            self.tokenizer = None
        
        # Initialize sentiment analyzer if needed (optional)
        # Note: Sentiment models are large (1GB+) and take time to download
        # The system works perfectly fine without them using NLP features
        self.sentiment_pipeline = None
        if use_sentiment:
            # Skip automatic download - sentiment is optional enhancement
            # Uncomment below if you want to download sentiment models (requires ~1GB download)
            # try:
            #     self.sentiment_pipeline = pipeline(
            #         "sentiment-analysis",
            #         model="ai4bharat/indic-bert",
            #         device=0 if torch.cuda.is_available() else -1
            #     )
            # except:
            #     self.sentiment_pipeline = None
            pass  # Sentiment analysis disabled by default to avoid large downloads
    
    def detect_sarcasm_level(self, text):
        """
        Detect sarcasm level in Hindi text
        
        Args:
            text: Input text in Hindi (Devanagari or Romanized)
            
        Returns:
            dict with 'sarcasm_level' (0-100), 'is_sarcastic' (bool), and 'confidence'
        """
        if not text or len(text.strip()) == 0:
            return {
                'sarcasm_level': 0,
                'is_sarcastic': False,
                'confidence': 0.0,
                'message': 'Empty text provided'
            }
        
        # ====================================================================
        # PRIMARY NLP-BASED DETECTION PIPELINE
        # ====================================================================
        if self.ml_model is not None and self.vectorizer is not None:
            try:
                text_processed = str(text).strip()
                
                # ============================================================
                # NLP TECHNIQUE: TF-IDF Vectorization (Inference)
                # ============================================================
                # Transform input text to same numerical feature space as training
                # Uses the same TF-IDF vectorizer learned during training
                text_tfidf = self.vectorizer.transform([text_processed])
                
                # ============================================================
                # NLP TECHNIQUE: Machine Learning Classification
                # ============================================================
                # Use trained Logistic Regression model to predict sarcasm probability
                # predict_proba() returns probability distribution [P(not_sarcastic), P(sarcastic)]
                sarcasm_prob = self.ml_model.predict_proba(text_tfidf)[0][1]  # P(sarcastic)
                
                # ============================================================
                # NLP TECHNIQUE: Sentiment Analysis (Optional Enhancement)
                # ============================================================
                # Analyze emotional tone to detect sentiment-text mismatches
                # Sarcasm often involves positive words with negative intent
                sentiment_score = None
                if self.use_sentiment and self.sentiment_pipeline is not None:
                    try:
                        # Uses transformer-based sentiment model (IndicBERT for Hindi)
                        sentiment_result = self.sentiment_pipeline(text_processed)[0]
                        sentiment_score = {
                            'label': sentiment_result['label'],      # POSITIVE/NEGATIVE/NEUTRAL
                            'score': sentiment_result['score']       # Confidence score
                        }
                    except:
                        pass  # Continue without sentiment if unavailable
                
                # ============================================================
                # NLP TECHNIQUE: Feature Engineering & Extraction
                # ============================================================
                # Extract linguistic, statistical, and punctuation features
                # Combines multiple NLP techniques for comprehensive analysis
                from nlp_features import extract_all_features
                nlp_features = extract_all_features(text_processed, sentiment_score)
                
                # Adjust probability based on NLP features (slight boost/reduction)
                adjustment = 0.0
                
                # High punctuation often indicates sarcasm
                if nlp_features.get('exclamation_count', 0) > 1:
                    adjustment += 0.05
                if nlp_features.get('exclamation_question', 0) > 0:
                    adjustment += 0.1
                
                # Sentiment contradiction (positive sentiment but negative context)
                if sentiment_score:
                    if sentiment_score['label'] == 'POSITIVE' and nlp_features.get('sarcastic_pattern_count', 0) > 0:
                        adjustment += 0.05
                
                # Pattern matches boost
                if nlp_features.get('sarcastic_pattern_count', 0) > 0:
                    adjustment += 0.03 * nlp_features['sarcastic_pattern_count']
                
                # Apply adjustment (clamp between 0 and 1)
                sarcasm_prob = max(0.0, min(1.0, sarcasm_prob + adjustment))
                
                # Convert to 0-100 scale
                sarcasm_level = int(sarcasm_prob * 100)
                # Use threshold of 53% to reduce false positives
                # 50-53% range indicates uncertainty
                is_sarcastic = sarcasm_level >= 53
                confidence = abs(sarcasm_level - 50) / 50
                
                result = {
                    'sarcasm_level': sarcasm_level,
                    'is_sarcastic': is_sarcastic,
                    'confidence': round(confidence, 2),
                    'message': f"{'Sarcastic' if is_sarcastic else 'Not Sarcastic'} (Level: {sarcasm_level}%)"
                }
                
                # Add NLP insights if available
                if sentiment_score:
                    result['sentiment'] = sentiment_score['label'].lower()
                    result['sentiment_confidence'] = round(sentiment_score['score'], 2)
                
                return result
            except Exception as e:
                print(f"Error using ML model: {e}, falling back to rule-based")
        
        # Fallback to rule-based detection
        sarcasm_score = self._rule_based_detection(text)
        
        # Normalize to 0-100 scale
        sarcasm_level = min(100, max(0, int(sarcasm_score * 100)))
        is_sarcastic = sarcasm_level >= 53
        confidence = abs(sarcasm_level - 50) / 50
        
        return {
            'sarcasm_level': sarcasm_level,
            'is_sarcastic': is_sarcastic,
            'confidence': round(confidence, 2),
            'message': f"{'Sarcastic' if is_sarcastic else 'Not Sarcastic'} (Level: {sarcasm_level}%)"
        }
    
    def _rule_based_detection(self, text):
        """
        Rule-based sarcasm detection using linguistic patterns
        """
        score = 0.0
        original_text = text
        
        # Convert to lowercase for pattern matching (but keep original for word lists)
        text_lower = text.lower()
        
        # ========== PATTERN-BASED DETECTION ==========
        # Sarcastic patterns in Hindi (case-insensitive)
        # Includes both Devanagari and Romanized patterns
        sarcastic_patterns = [
            # Devanagari patterns
            (r'वाह\s*!*', 0.25),  # Wow!
            (r'क्या\s+बात\s+है', 0.3),  # What a thing
            (r'बहुत\s+अच्छा', 0.25),  # Very good (often sarcastic)
            (r'शाबाश', 0.25),  # Well done (often sarcastic)
            (r'कितना\s+अच्छा', 0.25),  # How good
            (r'ज़रूर\s+है', 0.2),  # Definitely
            (r'हाँ\s+हाँ', 0.2),  # Yes yes (dismissive)
            (r'बिल्कुल', 0.2),  # Absolutely (often sarcastic)
            (r'बहुत\s+बढ़िया', 0.25),  # Very nice (sarcastic)
            (r'क्या\s+कमाल', 0.3),  # What amazing
            (r'तो\s+बहुत\s+अच्छा', 0.3),  # Then very good
            (r'बड़ा\s+अच्छा', 0.25),  # Big good
            (r'पता\s+ही\s+नहीं\s+था', 0.35),  # Didn't even know (very sarcastic)
            (r'पता\s+नहीं\s+था', 0.3),  # Didn't know
            (r'हमें\s+तो\s+.*\s+पता\s+ही\s+नहीं', 0.4),  # We didn't even know
            (r'तो\s+.*\s+पता\s+ही\s+नहीं', 0.35),  # Didn't even know (with तो)
            (r'हमें\s+तो\s+.*\s+पता\s+नहीं', 0.35),  # We didn't know
            # Romanized patterns
            (r'pata\s+hi\s+nahi\s+tha', 0.35),  # Didn't even know
            (r'pata\s+nahi\s+tha', 0.3),  # Didn't know
            (r'hame\s+to\s+.*\s+pata\s+hi\s+nahi', 0.4),  # We didn't even know
            (r'to\s+.*\s+pata\s+hi\s+nahi', 0.35),  # Didn't even know (with to)
            (r'hame\s+to\s+.*\s+pata\s+nahi', 0.35),  # We didn't know
            (r'wow\s*!*', 0.25),  # Wow (Romanized)
            (r'kya\s+bat\s+hai', 0.3),  # What a thing
            (r'bahut\s+accha', 0.25),  # Very good
            (r'shabash', 0.25),  # Well done
        ]
        
        # Check for sarcastic patterns
        for pattern, weight in sarcastic_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += weight
        
        # ========== GOOD/BAD WORD COMBINATION DETECTION ==========
        # Comprehensive lists of positive and negative words in Hindi
        # Includes both Devanagari and common Romanized forms
        positive_words = [
            # Devanagari
            'अच्छा', 'बढ़िया', 'शानदार', 'कमाल', 'बेहतरीन', 'उत्कृष्ट',
            'सुंदर', 'सुखद', 'खुशी', 'प्रसन्न', 'संतुष्ट', 'आनंद',
            'सफल', 'जीत', 'विजय', 'उपलब्धि', 'सफलता', 'अच्छे',
            'अच्छी', 'बेहतर', 'महान', 'श्रेष्ठ', 'उत्तम', 'बढ़िया',
            # Romanized
            'accha', 'badhiya', 'shandar', 'kamaal', 'behtareen', 'achha',
            'khoobsoorat', 'khushi', 'prasann', 'santusht', 'anand', 'safal',
            'jeet', 'vijay', 'safalta', 'mahan', 'shreshth', 'uttam'
        ]
        
        negative_words = [
            # Devanagari
            'बुरा', 'खराब', 'गलत', 'नहीं', 'कभी नहीं', 'बिल्कुल नहीं',
            'असफल', 'हार', 'असफलता', 'दुख', 'उदास', 'निराश',
            'क्रोध', 'गुस्सा', 'नफरत', 'घृणा', 'बेकार', 'फालतू',
            'बेमतलब', 'अनावश्यक', 'खराबी', 'समस्या', 'मुश्किल',
            'कठिन', 'बुरे', 'बुरी', 'नकारात्मक',
            # Romanized
            'bura', 'kharab', 'galat', 'nahi', 'kabhi nahi', 'bilkul nahi',
            'asafal', 'haar', 'asafalta', 'dukh', 'udaas', 'niraash',
            'krodh', 'gussa', 'nafrat', 'ghrina', 'bekar', 'faltu',
            'bematlab', 'anavashyak', 'kharabi', 'samasya', 'mushkil',
            'kathin', 'bure', 'buri', 'nakaratmak', 'pata nahi', 'pata hi nahi'
        ]
        
        # Split text into words (handle both Devanagari and Romanized)
        # Use word boundaries for better matching
        words = re.findall(r'\b\w+\b', text_lower)
        
        # Check for good and bad words in the same sentence
        # Use word boundary matching to avoid substring matches
        # For Hindi, also check without word boundaries since spacing can vary
        def word_in_text(word, text):
            """Check if word exists in text with word boundaries"""
            # Try with word boundaries first (for Romanized/English)
            pattern_boundary = r'\b' + re.escape(word) + r'\b'
            if re.search(pattern_boundary, text, re.IGNORECASE):
                return True
            # For Hindi, also check as standalone word (with spaces/punctuation around it)
            # This handles cases where Hindi words might not have clear word boundaries
            pattern_hindi = r'(^|\s|[^\w])' + re.escape(word) + r'(\s|[^\w]|$)'
            return bool(re.search(pattern_hindi, text, re.IGNORECASE))
        
        has_positive = any(word_in_text(word, text_lower) for word in positive_words)
        has_negative = any(word_in_text(word, text_lower) for word in negative_words)
        
        if has_positive and has_negative:
            # Strong indicator of sarcasm - good and bad words together
            score += 0.4
            # Count how many of each for stronger signal
            positive_count = sum(1 for word in positive_words if word_in_text(word, text_lower))
            negative_count = sum(1 for word in negative_words if word_in_text(word, text_lower))
            if positive_count > 1 and negative_count > 0:
                score += 0.15
            if negative_count > 1 and positive_count > 0:
                score += 0.15
        
        # Check for positive words with negative context indicators
        if has_positive:
            # Look for words that negate or question the positive
            negating_words = ['लेकिन', 'पर', 'मगर', 'हालांकि', 'किंतु', 'तो', 'भी']
            has_negator = any(word_in_text(word, text_lower) for word in negating_words)
            if has_negator:
                score += 0.2
        
        # ========== PUNCTUATION ANALYSIS ==========
        # Exclamation marks (often indicate sarcasm)
        exclamation_count = text.count('!')
        if exclamation_count > 0:
            score += min(0.25, exclamation_count * 0.12)
        
        # Question marks with exclamations (very sarcastic)
        if '?!' in text or '!?' in text:
            score += 0.3
        
        # Multiple exclamation/question marks
        if exclamation_count > 1:
            score += 0.15
        
        # Ellipsis (often used in sarcasm)
        if '...' in text or '…' in text:
            score += 0.15
        
        # Quotation marks (air quotes - sarcastic)
        if '"' in text or "'" in text:
            score += 0.15
        
        # ========== CONTEXTUAL CLUES ==========
        # Dismissive or questioning phrases
        dismissive_phrases = [
            'है ना', 'है न', 'है क्या', 'है कि', 'तो है', 'भी है',
            'ज़रूर', 'बिल्कुल', 'क्या बात', 'क्या कहना',
            # Romanized
            'hai na', 'hai n', 'hai kya', 'to hai', 'bhi hai',
            'zaroor', 'bilkul', 'kya bat', 'kya kahna'
        ]
        
        for phrase in dismissive_phrases:
            if phrase in text_lower:
                score += 0.15
        
        # Sarcastic "obviousness" patterns - very common in Hindi sarcasm
        obviousness_patterns = [
            (r'तो\s+.*\s+पता\s+ही\s+नहीं', 0.4),  # "to ... pata hi nahi" pattern
            (r'to\s+.*\s+pata\s+hi\s+nahi', 0.4),  # Romanized
            (r'हमें\s+तो', 0.25),  # "hame to" - dismissive start
            (r'hame\s+to', 0.25),  # Romanized
            (r'तो\s+.*\s+ही', 0.2),  # "to ... hi" - emphasis pattern
            (r'to\s+.*\s+hi', 0.2),  # Romanized
        ]
        
        for pattern, weight in obviousness_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                score += weight
        
        # Tone indicators (uppercase, repeated letters)
        if text.isupper() and len(text) > 5:
            score += 0.2
        
        # Repeated punctuation (e.g., !!!, ???)
        if re.search(r'[!?]{2,}', text):
            score += 0.15
        
        # ========== WORD FREQUENCY ANALYSIS ==========
        # Check for intensifiers with positive words (often sarcastic)
        intensifiers = ['बहुत', 'कितना', 'कितने', 'बड़ा', 'बड़ी', 'ज़्यादा', 'अत्यधिक']
        has_intensifier = any(word_in_text(word, text_lower) for word in intensifiers)
        if has_intensifier and has_positive:
            score += 0.2
        
        # ========== SENTENCE STRUCTURE ==========
        # Check if sentence starts with positive but ends with negative implication
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            if len(sentence_lower) > 5:
                # Check if starts with positive word
                first_words = sentence_lower.split()[:3]
                starts_positive = any(word in first_words for word in positive_words[:5])
                # Check if has negative words later
                has_neg_later = any(word_in_text(word, sentence_lower) for word in negative_words)
                if starts_positive and has_neg_later:
                    score += 0.25
        
        # ========== FINAL SCORING ==========
        # Ensure minimum score for texts with any sarcastic indicators
        if score > 0.1:
            # Boost slightly to ensure it's above threshold
            score = min(1.0, score * 1.1)
        
        # Base score adjustment - if text is very short and has patterns, it might be sarcastic
        if len(text.split()) < 5 and score > 0.2:
            score = min(1.0, score * 1.2)
        
        # If we have positive words with intensifiers but no clear negative, still consider it
        # (common sarcastic pattern: "बहुत अच्छा" without explicit negative)
        if has_positive and has_intensifier and score > 0.3 and score < 0.5:
            score = min(1.0, score * 1.15)
        
        return min(1.0, max(0.0, score))
    
    
    def batch_detect(self, texts):
        """
        Detect sarcasm for multiple texts
        """
        results = []
        for text in texts:
            results.append(self.detect_sarcasm_level(text))
        return results


def create_detector():
    """
    Factory function to create a sarcasm detector
    """
    return HindiSarcasmDetector()

