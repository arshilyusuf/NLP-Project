import torch
import numpy as np
from preprocess import preprocess_hindi_text
from nlp_features import extract_all_features, features_to_vector
from transformers import AlbertForSequenceClassification, AlbertTokenizer
import os
import joblib


class HindiSarcasmDetector:
    # main class for hindi sarcasm detection
    
    def __init__(self, use_sentiment=True, use_trained_model=True):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_sentiment = use_sentiment
        self.use_trained_model = use_trained_model
        
        self.ml_model = None
        self.vectorizer = None
        self.transformer_model = None
        self.tokenizer = None

        self.sentiment_model = None
        self.sentiment_tokenizer = None
        
        if use_trained_model:
            try:
                model_path = os.path.join('models', 'sarcasm_model.pkl')
                vectorizer_path = os.path.join('models', 'sarcasm_vectorizer.pkl')
                
                if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                    self.ml_model = joblib.load(model_path)
                    self.vectorizer = joblib.load(vectorizer_path)
                else:
                    pass
            except Exception as e:
                print(f"Could not load trained model: {e}")
                print("Using rule-based approach as fallback.")
        
        try:
            model_name = "ai4bharat/indic-bert"
            self.tokenizer = AlbertTokenizer.from_pretrained(model_name)
        except Exception:
            self.transformer_model = None
            self.tokenizer = None
        
        if self.use_sentiment:
            try:
                model_name = "ai4bharat/indic-bert"
                self.sentiment_tokenizer = AlbertTokenizer.from_pretrained(model_name)
                self.sentiment_model = AlbertForSequenceClassification.from_pretrained(model_name).to(self.device)
                print("Sentiment analysis model and tokenizer loaded for inference.")
            except Exception as e:
                print(f"Could not load sentiment analysis model for inference: {e}")
                print("Sentiment analysis will be skipped.")
    
        # detect sarcasm level in hindi text
    def detect_sarcasm_level(self, text):
        sarcasm_prob = 0.5
        sentiment_score = None

        text_processed = preprocess_hindi_text(text)
        
        if self.ml_model is not None and self.vectorizer is not None:
            try:  
                if self.use_sentiment and self.sentiment_model is not None:
                    try:
                        inputs = self.sentiment_tokenizer(text_processed, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
                        with torch.no_grad():
                            outputs = self.sentiment_model(**inputs)
                            logits = outputs.logits
                            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

                        if len(probabilities) == 3:
                            negative_prob = probabilities[0]
                            neutral_prob = probabilities[1]
                            positive_prob = probabilities[2]
                            
                            max_prob_idx = np.argmax(probabilities)
                            if max_prob_idx == 0:
                                sentiment_score = {'label': 'NEGATIVE', 'score': negative_prob}
                            elif max_prob_idx == 1:
                                sentiment_score = {'label': 'NEUTRAL', 'score': neutral_prob}
                            else:
                                sentiment_score = {'label': 'POSITIVE', 'score': positive_prob}
                        elif len(probabilities) == 2:
                            negative_prob = probabilities[0]
                            positive_prob = probabilities[1]
                            if positive_prob > negative_prob:
                                sentiment_score = {'label': 'POSITIVE', 'score': positive_prob}
                            else:
                                sentiment_score = {'label': 'NEGATIVE', 'score': negative_prob}

                    except Exception as e:
                        print(f"Error during sentiment prediction for inference: {e}")
                        sentiment_score = None

                X_tfidf = self.vectorizer.transform([text_processed])
                nlp_features = extract_all_features(text_processed, sentiment_score)
                X_nlp_features = features_to_vector(nlp_features)

                from scipy.sparse import hstack
                X_combined_features = hstack([X_tfidf, X_nlp_features.reshape(1, -1)])

                sarcasm_prob = self.ml_model.predict_proba(X_combined_features)[:, 1][0]

                adjustment = 0
                if sentiment_score:
                    if sentiment_score['label'] == 'POSITIVE' and nlp_features.get('sarcastic_pattern_count', 0) > 0:
                        adjustment += 0.05
                    elif sentiment_score['label'] == 'NEGATIVE' and nlp_features.get('positive_word_count', 0) > 0 and nlp_features.get('sarcastic_pattern_count', 0) == 0:
                        adjustment -= 0.02
                sarcasm_prob = max(0.0, min(1.0, sarcasm_prob + adjustment))
                
            except Exception as e:
                print(f"Error during ML model prediction: {e}")
                sarcasm_prob = self._rule_based_detection(text_processed)
        else:
            sarcasm_prob = self._rule_based_detection(text_processed)
            
        sarcastic = sarcasm_prob >= 0.5
        confidence = abs(sarcasm_prob - 0.5) * 2

        return {
            'text': text,
            'sarcastic': sarcastic,
            'confidence': confidence,
            'sarcasm_score': sarcasm_prob,
            'script_type': 'Unknown',
            'sentiment': sentiment_score
        }
    
    def _rule_based_detection(self, text):
        # rule-based sarcasm detection using linguistic patterns
        score = 0.0
        original_text = text
        
        text_lower = text.lower()
        
        sarcastic_patterns = [
            (r'वाह\s*!', 0.25),
            (r'क्या\s+बात\s+है', 0.3),
            (r'बहुत\s+अच्छा', 0.25),
            (r'शाबाश', 0.25),
            (r'कितना\s+अच्छा', 0.25),
            (r'ज़रूर\s+है', 0.2),
            (r'हाँ\s+हाँ', 0.2),
            (r'बिल्कुल', 0.2),
            (r'बहुत\s+बढ़िया', 0.25),
            (r'क्या\s+कमाल', 0.3),
            (r'तो\s+बहुत\s+अच्छा', 0.3),
            (r'बड़ा\s+अच्छा', 0.25),
            (r'पता\s+ही\s+नहीं\s+था', 0.35),
            (r'पता\s+नहीं\s+था', 0.3),
            (r'हमें\s+तो\s+.*\s+पता\s+ही\s+नहीं', 0.4),
            (r'तो\s+.*\s+पता\s+ही\s+नहीं', 0.35),
            (r'हमें\s+तो\s+.*\s+पता\s+नहीं', 0.35),
            (r'pata\s+hi\s+nahi\s+tha', 0.35),
            (r'pata\s+nahi\s+tha', 0.3),
            (r'hame\s+to\s+.*\s+pata\s+hi\s+nahi', 0.4),
            (r'to\s+.*\s+pata\s+hi\s+nahi', 0.35),
            (r'hame\s+to\s+.*\s+pata\s+nahi', 0.35),
            (r'wow\s*!', 0.25),
            (r'kya\s+bat\s+hai', 0.3),
            (r'bahut\s+accha', 0.25),
            (r'shabash', 0.25),
        ]
        
        for pattern, weight in sarcastic_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += weight
        
        positive_words = [
            'अच्छा', 'बढ़िया', 'शानदार', 'कमाल', 'बेहतरीन', 'उत्कृष्ट',
            'सुंदर', 'सुखद', 'खुशी', 'प्रसन्न', 'संतुष्ट', 'आनंद',
            'सफल', 'जीत', 'विजय', 'उपलब्धि', 'सफलता', 'अच्छे',
            'अच्छी', 'बेहतर', 'महान', 'श्रेष्ठ', 'उत्तम', 'बढ़िया',
            'accha', 'badhiya', 'shandar', 'kamaal', 'behtareen', 'achha',
            'khoobsoorat', 'khushi', 'prasann', 'santusht', 'anand', 'safal',
            'jeet', 'vijay', 'safalta', 'mahan', 'shreshth', 'uttam'
        ]
        
        negative_words = [
            'बुरा', 'खराब', 'गलत', 'नहीं', 'कभी नहीं', 'बिल्कुल नहीं',
            'असफल', 'हार', 'असफलता', 'दुख', 'उदास', 'निराश',
            'क्रोध', 'गुस्सा', 'नफरत', 'घृणा', 'बेकार', 'फालतू',
            'बेमतलब', 'अनावश्यक', 'खराबी', 'समस्या', 'मुश्किल',
            'कठिन', 'बुरे', 'बुरी', 'नकारात्मक',
            'bura', 'kharab', 'galat', 'nahi', 'kabhi nahi', 'bilkul nahi',
            'asafal', 'haar', 'asafalta', 'dukh', 'udaas', 'niraash',
            'krodh', 'gussa', 'nafrat', 'ghrina', 'bekar', 'faltu',
            'bematlab', 'anavashyak', 'kharabi', 'samasya', 'mushkil',
            'kathin', 'bure', 'buri', 'nakaratmak', 'pata nahi', 'pata hi nahi'
        ]
        
        words = re.findall(r'\b\w+\b', text_lower)
        
        def word_in_text(word, text):
            # check if word exists in text with word boundaries
            pattern_boundary = r'\b' + re.escape(word) + r'\b'
            if re.search(pattern_boundary, text, re.IGNORECASE):
                return True
            pattern_hindi = r'(^|\s|[^\w])' + re.escape(word) + r'(\s|[^\w]|$)'
            return bool(re.search(pattern_hindi, text, re.IGNORECASE))
        
        has_positive = any(word_in_text(word, text_lower) for word in positive_words)
        has_negative = any(word_in_text(word, text_lower) for word in negative_words)
        
        if has_positive and has_negative:
            score += 0.4
            positive_count = sum(1 for word in positive_words if word_in_text(word, text_lower))
            negative_count = sum(1 for word in negative_words if word_in_text(word, text_lower))
            if positive_count > 1 and negative_count > 0:
                score += 0.15
            if negative_count > 1 and positive_count > 0:
                score += 0.15
        
        if has_positive:
            negating_words = ['लेकिन', 'पर', 'मगर', 'हालांकि', 'किंतु', 'तो', 'भी']
            has_negator = any(word_in_text(word, text_lower) for word in negating_words)
            if has_negator:
                score += 0.2
        
        exclamation_count = text.count('!')
        if exclamation_count > 0:
            score += min(0.25, exclamation_count * 0.12)
        
        if '?!' in text or '!?' in text:
            score += 0.3
        
        if exclamation_count > 1:
            score += 0.15
        
        if '...' in text or '…' in text:
            score += 0.15
        
        if '"' in text or "'" in text:
            score += 0.15
        
        sarcastic_patterns = [
            (r'वाह\s*!', 0.25),
            (r'क्या\s+बात\s+है', 0.3),
            (r'बहुत\s+अच्छा', 0.25),
            (r'शाबाश', 0.25),
            (r'कितना\s+अच्छा', 0.25),
            (r'ज़रूर\s+है', 0.2),
            (r'हाँ\s+हाँ', 0.2),
            (r'बिल्कुल', 0.2),
            (r'बहुत\s+बढ़िया', 0.25),
            (r'क्या\s+कमाल', 0.3),
            (r'तो\s+बहुत\s+अच्छा', 0.3),
            (r'बड़ा\s+अच्छा', 0.25),
            (r'पता\s+ही\s+नहीं\s+था', 0.35),
            (r'पता\s+नहीं\s+था', 0.3),
            (r'हमें\s+तो\s+.*\s+पता\s+ही\s+नहीं', 0.4),
            (r'तो\s+.*\s+पता\s+ही\s+नहीं', 0.35),
            (r'हमें\s+तो\s+.*\s+पता\s+नहीं', 0.35),
            (r'pata\s+hi\s+nahi\s+tha', 0.35),
            (r'pata\s+nahi\s+tha', 0.3),
            (r'hame\s+to\s+.*\s+pata\s+hi\s+nahi', 0.4),
            (r'to\s+.*\s+pata\s+hi\s+nahi', 0.35),
            (r'hame\s+to\s+.*\s+pata\s+nahi', 0.35),
            (r'wow\s*!', 0.25),
            (r'kya\s+bat\s+hai', 0.3),
            (r'bahut\s+accha', 0.25),
            (r'shabash', 0.25),
        ]
        
        for pattern, weight in sarcastic_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                score += weight
        
        if text.isupper() and len(text) > 5:
            score += 0.2
        
        if re.search(r'[!?]{2,}', text):
            score += 0.15
        
        intensifiers = ['बहुत', 'कितना', 'कितने', 'बड़ा', 'बड़ी', 'ज़्यादा', 'अत्यधिक']
        has_intensifier = any(word_in_text(word, text_lower) for word in intensifiers)
        if has_intensifier and has_positive:
            score += 0.2
        
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            if len(sentence_lower) > 5:
                first_words = sentence_lower.split()[:3]
                starts_positive = any(word in first_words for word in positive_words[:5])
                has_neg_later = any(word_in_text(word, sentence_lower) for word in negative_words)
                if starts_positive and has_neg_later:
                    score += 0.25
        
        if score > 0.1:
            score = min(1.0, score * 1.1)
        
        if len(text.split()) < 5 and score > 0.2:
            score = min(1.0, score * 1.2)
        
        if has_positive and has_intensifier and score > 0.3 and score < 0.5:
            score = min(1.0, score * 1.15)
        
        return min(1.0, max(0.0, score))
    
    def batch_detect(self, texts):
        # detect sarcasm for multiple texts
        results = []
        for text in texts:
            results.append(self.detect_sarcasm_level(text))
        return results

def create_detector():
    # factory function to create a sarcasm detector
    return HindiSarcasmDetector()

