import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import re
from preprocess import preprocess_hindi_text, detect_script
from transformers import AlbertForSequenceClassification, AlbertTokenizer
import torch
from nlp_features import extract_all_features, features_to_vector


class HindiSarcasmTrainer:
    # train a sarcasm detection model for hindi text
    
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.feature_names = None
        self.sentiment_model = None
        self.sentiment_tokenizer = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            model_name = "ai4bharat/indic-bert"
            self.sentiment_tokenizer = AlbertTokenizer.from_pretrained(model_name)
            self.sentiment_model = AlbertForSequenceClassification.from_pretrained(model_name).to(self.device)
            print("Sentiment analysis model and tokenizer loaded.")
        except Exception as e:
            print(f"Could not load sentiment analysis model: {e}")
            print("Sentiment analysis will be skipped during training.")

    def _get_sentiment_score(self, text):
        # get sentiment score for text
        if self.sentiment_model is None or self.sentiment_tokenizer is None:
            return None
        
        try:
            inputs = self.sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

            if len(probabilities) == 2:
                negative_prob = probabilities[0]
                positive_prob = probabilities[1]
                if positive_prob > negative_prob:
                    return {'label': 'POSITIVE', 'score': positive_prob}
                else:
                    return {'label': 'NEGATIVE', 'score': negative_prob}
            elif len(probabilities) == 3:
                negative_prob = probabilities[0]
                neutral_prob = probabilities[1]
                positive_prob = probabilities[2]
                
                max_prob_idx = np.argmax(probabilities)
                if max_prob_idx == 0:
                    return {'label': 'NEGATIVE', 'score': negative_prob}
                elif max_prob_idx == 1:
                    return {'label': 'NEUTRAL', 'score': neutral_prob}
                else:
                    return {'label': 'POSITIVE', 'score': positive_prob}
            else:
                return None

        except Exception as e:
            print(f"Error during sentiment prediction: {e}")
            return None
    
    def load_dataset(self, csv_path='dataset/hindi_sarcasm_dataset.csv'):
        # load dataset from csv file
        if not os.path.exists(csv_path):
            print(f"Dataset not found at {csv_path}")
            print("Creating sample dataset...")
            self.create_sample_dataset(csv_path)
        
        try:
            df = pd.read_csv(csv_path, encoding='utf-8', quotechar='"', skipinitialspace=True, on_bad_lines='skip')
        except:
            try:
                df = pd.read_csv(csv_path, encoding='utf-8', sep=',', quotechar='"', on_bad_lines='skip', engine='python')
            except:
                import csv
                rows = []
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f, quotechar='"', skipinitialspace=True)
                    header = next(reader)
                    for row in reader:
                        if len(row) >= 2:
                            rows.append({'text': row[0], 'label': row[1]})
                df = pd.DataFrame(rows)
        
        print(f"Loaded dataset: {len(df)} samples")
        
        df = self.clean_dataset(df)
        
        print(f"After cleaning: {len(df)} samples")
        print(f"Distribution: {df['label'].value_counts().to_dict()}")
        
        return df['text'].values, df['label'].values
    
    def clean_dataset(self, df):
        # clean the dataset
        original_len = len(df)
        
        df = df.dropna(subset=['text', 'label'])
        
        df['label'] = df['label'].astype(int)
        
        df = df[df['label'].isin([0, 1])]
        
        duplicates_removed = df.duplicated(subset=['text']).sum()
        df = df.drop_duplicates(subset=['text'], keep='first')
        
        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate texts")
        
        df = df[df['text'].str.len() >= 2]
        
        df = df.reset_index(drop=True)
        
        removed = original_len - len(df)
        if removed > 0:
            print(f"Removed {removed} invalid rows")
        
        return df
    
    def create_sample_dataset(self, csv_path):
        # create a sample dataset if none exists
        os.makedirs('dataset', exist_ok=True)
        
        sarcastic_texts = [
            "वाह! क्या बात है, तुमने तो कमाल कर दिया!",
            "बहुत अच्छा काम किया है तुमने",
            "शाबाश! तुमने तो सच में कमाल कर दिया!",
            "कितना अच्छा है यह मौसम...",
            "अच्छा है लेकिन बुरा भी है",
            "यह बहुत अच्छा है और बहुत खराब भी",
            "हमें तो ये पता ही नहीं था",
            "hame to ye pata hi nahi tha",
            "बिल्कुल सही कह रहे हो",
            "ज़रूर है, मैं मानता हूँ",
            "क्या कमाल है यह",
            "बड़ा अच्छा काम किया",
            "तो बहुत अच्छा है",
            "हाँ हाँ, बिल्कुल सही",
            "क्या बात है यार",
            "बहुत बढ़िया काम है",
            "शाबाश बेटा",
            "कितना बढ़िया है",
            "वाह वाह, क्या बात है",
            "बिल्कुल बढ़िया",
        ]
        
        non_sarcastic_texts = [
            "आज मौसम बहुत सुहावना है",
            "मैं आज स्कूल जा रहा हूँ",
            "यह एक बहुत अच्छी किताब है",
            "कल मेरी परीक्षा है",
            "मुझे खाना बहुत पसंद है",
            "आज का दिन बहुत अच्छा है",
            "मैं पढ़ाई कर रहा हूँ",
            "यह फिल्म बहुत अच्छी है",
            "मुझे संगीत सुनना पसंद है",
            "आज बारिश हो रही है",
            "मैं घर जा रहा हूँ",
            "यह एक सुंदर जगह है",
            "मुझे यह पसंद है",
            "आज का खाना बहुत स्वादिष्ट था",
            "मैं खुश हूँ",
            "यह एक अच्छा दिन है",
            "मुझे यह काम पसंद है",
            "आज मैंने कुछ नया सीखा",
            "यह बहुत उपयोगी है",
            "मैं इससे संतुष्ट हूँ",
        ]
        
        texts = sarcastic_texts + non_sarcastic_texts
        labels = [1] * len(sarcastic_texts) + [0] * len(non_sarcastic_texts)
        
        df = pd.DataFrame({'text': texts, 'label': labels})
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"Created sample dataset at {csv_path}")
        print(f"Total samples: {len(df)} (Sarcastic: {len(sarcastic_texts)}, Non-sarcastic: {len(non_sarcastic_texts)})")
    
    def preprocess_texts(self, texts):
        # preprocess texts for training
        processed = []
        for text in texts:
            text = str(text).strip()
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            processed.append(text)
        
        return processed
    
    def train(self, X, y, model_type='logistic', test_size=0.2):
        # train the sarcasm detection model
        print("\n" + "="*60)
        print("TRAINING HINDI SARCASM DETECTION MODEL")
        print("="*60)
        
        print("Preprocessing texts...")
        X_processed = self.preprocess_texts(X)
        
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training samples: {len(X_train_raw)}")
        print(f"Test samples: {len(X_test_raw)}")

        print("\n[NLP] Performing Sentiment Analysis...")
        train_sentiment_scores = []
        test_sentiment_scores = []
        if self.sentiment_model:
            train_sentiment_results = [self._get_sentiment_score(text) for text in X_train_raw]
            test_sentiment_results = [self._get_sentiment_score(text) for text in X_test_raw]
            for res in train_sentiment_results:
                train_sentiment_scores.append(res)
            for res in test_sentiment_results:
                test_sentiment_scores.append(res)
        else:
            print("Sentiment model not loaded, skipping sentiment feature extraction.")

        print("[NLP] Extracting comprehensive NLP features...")
        X_train_features = []
        for i, text in enumerate(X_train_raw):
            sentiment_score = train_sentiment_scores[i] if self.sentiment_model else None
            X_train_features.append(features_to_vector(extract_all_features(text, sentiment_score)))
        X_test_features = []
        for i, text in enumerate(X_test_raw):
            sentiment_score = test_sentiment_scores[i] if self.sentiment_model else None
            X_test_features.append(features_to_vector(extract_all_features(text, sentiment_score)))
        
        X_train_features = np.array(X_train_features)
        X_test_features = np.array(X_test_features)
        print(f"Engineered feature dimensions: {X_train_features.shape}")

        print("\n[NLP] Creating TF-IDF vectorizer (text -> numerical features)...")
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 4),
            min_df=2,
            max_df=0.95,
            lowercase=True,
            analyzer='char_wb',
            sublinear_tf=True,
            strip_accents='unicode'
        )
        
        print("[NLP] Vectorizing training data (text -> numerical features)...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train_raw)
        X_test_tfidf = self.vectorizer.transform(X_test_raw)
        
        print(f"TF-IDF feature dimensions: {X_train_tfidf.shape}")

        from scipy.sparse import hstack
        X_train_final = hstack([X_train_tfidf, X_train_features])
        X_test_final = hstack([X_test_tfidf, X_test_features])

        print(f"Combined feature dimensions: {X_train_final.shape}")
        
        print(f"\nTraining {model_type} classifier...")
        
        if model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000,
                C=1.0,
                class_weight='balanced',
                random_state=42
            )
        elif model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                class_weight='balanced',
                probability=True,
                random_state=42
            )
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                class_weight='balanced',
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.fit(X_train_final, y_train)
        
        print("\nEvaluating model...")
        y_train_pred = self.model.predict(X_train_final)
        y_test_pred = self.model.predict(X_test_final)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        print(f"\nTraining Accuracy: {train_acc:.2%}")
        print(f"Test Accuracy: {test_acc:.2%}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_test_pred, target_names=['Not Sarcastic', 'Sarcastic']))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_test_pred))
        
        return test_acc
    
    def predict_proba(self, text):
        # predict sarcasm probability for a single text
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained. Call train() first.")
        
        text_processed = self.preprocess_texts([text])[0]
        text_tfidf = self.vectorizer.transform([text_processed])
        proba = self.model.predict_proba(text_tfidf)[0]
        
        return proba[1]
    
    def save_model(self, model_dir='models'):
        # save the trained model and vectorizer
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'sarcasm_model.pkl')
        vectorizer_path = os.path.join(model_dir, 'sarcasm_vectorizer.pkl')
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        
        print(f"\nModel saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
    
    def load_model(self, model_dir='models'):
        # load a trained model and vectorizer
        model_path = os.path.join(model_dir, 'sarcasm_model.pkl')
        vectorizer_path = os.path.join(model_dir, 'sarcasm_vectorizer.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Model not found. Train a model first using train_model.py")
        
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        
        print("Model loaded successfully!")


def main():
    # main training function
    trainer = HindiSarcasmTrainer()
    
    X, y = trainer.load_dataset()
    accuracy = trainer.train(X, y, model_type='logistic')
    trainer.save_model()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Final Test Accuracy: {accuracy:.2%}")
    print("\nYou can now use the trained model in model.py")


if __name__ == '__main__':
    main()

