"""
Training Script for Hindi Sarcasm Detection Model
Uses TF-IDF features with a classifier (SVM/Logistic Regression)
"""

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


class HindiSarcasmTrainer:
    """Train a sarcasm detection model for Hindi text"""
    
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.feature_names = None
    
    def load_dataset(self, csv_path='dataset/hindi_sarcasm_dataset.csv'):
        """
        Load dataset from CSV file
        Expected format: text, label (0=not sarcastic, 1=sarcastic)
        Handles duplicates, emojis, and various text formats
        """
        if not os.path.exists(csv_path):
            print(f"Dataset not found at {csv_path}")
            print("Creating sample dataset...")
            self.create_sample_dataset(csv_path)
        
        # Read CSV with proper handling of commas in text
        try:
            df = pd.read_csv(csv_path, encoding='utf-8', quotechar='"', skipinitialspace=True, on_bad_lines='skip')
        except:
            try:
                df = pd.read_csv(csv_path, encoding='utf-8', sep=',', quotechar='"', on_bad_lines='skip', engine='python')
            except:
                # Last resort: read line by line
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
        
        # Clean the data
        df = self.clean_dataset(df)
        
        print(f"After cleaning: {len(df)} samples")
        print(f"Distribution: {df['label'].value_counts().to_dict()}")
        
        return df['text'].values, df['label'].values
    
    def clean_dataset(self, df):
        """
        Clean the dataset: remove duplicates, handle missing values, normalize
        """
        original_len = len(df)
        
        # Remove rows with missing text or label
        df = df.dropna(subset=['text', 'label'])
        
        # Convert label to int
        df['label'] = df['label'].astype(int)
        
        # Keep only valid labels (0 or 1)
        df = df[df['label'].isin([0, 1])]
        
        # Remove duplicate texts (keep first occurrence)
        duplicates_removed = df.duplicated(subset=['text']).sum()
        df = df.drop_duplicates(subset=['text'], keep='first')
        
        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate texts")
        
        # Remove very short texts (less than 2 characters)
        df = df[df['text'].str.len() >= 2]
        
        # Reset index
        df = df.reset_index(drop=True)
        
        removed = original_len - len(df)
        if removed > 0:
            print(f"Removed {removed} invalid rows")
        
        return df
    
    def create_sample_dataset(self, csv_path):
        """Create a sample dataset if none exists"""
        os.makedirs('dataset', exist_ok=True)
        
        # Sample sarcastic and non-sarcastic examples
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
        
        # Create DataFrame
        texts = sarcastic_texts + non_sarcastic_texts
        labels = [1] * len(sarcastic_texts) + [0] * len(non_sarcastic_texts)
        
        df = pd.DataFrame({'text': texts, 'label': labels})
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"Created sample dataset at {csv_path}")
        print(f"Total samples: {len(df)} (Sarcastic: {len(sarcastic_texts)}, Non-sarcastic: {len(non_sarcastic_texts)})")
    
    def preprocess_texts(self, texts):
        """
        Preprocess texts for training
        Handles emojis, extra spaces, and normalization
        """
        processed = []
        for text in texts:
            # Convert to string and strip
            text = str(text).strip()
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Keep emojis (they can be useful for sarcasm detection)
            # But normalize multiple spaces around them
            text = text.strip()
            
            # Basic normalization - keep original text structure
            # Don't remove emojis as they can indicate sarcasm
            processed.append(text)
        
        return processed
    
    def train(self, X, y, model_type='logistic', test_size=0.2):
        """
        Train the sarcasm detection model
        
        Args:
            X: List of texts
            y: Labels (0 or 1)
            model_type: 'logistic', 'svm', or 'random_forest'
            test_size: Proportion of data for testing
        """
        print("\n" + "="*60)
        print("TRAINING HINDI SARCASM DETECTION MODEL")
        print("="*60)
        
        # Preprocess texts
        print("Preprocessing texts...")
        X_processed = self.preprocess_texts(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # ====================================================================
        # NLP TECHNIQUE: TF-IDF (Term Frequency-Inverse Document Frequency)
        # ====================================================================
        # TF-IDF is a statistical measure used to evaluate how important a word
        # is to a document in a collection. It helps identify distinctive features.
        #
        # Formula: TF-IDF(t,d) = TF(t,d) × IDF(t)
        # - TF (Term Frequency): How often a term appears in a document
        # - IDF (Inverse Document Frequency): How rare/common a term is across all documents
        #
        # Why TF-IDF for Hindi Sarcasm Detection:
        # - Captures important character patterns that indicate sarcasm
        # - Works with both Devanagari and Romanized Hindi
        # - Handles emojis and mixed scripts effectively
        # ====================================================================
        print("\n[NLP] Creating TF-IDF vectorizer...")
        self.vectorizer = TfidfVectorizer(
            max_features=10000,      # Limit to top 10,000 most important features
            ngram_range=(1, 4),      # Character n-grams: 1-char, 2-char, 3-char, 4-char sequences
                                     # Example: "वाह" → ['व', 'ा', 'ह', 'वा', 'ाह', 'वाह']
            min_df=2,                 # Ignore features appearing in < 2 documents (noise reduction)
            max_df=0.95,              # Ignore features in >95% documents (too common)
            lowercase=True,           # Normalize case for consistent features
            analyzer='char_wb',       # Character-level n-grams with word boundaries
                                     # Better for Hindi (agglutinative language) than word-level
            sublinear_tf=True,        # Apply log scaling: 1 + log(tf) instead of raw tf
                                     # Reduces impact of very frequent terms
            strip_accents='unicode'   # Handle Hindi diacritics properly
        )
        
        # ====================================================================
        # NLP TECHNIQUE: VECTORIZATION (Text to Numerical Features)
        # ====================================================================
        # Transform text data into numerical vectors that ML models can process
        # - fit_transform(): Learn vocabulary from training data and transform it
        # - transform(): Use learned vocabulary to transform test data
        # ====================================================================
        print("[NLP] Vectorizing training data (text → numerical features)...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)  # Learn + Transform
        X_test_tfidf = self.vectorizer.transform(X_test)       # Transform only
        
        print(f"Feature dimensions: {X_train_tfidf.shape}")
        
        # Train model
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
        
        # Train
        self.model.fit(X_train_tfidf, y_train)
        
        # Evaluate
        print("\nEvaluating model...")
        y_train_pred = self.model.predict(X_train_tfidf)
        y_test_pred = self.model.predict(X_test_tfidf)
        
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
        """Predict sarcasm probability for a single text"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Preprocess
        text_processed = self.preprocess_texts([text])[0]
        
        # Transform
        text_tfidf = self.vectorizer.transform([text_processed])
        
        # Predict probability
        proba = self.model.predict_proba(text_tfidf)[0]
        
        return proba[1]  # Probability of being sarcastic
    
    def save_model(self, model_dir='models'):
        """Save the trained model and vectorizer"""
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'sarcasm_model.pkl')
        vectorizer_path = os.path.join(model_dir, 'sarcasm_vectorizer.pkl')
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        
        print(f"\nModel saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
    
    def load_model(self, model_dir='models'):
        """Load a trained model and vectorizer"""
        model_path = os.path.join(model_dir, 'sarcasm_model.pkl')
        vectorizer_path = os.path.join(model_dir, 'sarcasm_vectorizer.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Model not found. Train a model first using train_model.py")
        
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        
        print("Model loaded successfully!")


def main():
    """Main training function"""
    trainer = HindiSarcasmTrainer()
    
    # Load dataset
    X, y = trainer.load_dataset()
    
    # Train model
    accuracy = trainer.train(X, y, model_type='logistic')
    
    # Save model
    trainer.save_model()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Final Test Accuracy: {accuracy:.2%}")
    print("\nYou can now use the trained model in model.py")


if __name__ == '__main__':
    main()

