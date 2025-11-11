# Hindi Sarcasm Detection Project Workflow

This document details the complete workflow of the Hindi Sarcasm Detection project, from data preparation and preprocessing to model training, NLP techniques employed, library choices, and model evaluation.

## 1. Workflow Overview

This project is designed to detect sarcasm in Hindi text, supporting both Devanagari script and Romanized Hindi. The core workflow involves:

1.  **Data Preparation**: Loading and cleaning the Hindi sarcasm dataset.
2.  **Preprocessing**: Normalizing text, handling emojis, and preparing data for feature extraction.
3.  **Feature Engineering**: Extracting various numerical features using advanced NLP techniques.
4.  **Model Training**: Training a machine learning classifier on the engineered features.
5.  **Model Persistence**: Saving the trained model and vectorizer for future use.
6.  **Prediction/Inference**: Using the trained model to detect sarcasm in new Hindi text.
7.  **Deployment/Interfaces**: Providing multiple interfaces (CLI, Web App, API) for interaction.
8.  **Model Evaluation**: Testing the model's performance on unseen data and generating detailed reports.

## 2. Dataset Used for Training

The primary dataset used for training is `dataset/hindi_sarcasm_dataset.csv`. This CSV file contains Hindi text samples labeled as either sarcastic (1) or not sarcastic (0). The `train_model.py` script handles the loading and initial cleaning of this dataset.

**File**: `dataset/hindi_sarcasm_dataset.csv`
**Format**: `text,label`

**Code for Loading Dataset (`train_model.py` - `load_dataset` method)**:

```28:57:train_model.py
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
```

**Rationale**: `pandas` is used for efficient data loading and manipulation. The `try-except` blocks ensure robust CSV reading, handling various delimiters and malformed lines, while `on_bad_lines='skip'` prevents crashes due to corrupted entries. `encoding='utf-8'` is crucial for correctly handling Hindi characters.

## 3. Data Preprocessing

Preprocessing involves cleaning and normalizing the text data to make it suitable for feature extraction and model training. The `clean_dataset` and `preprocess_texts` methods within `HindiSarcasmTrainer` in `train_model.py` handle these steps.

**Code for Cleaning Dataset (`train_model.py` - `clean_dataset` method)**:

```67:99:train_model.py
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
```

**Rationale**: This function uses `pandas` operations to ensure data quality: dropping missing values, converting labels to integers, filtering invalid labels, removing duplicate text entries, and discarding very short texts. This prevents noisy data from negatively impacting model training.

**Code for Preprocessing Text (`train_model.py` - `preprocess_texts` method)**:

```161:182:train_model.py
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
```

**Rationale**: This method performs basic text normalization. `re.sub(r'\s+', ' ', text)` uses the `re` (regular expression) library to replace multiple whitespace characters with a single space, standardizing spacing. Emojis are intentionally kept as they can be crucial indicators of sarcasm in social media text.

## 4. NLP Techniques and Libraries

This project employs a combination of traditional and advanced NLP techniques to build a robust sarcasm detection model.

### 4.1 TF-IDF (Term Frequency-Inverse Document Frequency)

**What it is**: TF-IDF is a numerical statistic reflecting the importance of a word in a document relative to a collection of documents. It's calculated by multiplying the Term Frequency (how often a word appears in a document) by the Inverse Document Frequency (a measure of how rare the word is across all documents).

**Why it helps**: For Hindi sarcasm detection, TF-IDF with character-level n-grams is highly effective. Hindi is an agglutinative language, where words can be formed by combining morphemes. Character n-grams capture sub-word patterns that might be indicative of sarcasm, regardless of precise word boundaries. It also handles variations in Romanized Hindi.

**Libraries**: `sklearn.feature_extraction.text.TfidfVectorizer`

**Code (`train_model.py` - `train` method)**:

```210:249:train_model.py
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
```

**Explanation**: The `TfidfVectorizer` is configured with `ngram_range=(1, 4)` and `analyzer='char_wb'` to create character-level n-grams with word boundaries. `max_features` limits the vocabulary size, `min_df` and `max_df` filter out very rare or very common terms, and `strip_accents='unicode'` helps normalize Hindi diacritics. `sublinear_tf=True` applies a logarithmic scaling to term frequency, reducing the impact of highly frequent terms.

### 4.2 Feature Engineering

**What it is**: Feature engineering is the process of using domain knowledge to create new input features from raw data to improve the performance of machine learning models. In this project, `nlp_features.py` combines various NLP techniques to extract a rich set of features.

**Why it helps**: Sarcasm is complex and often subtle. A single type of feature might not capture all its nuances. By combining diverse features—linguistic, statistical, punctuation-based, sentiment-based, and emoji-based—the model gains a more comprehensive understanding of the text's potential sarcastic intent.

**Libraries**: `re`, `numpy`, `collections.Counter`

**File**: `nlp_features.py`

**Overview of Features Extracted**:

*   **Text Statistics** (e.g., character count, word count, average word length, punctuation ratio). Sarcastic text often exhibits unusual statistical properties. (See `extract_text_statistics`)
*   **Punctuation Pattern Analysis** (e.g., exclamation count, question count, ellipsis count, presence of repeated punctuation). Punctuation is critical for conveying tone in sarcasm. (See `extract_punctuation_features`)
*   **Linguistic Pattern Matching** (e.g., counts of specific sarcastic phrases, presence of intensifiers or dismissive words). Hindi has particular expressions and word usages that indicate sarcasm. (See `extract_linguistic_features`)
*   **Sentiment Analysis Integration** (e.g., sentiment polarity, sentiment score). Contradiction between apparent sentiment and actual intent is a hallmark of sarcasm. (See `extract_sentiment_features`)
*   **Emoji Detection & Analysis** (e.g., emoji count, presence of emojis). Emojis are frequently used in digital communication to convey sarcasm. (See `extract_emoji_features`)

**Code Example (`nlp_features.py` - `extract_punctuation_features`)**:

```44:67:nlp_features.py
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
```

**Explanation**: This function directly counts specific punctuation marks and checks for patterns (like `?!` or repeated marks) that are strongly associated with sarcastic tones. The `re` library is used for regex-based pattern matching.

### 4.3 Sentiment Analysis

**What it is**: Sentiment analysis is the process of identifying and extracting subjective information from text, determining the emotional tone (positive, negative, or neutral).

**Why it helps**: Sarcasm often presents a contradiction where positive words are used to convey negative sentiment, or vice versa. By integrating a Hindi-specific sentiment model, the project can detect these mismatches, which are strong indicators of sarcasm.

**Libraries**: `transformers` (HuggingFace, specifically `AutoTokenizer`, `AutoModelForSequenceClassification`, `pipeline`), `torch`.

**Code (`model.py` - `detect_sarcasm_level` method)**:

```120:134:model.py
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
```

**Explanation**: This section conditionally uses a pre-trained transformer model (like `ai4bharat/indic-bert`) from the HuggingFace `transformers` library to perform sentiment analysis. The `pipeline` function simplifies this process. If a sentiment model is available, it extracts the sentiment label (POSITIVE/NEGATIVE/NEUTRAL) and its confidence score, which can then be used as a feature to adjust the sarcasm probability.

### 4.4 Machine Learning Classification

**What it is**: Machine learning classification is the process of training an algorithm to categorize data into predefined classes. In this project, a `LogisticRegression` classifier is used to distinguish between sarcastic and non-sarcastic text.

**Why it helps**: After text is converted into numerical features (TF-IDF vectors and engineered features), a supervised learning algorithm like Logistic Regression can learn the complex patterns and relationships between these features and the sarcasm labels. It's a robust and interpretable model suitable for binary classification tasks.

**Libraries**: `sklearn.linear_model.LogisticRegression`, `numpy`, `pandas`.

**Code (`train_model.py` - `train` method)**:

```253:262:train_model.py
        # Train model
        print(f"\nTraining {model_type} classifier...")
        
        if model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000,
                C=1.0,
                class_weight='balanced',
                random_state=42
            )
```

**Explanation**: A `LogisticRegression` model is initialized with parameters like `max_iter` (maximum iterations for convergence), `C` (inverse of regularization strength), `class_weight='balanced'` (to handle potential class imbalance), and `random_state` for reproducibility. The model is then trained using `self.model.fit(X_train_tfidf, y_train)`.

### 4.5 Linguistic Pattern Matching (Rule-Based Fallback)

**What it is**: This technique involves using predefined rules and regular expressions to identify specific words, phrases, or structural patterns known to indicate sarcasm.

**Why it helps**: While machine learning models are powerful, explicit linguistic patterns can provide strong, direct signals of sarcasm, especially in languages like Hindi where certain phrases are conventionally used sarcastically. This rule-based approach serves as a robust fallback mechanism and also contributes to feature engineering.

**Libraries**: `re`

**Code (`model.py` - `_rule_based_detection` method)**:

```213:250:model.py
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
```

**Explanation**: This code block defines a list of Hindi and Romanized regex patterns that are commonly indicative of sarcasm. The `re.search` function is used to find these patterns in the input text, and a score is accumulated based on the presence of these patterns. This provides a direct, rule-based detection mechanism that complements the machine learning model.

## 5. Model Training and Persistence

After feature extraction, the `train_model.py` script trains the chosen machine learning model and then saves both the trained model and the TF-IDF vectorizer.

**Code (`train_model.py` - `main` function and `save_model` method)**:

```353:357:train_model.py
    # Train model
    accuracy = trainer.train(X, y, model_type='logistic')
    
    # Save model
    trainer.save_model()
```

```319:330:train_model.py
    def save_model(self, model_dir='models'):
        """Save the trained model and vectorizer"""
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'sarcasm_model.pkl')
        vectorizer_path = os.path.join(model_dir, 'sarcasm_vectorizer.pkl')
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        
        print(f"\nModel saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
```

**Rationale**: The `joblib` library is used for efficient serialization and deserialization of Python objects, making it ideal for saving and loading large `scikit-learn` models and vectorizers. This allows the trained model to be reused without retraining every time the application runs, which is crucial for deployment.

## 6. Model Testing and Evaluation

Model testing is crucial to assess the generalization performance of the trained model on unseen data. The `show_model_performance.py` script is responsible for this, generating comprehensive evaluation reports.

**Datasets Used for Testing**: The model has been tested on several datasets:

*   `dataset/hindi_sarcasm_dataset.csv` (the full training dataset, though typically evaluated on a split portion, here used as a test set for a full evaluation report).
*   `dataset/test_set.csv` (a dedicated test set).
*   `dataset/combined_test_set.csv` (a newly created dataset combining 45% random samples from `hindi_sarcasm_dataset.csv` and all of `test_set.csv`).

**Primary Evaluation Output**: The most recent comprehensive evaluation report is saved in `evaluation.txt`.

**Code for Evaluation (`show_model_performance.py` - `show_performance` function)**:

```55:59:show_model_performance.py
def show_performance(test_csv_path='dataset/test_set.csv', output_file_path=None):
    """Display model performance metrics"""
    
    if output_file_path:
        f = open(output_file_path, 'w', encoding='utf-8')
        def print_to_file(*args, **kwargs):
            print(*args, file=f, **kwargs)
        print_output = print_to_file
    else:
        print_output = print

    print_output("="*70)
    print_output("HINDI SARCASM DETECTION MODEL - PERFORMANCE METRICS")
    print_output("="*70)
```

**Rationale**: The `show_model_performance.py` script loads the trained model and vectorizer, preprocesses the test data, transforms it using the loaded vectorizer, and then makes predictions. It uses `sklearn.metrics` (e.g., `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `classification_report`, `confusion_matrix`, `roc_auc_score`) to calculate and display various performance metrics. The script is flexible enough to take any CSV file as a test set and can output results to a specified file, which is crucial for tracking model performance over time and across different datasets.
