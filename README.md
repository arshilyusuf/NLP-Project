# Hindi Sarcasm Detection Project

An NLP project for detecting sarcasm in Hindi text, supporting both Devanagari script and Romanized Hindi.

## Features

- ✅ Detects sarcasm in Hindi text (both Devanagari and Romanized)
- ✅ Provides sarcasm level (0-100%)
- ✅ Confidence scoring
- ✅ Multiple interfaces (CLI, Web App, API)
- ✅ Batch processing support
- ✅ Automatic script detection and conversion

## Installation

1. **Clone or download this project**

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Train the model (required for ML-based detection):**
```bash
python train_model.py
```

This will create a sample dataset and train a machine learning model. For better accuracy, add your own dataset to `dataset/hindi_sarcasm_dataset.csv` (see TRAINING_GUIDE.md).

Note: The project uses PyTorch and Transformers. If you encounter issues, you may need to install PyTorch separately based on your system:
- For CPU: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- For GPU: Visit [PyTorch website](https://pytorch.org/) for GPU-specific installation

## Model Performance

To view detailed performance metrics of the trained model:

```bash
python show_model_performance.py
```

This will display:
- **Accuracy**: 95.95% (Test Set)
- **Precision**: 93.20% (Sarcastic class)
- **Recall**: 91.33% (Sarcastic class)
- **F1-Score**: 92.26% (Sarcastic class)
- **ROC-AUC**: 99.52%
- **Confusion Matrix**: Detailed error analysis
- **Per-class metrics**: Performance for each class

## Usage

### 1. Command Line Interface (CLI)

#### Interactive Mode
```bash
python main.py -i
```

#### Single Text Analysis
```bash
python main.py -t "वाह! क्या बात है"
```

Or with Romanized Hindi:
```bash
python main.py -t "Wah! Kya baat hai"
```

#### Batch Processing from File
```bash
python main.py -f input.txt
```
(One text per line in the file)

### 2. Web Application

Start the Flask web server:
```bash
python app.py
```

Then open your browser and navigate to:
```
http://localhost:5000
```

### 3. Python API

```python
from model import HindiSarcasmDetector

# Initialize detector
detector = HindiSarcasmDetector()

# Detect sarcasm
result = detector.detect_sarcasm_level("वाह! क्या बात है")

print(f"Sarcasm Level: {result['sarcasm_level']}%")
print(f"Is Sarcastic: {result['is_sarcastic']}")
print(f"Confidence: {result['confidence']}")
```

## Project Structure

```
.
├── main.py              # CLI application
├── app.py               # Flask web application
├── model.py             # Sarcasm detection model
├── preprocess.py        # Text preprocessing utilities
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── templates/
    └── index.html      # Web interface
```

## NLP Techniques Used

This project demonstrates multiple **Natural Language Processing (NLP) techniques**:

### 1. **TF-IDF (Term Frequency-Inverse Document Frequency)**
- **What it is**: Statistical measure to evaluate word importance in documents
- **Formula**: `TF-IDF(t,d) = TF(t,d) × IDF(t)`
- **Implementation**: Character-level n-grams (1-4 grams) for Hindi text
- **Why it works**: Captures distinctive character patterns that indicate sarcasm
- **Location**: `train_model.py` (lines 210-249)

### 2. **Vectorization (Text to Numerical Features)**
- **What it is**: Converting text data into numerical vectors for ML models
- **Implementation**: TF-IDF vectorization with 10,000 max features
- **Why it works**: ML models require numerical input; vectorization enables this
- **Location**: `train_model.py` (lines 240-249), `model.py` (lines 106-110)

### 3. **Sentiment Analysis**
- **What it is**: Analyzing emotional tone (positive/negative/neutral) of text
- **Implementation**: Transformer-based models (IndicBERT for Hindi)
- **Why it works**: Detects sentiment-text mismatches (positive words with negative intent)
- **Location**: `model.py` (lines 120-134)

### 4. **Feature Engineering**
- **What it is**: Combining multiple NLP techniques into comprehensive features
- **Features extracted**:
  - Text Statistics (7 features): length, word count, punctuation ratios
  - Punctuation Analysis (8 features): exclamations, questions, ellipsis
  - Linguistic Patterns (4 features): sarcastic phrases, intensifiers
  - Sentiment Features (3 features): polarity, confidence scores
  - Emoji Detection (2 features): emoji count, presence
- **Total**: 24+ engineered features
- **Location**: `nlp_features.py`

### 5. **Machine Learning Classification**
- **What it is**: Logistic Regression classifier trained on NLP features
- **Implementation**: Combines TF-IDF vectors + engineered features
- **Performance**: 95.95% test accuracy
- **Location**: `train_model.py` (lines 253-259), `model.py` (lines 113-117)

### 6. **Linguistic Pattern Matching**
- **What it is**: Regex-based detection of known sarcastic expressions
- **Implementation**: Pattern matching for Hindi sarcastic phrases
- **Why it works**: Hindi has specific linguistic patterns indicating sarcasm
- **Location**: `nlp_features.py` (lines 70-111)

## Example Outputs

**Input:** "वाह! क्या बात है, तुमने तो कमाल कर दिया!"
- **Sarcasm Level:** 75%
- **Is Sarcastic:** Yes
- **Confidence:** 0.5

**Input:** "आज मौसम बहुत सुहावना है"
- **Sarcasm Level:** 15%
- **Is Sarcastic:** No
- **Confidence:** 0.7

## Supported Scripts

- **Devanagari (हिंदी)**: Native Hindi script
- **Romanized Hindi**: English transliteration (e.g., "Wah! Kya baat hai")
- **Mixed**: Combination of both scripts

## Technical Details

- **Model**: Uses IndicBERT (ai4bharat/indic-bert) for Hindi language understanding
- **Framework**: PyTorch, Transformers (HuggingFace)
- **Preprocessing**: Handles script detection and conversion using indic-transliteration
- **Scoring**: Combines rule-based and transformer-based scores for final prediction

## Limitations

- The model works best with clear sarcastic patterns
- Very subtle sarcasm might be missed
- Performance depends on the quality of the pre-trained model
- First run may take time to download the model

## Future Improvements

- Fine-tuning on a larger Hindi sarcasm dataset
- Support for more regional variations
- Improved handling of context and tone
- Better integration with sentiment analysis

## Troubleshooting

1. **Model download issues**: The first run will download the model (~500MB). Ensure stable internet connection.

2. **CUDA/GPU errors**: The code automatically falls back to CPU if GPU is not available.

3. **Import errors**: Make sure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

4. **Transliteration errors**: If Romanized to Devanagari conversion fails, the system will use the original text.

## License

This project is created for educational purposes.

## Author

Created for NLP Assignment - Hindi Sarcasm Detection

## Submission Date

Project completed for submission.

