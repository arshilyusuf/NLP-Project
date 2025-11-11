"""
Model Performance Evaluation Script
Displays comprehensive performance metrics for the trained Hindi Sarcasm Detection model

This script demonstrates NLP evaluation techniques:
1. Train/Test Split - Separates data for unbiased evaluation
2. TF-IDF Vectorization - Transforms test data using learned features
3. Classification Metrics - Accuracy, Precision, Recall, F1-Score
4. Confusion Matrix - Detailed error analysis
5. ROC-AUC - Area under ROC curve for binary classification
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import train_test_split
import joblib
import os
import re
from train_model import HindiSarcasmTrainer


def load_dataset(csv_path):
    """Load and clean dataset"""
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
    
    # Clean
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['text', 'label'])
    df['label'] = df['label'].astype(int)
    df = df[df['label'].isin([0, 1])]
    df = df.drop_duplicates(subset=['text'], keep='first')
    df = df[df['text'].str.len() >= 2]
    df = df.reset_index(drop=True)
    
    return df['text'].values, df['label'].values


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
    
    # Check if model exists
    model_path = os.path.join('models', 'sarcasm_model.pkl')
    vectorizer_path = os.path.join('models', 'sarcasm_vectorizer.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print_output("\n❌ Model not found!")
        print_output("Please train the model first by running: python train_model.py")
        return
    
    # Load model and vectorizer
    print_output("\n📦 Loading trained model...")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print_output("✓ Model loaded successfully!")
    
    # Load test dataset
    print_output("\n📊 Loading test dataset from {test_csv_path}...")
    X_test_raw, y_test = load_dataset(test_csv_path)
    print_output(f"✓ Loaded {len(X_test_raw)} test samples")
    
    # Preprocess texts
    print_output("\n🔧 Preprocessing texts...")
    trainer = HindiSarcasmTrainer()
    X_test_processed = trainer.preprocess_texts(X_test_raw)
    
    print_output(f"\n📈 Test samples: {len(X_test_processed)}")
    
    # Transform texts
    print_output("\n🔄 Transforming texts to TF-IDF features...")
    X_test_tfidf = vectorizer.transform(X_test_processed)
    
    # Predictions
    print_output("\n🔮 Making predictions...")
    y_test_pred = model.predict(X_test_tfidf)
    y_test_proba = model.predict_proba(X_test_tfidf)[:, 1]
    
    # Calculate metrics
    print_output("\n" + "="*70)
    print_output("PERFORMANCE METRICS")
    print_output("="*70)
    
    # Test metrics
    test_acc = accuracy_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred, zero_division=0)
    test_rec = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    
    # ROC AUC
    try:
        roc_auc = roc_auc_score(y_test, y_test_proba)
    except Exception as e:
        print_output(f"Could not calculate ROC-AUC: {e}")
        roc_auc = None
    
    print_output("\n📊 TEST SET PERFORMANCE:")
    print_output(f"   Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print_output(f"   Precision: {test_prec:.4f} ({test_prec*100:.2f}%)")
    print_output(f"   Recall:    {test_rec:.4f} ({test_rec*100:.2f}%)")
    print_output(f"   F1-Score:  {test_f1:.4f} ({test_f1*100:.2f}%)")
    if roc_auc:
        print_output(f"   ROC-AUC:   {roc_auc:.4f} ({roc_auc*100:.2f}%)")
    
    # Classification Report
    print_output("\n" + "="*70)
    print_output("DETAILED CLASSIFICATION REPORT (Test Set)")
    print_output("="*70)
    print_output(classification_report(y_test, y_test_pred, 
                                target_names=['Not Sarcastic', 'Sarcastic'],
                                digits=4))
    
    # Confusion Matrix
    print_output("\n" + "="*70)
    print_output("CONFUSION MATRIX (Test Set)")
    print_output("="*70)
    cm = confusion_matrix(y_test, y_test_pred)
    print_output("\n                Predicted")
    print_output("              Not Sarc  Sarcastic")
    print_output(f"Actual Not Sarc  {cm[0][0]:5d}     {cm[0][1]:5d}")
    print_output(f"      Sarcastic  {cm[1][0]:5d}     {cm[1][1]:5d}")
    
    print_output("\n" + "-"*70)
    print_output("Interpretation:")
    print_output(f"  True Negatives (TN):  {cm[0][0]} - Correctly identified as Not Sarcastic")
    print_output(f"  False Positives (FP): {cm[0][1]} - Incorrectly identified as Sarcastic")
    print_output(f"  False Negatives (FN): {cm[1][0]} - Missed Sarcastic texts")
    print_output(f"  True Positives (TP):  {cm[1][1]} - Correctly identified as Sarcastic")
    
    # Per-class metrics
    print_output("\n" + "="*70)
    print_output("PER-CLASS PERFORMANCE (Test Set)")
    print_output("="*70)
    
    tn, fp, fn, tp = cm.ravel()
    
    # Not Sarcastic class
    not_sarc_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
    not_sarc_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    not_sarc_f1 = 2 * (not_sarc_precision * not_sarc_recall) / (not_sarc_precision + not_sarc_recall) if (not_sarc_precision + not_sarc_recall) > 0 else 0
    
    print_output("\nNot Sarcastic Class:")
    print_output(f"   Precision: {not_sarc_precision:.4f} ({not_sarc_precision*100:.2f}%)")
    print_output(f"   Recall:    {not_sarc_recall:.4f} ({not_sarc_recall*100:.2f}%)")
    print_output(f"   F1-Score:  {not_sarc_f1:.4f} ({not_sarc_f1*100:.2f}%)")
    
    print_output("\nSarcastic Class:")
    print_output(f"   Precision: {test_prec:.4f} ({test_prec*100:.2f}%)")
    print_output(f"   Recall:    {test_rec:.4f} ({test_rec*100:.2f}%)")
    print_output(f"   F1-Score:  {test_f1:.4f} ({test_f1*100:.2f}%)")
    
    # Model Summary
    print_output("\n" + "="*70)
    print_output("MODEL SUMMARY")
    print_output("="*70)
    print_output(f"✅ Overall Test Accuracy: {test_acc*100:.2f}%")
    print_output(f"✅ Model Type: Logistic Regression with TF-IDF features")
    print_output(f"✅ Feature Extraction: Character n-grams (1-4 grams)")
    print_output(f"✅ Max Features: 10,000")
    print_output(f"✅ Test Samples: {len(X_test_processed)}")
    
    if test_acc >= 0.95:
        performance_level = "Excellent"
    elif test_acc >= 0.90:
        performance_level = "Very Good"
    elif test_acc >= 0.85:
        performance_level = "Good"
    else:
        performance_level = "Moderate"
    
    print_output(f"✅ Performance Level: {performance_level}")
    print_output("="*70)

    if output_file_path:
        f.close()
        print(f"Performance metrics saved to {output_file_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Hindi Sarcasm Detection Model Performance.")
    parser.add_argument('--test_set', type=str, default='dataset/test_set.csv',
                        help='Path to the CSV file containing the test set.')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to a file where performance metrics will be saved.')
    args = parser.parse_args()
    
    show_performance(test_csv_path=args.test_set, output_file_path=args.output_file)

