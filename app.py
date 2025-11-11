"""
Flask Web Application for Hindi Sarcasm Detection
"""

from flask import Flask, render_template, request, jsonify
from model import HindiSarcasmDetector
from preprocess import detect_script, preprocess_hindi_text
import os

app = Flask(__name__)

# Initialize detector (silently - no verbose output)
detector = HindiSarcasmDetector()


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/api/detect', methods=['POST'])
def detect_sarcasm():
    """API endpoint for sarcasm detection"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({
                'error': 'No text provided'
            }), 400
        
        # Detect sarcasm
        result = detector.detect_sarcasm_level(text)
        
        # Add script type
        result['script_type'] = detect_script(text)
        result['processed_text'] = preprocess_hindi_text(text)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

