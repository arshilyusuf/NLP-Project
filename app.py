from flask import Flask, render_template, request, jsonify
from model import HindiSarcasmDetector
from preprocess import detect_script, preprocess_hindi_text
import os
import numpy as np

app = Flask(__name__)

detector = HindiSarcasmDetector()


def convert_booleans_to_strings(obj):
    # recursively converts boolean and numpy types to standard python types for json serialization.
    if isinstance(obj, dict):
        return {k: convert_booleans_to_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_booleans_to_strings(elem) for elem in obj]
    elif isinstance(obj, np.bool_): # catch numpy booleans
        return bool(obj) # convert to standard python bool
    elif isinstance(obj, np.floating): # catch numpy floats
        return float(obj) # convert to standard python float
    elif isinstance(obj, bool): # already handles standard python booleans, but keep for clarity
        return str(obj) # convert to string as originally intended for display
    return obj

@app.route('/')
def index():
    # main page
    return render_template('index.html')


@app.route('/api/detect', methods=['POST'])
def detect_sarcasm():
    # api endpoint for sarcasm detection
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({
                'error': 'no text provided'
            }), 400
        
        result = detector.detect_sarcasm_level(text)
        
        # recursively convert all boolean values to strings for json serialization
        result = convert_booleans_to_strings(result)

        # debug: print types of result dictionary items
        print("\n--- debugging json serialization ---")
        for k, v in result.items():
            print(f"key: {k}, type: {type(v)}, value: {v}")
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    print(f"  sub_key: {sub_k}, sub_type: {type(sub_v)}, sub_value: {sub_v}")
        print("--- end debugging ---\n")

        result['script_type'] = detect_script(text)
        result['processed_text'] = preprocess_hindi_text(text)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

