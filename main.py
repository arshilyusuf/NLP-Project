"""
Main Application for Hindi Sarcasm Detection
Provides both CLI and Web interface
"""

import argparse
import sys
from model import HindiSarcasmDetector
from preprocess import preprocess_hindi_text, detect_script


def print_result(text, result):
    """
    Pretty print the detection result
    """
    print("\n" + "="*60)
    print("HINDI SARCASM DETECTION RESULT")
    print("="*60)
    print(f"Input Text: {text}")
    print(f"Script Type: {detect_script(text)}")
    print(f"\nSarcasm Level: {result['sarcasm_level']}%")
    print(f"Is Sarcastic: {'Yes' if result['is_sarcastic'] else 'No'}")
    print(f"Confidence: {result['confidence']*100:.1f}%")
    print(f"Message: {result['message']}")
    print("="*60 + "\n")


def interactive_mode():
    """
    Interactive CLI mode
    """
    print("\n" + "="*60)
    print("Hindi Sarcasm Detection - Interactive Mode")
    print("="*60)
    print("Enter Hindi text (Devanagari or Romanized)")
    print("Type 'quit' or 'exit' to stop\n")
    
    detector = HindiSarcasmDetector()
    
    while True:
        try:
            text = input("Enter text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not text:
                print("Please enter some text.\n")
                continue
            
            result = detector.detect_sarcasm_level(text)
            print_result(text, result)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def single_text_mode(text):
    """
    Process a single text input
    """
    detector = HindiSarcasmDetector()
    result = detector.detect_sarcasm_level(text)
    print_result(text, result)
    return result


def file_mode(file_path):
    """
    Process texts from a file (one per line)
    """
    detector = HindiSarcasmDetector()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"\nProcessing {len(lines)} texts from file...\n")
        
        results = []
        for i, line in enumerate(lines, 1):
            text = line.strip()
            if text:
                print(f"\n[{i}/{len(lines)}] Processing: {text[:50]}...")
                result = detector.detect_sarcasm_level(text)
                results.append(result)
                print(f"Result: {result['message']}")
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        sarcastic_count = sum(1 for r in results if r['is_sarcastic'])
        avg_level = sum(r['sarcasm_level'] for r in results) / len(results) if results else 0
        print(f"Total texts: {len(results)}")
        print(f"Sarcastic: {sarcastic_count}")
        print(f"Non-sarcastic: {len(results) - sarcastic_count}")
        print(f"Average sarcasm level: {avg_level:.1f}%")
        print("="*60 + "\n")
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)


def main():
    """
    Main entry point
    """
    parser = argparse.ArgumentParser(
        description='Hindi Sarcasm Detection Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py -t "वाह! क्या बात है"
  python main.py -t "Wah! Kya baat hai"
  python main.py -i
  python main.py -f input.txt
        """
    )
    
    parser.add_argument(
        '-t', '--text',
        type=str,
        help='Input Hindi text to analyze (Devanagari or Romanized)'
    )
    
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '-f', '--file',
        type=str,
        help='Path to file containing texts (one per line)'
    )
    
    args = parser.parse_args()
    
    # Determine mode
    if args.interactive:
        interactive_mode()
    elif args.file:
        file_mode(args.file)
    elif args.text:
        single_text_mode(args.text)
    else:
        # Default to interactive mode
        interactive_mode()


if __name__ == '__main__':
    main()

