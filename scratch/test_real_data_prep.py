import os
import sys
import random
from pathlib import Path

# Add workspace root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from real_data_annotator.annotate_ocr import parse_xml_annotations
from src.training.prepare_dataset import mask_latex_in_real_data

def test_xml_parsing():
    print("--- Testing XML Parsing ---")
    tagged_text = (
        "<question_label>Câu 1:</question_label><stem>Tìm tập nghiệm của bất phương trình "
        "$\\log_2(x - 1) < 3$.</stem>\n"
        "<option_label>A.</option_label><option_text>$(1; 9)$.</option_text>\n"
        "<option_label>B.</option_label><option_text>$(-\\infty; 9)$.</option_text>"
    )
    
    raw_text, spans = parse_xml_annotations(tagged_text)
    print("Raw Text:")
    print(repr(raw_text))
    print("\nSpans:")
    for span in spans:
        print(f"  {span['label']}: [{span['start']}:{span['end']}] -> {repr(span['text'])}")
        # Verify span text matches raw_text slice
        assert raw_text[span['start']:span['end']] == span['text']
        
    print("XML parsing test PASSED!")

def test_latex_masking():
    print("\n--- Testing LaTeX Masking & Offset Shifts ---")
    raw_text = "Câu 1: Cho hàm số $y = f(x)$ đồng biến trên $\\mathbb{R}$.\nA. $y = x^2$.\nB. $y = x$."
    spans = [
        {"start": 0, "end": 7, "label": "question_label", "text": "Câu 1: "},
        {"start": 7, "end": 56, "label": "stem", "text": "Cho hàm số $y = f(x)$ đồng biến trên $\\mathbb{R}$."},
        {"start": 57, "end": 60, "label": "option_label", "text": "A. "},
        {"start": 60, "end": 71, "label": "option_text", "text": "$y = x^2$."},
        {"start": 72, "end": 75, "label": "option_label", "text": "B. "},
        {"start": 75, "end": 84, "label": "option_text", "text": "$y = x$."}
    ]
    
    # Let's seed the random generator so that all LaTeX equations get masked (prob=1.0)
    rng = random.Random(42)
    masked_text, shifted_spans = mask_latex_in_real_data(
        raw_text, spans, "[LATEX]", mask_prob=1.0, rng=rng
    )
    
    print("Masked Text:")
    print(repr(masked_text))
    print("\nShifted Spans:")
    for span in shifted_spans:
        print(f"  {span['label']}: [{span['start']}:{span['end']}] -> {repr(span['text'])}")
        # Verify span text matches masked_text slice
        assert masked_text[span['start']:span['end']] == span['text']
        
    print("LaTeX masking offset shift test PASSED!")

if __name__ == "__main__":
    test_xml_parsing()
    test_latex_masking()
