import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

def debug_chunk_alignment():
    # Load label mapping
    mapping_path = Path("output/dataset/label_mapping.json")
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    id_to_tag = mapping["id_to_tag"]
    
    # Directly load target exam
    exam_file = Path("output/exams/exam_economics_law_g12_20260602_024154_5521dcaf.json")
    with open(exam_file, "r", encoding="utf-8") as f:
        exam_data = json.load(f)
        
    # Reconstruct exam to get raw_text and spans
    from src.generation.reconstructor import reconstruct_exam, ReconstructorConfig
    
    # Reconstruct the exam with the same config
    config = ReconstructorConfig(
        typo_rate=0.02,
        space_noise_rate=0.15,
        latex_mask_prob=0.5,
        latex_placeholder="[LATEX]",
        casing_noise_prob=0.10,
        synonym_swap_prob=0.10,
        formatting_noise_prob=0.10
    )
    # The random seed for the exam reconstruction is exam_id or str(exam_data)
    config.seed = exam_data.get("exam_id", "") or str(exam_data)
    exam_reconstructed = reconstruct_exam(exam_data, config)
    raw_text = exam_reconstructed["raw_text"]
    spans = exam_reconstructed["spans"]
    
    print(f"Reconstructed raw text length: {len(raw_text)}")
    
    # Let's find the first span where the slice raw_text[start:end] does not match span["text"]
    mismatch_found = False
    for i, s in enumerate(spans):
        start = s["start"]
        end = s["end"]
        label = s["label"]
        expected_text = s.get("text", "")
        actual_text = raw_text[start:end]
        
        if expected_text != actual_text:
            print(f"\nFirst mismatch found at span index {i}:")
            print(f"Label: {label}")
            print(f"Offsets: ({start}, {end})")
            print(f"Expected text (from span['text']): {repr(expected_text)}")
            print(f"Actual text (from raw_text slice): {repr(actual_text)}")
            
            # Print surrounding raw text
            surrounding_start = max(0, start - 50)
            surrounding_end = min(len(raw_text), end + 50)
            print(f"\nSurrounding raw text (indices {surrounding_start} to {surrounding_end}):")
            print(repr(raw_text[surrounding_start:surrounding_end]))
            
            # Print previous spans
            print("\nPrevious 3 spans:")
            for j in range(max(0, i - 3), i):
                prev_s = spans[j]
                print(f"Span {j} | ({prev_s['start']},{prev_s['end']}) | {prev_s['label']}: {repr(prev_s.get('text', ''))}")
                
            mismatch_found = True
            break
            
    if not mismatch_found:
        print("\nNo mismatches found between spans and raw_text!")
    return

if __name__ == "__main__":
    debug_chunk_alignment()
