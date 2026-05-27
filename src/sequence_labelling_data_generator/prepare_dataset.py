import os
import sys
import json
import re
import argparse
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Set up local import path if needed
sys.path.append(str(Path(__file__).parent.parent))

from sequence_labelling_data_generator.reconstructor import reconstruct_question, ReconstructorConfig

# Define base tags and generate tag mapping
BASE_TAGS = [
    "question_label",
    "stem",
    "option_label",
    "option_text",
    "context",
    "ordering_item_label",
    "ordering_item_text"
]

LATEX_REGEX = re.compile(r'\$[^$]+\$')

def get_tag_mappings() -> Tuple[Dict[str, int], Dict[int, str]]:
    tag_to_id = {"O": 0}
    for tag in BASE_TAGS:
        tag_to_id[f"B-{tag}"] = len(tag_to_id)
        tag_to_id[f"I-{tag}"] = len(tag_to_id)
    id_to_tag = {v: k for k, v in tag_to_id.items()}
    return tag_to_id, id_to_tag

def replace_latex_in_text(text: str, placeholder: str) -> str:
    if not text:
        return text
    return LATEX_REGEX.sub(placeholder, text)

def replace_latex_in_question(q_data: Dict[str, Any], placeholder: str) -> Dict[str, Any]:
    """Replaces all LaTeX equations ($...$) inside question text fields with a placeholder."""
    q_copy = json.loads(json.dumps(q_data))
    
    if q_copy.get("is_group", False):
        if "context" in q_copy:
            q_copy["context"] = replace_latex_in_text(q_copy["context"], placeholder)
        if "questions" in q_copy:
            for sub_q in q_copy["questions"]:
                if "stem" in sub_q:
                    sub_q["stem"] = replace_latex_in_text(sub_q["stem"], placeholder)
                if "options" in sub_q:
                    sub_q["options"] = [replace_latex_in_text(opt, placeholder) for opt in sub_q["options"]]
    else:
        if "stem" in q_copy:
            q_copy["stem"] = replace_latex_in_text(q_copy["stem"], placeholder)
        if "options" in q_copy:
            q_copy["options"] = [replace_latex_in_text(opt, placeholder) for opt in q_copy["options"]]
            
    return q_copy

def align_tokens_to_spans(offset_mapping: List[Tuple[int, int]], spans: List[Dict[str, Any]], tag_to_id: Dict[str, int]) -> List[int]:
    """
    Aligns tokenizer offset mapping with character-level spans to assign token-level labels.
    Special tokens (with offset (0, 0)) are assigned -100.
    """
    labels = []
    for start, end in offset_mapping:
        if start == 0 and end == 0:
            labels.append(-100)
            continue
            
        # Find which span this token overlaps with
        matching_span = None
        for span in spans:
            overlap_start = max(start, span["start"])
            overlap_end = min(end, span["end"])
            if overlap_start < overlap_end:
                matching_span = span
                break
                
        if matching_span is None:
            labels.append(tag_to_id["O"])
        else:
            span_label = matching_span["label"]
            # Assign B- tag if this token contains the start character of the span, otherwise I-
            if start <= matching_span["start"] < end:
                tag = f"B-{span_label}"
            else:
                tag = f"I-{span_label}"
            labels.append(tag_to_id.get(tag, tag_to_id["O"]))
            
    return labels

def process_single_question(
    q_data: Dict[str, Any], 
    tokenizer: Any, 
    tag_to_id: Dict[str, int], 
    id_to_tag: Dict[int, str],
    latex_placeholder: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Processes a single question: replaces LaTeX equations if specified, ensures it is reconstructed, 
    tokenizes it, aligns the labels, and returns the tokenized sample.
    """
    # 1. Replace LaTeX if placeholder is specified
    if latex_placeholder:
        q_data = replace_latex_in_question(q_data, latex_placeholder)
        # Discard pre-reconstructed values to force reconstruction with placeholders and correct offsets
        q_data.pop("raw_text", None)
        q_data.pop("spans", None)

    # 2. Reconstruct raw text and character spans
    if "raw_text" not in q_data or "spans" not in q_data:
        q_data = reconstruct_question(q_data, ReconstructorConfig())
        
    raw_text = q_data["raw_text"]
    spans = q_data["spans"]
    
    # 3. Tokenize with offset mapping
    tokenized = tokenizer(
        raw_text,
        return_offsets_mapping=True,
        truncation=True,
        add_special_tokens=True
    )
    
    offset_mapping = tokenized["offset_mapping"]
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    
    # 4. Align labels
    labels = align_tokens_to_spans(offset_mapping, spans, tag_to_id)
    
    # 5. Generate human-readable tags list (for debugging/validation)
    tags = [id_to_tag.get(label_id, "O") if label_id != -100 else "IGNORE" for label_id in labels]
    
    # 6. Extract tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    return {
        "tokens": tokens,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "tags": tags,
        "metadata": {
            "subject": q_data.get("subject"),
            "grade": q_data.get("grade"),
            "question_type": q_data.get("question_type"),
            "difficulty": q_data.get("difficulty"),
            "is_group": q_data.get("is_group", False)
        }
    }

def main():
    parser = argparse.ArgumentParser(description="XLM-RoBERTa Sequence Labelling Dataset Preparer")
    parser.add_argument(
        "-i", "--input-dir",
        type=str,
        default="output",
        help="Directory containing the input question JSON files (default: 'output')"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="dataset_output",
        help="Directory to save the output dataset splits (default: 'dataset_output')"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="FacebookAI/xlm-roberta-base",
        help="Hugging Face model / tokenizer name (default: 'FacebookAI/xlm-roberta-base')"
    )
    parser.add_argument(
        "--latex-placeholder",
        type=str,
        default="[LATEX]",
        help="Special token placeholder for LaTeX equations. Set to empty string '' to keep LaTeX unchanged. (default: '[LATEX]')"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of training set (default: 0.8)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Ratio of validation set (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset splitting (default: 42)"
    )
    args = parser.parse_args()
    
    # Validation of ratio inputs
    if args.train_ratio + args.val_ratio > 1.0 or args.train_ratio < 0.0 or args.val_ratio < 0.0:
        print("Error: train-ratio and val-ratio must sum to <= 1.0 and be non-negative.")
        sys.exit(1)
        
    test_ratio = 1.0 - (args.train_ratio + args.val_ratio)
    
    # Setup paths
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        sys.exit(1)
        
    json_files = list(input_path.glob("question_*.json"))
    if not json_files:
        print(f"No question JSON files found in '{args.input_dir}'. Please run the generator first.")
        sys.exit(1)
        
    # Import Hugging Face Transformers inside main so CLI help works without it installed
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("Error: 'transformers' is not installed. Please install it or use the active Pixi environment.")
        sys.exit(1)
        
    print(f"Loading tokenizer '{args.model}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        sys.exit(1)
        
    # Process LaTeX placeholder
    latex_placeholder = args.latex_placeholder if args.latex_placeholder.strip() else None
    if latex_placeholder:
        print(f"LaTeX equations will be replaced with placeholder token: '{latex_placeholder}'")
        # Add placeholder to tokenizer as a special token
        tokenizer.add_special_tokens({"additional_special_tokens": [latex_placeholder]})
        
    tag_to_id, id_to_tag = get_tag_mappings()
    
    # Save label mapping to output directory for training setup later
    label_mapping = {
        "tag_to_id": tag_to_id,
        "id_to_tag": id_to_tag
    }
    with open(output_path / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=2)
        
    print(f"Processing {len(json_files)} files...")
    processed_samples = []
    
    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                q_data = json.load(f)
            sample = process_single_question(q_data, tokenizer, tag_to_id, id_to_tag, latex_placeholder)
            if sample:
                sample["metadata"]["source_file"] = file_path.name
                processed_samples.append(sample)
        except Exception as e:
            print(f"Warning: Failed to process {file_path.name}: {e}")
            
    print(f"Successfully processed {len(processed_samples)} / {len(json_files)} questions.")
    
    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(processed_samples)
    
    n_total = len(processed_samples)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)
    
    train_samples = processed_samples[:n_train]
    val_samples = processed_samples[n_train:n_train+n_val]
    test_samples = processed_samples[n_train+n_val:]
    
    splits = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples
    }
    
    for split_name, samples in splits.items():
        split_file = output_path / f"{split_name}.jsonl"
        with open(split_file, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"Saved {len(samples)} samples to '{split_file}'")
        
    print("\nDataset preparation completed successfully!")
    print(f"Total samples: {n_total} (Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)})")
    print(f"Label mapping saved to '{output_path / 'label_mapping.json'}'")

if __name__ == "__main__":
    main()
