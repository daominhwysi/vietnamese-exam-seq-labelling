import os
import sys
import json
import argparse
import random
from pathlib import Path

# Setup local import paths
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.generation.reconstructor import (
    reconstruct_question, 
    reconstruct_exam, 
    ReconstructorConfig
)
from src.training.dataset import (
    get_tag_mappings,
    spans_to_xml,
    align_tokens_to_spans,
    process_exam_level,
    process_question_as_exam_level,
    process_single_question_legacy,
    process_single_question,
    replace_latex_in_question,
    scan_input_files,
    save_jsonl_split
)

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
        default="output/dataset",
        help="Directory to save the output dataset splits (default: 'output/dataset')"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="aisingapore/SEA-LION-ModernBERT-300M",
        help="Hugging Face model / tokenizer name (default: 'aisingapore/SEA-LION-ModernBERT-300M')"
    )
    parser.add_argument(
        "--latex-placeholder",
        type=str,
        default="[LATEX]",
        help="Special token placeholder for LaTeX equations (default: '[LATEX]')"
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
    
    # Advanced data prep features
    parser.add_argument(
        "--exam-level",
        action="store_true",
        help="Process datasets at the exam level with multi-scale sliding windows (default: False)"
    )
    parser.add_argument(
        "--max-len",
        type=str,
        default="512,768,1024,2048",
        help="Comma-separated sequence lengths for tokenization (default: '512,768,1024,2048')"
    )
    parser.add_argument(
        "--stride",
        type=str,
        default="128,192,256,512",
        help="Comma-separated strides for tokenization (default: '128,192,256,512')"
    )
    
    # Advanced Data Augmentations
    parser.add_argument(
        "--typo-rate",
        type=float,
        default=0.02,
        help="Spelling mistake typo injection rate (default: 0.02)"
    )
    parser.add_argument(
        "--space-noise-rate",
        type=float,
        default=0.15,
        help="Spacing noise injection rate (default: 0.15)"
    )
    parser.add_argument(
        "--latex-mask-prob",
        type=float,
        default=0.5,
        help="Probability of masking LaTeX formulas (default: 0.5)"
    )
    parser.add_argument(
        "--enable-permutations",
        action="store_true",
        help="Enable random permutations of questions and options (default: False)"
    )
    parser.add_argument(
        "--option-drop-prob",
        type=float,
        default=0.05,
        help="Probability of dropping 1 to 3 options to simulate OCR cuts (default: 0.05)"
    )
    parser.add_argument(
        "--casing-noise-prob",
        type=float,
        default=0.10,
        help="Probability of random casing/capitalization noise (default: 0.10)"
    )
    parser.add_argument(
        "--synonym-swap-prob",
        type=float,
        default=0.10,
        help="Probability of random prefix synonym swap (default: 0.10)"
    )
    parser.add_argument(
        "--formatting-noise-prob",
        type=float,
        default=0.10,
        help="Probability of wrapping labels in random Markdown/HTML formatting (default: 0.10)"
    )
    parser.add_argument(
        "--inline-option-prob",
        type=float,
        default=0.0,
        help="Probability of formatting options inline (default: 0.0)"
    )
    parser.add_argument(
        "--min-inline-spaces",
        type=int,
        default=5,
        help="Minimum random spaces to inject between inline options (default: 5)"
    )
    parser.add_argument(
        "--max-inline-spaces",
        type=int,
        default=30,
        help="Maximum random spaces to inject between inline options (default: 30)"
    )
    parser.add_argument(
        "--min-inline-tabs",
        type=int,
        default=1,
        help="Minimum random tabs to inject between inline options (default: 1)"
    )
    parser.add_argument(
        "--max-inline-tabs",
        type=int,
        default=3,
        help="Maximum random tabs to inject between inline options (default: 3)"
    )
    parser.add_argument(
        "--answer-table-format",
        type=str,
        choices=["md", "html", "csv", "random"],
        default="random",
        help="Answer table format (default: random)"
    )
    parser.add_argument(
        "--answer-table-direction",
        type=str,
        choices=["horizontal", "vertical", "random"],
        default="random",
        help="Answer table direction (default: random)"
    )
    parser.add_argument(
        "--answer-table-chunk-size",
        type=int,
        default=None,
        help="Answer table chunk size (default: None)"
    )
    parser.add_argument(
        "--no-paraphrase-section-titles",
        action="store_false",
        dest="paraphrase_section_titles",
        default=True,
        help="Disable random paraphrasing of section titles (default: enabled)"
    )

    args = parser.parse_args()
    run_prepare_dataset(args)

def run_prepare_dataset(args):
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
        
    json_files, exam_files = scan_input_files(args.input_dir)
    if not json_files and not exam_files:
        print(f"No question JSON files or exam JSON files found in '{args.input_dir}'. Please generate data first.")
        sys.exit(1)
        
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
        
    # Process LaTeX placeholder and special tokens
    latex_placeholder = args.latex_placeholder if args.latex_placeholder and args.latex_placeholder.strip() else "[LATEX]"
    special_tokens = ["<blank />", "<blank/>", "[BLANK]"]
    if latex_placeholder:
        special_tokens.append(latex_placeholder)
    
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        
    tag_to_id, id_to_tag = get_tag_mappings()
    
    label_mapping = {
        "tag_to_id": tag_to_id,
        "id_to_tag": id_to_tag
    }
    with open(output_path / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=2)
        
    # Parse window configurations
    max_lens = [int(x.strip()) for x in getattr(args, "max_len", "512").split(",") if x.strip()]
    strides = [int(x.strip()) for x in getattr(args, "stride", "128").split(",") if x.strip()]
    while len(strides) < len(max_lens):
        strides.append(strides[-1] if strides else 128)
    strides = strides[:len(max_lens)]
    window_configs = list(zip(max_lens, strides))
    
    # Build the shared ReconstructorConfig
    reconstructor_config = ReconstructorConfig(
        typo_rate=args.typo_rate,
        space_noise_rate=args.space_noise_rate,
        latex_mask_prob=args.latex_mask_prob,
        latex_placeholder=latex_placeholder,
        enable_permutations=args.enable_permutations,
        option_drop_prob=args.option_drop_prob,
        casing_noise_prob=args.casing_noise_prob,
        synonym_swap_prob=args.synonym_swap_prob,
        formatting_noise_prob=args.formatting_noise_prob,
        inline_option_prob=getattr(args, "inline_option_prob", 0.0),
        min_inline_spaces=getattr(args, "min_inline_spaces", 5),
        max_inline_spaces=getattr(args, "max_inline_spaces", 30),
        min_inline_tabs=getattr(args, "min_inline_tabs", 1),
        max_inline_tabs=getattr(args, "max_inline_tabs", 3),
        answer_table_format=getattr(args, "answer_table_format", "random"),
        answer_table_direction=getattr(args, "answer_table_direction", "random"),
        answer_table_chunk_size=getattr(args, "answer_table_chunk_size", None),
        paraphrase_section_titles=getattr(args, "paraphrase_section_titles", True)
    )
        
    print(f"Processing data: found {len(json_files)} question file(s) and {len(exam_files)} exam file(s)...")
    processed_samples = []

    # Prepare xml output directory for annotated XML files
    xml_output_path = output_path / "xml"
    xml_output_path.mkdir(parents=True, exist_ok=True)

    def _save_xml(stem: str, raw_text: str, spans: List[Dict[str, Any]]) -> None:
        """Write ground-truth inline-tagged XML for a source file."""
        try:
            xml_content = spans_to_xml(raw_text, spans)
            xml_file = xml_output_path / f"{stem}_annotated.xml"
            with open(xml_file, "w", encoding="utf-8") as xf:
                xf.write(xml_content)
        except Exception as xe:
            print(f"Warning: Could not write XML for '{stem}': {xe}")

    # 1. Process individual question files
    num_q_files = len(json_files)
    for q_idx, file_path in enumerate(json_files):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                q_data = json.load(f)

            if args.exam_level:
                samples = process_question_as_exam_level(q_data, tokenizer, tag_to_id, id_to_tag, window_configs, reconstructor_config)
                for s in samples:
                    s["metadata"]["source_file"] = file_path.name
                    processed_samples.append(s)
            else:
                sample = process_single_question_legacy(q_data, tokenizer, tag_to_id, id_to_tag, reconstructor_config)
                if sample:
                    sample["metadata"]["source_file"] = file_path.name
                    processed_samples.append(sample)

            # Generate XML from ground-truth spans
            try:
                q_rec = reconstruct_question(q_data, ReconstructorConfig())
                _save_xml(file_path.stem, q_rec["raw_text"], q_rec["spans"])
            except Exception:
                pass

        except Exception as e:
            print(f"Warning: Failed to process question {file_path.name}: {e}")

        if (q_idx + 1) % 50 == 0 or (q_idx + 1) == num_q_files:
            print(f"[Progress] Processed {q_idx + 1}/{num_q_files} individual questions...")
            
    # 2. Process exam files
    exam_q_count = 0
    num_exam_files = len(exam_files)
    for exam_idx, file_path in enumerate(exam_files):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                exam_data = json.load(f)

            if args.exam_level:
                samples = process_exam_level(exam_data, tokenizer, tag_to_id, id_to_tag, window_configs, reconstructor_config)
                for s in samples:
                    s["metadata"]["source_file"] = file_path.name
                    processed_samples.append(s)
                    exam_q_count += 1
            else:
                sections = exam_data.get("sections", {})
                for section_title, questions in sections.items():
                    for idx, q_data in enumerate(questions):
                        q_copy = dict(q_data)
                        if "subject" not in q_copy and "subject" in exam_data:
                            q_copy["subject"] = exam_data["subject"]
                        if "grade" not in q_copy and "grade" in exam_data:
                            q_copy["grade"] = exam_data["grade"]

                        sample = process_single_question_legacy(q_copy, tokenizer, tag_to_id, id_to_tag, reconstructor_config)
                        if sample:
                            sample["metadata"]["source_file"] = f"{file_path.name}::{section_title}::q_{idx}"
                            processed_samples.append(sample)
                            exam_q_count += 1

            # Generate XML from ground-truth spans (one XML per source exam)
            try:
                if exam_data.get("is_real", False) and "raw_text" in exam_data and "spans" in exam_data:
                    _save_xml(file_path.stem, exam_data["raw_text"], exam_data["spans"])
                else:
                    exam_rec = reconstruct_exam(exam_data, ReconstructorConfig())
                    _save_xml(file_path.stem, exam_rec["raw_text"], exam_rec["spans"])
            except Exception:
                pass

        except Exception as e:
            print(f"Warning: Failed to process exam {file_path.name}: {e}")

        if (exam_idx + 1) % 10 == 0 or (exam_idx + 1) == num_exam_files:
            print(f"[Progress] Processed {exam_idx + 1}/{num_exam_files} exams...")

            
    print(f"Successfully prepared {len(processed_samples)} training samples (sources include individual question files and compiled exam files).")
    
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
        save_jsonl_split(samples, split_file)
        print(f"Saved {len(samples)} samples to '{split_file}'")
        
    print("\nDataset preparation completed successfully!")
    print(f"Total samples: {n_total} (Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)})")
    print(f"Label mapping saved to '{output_path / 'label_mapping.json'}'")

if __name__ == "__main__":
    main()
