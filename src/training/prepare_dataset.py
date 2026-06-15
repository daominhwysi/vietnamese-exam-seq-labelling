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

from src.generation.reconstructor import (
    reconstruct_question, 
    reconstruct_exam, 
    ReconstructorConfig
)

# Define base tags and generate tag mapping
BASE_TAGS = [
    "question_label",
    "stem",
    "option_label",
    "option_text",
    "context"
]

def get_tag_mappings() -> Tuple[Dict[str, int], Dict[int, str]]:
    tag_to_id = {"O": 0}
    for tag in BASE_TAGS:
        tag_to_id[f"B-{tag}"] = len(tag_to_id)
        tag_to_id[f"I-{tag}"] = len(tag_to_id)
    id_to_tag = {v: k for k, v in tag_to_id.items()}
    return tag_to_id, id_to_tag

def align_tokens_to_spans(
    offset_mapping: List[Tuple[int, int]], 
    spans: List[Dict[str, Any]], 
    tag_to_id: Dict[str, int],
    raw_text: Optional[str] = None
) -> List[int]:
    """
    Aligns tokenizer offset mapping with character-level spans to assign token-level labels
    using the V2 Character-Anchor Lookup method.
    """
    # 1. Clean spans (strip whitespaces/tabs)
    clean_spans = []
    for span in spans:
        span_text = span.get("text", "")
        start = span["start"]
        end = span["end"]
        
        if span_text:
            stripped = span_text.strip()
            if not stripped:
                continue
            leading = len(span_text) - len(span_text.lstrip())
            trailing = len(span_text) - len(span_text.rstrip())
            clean_start = start + leading
            clean_end = end - trailing
        else:
            if raw_text is not None:
                raw_span_text = raw_text[start:end]
                stripped = raw_span_text.strip()
                if not stripped:
                    continue
                leading = len(raw_span_text) - len(raw_span_text.lstrip())
                trailing = len(raw_span_text) - len(raw_span_text.rstrip())
                clean_start = start + leading
                clean_end = end - trailing
            else:
                clean_start = start
                clean_end = end
                
        clean_spans.append({
            "start": clean_start,
            "end": clean_end,
            "label": span["label"]
        })
        
    labels = []
    
    # 2. Map tokens based on first non-whitespace character offset lookup
    for start, end in offset_mapping:
        if start == 0 and end == 0:
            labels.append(-100)
            continue
            
        non_space_char_idx = -1
        if raw_text is not None:
            for char_idx in range(start, end):
                if char_idx < len(raw_text) and not raw_text[char_idx].isspace():
                    non_space_char_idx = char_idx
                    break
        else:
            # Fallback if raw_text is not provided
            non_space_char_idx = start
            
        if non_space_char_idx == -1:
            # It's a whitespace-only token. Check if it falls entirely within some span.
            # If so, label it as part of that span (using "I-label" so that the entity is contiguous).
            matched_span = None
            for span in clean_spans:
                if span["start"] <= start and end <= span["end"]:
                    matched_span = span
                    break
            if matched_span is not None:
                labels.append(tag_to_id.get(f"I-{matched_span['label']}", tag_to_id["O"]))
            else:
                labels.append(tag_to_id["O"])
            continue
            
        matched_span = None
        for span in clean_spans:
            if span["start"] <= non_space_char_idx < span["end"]:
                matched_span = span
                break
                
        if matched_span is None:
            labels.append(tag_to_id["O"])
        else:
            span_label = matched_span["label"]
            if non_space_char_idx == matched_span["start"]:
                tag = f"B-{span_label}"
            else:
                tag = f"I-{span_label}"
            labels.append(tag_to_id.get(tag, tag_to_id["O"]))
            
    return labels

def mask_latex_in_real_data(
    raw_text: str,
    spans: List[Dict[str, Any]],
    placeholder: str,
    mask_prob: float,
    rng: random.Random
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Finds LaTeX formulas ($...$ and $$...$$) in raw_text, masks them with placeholder with probability mask_prob,
    and shifts span offsets accordingly.
    """
    if mask_prob <= 0.0 or not raw_text:
        return raw_text, spans

    pattern = re.compile(r"\$\$.*?\$\$|\$.*?\$", re.DOTALL)
    matches = list(pattern.finditer(raw_text))
    if not matches:
        return raw_text, spans
        
    current_text = raw_text
    new_spans = [dict(s) for s in spans]
    
    # Process from back to front to avoid shifting indices of earlier matches
    for match in reversed(matches):
        if rng.random() > mask_prob:
            continue
            
        m_start, m_end = match.span()
        diff = len(placeholder) - (m_end - m_start)
        
        # Replace in text
        current_text = current_text[:m_start] + placeholder + current_text[m_end:]
        
        # Adjust spans
        updated_spans = []
        for span in new_spans:
            s_start = span["start"]
            s_end = span["end"]
            
            # If the span starts after the replaced segment, shift it
            if s_start >= m_end:
                span["start"] += diff
                span["end"] += diff
            # If the span starts before but ends after/during
            elif s_start < m_start and s_end > m_end:
                span["end"] += diff
            # If the span is entirely within the masked LaTeX
            elif s_start >= m_start and s_end <= m_end:
                span["start"] = m_start
                span["end"] = m_start + len(placeholder)
            # If the span overlaps the beginning but not the end
            elif s_start < m_start and s_end > m_start:
                span["end"] = m_start + len(placeholder)
            
            if "text" in span:
                span["text"] = current_text[span["start"]:span["end"]]
            updated_spans.append(span)
        new_spans = updated_spans
        
    return current_text, new_spans

def process_exam_level(
    exam_data: Dict[str, Any],
    tokenizer: Any,
    tag_to_id: Dict[str, int],
    id_to_tag: Dict[int, str],
    window_configs: List[Tuple[int, int]],
    reconstructor_config: ReconstructorConfig
) -> List[Dict[str, Any]]:
    """
    Reconstructs the full exam document, tokenizes it across multiple sliding window configs,
    aligns labels, and returns a list of prepared samples.
    """
    if exam_data.get("is_real", False) and "raw_text" in exam_data and "spans" in exam_data:
        raw_text = exam_data["raw_text"]
        spans = exam_data["spans"]
        if reconstructor_config.latex_mask_prob > 0.0:
            import hashlib
            h = hashlib.md5((exam_data.get("exam_id", "") or raw_text).encode("utf-8")).hexdigest()
            rng = random.Random(int(h, 16) & 0xFFFFFFFF)
            raw_text, spans = mask_latex_in_real_data(
                raw_text, spans, reconstructor_config.latex_placeholder, reconstructor_config.latex_mask_prob, rng
            )
    else:
        exam_reconstructed = reconstruct_exam(exam_data, reconstructor_config)
        raw_text = exam_reconstructed["raw_text"]
        spans = exam_reconstructed["spans"]
    
    samples = []
    
    for max_len, stride in window_configs:
        tokenized = tokenizer(
            raw_text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_len,
            stride=stride,
            return_overflowing_tokens=True,
            add_special_tokens=True
        )
        
        num_chunks = len(tokenized["input_ids"])
        for chunk_idx in range(num_chunks):
            input_ids = tokenized["input_ids"][chunk_idx]
            attention_mask = tokenized["attention_mask"][chunk_idx]
            offset_mapping = tokenized["offset_mapping"][chunk_idx]
            
            labels = align_tokens_to_spans(offset_mapping, spans, tag_to_id, raw_text)
            tags = [id_to_tag.get(label_id, "O") if label_id != -100 else "IGNORE" for label_id in labels]
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            
            samples.append({
                "tokens": tokens,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "tags": tags,
                "metadata": {
                    "subject": exam_data.get("subject"),
                    "grade": exam_data.get("grade"),
                    "exam_id": exam_data.get("exam_id"),
                    "max_len": max_len,
                    "stride": stride,
                    "chunk_idx": chunk_idx,
                    "total_chunks": num_chunks
                }
            })
            
    return samples

def process_question_as_exam_level(
    q_data: Dict[str, Any],
    tokenizer: Any,
    tag_to_id: Dict[str, int],
    id_to_tag: Dict[int, str],
    window_configs: List[Tuple[int, int]],
    reconstructor_config: ReconstructorConfig
) -> List[Dict[str, Any]]:
    """
    Treats an individual question as a mini-exam and tokenizes it using sliding window configs.
    """
    if q_data.get("is_real", False) and "raw_text" in q_data and "spans" in q_data:
        raw_text = q_data["raw_text"]
        spans = q_data["spans"]
        if reconstructor_config.latex_mask_prob > 0.0:
            import hashlib
            h = hashlib.md5((q_data.get("exam_id", "") or raw_text).encode("utf-8")).hexdigest()
            rng = random.Random(int(h, 16) & 0xFFFFFFFF)
            raw_text, spans = mask_latex_in_real_data(
                raw_text, spans, reconstructor_config.latex_placeholder, reconstructor_config.latex_mask_prob, rng
            )
    else:
        q_reconstructed = reconstruct_question(q_data, reconstructor_config)
        raw_text = q_reconstructed["raw_text"]
        spans = q_reconstructed["spans"]
    
    samples = []
    
    for max_len, stride in window_configs:
        tokenized = tokenizer(
            raw_text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_len,
            stride=stride,
            return_overflowing_tokens=True,
            add_special_tokens=True
        )
        
        num_chunks = len(tokenized["input_ids"])
        for chunk_idx in range(num_chunks):
            input_ids = tokenized["input_ids"][chunk_idx]
            attention_mask = tokenized["attention_mask"][chunk_idx]
            offset_mapping = tokenized["offset_mapping"][chunk_idx]
            
            labels = align_tokens_to_spans(offset_mapping, spans, tag_to_id, raw_text)
            tags = [id_to_tag.get(label_id, "O") if label_id != -100 else "IGNORE" for label_id in labels]
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            
            samples.append({
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
                    "max_len": max_len,
                    "stride": stride,
                    "chunk_idx": chunk_idx,
                    "total_chunks": num_chunks
                }
            })
            
    return samples

def process_single_question_legacy(
    q_data: Dict[str, Any], 
    tokenizer: Any, 
    tag_to_id: Dict[str, int], 
    id_to_tag: Dict[int, str],
    reconstructor_config: ReconstructorConfig
) -> Optional[Dict[str, Any]]:
    """
    Legacy method for single question parsing (keeps original question-level split layout).
    """
    if q_data.get("is_real", False) and "raw_text" in q_data and "spans" in q_data:
        raw_text = q_data["raw_text"]
        spans = q_data["spans"]
        if reconstructor_config.latex_mask_prob > 0.0:
            import hashlib
            h = hashlib.md5((q_data.get("exam_id", "") or raw_text).encode("utf-8")).hexdigest()
            rng = random.Random(int(h, 16) & 0xFFFFFFFF)
            raw_text, spans = mask_latex_in_real_data(
                raw_text, spans, reconstructor_config.latex_placeholder, reconstructor_config.latex_mask_prob, rng
            )
    else:
        q_reconstructed = reconstruct_question(q_data, reconstructor_config)
        raw_text = q_reconstructed["raw_text"]
        spans = q_reconstructed["spans"]
    
    tokenized = tokenizer(
        raw_text,
        return_offsets_mapping=True,
        truncation=True,
        add_special_tokens=True
    )
    
    offset_mapping = tokenized["offset_mapping"]
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    
    labels = align_tokens_to_spans(offset_mapping, spans, tag_to_id, raw_text)
    tags = [id_to_tag.get(label_id, "O") if label_id != -100 else "IGNORE" for label_id in labels]
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
            "is_group": q_data.get("is_group", False),
            "chapter": q_data.get("chapter"),
            "unit": q_data.get("unit"),
            "problem_type_id": q_data.get("problem_type_id"),
            "problem_type_name": q_data.get("problem_type_name"),
            "problem_type_level": q_data.get("problem_type_level")
        }
    }

def replace_latex_in_question(q_data: Dict[str, Any], placeholder: str) -> Dict[str, Any]:
    import copy
    q_copy = copy.deepcopy(q_data)
    
    def process_field(val):
        if isinstance(val, str):
            return re.sub(r"\$\$.*?\$\$|\$.*?\$", placeholder, val)
        elif isinstance(val, list):
            return [process_field(x) for x in val]
        return val

    if q_copy.get("is_group", False):
        if "context" in q_copy:
            q_copy["context"] = process_field(q_copy["context"])
        if "questions" in q_copy and isinstance(q_copy["questions"], list):
            for sub_q in q_copy["questions"]:
                if "stem" in sub_q:
                    sub_q["stem"] = process_field(sub_q["stem"])
                if "options" in sub_q:
                    sub_q["options"] = process_field(sub_q["options"])
    else:
        if "stem" in q_copy:
            q_copy["stem"] = process_field(q_copy["stem"])
        if "options" in q_copy:
            q_copy["options"] = process_field(q_copy["options"])
            
    return q_copy

def process_single_question(
    q_data: Dict[str, Any],
    tokenizer: Any,
    tag_to_id: Dict[str, int],
    id_to_tag: Dict[int, str],
    latex_placeholder: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    config = ReconstructorConfig()
    if latex_placeholder is not None:
        config.latex_placeholder = latex_placeholder
        config.latex_mask_prob = 1.0
    return process_single_question_legacy(q_data, tokenizer, tag_to_id, id_to_tag, config)

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
        default="FacebookAI/xlm-roberta-base",
        help="Hugging Face model / tokenizer name (default: 'FacebookAI/xlm-roberta-base')"
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
        
    json_files = list(input_path.glob("question_*.json"))
    exam_files = list(input_path.glob("**/exam_*.json"))
    real_exam_files = list(input_path.glob("**/real_exam_*.json"))
    if not json_files and not exam_files and not real_exam_files:
        print(f"No question JSON files, exam JSON files, or real exam JSON files found in '{args.input_dir}'. Please generate data first.")
        sys.exit(1)
    # Combine real exams into exam_files list
    exam_files.extend(real_exam_files)
        
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
        max_inline_tabs=getattr(args, "max_inline_tabs", 3)
    )
        
    print(f"Processing data: found {len(json_files)} question file(s) and {len(exam_files)} exam file(s)...")
    processed_samples = []
    
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
                # Compile exam level
                samples = process_exam_level(exam_data, tokenizer, tag_to_id, id_to_tag, window_configs, reconstructor_config)
                for s in samples:
                    s["metadata"]["source_file"] = file_path.name
                    processed_samples.append(s)
                    exam_q_count += 1
            else:
                # Process exam at question level (legacy fallback)
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
        with open(split_file, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"Saved {len(samples)} samples to '{split_file}'")
        
    print("\nDataset preparation completed successfully!")
    print(f"Total samples: {n_total} (Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)})")
    print(f"Label mapping saved to '{output_path / 'label_mapping.json'}'")

if __name__ == "__main__":
    main()
