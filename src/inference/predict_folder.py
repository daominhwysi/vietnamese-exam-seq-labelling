#!/usr/bin/env python3
import os
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import json
import re
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Bypass torchao compatibility check bug on older pre-installed versions in Google Colab
try:
    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
except Exception:
    pass

def load_label_mapping(model_dir):
    mapping_path = os.path.join(model_dir, "label_mapping.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
            return mapping["tag_to_id"], {int(k): v for k, v in mapping["id_to_tag"].items()}
            
    # Fallback to standard base tags mapping
    base_tags = ["question_label", "stem", "option_label", "option_text", "stimulus", "section", "explanation"]
    tag_to_id = {"O": 0}
    for tag in base_tags:
        tag_to_id[f"B-{tag}"] = len(tag_to_id)
        tag_to_id[f"I-{tag}"] = len(tag_to_id)
    id_to_tag = {v: k for k, v in tag_to_id.items()}
    return tag_to_id, id_to_tag

def is_valid_latex(content):
    content_stripped = content.strip()
    if not content_stripped:
        return False
    
    # Case 1: Single variable or number (e.g. $x$, $a$, $1$)
    if len(content_stripped) == 1:
        return content_stripped.isalnum()
        
    # Case 2: Mathematical interval notation like (-\infty; 0] or [8; +\infty) or (1; 2]
    if re.match(r'^[\[\(][^\[\]\(\)]+[\]\)]$', content_stripped) and (';' in content_stripped or ',' in content_stripped):
        return True

    # Case 3: Balanced braces, brackets, and parentheses
    brackets = {'{': '}', '(': ')', '[': ']'}
    stack = []
    for char in content_stripped:
        if char in brackets:
            stack.append(char)
        elif char in brackets.values():
            if not stack:
                return False
            last = stack.pop()
            if brackets[last] != char:
                return False
    if stack:
        return False
        
    # Case 4: Contains standard math/latex character indicators
    math_indicators = ['\\', '^', '_', '+', '-', '*', '/', '=', '<', '>', '{', '}', '[', ']']
    if any(ind in content_stripped for ind in math_indicators):
        return True
        
    # Case 5: Short alphanumeric math terms without spaces (e.g. $2a$, $x1$, $100$)
    if len(content_stripped) < 10 and re.match(r'^[a-zA-Z0-9]+$', content_stripped):
        return True
        
    return False

def get_latex_spans(text):
    spans = []
    # 1. Matches $$...$$ (display math)
    for match in re.finditer(r"\$\$.*?\$\$", text, re.DOTALL):
        # Verify content inside $$
        content = match.group(0)[2:-2]
        if is_valid_latex(content):
            spans.append(match.span())
        
    # 2. Matches $...$ (inline math)
    for match in re.finditer(r"\$(?!\s)[^\$\n]+?(?<!\s)\$", text):
        span = match.span()
        content = match.group(0)[1:-1]
        if not is_valid_latex(content):
            continue
            
        # Avoid overlapping with display math
        overlap = False
        for d_start, d_end in spans:
            if not (span[1] <= d_start or span[0] >= d_end):
                overlap = True
                break
        if not overlap:
            spans.append(span)
            
    spans.sort(key=lambda x: x[0])
    return spans

def extract_segments(raw_text, predictions, offsets, attention_mask, id_to_tag):
    """
    Groups contiguous token classifications using the offset mapping to extract
    exact text segments from the original raw string. Skips masked/ignored tokens.
    Implements a robust BIO state machine with automatic orphaned I-tag promotion.
    """
    segments = []
    current_label = None
    current_start = -1
    current_end = -1
    
    for idx, (pred_id, offset) in enumerate(zip(predictions, offsets)):
        start, end = offset
        # Skip special tokens, padding, and masked tokens (where attention_mask is 0)
        if (start == 0 and end == 0) or (attention_mask[idx] == 0):
            continue
            
        label = id_to_tag[pred_id]
        
        if label.startswith("B-"):
            # Save the previous segment if tracking
            if current_label and current_start < current_end:
                segments.append({
                    "label": current_label,
                    "start": current_start,
                    "end": current_end,
                    "text": raw_text[current_start:current_end]
                })
            current_label = label[2:]
            current_start = start
            current_end = end
        elif label.startswith("I-"):
            tag_name = label[2:]
            if current_label == tag_name:
                # Extend the current segment
                current_end = end
            else:
                # Save the previous segment if tracking
                if current_label and current_start < current_end:
                    segments.append({
                        "label": current_label,
                        "start": current_start,
                        "end": current_end,
                        "text": raw_text[current_start:current_end]
                    })
                # Auto-promote orphaned I-tag to start a new segment
                current_label = tag_name
                current_start = start
                current_end = end
        else:  # Outside tag "O" or non-entity tag
            if current_label and current_start < current_end:
                segments.append({
                    "label": current_label,
                    "start": current_start,
                    "end": current_end,
                    "text": raw_text[current_start:current_end]
                })
                current_label = None
                current_start = -1
                current_end = -1
                
    # Save any remaining segment
    if current_label and current_start < current_end:
        segments.append({
            "label": current_label,
            "start": current_start,
            "end": current_end,
            "text": raw_text[current_start:current_end]
        })

    # Filter out empty and spurious non-alphanumeric label segments (e.g. '*' or '-' tagged as option_label)
    cleaned_segments = []
    for s in segments:
        txt = s["text"].strip()
        label = s["label"]
        if not txt:
            continue
        if label in ["option_label", "question_label"] and not any(c.isalnum() for c in txt):
            continue
        cleaned_segments.append(s)
        
    return cleaned_segments

def segments_to_xml(raw_text: str, segments: list) -> str:
    """
    Reconstructs the full raw text with inline XML tags around each labeled segment,
    matching the ground-truth format using character offsets directly to avoid runaway offset drift:
        <question_label>Câu 1.</question_label> <stem>Nội dung câu hỏi...</stem>

    Untagged text between segments (e.g. page headers, separators) is preserved
    verbatim outside any tag.
    """
    sorted_segs = sorted(segments, key=lambda x: (x.get("start", 0), -x.get("end", 0)))
    result = []
    cursor = 0

    for seg in sorted_segs:
        label = seg["label"]
        start = seg.get("start", -1)
        end = seg.get("end", -1)
        text = seg.get("text", "")

        if start >= 0 and end >= 0 and start >= cursor:
            if start > cursor:
                result.append(raw_text[cursor:start])
            result.append(f"<{label}>{raw_text[start:end]}</{label}>")
            cursor = end
        elif start == -1 or end == -1:
            idx = raw_text.find(text, cursor)
            if idx != -1:
                if idx > cursor:
                    result.append(raw_text[cursor:idx])
                result.append(f"<{label}>{text}</{label}>")
                cursor = idx + len(text)

    # Append any trailing untagged text after the last segment
    if cursor < len(raw_text):
        result.append(raw_text[cursor:])

    return "".join(result)

def main():
    parser = argparse.ArgumentParser(description="Batch inference for exam document segmentation from a folder of text files")
    parser.add_argument(
        "-i", "--input-dir",
        type=str,
        required=True,
        help="Path to folder containing the raw text (.txt) files"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="inference_output",
        help="Path to folder where prediction results will be saved (default: 'inference_output')"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./results",
        help="Directory of the trained model or HF repo (default: './results' with fallback to HF)"
    )
    parser.add_argument(
        "--base-model-name",
        type=str,
        default="jhu-clsp/mmBERT-base",
        help="Base model/tokenizer name used (default: 'jhu-clsp/mmBERT-base')"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum sequence length for tokenization window (default: 1024)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=256,
        help="Overlap stride for sliding window tokenization (default: 256)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    
    if not input_path.exists() or not input_path.is_dir():
        print(f"Error: Input directory '{args.input_dir}' does not exist or is not a directory.")
        sys.exit(1)
        
    output_path.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Auto-detect default model path if default "./results" doesn't exist
    model_dir = args.model_dir
    if model_dir.startswith("./content/"):
        alt_path = model_dir[1:]  # /content/...
        if os.path.exists(alt_path):
            model_dir = alt_path
        elif os.path.exists(model_dir[10:]):  # stripped ./content/
            model_dir = model_dir[10:]
            
    if model_dir == "./results" and not os.path.exists("./results"):
        if os.path.exists("./results_enhanced_v3"):
            model_dir = "./results_enhanced_v3"
        elif os.path.exists("./results_full"):
            model_dir = "./results_full"
        else:
            model_dir = "daominhwysi/results_full"
            
    if not os.path.exists(model_dir) and (model_dir.startswith((".", "/", "\\")) or "/" in model_dir and os.path.exists(os.path.abspath(model_dir))):
        if os.path.exists(os.path.abspath(model_dir)):
            model_dir = os.path.abspath(model_dir)
    
    # 1. Load Tokenizer (prefer local checkpoint, fallback to base model with correct special tokens)
    print(f"Loading tokenizer from: {model_dir}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        print(f"  Successfully loaded tokenizer from checkpoint: {model_dir}")
    except Exception:
        print(f"  Local tokenizer files not found. Loading base tokenizer: {args.base_model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, use_fast=True)
        # Add the exact same special tokens in the exact same order as during training
        special_tokens = ["<blank />", "<blank/>", "[BLANK]", "[LATEX]"]
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        print(f"  Added special tokens: {special_tokens}")
    
    # 2. Load Label Mappings
    tag_to_id, id_to_tag = load_label_mapping(model_dir)
    print(f"Loaded {len(tag_to_id)} labels.")
    
    # 3. Load Model (Supports Enhanced Head, full fine-tuned, and PEFT/LoRA adapters)
    print(f"Loading model weights from: {model_dir}...")
    is_lora = os.path.exists(os.path.join(model_dir, "adapter_config.json"))
    has_enhanced_head = os.path.exists(os.path.join(model_dir, "enhanced_head_config.json"))
    if not has_enhanced_head:
        try:
            from transformers.utils.hub import cached_file
            st_file = os.path.join(model_dir, "model.safetensors") if os.path.isdir(model_dir) else cached_file(model_dir, "model.safetensors")
            if st_file and os.path.exists(st_file):
                from safetensors import safe_open
                with safe_open(st_file, framework="pt") as f:
                    st_keys = f.keys()
                    if any("head.layer_weights" in k or "head.dense" in k for k in st_keys):
                        has_enhanced_head = True
        except Exception:
            pass
    if not is_lora and not os.path.exists(model_dir):
        try:
            from transformers.utils.hub import cached_file
            hf_lora = cached_file(model_dir, "adapter_config.json")
            if hf_lora is not None:
                is_lora = True
        except Exception:
            pass
    
    if has_enhanced_head:
        print("Detected Enhanced Head model checkpoint. Loading with EnhancedTokenClassifierModel...")
        from src.model.head import EnhancedTokenClassifierModel
        model = EnhancedTokenClassifierModel.from_pretrained(
            model_dir,
            num_labels=len(tag_to_id),
            id2label=id_to_tag,
            label2id=tag_to_id
        )
        if hasattr(model.config, "id2label") and model.config.id2label:
            id_to_tag = {int(k): v for k, v in model.config.id2label.items()}
            tag_to_id = {v: k for k, v in id_to_tag.items()}
    elif is_lora:
        print("Detected LoRA adapter checkpoint. Loading base model first...")
        # Auto-detect base model name from adapter config
        base_model_name = args.base_model_name
        try:
            with open(os.path.join(model_dir, "adapter_config.json"), "r", encoding="utf-8") as f:
                config = json.load(f)
                base_model_name = config.get("base_model_name_or_path", base_model_name)
                print(f"  Auto-detected base model name from adapter config: {base_model_name}")
        except Exception:
            pass
            
        from peft import PeftModel
        base_model = AutoModelForTokenClassification.from_pretrained(
            base_model_name,
            num_labels=len(tag_to_id),
            id2label=id_to_tag,
            label2id=tag_to_id
        )
        base_model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(base_model, model_dir)
    else:
        print("Loading full fine-tuned model checkpoint...")
        # Load directly (it already contains the correctly saved, resized embeddings config)
        model = AutoModelForTokenClassification.from_pretrained(model_dir)
        if hasattr(model.config, "id2label") and model.config.id2label:
            id_to_tag = {int(k): v for k, v in model.config.id2label.items()}
            tag_to_id = {v: k for k, v in id_to_tag.items()}
        
    model.to(device)
    model.eval()
    
    # 4. Process all text/md files in the input folder
    txt_files = sorted(list(input_path.glob("*.txt")) + list(input_path.glob("*.md")))
    if not txt_files:
        print(f"No .txt files found in '{args.input_dir}'.")
        sys.exit(0)
        
    print(f"Found {len(txt_files)} file(s). Starting batch inference...")
    
    for file_path in txt_files:
        print(f"Processing '{file_path.name}'...")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
                
            if not raw_text.strip():
                print(f"Warning: File '{file_path.name}' is empty, skipping.")
                continue
                
            # Get LaTeX spans from the raw text
            latex_spans = get_latex_spans(raw_text)
            
            # Pre-process raw_text by replacing LaTeX equations with '[LATEX]'
            processed_text = ""
            last_idx = 0
            for start, end in latex_spans:
                processed_text += raw_text[last_idx:start] + "[LATEX]"
                last_idx = end
            processed_text += raw_text[last_idx:]
            
            # Tokenize the processed text using sliding window config
            tokenized = tokenizer(
                processed_text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                stride=args.stride,
                return_overflowing_tokens=True
            )
            
            # Map modified offsets back to original raw_text character positions helper
            def map_idx(idx):
                mod_pos = 0
                orig_pos = 0
                for o_start, o_end in latex_spans:
                    segment_len = o_start - orig_pos
                    if idx <= mod_pos + segment_len:
                        return orig_pos + (idx - mod_pos)
                    orig_pos = o_end
                    mod_pos += segment_len + len("[LATEX]")
                    if idx < mod_pos:
                        return o_start
                return orig_pos + (idx - mod_pos)

            span_logits = {}
            span_tokens = {}
            
            num_chunks = len(tokenized["input_ids"])
            print(f"  Tokenized into {num_chunks} chunks.")
            
            for chunk_idx in range(num_chunks):
                chunk_input_ids = tokenized["input_ids"][chunk_idx]
                chunk_attention_mask = tokenized["attention_mask"][chunk_idx]
                chunk_mod_offsets = tokenized["offset_mapping"][chunk_idx]
                
                # Map chunk offsets back to original positions
                chunk_offsets = []
                for start, end in chunk_mod_offsets:
                    if start == 0 and end == 0:
                        chunk_offsets.append((0, 0))
                    else:
                        chunk_offsets.append((map_idx(start), map_idx(end)))
                
                inputs = {
                    "input_ids": torch.tensor([chunk_input_ids]).to(device),
                    "attention_mask": torch.tensor([chunk_attention_mask]).to(device)
                }
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    
                chunk_logits = outputs.logits[0].cpu()
                chunk_tokens = tokenizer.convert_ids_to_tokens(chunk_input_ids)
                
                for i, (token, offset, mask) in enumerate(zip(chunk_tokens, chunk_offsets, chunk_attention_mask)):
                    start, end = offset
                    # Skip special tokens and padding tokens
                    if (start == 0 and end == 0) or mask == 0:
                        continue
                        
                    span = (start, end)
                    if span not in span_logits:
                        span_logits[span] = []
                        span_tokens[span] = token
                    span_logits[span].append(chunk_logits[i])
                    
            # Reconstruct unique token sequence sorted by start position
            sorted_spans = sorted(span_logits.keys(), key=lambda x: (x[0], x[1]))
            
            predictions = []
            offsets = []
            tokens = []
            attention_mask = []
            
            for span in sorted_spans:
                start, end = span
                # Average predictions from multiple windows
                avg_logits = torch.mean(torch.stack(span_logits[span]), dim=0)
                pred_id = torch.argmax(avg_logits).item()
                
                predictions.append(pred_id)
                offsets.append(span)
                tokens.append(span_tokens[span])
                attention_mask.append(1)
            
            # 1. Structured JSON (Parsing Version)
            segments = extract_segments(raw_text, predictions, offsets, attention_mask, id_to_tag)
            result_data = {
                "file_name": file_path.name,
                "original_text_length": len(raw_text),
                "segments": segments
            }
            output_json_file = output_path / f"{file_path.stem}_structured.json"
            with open(output_json_file, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
                
            # 2. Human-Readable Token-Class Predictions (.txt)
            lines = []
            lines.append(f"File: {file_path.name}")
            lines.append(f"Original Text Length: {len(raw_text)}")
            lines.append(f"Window Size: {args.max_length} | Overlap Stride: {args.stride}")
            lines.append("-" * 75)
            lines.append(f"{'Token':<30} | {'Prediction':<20} | Offsets")
            lines.append("-" * 75)
            
            for token, pred_id, offset in zip(tokens, predictions, offsets):
                start, end = offset
                tag = id_to_tag[pred_id]
                
                readable_token = token.replace(" ", " ").replace("▁", "")
                lines.append(f"{readable_token:<30} | {tag:<20} | [{start}, {end}]")
            
            lines.append("-" * 75)
            
            output_txt_file = output_path / f"{file_path.stem}_predictions.txt"
            with open(output_txt_file, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
                
            print(f"  [Success] Saved structured JSON to '{output_json_file.name}'")
            print(f"  [Success] Saved readable predictions to '{output_txt_file.name}'")

            # 3. Inline-tagged XML (matches annotate_ocr.py format)
            xml_content = segments_to_xml(raw_text, segments)
            output_xml_file = output_path / f"{file_path.stem}_annotated.xml"
            with open(output_xml_file, "w", encoding="utf-8") as f:
                f.write(xml_content)
            print(f"  [Success] Saved annotated XML to '{output_xml_file.name}'")
            
        except Exception as e:
            print(f"  [Error] Failed to process '{file_path.name}': {e}")
            
    print("\nBatch inference completed successfully!")

if __name__ == "__main__":
    main()
