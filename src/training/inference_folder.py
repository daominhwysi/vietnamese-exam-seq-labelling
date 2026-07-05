#!/usr/bin/env python3
import os
import sys
import json
import re
import torch
import argparse
from pathlib import Path
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
    base_tags = ["question_label", "stem", "option_label", "option_text", "context", "section", "explanation"]
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
        
    # Case 2: Balanced braces, brackets, and parentheses
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
        
    # Case 3: Contains standard math/latex character indicators
    math_indicators = ['\\', '^', '_', '+', '-', '*', '/', '=', '<', '>', '{', '}', '[', ']']
    if any(ind in content_stripped for ind in math_indicators):
        return True
        
    # Case 4: Short alphanumeric math terms without spaces (e.g. $2a$, $x1$, $100$)
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
                    "text": raw_text[current_start:current_end].strip()
                })
            current_label = label[2:]
            current_start = start
            current_end = end
        elif label.startswith("I-") and current_label == label[2:]:
            # Extend the current segment
            if current_start == -1:
                current_start = start
            current_end = end
        else:  # Outside tag "O" or mismatching tag
            if current_label and current_start < current_end:
                segments.append({
                    "label": current_label,
                    "text": raw_text[current_start:current_end].strip()
                })
                current_label = None
                current_start = -1
                current_end = -1
                
    # Save any remaining segment
    if current_label and current_start < current_end:
        segments.append({
            "label": current_label,
            "text": raw_text[current_start:current_end].strip()
        })
        
    return segments

def segments_to_xml(raw_text: str, segments: list) -> str:
    """
    Reconstructs the full raw text with inline XML tags around each labeled segment,
    matching the format produced by annotate_ocr.py:
        <question_label>Câu 1.</question_label> <stem>Nội dung câu hỏi...</stem>

    Untagged text between segments (e.g. page headers, separators) is preserved
    verbatim outside any tag. Character offsets on the segments are used to find
    the exact gap text from raw_text.
    """
    # Re-derive character offsets by searching for each segment's text in order.
    # The segments list from extract_segments() only carries stripped text,
    # so we locate each one forward from the previous match end.
    result = []
    cursor = 0

    for seg in segments:
        label = seg["label"]
        text = seg["text"]
        if not text:
            continue

        # Find next occurrence of this text starting from cursor
        idx = raw_text.find(text, cursor)
        if idx == -1:
            # Fallback: skip this segment gracefully
            continue

        # Append any untagged gap text before this segment
        if idx > cursor:
            result.append(raw_text[cursor:idx])

        # Append the tagged segment
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
        help="Directory of the trained model (default: './results')"
    )
    parser.add_argument(
        "--base-model-name",
        type=str,
        default="aisingapore/SEA-LION-ModernBERT-300M",
        help="Base model/tokenizer name used (default: 'aisingapore/SEA-LION-ModernBERT-300M')"
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
    
    # 1. Load Tokenizer (prefer local checkpoint, fallback to base model with correct special tokens)
    print(f"Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
        print(f"  Successfully loaded tokenizer from local checkpoint: {args.model_dir}")
    except Exception:
        print(f"  Local tokenizer files not found. Loading base tokenizer: {args.base_model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, use_fast=True)
        # Add the exact same special tokens in the exact same order as during training
        special_tokens = ["<blank />", "<blank/>", "[BLANK]", "[LATEX]"]
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        print(f"  Added special tokens: {special_tokens}")
    
    # 2. Load Label Mappings
    tag_to_id, id_to_tag = load_label_mapping(args.model_dir)
    print(f"Loaded {len(tag_to_id)} labels.")
    
    # 3. Load Model (Supports both full fine-tuned and PEFT/LoRA adapters)
    print(f"Loading model weights from: {args.model_dir}...")
    is_lora = os.path.exists(os.path.join(args.model_dir, "adapter_config.json"))
    
    if is_lora:
        print("Detected LoRA adapter checkpoint. Loading base model first...")
        # Auto-detect base model name from adapter config
        base_model_name = args.base_model_name
        try:
            with open(os.path.join(args.model_dir, "adapter_config.json"), "r", encoding="utf-8") as f:
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
        model = PeftModel.from_pretrained(base_model, args.model_dir)
    else:
        print("Loading full fine-tuned model checkpoint...")
        # Load directly (it already contains the correctly saved, resized embeddings config)
        model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
        
    model.to(device)
    model.eval()
    
    # 4. Process all text files in the input folder
    txt_files = list(input_path.glob("*.txt"))
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
            
            # Offset mapper helper back to original raw_text character positions using efficient binary search
            mapper = OffsetMapper(latex_spans)
            map_idx = mapper.map_idx
 
            span_logits = {}
            span_tokens = {}
            
            num_chunks = len(tokenized["input_ids"])
            print(f"  Tokenized into {num_chunks} chunks.")
            sigma = 64  # boundary margin for trapezoidal/Gaussian weight decay
            
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
                chunk_len = len(chunk_input_ids)
                
                for i, (token, offset, mask) in enumerate(zip(chunk_tokens, chunk_offsets, chunk_attention_mask)):
                    start, end = offset
                    # Skip special tokens and padding tokens
                    if (start == 0 and end == 0) or mask == 0:
                        continue
                        
                    span = (start, end)
                    
                    # Compute Gaussian/trapezoidal boundary weight for position i
                    dist_to_boundary = min(i, chunk_len - 1 - i)
                    w = min(1.0, dist_to_boundary / sigma)
                    w = max(0.0001, w)
                    
                    if span not in span_logits:
                        span_logits[span] = {"logits_sum": chunk_logits[i] * w, "weight_sum": w}
                        span_tokens[span] = token
                    else:
                        span_logits[span]["logits_sum"] += chunk_logits[i] * w
                        span_logits[span]["weight_sum"] += w
                    
            # Reconstruct unique token sequence sorted by start position
            sorted_spans = sorted(span_logits.keys(), key=lambda x: (x[0], x[1]))
            
            predictions = []
            offsets = []
            tokens = []
            attention_mask = []
            
            for span in sorted_spans:
                start, end = span
                logits_sum = span_logits[span]["logits_sum"]
                weight_sum = span_logits[span]["weight_sum"]
                avg_logits = logits_sum / weight_sum
                
                pred_id = torch.argmax(avg_logits).item()
                
                predictions.append(pred_id)
                offsets.append(span)
                tokens.append(span_tokens[span])
                attention_mask.append(1)
                
            # Resolve any invalid BIO transitions to guarantee syntactic validity
            tag_to_id = {v: k for k, v in id_to_tag.items()}
            predictions = resolve_bio_violations(predictions, id_to_tag, tag_to_id)
            
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

class OffsetMapper:
    def __init__(self, latex_spans: list[tuple[int, int]]):
        self.shifts = []
        mod_pos = 0
        orig_pos = 0
        for o_start, o_end in latex_spans:
            segment_len = o_start - orig_pos
            mod_start = mod_pos + segment_len
            mod_end = mod_start + len("[LATEX]")
            shift = o_end - mod_end
            self.shifts.append((mod_start, mod_end, shift, o_start))
            
            orig_pos = o_end
            mod_pos = mod_end
            
        self.starts = [s[0] for s in self.shifts]

    def map_idx(self, idx: int) -> int:
        if not self.shifts:
            return idx
        import bisect
        pos = bisect.bisect_right(self.starts, idx) - 1
        if pos >= 0:
            mod_start, mod_end, shift, o_start = self.shifts[pos]
            if idx < mod_end:
                return o_start
            return idx + shift
        return idx


def resolve_bio_violations(predictions: list[int], id_to_tag: dict[int, str], tag_to_id: dict[str, int]) -> list[int]:
    corrected_predictions = list(predictions)
    prev_tag = "O"
    for idx, pred_id in enumerate(corrected_predictions):
        tag = id_to_tag[pred_id]
        if tag.startswith("I-"):
            tag_class = tag[2:]
            if prev_tag != f"B-{tag_class}" and prev_tag != f"I-{tag_class}":
                b_tag = f"B-{tag_class}"
                if b_tag in tag_to_id:
                    corrected_predictions[idx] = tag_to_id[b_tag]
                    tag = b_tag
                else:
                    corrected_predictions[idx] = tag_to_id.get("O", 0)
                    tag = "O"
        prev_tag = tag
    return corrected_predictions


if __name__ == "__main__":
    main()


