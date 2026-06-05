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
    base_tags = ["question_label", "stem", "option_label", "option_text", "context"]
    tag_to_id = {"O": 0}
    for tag in base_tags:
        tag_to_id[f"B-{tag}"] = len(tag_to_id)
        tag_to_id[f"I-{tag}"] = len(tag_to_id)
    id_to_tag = {v: k for k, v in tag_to_id.items()}
    return tag_to_id, id_to_tag

def get_latex_spans(text):
    spans = []
    # Matches $$...$$ and $...$
    for match in re.finditer(r"\$\$.*?\$\$|\$.*?\$", text, re.DOTALL):
        spans.append(match.span())
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
        help="Path to folder where structured JSON results will be saved (default: 'inference_output')"
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
        default="jhu-clsp/mmbert-base",
        help="Base model/tokenizer name used (default: 'jhu-clsp/mmbert-base')"
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
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        print(f"  Successfully loaded tokenizer from local checkpoint: {args.model_dir}")
    except Exception:
        print(f"  Local tokenizer files not found. Loading base tokenizer: {args.base_model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
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
            
            # Tokenize the processed text to match training format
            tokenized = tokenizer(
                processed_text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=512,  # Window size for inference segment
                return_tensors="pt"
            )
            
            input_ids = tokenized["input_ids"][0].numpy()
            attention_mask = tokenized["attention_mask"][0].numpy()
            mod_offsets = tokenized["offset_mapping"][0].numpy()
            
            # Map modified offsets back to original raw_text character positions
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

            offsets = []
            for start, end in mod_offsets:
                if start == 0 and end == 0:
                    offsets.append((0, 0))
                else:
                    offsets.append((map_idx(start), map_idx(end)))
            
            # Convert modified arrays back to tensors and send to model device
            inputs = {
                "input_ids": torch.tensor([input_ids]).to(device),
                "attention_mask": torch.tensor([attention_mask]).to(device)
            }
            
            with torch.no_grad():
                outputs = model(**inputs)
                
            predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()
            
            # Extract structured sections
            segments = extract_segments(raw_text, predictions, offsets, attention_mask, id_to_tag)
            
            # Save output as structured JSON
            result_data = {
                "file_name": file_path.name,
                "original_text_length": len(raw_text),
                "segments": segments
            }
            
            output_file = output_path / f"{file_path.stem}_structured.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
                
            print(f"  [Success] Saved structured segments to '{output_file.name}'")
            
        except Exception as e:
            print(f"  [Error] Failed to process '{file_path.name}': {e}")
            
    print("\nBatch inference completed successfully!")

if __name__ == "__main__":
    main()
