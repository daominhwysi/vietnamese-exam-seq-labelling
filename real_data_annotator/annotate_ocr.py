import os
import sys
import json
import re
import hashlib
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Set up path to import src modules (like token_tracker)
script_dir = Path(__file__).resolve().parent
workspace_dir = script_dir.parent
sys.path.append(str(workspace_dir))

from dotenv import load_dotenv
from openai import OpenAI
from src.token_tracker import log_response

# Reconfigure stdout for UTF-8 to handle Vietnamese terminal logging
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

SYSTEM_PROMPT = """You are an expert NLP data annotator for Vietnamese educational exam papers.
Your task is to annotate the provided raw OCR text of an exam paper by wrapping specific components in XML tags.

You MUST wrap the following entities:
1. <question_label>...</question_label>: Wrap question prefix indicators (e.g. "Câu 1:", "Câu 12.", "Question 1:", "C1.", "1.", "a) ", "1) ").
2. <stem>...</stem>: Wrap the main text body of a question or sub-question (including any ordering items or list of items to order).
3. <option_label>...</option_label>: Wrap options letters/prefixes (e.g. "A.", "B.", "C.", "D.", "a.", "b.", "c.", "d.").
4. <option_text>...</option_text>: Wrap the textual content of options.
5. <context>...</context>: Wrap the shared passage/context block in group questions (passages, reading texts, shared diagrams description).

CRITICAL RULES:
1. Do NOT modify, correct, rephrase, or change any part of the input text. Preserve all spelling mistakes, typos, symbols, page markers (like "<|page|>Page X"), LaTeX formulas (enclosed in $...$ or $$...$$), and formatting exactly as they are in the input. ONLY insert the opening and closing XML tags.
2. Every tag MUST be properly closed. Do not nest tags. Tags must be strictly sequential.
3. Text that does not belong to any entity (e.g., page numbers, header information like "SỞ GD-ĐT...", "ĐỀ CHÍNH THỨC", instructions, horizontal separators) must NOT be wrapped in any tags. Keep it outside the XML tags.
4. Output ONLY the annotated text. Do not write any markdown code blocks (e.g. ```xml), introduction, or conversational filler.
"""

def clean_llm_response(text: str) -> str:
    # 1. Prune think tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    
    # 2. Strip markdown code blocks if the model returned them
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            if lines[-1].strip() == "```":
                text = "\n".join(lines[1:-1])
            else:
                text = "\n".join(lines[1:])
    return text.strip()

def parse_xml_annotations(tagged_text: str) -> Tuple[str, List[Dict[str, Any]]]:
    allowed_tags = {"question_label", "stem", "option_label", "option_text", "context"}
    raw_chars = []
    spans = []
    
    tag_pattern = re.compile(r"<(/)?([a-zA-Z_0-9]+)>")
    
    pos = 0
    current_open_tag = None
    tag_start_idx = -1
    
    for match in tag_pattern.finditer(tagged_text):
        start, end = match.span()
        # Add the text before the tag to raw_chars
        text_before = tagged_text[pos:start]
        raw_chars.append(text_before)
        
        is_closing = bool(match.group(1))
        tag_name = match.group(2)
        
        if tag_name in allowed_tags:
            if not is_closing:
                # Open tag
                current_open_tag = tag_name
                tag_start_idx = len("".join(raw_chars))
            else:
                # Close tag
                if current_open_tag == tag_name and tag_start_idx != -1:
                    tag_end_idx = len("".join(raw_chars))
                    span_text = "".join(raw_chars)[tag_start_idx:tag_end_idx]
                    spans.append({
                        "start": tag_start_idx,
                        "end": tag_end_idx,
                        "label": tag_name,
                        "text": span_text
                    })
                current_open_tag = None
                tag_start_idx = -1
        else:
            # If it's an unallowed tag, treat it as literal text
            raw_chars.append(match.group(0))
            
        pos = end
        
    raw_chars.append(tagged_text[pos:])
    raw_text = "".join(raw_chars)
    
    return raw_text, spans

def get_fallback_metadata(file_path: Path) -> Tuple[str, int]:
    # Subject mapping from folder structure
    parent_name = file_path.parent.name.lower()
    subject_map = {
        "toan": "math_algebra",
        "ta": "english",
        "vl": "physics",
        "vumaiphuongta": "english",
        "hoa": "chemistry",
        "su": "history",
        "dia": "geography",
        "gdcd": "economics_law",
        "van": "literature"
    }
    subject = subject_map.get(parent_name, "math_algebra")
    
    # Grade extraction
    grade = 12
    path_str = str(file_path).lower()
    grade_match = re.search(r'\b(lop|lớp|g)?(10|11|12|8|9)\b', path_str)
    if grade_match:
        grade = int(grade_match.group(2))
    return subject, grade

def main():
    parser = argparse.ArgumentParser(description="Annotate raw OCR Vietnamese exams using LLMs")
    parser.add_argument(
        "--input", "-i",
        default=str(script_dir / "out"),
        help="Directory containing output markdown files from OCR (default: real_data_annotator/out)"
    )
    parser.add_argument(
        "--output", "-o",
        default=str(workspace_dir / "output" / "real_exams"),
        help="Directory to save the annotated JSON files (default: output/real_exams)"
    )
    parser.add_argument(
        "--subject", "-s",
        default=None,
        help="Subject tag for the real data (e.g. math_algebra, physics, english)"
    )
    parser.add_argument(
        "--grade", "-g",
        type=int,
        default=None,
        help="Grade level for the real data (e.g. 10, 11, 12)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit the number of files to process"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-annotation even if the output file already exists"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model identifier (default: auto-detected based on environment keys)"
    )
    
    args = parser.parse_args()
    
    # 1. Setup Client based on available keys
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    llm_key = os.environ.get("LLM_API_KEY")
    
    if deepseek_key:
        print("Using DeepSeek API for annotation.")
        client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com")
        default_model = "deepseek-chat"
    elif llm_key:
        print("Using Vilao.ai (Minimax-M3) API for annotation.")
        client = OpenAI(api_key=llm_key, base_url="https://api.vilao.ai/v1")
        default_model = "mn/Minimax-M3"
    else:
        print("Error: Neither DEEPSEEK_API_KEY nor LLM_API_KEY is configured in the environment.")
        sys.exit(1)
        
    model_name = args.model or default_model
    print(f"Using model: {model_name}")
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        print(f"Error: Input directory '{args.input}' does not exist.")
        sys.exit(1)
        
    # Find all .md files
    md_files = []
    for root, _, files in os.walk(input_path):
        for file in files:
            if file.lower().endswith(".md"):
                md_files.append(Path(root) / file)
                
    if not md_files:
        print(f"No .md files found in '{args.input}'.")
        sys.exit(0)
        
    if args.limit is not None:
        md_files = md_files[:args.limit]
        
    print(f"Found {len(md_files)} file(s) to process.\n")
    
    success_count = 0
    skipped_count = 0
    
    for idx, file_path in enumerate(md_files):
        print(f"[{idx+1}/{len(md_files)}] Processing: {file_path.relative_to(input_path)}")
        
        # Determine subject & grade
        subject = args.subject
        grade = args.grade
        
        fallback_subject, fallback_grade = get_fallback_metadata(file_path)
        if subject is None:
            subject = fallback_subject
        if grade is None:
            grade = fallback_grade
            
        # Create output filename based on a hash of the relative path to avoid collisions
        rel_sig = str(file_path.relative_to(input_path))
        path_hash = hashlib.md5(rel_sig.encode("utf-8")).hexdigest()[:8]
        out_filename = f"real_exam_{subject}_g{grade}_{path_hash}.json"
        out_file_path = output_path / out_filename
        
        if out_file_path.exists() and not args.force:
            print(f"  Skipping: Annotated output already exists at '{out_file_path.name}'.")
            skipped_count += 1
            continue
            
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_ocr_text = f.read()
                
            if not raw_ocr_text.strip():
                print("  Warning: File is empty. Skipping.")
                continue
                
            print(f"  Calling LLM to annotate text ({len(raw_ocr_text)} chars)...")
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": raw_ocr_text}
                ],
                temperature=0.0
            )
            log_response(response, model=model_name)
            
            raw_result = response.choices[0].message.content
            tagged_text = clean_llm_response(raw_result)
            
            # Parse XML
            raw_text, spans = parse_xml_annotations(tagged_text)
            
            # Simple length validation (should be similar to original text)
            original_len = len(raw_ocr_text)
            stripped_len = len(raw_text)
            ratio = stripped_len / max(1, original_len)
            
            if ratio < 0.85 or ratio > 1.15:
                print(f"  Warning: Text mismatch detected (original: {original_len}, stripped: {stripped_len}, ratio: {ratio:.2f}).")
                print("  Skipping this file to avoid bad annotations.")
                continue
                
            # Build and save result
            result_data = {
                "exam_id": f"real_{path_hash}",
                "subject": subject,
                "grade": grade,
                "created_at": datetime.now().isoformat(),
                "is_real": True,
                "raw_text": raw_text,
                "spans": spans
            }
            
            with open(out_file_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
                
            print(f"  Saved annotation to: {out_file_path.name} ({len(spans)} spans)")
            success_count += 1
            
        except Exception as e:
            print(f"  Error processing file: {e}")
            
    print("\n==================================================")
    print(f"Finished processing real OCR exams.")
    print(f"  Successfully annotated : {success_count} file(s)")
    print(f"  Skipped (already exist): {skipped_count} file(s)")
    print(f"  All outputs saved to   : {output_path}")
    print("==================================================")

if __name__ == "__main__":
    main()
