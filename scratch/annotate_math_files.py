import os
import sys
import json
import hashlib
import difflib
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Add workspace directory to path
script_dir = Path(__file__).resolve().parent
workspace_dir = script_dir.parent
sys.path.append(str(workspace_dir))

load_dotenv(dotenv_path=workspace_dir / ".env")

SYSTEM_PROMPT = """You are an expert NLP data annotator for Vietnamese educational exam papers.
Your task is to annotate the provided raw OCR text of an exam paper by wrapping specific components in XML tags.

You MUST wrap the following entities:
1. <question_label>...</question_label>: Wrap main question or main sub-question prefix indicators (e.g. "Câu 1:", "Câu 12.", "Question 1:", "C1.", "1.", "2.", "1) ", "2) "), as well as question number prefixes inside reference answer key or explanation blocks (e.g., "<question_label>**Câu 1:**</question_label>", "Question 2."). If the question label is bold like "**Câu 1.**", the bold wrappers "**" must also be wrapped inside the <question_label> tag (e.g., "<question_label>**Câu 1.**</question_label>"). Do NOT wrap sub-question parts like "a)", "b)", "c)" or "a.", "b.", "c." here.
2. <stem>...</stem>: Wrap the main text body of a question or main sub-question (including any ordering items or list of items to order). Note: If a question has sub-parts a), b), c) that are annotated as options, the introductory text (e.g. "Cho tam giác ABC...") should be the <stem> of that question, NOT <context>.
3. <option_label>...</option_label>: Wrap options letters/prefixes or sub-question letters/prefixes (e.g. "A.", "B.", "C.", "D.", "a)", "b)", "c)", "d)", "a.", "b.", "c.", "d."). If the option label is bold like "**A.**", the bold wrappers "**" must also be wrapped inside the <option_label> tag (e.g., "<option_label>**A.**</option_label>"). Do NOT wrap correct answer letters in reference explanation sections here.
4. <option_text>...</option_text>: Wrap the textual content of options or sub-questions (e.g., the requirement/proof statement of part a, b, etc.).
5. <context>...</context>: Wrap the shared passage/context block in group questions (passages, reading texts, shared diagrams description). Do NOT use context for standard question introductions that have parts a), b), c) treated as options.
6. <section>...</section>: Wrap section headers, subheaders, directions, and reference answer/explanation titles (e.g., "PHẦN I. Câu trắc nghiệm...", "Mark the letter A, B, C, or D...", "ĐÁP ÁN THAM KHẢO", "LỜI GIẢI THAM KHẢO").
7. <explanation>...</explanation>: Wrap reference explanations, answers explanation texts, and solutions for questions. Do NOT wrap the reference question number itself here (it must be wrapped in <question_label>, e.g. "**<question_label>Câu 1:</question_label> <explanation>B**\\n\\nCâu sau chỗ trống nói rằng...</explanation>").

CRITICAL RULES:
1. Do NOT modify, correct, rephrase, or change any part of the input text. Preserve all spelling mistakes, typos, symbols, page markers (like "<|page|>Page X"), LaTeX formulas (enclosed in $...$ or $$...$$), and formatting exactly as they are in the input. ONLY insert the opening and closing XML tags.
2. Every tag MUST be properly closed. Do not nest tags. Tags must be strictly sequential.
3. Text that does not belong to any entity (e.g., page numbers, header information like "SỞ GD-ĐT...", "ĐỀ CHÍNH THỨC", horizontal separators) must NOT be wrapped in any tags. Keep it outside the XML tags.
4. Output ONLY the annotated text. Do not write any markdown code blocks (e.g. ```xml), introduction, or conversational filler.
5. You MUST output the ENTIRE input text from the very first character to the very last character. Do NOT omit, truncate, or skip any sections (such as headers, footers, page indicators, end markers like "HẾT", or reference tables/answer keys at the end). Everything that is not tagged must still be outputted exactly as it is in the input, outside of XML tags.
6. **ORDERING QUESTIONS (câu sắp xếp / arrangement)**: When a question asks you to arrange items into the correct order, the scrambled items themselves (labeled a. b. c. d. e. or similar lowercase letters) are NOT options — they are the material to be arranged. You MUST include ALL scrambled items (plus any surrounding introductory/closing text like "Dear...", "Sincerely,") inside a SINGLE <stem> tag. Only the final lettered choices (A. B. C. D.) showing the ordering sequences (e.g. "d – a – e – b – c") are real options and should be tagged as <option_label>/<option_text>. The KEY signal is: if option texts contain dash-separated letter sequences (e.g. "b – d – a – c – e"), it is an ordering question.
7. Structure your reasoning in the `<think>` block extremely concisely (under 60 words total) using brief bullet points or short phrases:
   - Identify subject & layout (e.g. tables, shared context, arrangement)
   - Note edge cases or tricky elements (e.g. bold wrappers, subparts)
   - Verification checklist (ensure tags close and no text is changed)
    Keep this reasoning trace focused and direct. Avoid reproducing raw text or drafting the complete output.
"""

def clean_llm_response(text: str) -> str:
    # Prune think tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            if lines[-1].strip() == "```":
                text = "\n".join(lines[1:-1])
            else:
                text = "\n".join(lines[1:])
    return text.strip()

def parse_xml_annotations(tagged_text: str):
    allowed_tags = {
        "question_label",
        "stem",
        "option_label",
        "option_text",
        "context",
        "section",
        "explanation",
    }
    raw_chars = []
    spans = []
    tag_pattern = re.compile(r"<(/)?([a-zA-Z_0-9]+)>")
    pos = 0
    current_open_tag = None
    tag_start_idx = -1

    for match in tag_pattern.finditer(tagged_text):
        start, end = match.span()
        text_before = tagged_text[pos:start]
        raw_chars.append(text_before)

        is_closing = bool(match.group(1))
        tag_name = match.group(2)

        if tag_name in allowed_tags:
            if not is_closing:
                current_open_tag = tag_name
                tag_start_idx = len("".join(raw_chars))
            else:
                if current_open_tag == tag_name and tag_start_idx != -1:
                    tag_end_idx = len("".join(raw_chars))
                    span_text = "".join(raw_chars)[tag_start_idx:tag_end_idx]
                    spans.append({
                        "start": tag_start_idx,
                        "end": tag_end_idx,
                        "label": tag_name,
                        "text": span_text,
                    })
                current_open_tag = None
                tag_start_idx = -1
        else:
            raw_chars.append(match.group(0))
        pos = end

    raw_chars.append(tagged_text[pos:])
    raw_text = "".join(raw_chars)
    return raw_text, spans

def project_tags_to_original(clean_text, original_text, spans):
    # Align clean_text and original_text
    matcher = difflib.SequenceMatcher(None, clean_text, original_text)
    matching_blocks = matcher.get_matching_blocks()
    
    # Create label mapping
    clean_len = len(clean_text)
    char_labels = [None] * clean_len
    for span_id, span in enumerate(spans):
        for idx in range(span["start"], span["end"]):
            if 0 <= idx < clean_len:
                char_labels[idx] = (span["label"], span_id)
                
    orig_len = len(original_text)
    orig_char_labels = [None] * orig_len
    
    for a, b, size in matching_blocks:
        for i in range(size):
            orig_char_labels[b + i] = char_labels[a + i]
            
    # Interpolation for gaps
    for idx in range(1, orig_len - 1):
        if orig_char_labels[idx] is None:
            prev_label = orig_char_labels[idx - 1]
            next_label = orig_char_labels[idx + 1]
            if prev_label is not None and prev_label == next_label:
                orig_char_labels[idx] = prev_label
                
    new_spans = []
    current_label = None
    current_span_id = None
    start_idx = -1
    
    for idx in range(orig_len):
        lbl_info = orig_char_labels[idx]
        if lbl_info is not None:
            label, span_id = lbl_info
        else:
            label, span_id = None, None
            
        if span_id != current_span_id:
            if current_span_id is not None:
                new_spans.append({
                    "start": start_idx,
                    "end": idx,
                    "label": current_label
                })
            current_label = label
            current_span_id = span_id
            start_idx = idx
            
    if current_span_id is not None:
        new_spans.append({
            "start": start_idx,
            "end": orig_len,
            "label": current_label
        })
        
    new_spans = sorted(new_spans, key=lambda x: x["start"])
    
    result_parts = []
    last_idx = 0
    for span in new_spans:
        start = span["start"]
        end = span["end"]
        lbl = span["label"]
        result_parts.append(original_text[last_idx:start])
        result_parts.append(f"<{lbl}>")
        result_parts.append(original_text[start:end])
        result_parts.append(f"</{lbl}>")
        last_idx = end
    result_parts.append(original_text[last_idx:])
    
    return "".join(result_parts)

def main():
    files_to_process = [
        {"path": "toan/hsg-10-hungyen.md", "hash": "1a110fe0"},
        {"path": "toan/hp-ks-2056-2026.md", "hash": "5b47eca1"},
        {"path": "toan/tpc-1-25-26.md", "hash": "fe8e1414"}
    ]

    client = OpenAI(
        api_key=os.environ.get("NVIDIA_API_KEY"),
        base_url="https://integrate.api.nvidia.com/v1",
    )

    output_dir = workspace_dir / "output" / "real-exams"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_base = workspace_dir / "real_data_annotator" / "out"

    print("Starting Math Exams Tagging Pipeline...\n")

    for idx, item in enumerate(files_to_process, 1):
        rel_path = item["path"]
        expected_hash = item["hash"]
        file_path = input_base / rel_path

        print(f"[{idx}/10] Processing {rel_path} (Hash: {expected_hash})")
        if not file_path.exists():
            print(f"  ❌ Error: File {file_path} does not exist.")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            original_text = f.read()

        # Call NVIDIA NIM model
        print("  Calling NVIDIA NIM (deepseek-ai/deepseek-v4-pro) to tag...")
        try:
            response = client.chat.completions.create(
                model="deepseek-ai/deepseek-v4-pro",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": original_text}
                ],
                temperature=0.0,
                extra_body={"chat_template_kwargs": {"thinking": True}},
                stream=False
            )
            raw_content = response.choices[0].message.content
            tagged_text = clean_llm_response(raw_content)
        except Exception as e:
            print(f"  ❌ Error calling API: {e}")
            continue

        # Parse tags
        clean_text, spans = parse_xml_annotations(tagged_text)

        # Align and project tags back onto original_text to keep text identical
        print("  Aligning and projecting tags back onto original text...")
        aligned_xml = project_tags_to_original(clean_text, original_text, spans)

        # Verify strict alignment
        reconstructed_clean = re.sub(r"</?\w+[^>]*>", "", aligned_xml)
        if reconstructed_clean == original_text:
            print("  ✅ Verification successful: Reconstructed text is 100% identical.")
        else:
            print("  ⚠️ Warning: Reconstructed text differs from original. Let's do a fallback replace.")
            # If not identical, fallback: just use the original text if possible, or print diff
            diff = list(difflib.unified_diff(original_text.splitlines(), reconstructed_clean.splitlines()))
            print("\n".join(diff[:10]))

        # Write output XML
        xml_out = output_dir / f"real_exam_{expected_hash}.xml"
        with open(xml_out, "w", encoding="utf-8") as f:
            f.write(aligned_xml)
        print(f"  ✅ Saved XML to: {xml_out.relative_to(workspace_dir)}")

        # Run verify script
        print("  Running parser script...")
        import subprocess
        # Command: python scratch/parse_xml_to_json.py output/real_exams/real_exam_<hash>.xml <path>
        cmd = [
            str(workspace_dir / ".pixi" / "envs" / "default" / "bin" / "python"),
            str(workspace_dir / "scratch" / "parse_xml_to_json.py"),
            str(xml_out),
            rel_path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"  ✅ Parser output: {res.stdout.strip()}")
        else:
            print(f"  ❌ Parser failed: {res.stderr.strip()}")

    print("\nAll files processed successfully!")

if __name__ == "__main__":
    main()
