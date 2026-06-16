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
from tqdm import tqdm
from src.token_tracker import log_response

# Reconfigure stdout for UTF-8 to handle Vietnamese terminal logging
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(dotenv_path=workspace_dir / ".env")

SYSTEM_PROMPT = """You are an expert NLP data annotator for Vietnamese educational exam papers.
Your task is to annotate the provided raw OCR text of an exam paper by wrapping specific components in XML tags.

You MUST wrap the following entities:
1. <question_label>...</question_label>: Wrap main question or main sub-question prefix indicators (e.g. "Câu 1:", "Câu 12.", "Question 1:", "C1.", "1.", "2.", "1) ", "2) "). Do NOT wrap sub-question parts like "a)", "b)", "c)" or "a.", "b.", "c." here.
2. <stem>...</stem>: Wrap the main text body of a question or main sub-question (including any ordering items or list of items to order). Note: If a question has sub-parts a), b), c) that are annotated as options, the introductory text (e.g. "Cho tam giác ABC...") should be the <stem> of that question, NOT <context>.
3. <option_label>...</option_label>: Wrap options letters/prefixes or sub-question letters/prefixes (e.g. "A.", "B.", "C.", "D.", "a)", "b)", "c)", "d)", "a.", "b.", "c.", "d.").
4. <option_text>...</option_text>: Wrap the textual content of options or sub-questions (e.g., the requirement/proof statement of part a, b, etc.).
5. <context>...</context>: Wrap the shared passage/context block in group questions (passages, reading texts, shared diagrams description). Do NOT use context for standard question introductions that have parts a), b), c) treated as options.
6. <instruction>...</instruction>: Wrap section headers, subheaders, and directions (e.g., "PHẦN I. Câu trắc nghiệm...", "Mark the letter A, B, C, or D...", "Phần II: Đúng sai", "Đọc đoạn văn sau và trả lời...").

CRITICAL RULES:
1. Do NOT modify, correct, rephrase, or change any part of the input text. Preserve all spelling mistakes, typos, symbols, page markers (like "<|page|>Page X"), LaTeX formulas (enclosed in $...$ or $$...$$), and formatting exactly as they are in the input. ONLY insert the opening and closing XML tags.
2. Every tag MUST be properly closed. Do not nest tags. Tags must be strictly sequential.
3. Text that does not belong to any entity (e.g., page numbers, header information like "SỞ GD-ĐT...", "ĐỀ CHÍNH THỨC", horizontal separators) must NOT be wrapped in any tags. Keep it outside the XML tags.
4. Output ONLY the annotated text. Do not write any markdown code blocks (e.g. ```xml), introduction, or conversational filler.
5. You MUST output the ENTIRE input text from the very first character to the very last character. Do NOT omit, truncate, or skip any sections (such as headers, footers, page indicators, end markers like "HẾT", or reference tables/answer keys at the end). Everything that is not tagged must still be outputted exactly as it is in the input, outside of XML tags.
6. Structure your reasoning in the `<think>` block systematically using this framework:
   - **Document Structure**: Briefly state the overall exam sections, formatting style (e.g. standard layout, inline options, tables, list items), and presence of shared contexts/passages.
   - **Edge Cases & Ambiguities**: Identify tricky layouts (e.g., table cells, arrangement questions, sub-questions inside stems, non-annotated elements like page indicators) and explicitly plan how to handle them.
   - **Sequential Verification plan**: Outline the tagging sequence to ensure no tags nest, all open tags close, and every single character of the original text is preserved exactly.
    Keep this reasoning trace focused and direct. Avoid reproducing the entire raw text or drafting the complete output inside the thoughts.

---
FEW-SHOT EXAMPLES:

### Example 1: Mathematics (Stems with LaTeX formulas and nested sub-questions)

#### Input:
Câu 1 (2,00 điểm). Cho phương trình $x^2 - 2(m-1)x + m^2 - 6 = 0$.
1. Giải phương trình khi $m = 3$.
2. Tìm $m$ để phương trình có hai nghiệm phân biệt $x_1, x_2$.

#### Output:
<think>
**Document Structure Analysis:**
- Subject: Math exam with LaTeX formulas ($...$).
- Layout: Contains a main question "Câu 1" with two sub-questions "1." and "2.".
- Preserving all LaTeX formulas exactly as is.

**Layout & Tagging Strategy by Section:**
- "Câu 1 (2,00 điểm)." is the main question prefix -> Tag as <question_label>.
- The introductory statement "Cho phương trình..." -> Tag as <stem>.
- Sub-question prefixes "1." and "2." -> Tag as <question_label>.
- Sub-question content -> Tag as <stem>.

**Sequential Verification:**
- Ensure all tags are correctly nested, all open tags close, and text inside LaTeX remains unaltered.
</think><question_label>Câu 1 (2,00 điểm).</question_label> <stem>Cho phương trình $x^2 - 2(m-1)x + m^2 - 6 = 0$.</stem>
<question_label>1.</question_label> <stem>Giải phương trình khi $m = 3$.</stem>
<question_label>2.</question_label> <stem>Tìm $m$ để phương trình có hai nghiệm phân biệt $x_1, x_2$.</stem>

---

### Example 2: English (Pronunciation, passages, and inline options)

#### Input:
**Question 1.** Choose the word whose underlined part is pronounced differently:
    A. fini<u>sh</u>ed    B. expla<u>i</u>ned

*Read the passage and choose the correct answer:*
In today's digital age, Vietnamese young people are depending on the virtual world.
**Question 2.** What is the main topic of the passage?
    A. Tech trends.    B. Virtual reality.

#### Output:
<think>
**Document Structure Analysis:**
- Subject: English exam with inline options and a shared reading passage.
- Layout: Question 1 (inline options A, B with HTML underline markup), followed by a reading passage instruction, context, and Question 2.

**Layout & Tagging Strategy by Section:**
- "**Question 1.**" -> Tag as <question_label>.
- "Choose the word whose underlined part is pronounced differently:" -> Tag as <stem>.
- Inline prefixes "A." and "B." -> Tag as <option_label>.
- Inline option text "finished" / "explained" containing <u> markup -> Tag as <option_text>, keeping <u> tags exactly intact.
- Passage text "In today's digital age..." -> Tag as <context> since it is a reading block.
- "**Question 2.**" -> Tag as <question_label>.
- "What is the main topic..." -> Tag as <stem>.
- Option prefixes and texts -> Tag as <option_label> and <option_text>.

**Sequential Verification:**
- Check tag matching and nesting boundaries. Ensure no text is altered.
</think><question_label>**Question 1.**</question_label> <stem>Choose the word whose underlined part is pronounced differently:</stem>
    <option_label>A.</option_label> <option_text>fini<u>sh</u>ed</option_text>    <option_label>B.</option_label> <option_text>expla<u>i</u>ned</option_text>

<instruction>*Read the passage and choose the correct answer:*</instruction>
<context>In today's digital age, Vietnamese young people are depending on the virtual world.</context>
<question_label>**Question 2.**</question_label> <stem>What is the main topic of the passage?</stem>
    <option_label>A.</option_label> <option_text>Tech trends.</option_text>    <option_label>B.</option_label> <option_text>Virtual reality.</option_text>

---

### Example 3: Tabular Questions (OCR-style tables representing multiple-choice questions)

#### Input:
| Question 1: A. <u>fa</u>ce | B. <u>pa</u>ge | C. <u>ba</u>ke | D. <u>mar</u>k |
|---|---|---|---|
| **Question 2: A.** cartoon | **B.** practice | **C.** picture | **D.** maintain |

| Câu hỏi | A | B | C | D |
|----------|---|---|---|---|
| **Question 5.** | a | the | an | Ø (no article) |
| **Question 6.** | whom | who | whose | which |

Câu 3. Kim loại nào dưới đây hoạt động hóa học mạnh nhất?
| | |
|---|---|
| A. Al | B. Fe |
| C. Cu | D. Zn |

--- HẾT ---

#### Output:
<think>
**Document Structure Analysis:**
- Subject: Educational questions formatted in markdown tables.
- Layout:
  - Table 1 (Questions 1 and 2): A horizontal option layout inside a 4-column grid. Column 1 includes the question prefix and option A. Subsequent columns contain options B, C, D. Column 1 of row 2 includes the question prefix for Question 2 and option A inside bold markdown.
  - Table 2 (Questions 5 and 6): Standard table with column headers for options A, B, C, D. Row cells contain the raw option texts without individual option letters.
  - Question 3: Standard question stem with options arranged in a 2x2 table grid.

**Layout & Tagging Strategy by Section:**
- Table 1 (Row 1):
  - Cell 1: "Question 1: A. <u>fa</u>ce" -> Tag prefix as <question_label>Question 1:</question_label>, letter as <option_label>A.</option_label>, and word as <option_text><u>fa</u>ce</option_text> (keeping underline tags).
  - Cells 2-4: Tag letters as <option_label> and words as <option_text>.
- Table 1 (Row 2):
  - Cell 1: "**Question 2: A.** cartoon" -> Tag inside the bold tags: "**<question_label>Question 2:</question_label> <option_label>A.</option_label>** <option_text>cartoon</option_text>".
  - Cells 2-4: Tag letters inside bold tags as <option_label> and option text as <option_text>.
- Table 2:
  - Keep headers "| Câu hỏi | A | B | C | D |" and separators untagged.
  - In each row: row label (e.g. "**Question 5.**") -> Tag as <question_label>. Cell options -> Tag entire cell text as <option_text> since there are no option letters inside cells.
- Question 3:
  - "Câu 3." -> Tag as <question_label>.
  - Stem text -> Tag as <stem>.
  - Grid table: Tag option prefixes as <option_label> and option texts as <option_text> in each cell.

**Sequential Verification:**
- Ensure all markdown table characters (`|`, dashes) and styling tags (`**`, `<u>`) are preserved exactly, and no tags nest.
- Trailing non-question text like "--- HẾT ---" is preserved verbatim at the end outside of tags.
</think>| <question_label>Question 1:</question_label> <option_label>A.</option_label> <option_text><u>fa</u>ce</option_text> | <option_label>B.</option_label> <option_text><u>pa</u>ge</option_text> | <option_label>C.</option_label> <option_text><u>ba</u>ke</option_text> | <option_label>D.</option_label> <option_text><u>mar</u>k</option_text> |
|---|---|---|---|
| **<question_label>Question 2:</question_label> <option_label>A.</option_label>** <option_text>cartoon</option_text> | **<option_label>B.</option_label>** <option_text>practice</option_text> | **<option_label>C.</option_label>** <option_text>picture</option_text> | **<option_label>D.</option_label>** <option_text>maintain</option_text> |

| Câu hỏi | A | B | C | D |
|----------|---|---|---|---|
| <question_label>**Question 5.**</question_label> | <option_text>a</option_text> | <option_text>the</option_text> | <option_text>an</option_text> | <option_text>Ø (no article)</option_text> |
| <question_label>**Question 6.**</question_label> | <option_text>whom</option_text> | <option_text>who</option_text> | <option_text>whose</option_text> | <option_text>which</option_text> |

<question_label>Câu 3.</question_label> <stem>Kim loại nào dưới đây hoạt động hóa học mạnh nhất?</stem>
| | |
|---|---|
| <option_label>A.</option_label> <option_text>Al</option_text> | <option_label>B.</option_label> <option_text>Fe</option_text> |
| <option_label>C.</option_label> <option_text>Cu</option_text> | <option_label>D.</option_label> <option_text>Zn</option_text> |

--- HẾT ---
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
    allowed_tags = {"question_label", "stem", "option_label", "option_text", "context", "instruction"}
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
                    spans.append(
                        {
                            "start": tag_start_idx,
                            "end": tag_end_idx,
                            "label": tag_name,
                            "text": span_text,
                        }
                    )
                current_open_tag = None
                tag_start_idx = -1
        else:
            # If it's an unallowed tag, treat it as literal text
            raw_chars.append(match.group(0))

        pos = end

    raw_chars.append(tagged_text[pos:])
    raw_text = "".join(raw_chars)

    return raw_text, spans


def get_client_and_model(
    model_name: str | None, deepseek_key: str | None, llm_key: str | None, provider: str | None = None
) -> Tuple[OpenAI, str]:
    """
    Determine the appropriate OpenAI client and model name to use.
    Supports routing based on explicit provider argument, falling back to auto-detection.
    """
    nvidia_key = os.environ.get("NVIDIA_API_KEY") or deepseek_key

    is_nvidia = False
    use_vilao = False
    is_deepseek = False

    if provider == "nvidia":
        is_nvidia = True
    elif provider == "vilao":
        use_vilao = True
    elif provider == "deepseek":
        is_deepseek = True
    else:
        # Auto-detect routing
        if model_name in ["deepseek-v4-pro", "deepseek-ai/deepseek-v4-pro"]:
            is_nvidia = True
        elif not model_name:
            if deepseek_key:
                is_deepseek = True
            elif llm_key:
                use_vilao = True
            else:
                print("Error: Neither DEEPSEEK_API_KEY nor LLM_API_KEY is configured in the environment.")
                sys.exit(1)
        else:
            if "/" in model_name:
                use_vilao = True
            elif "minimax" in model_name.lower() or model_name.startswith("mn/"):
                use_vilao = True
            elif not deepseek_key and llm_key:
                use_vilao = True
            else:
                is_deepseek = True

    if is_nvidia:
        if not nvidia_key:
            print("Error: Provider requires NVIDIA_API_KEY or DEEPSEEK_API_KEY but neither is set.")
            sys.exit(1)
        target_model = model_name or "deepseek-ai/deepseek-v4-pro"
        if target_model in ["deepseek-v4-pro", "deepseek-ai/deepseek-v4-pro"]:
            target_model = "deepseek-ai/deepseek-v4-pro"
        print(f"Routing to NVIDIA NIM for annotation with model: {target_model}")
        return OpenAI(api_key=nvidia_key, base_url="https://integrate.api.nvidia.com/v1"), target_model

    elif use_vilao:
        if not llm_key:
            if deepseek_key:
                print("Warning: Model might require Vilao.ai (LLM_API_KEY), but only DEEPSEEK_API_KEY is available. Using DeepSeek API.")
                return OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com"), model_name or "deepseek-chat"
            print("Error: Model requires LLM_API_KEY (Vilao.ai) but it is not set in the environment.")
            sys.exit(1)
        
        # Prepare the final model name with proper provider prefix
        final_model = model_name or "mn/Minimax-M3"
        if "/" not in final_model:
            if "minimax" in final_model.lower():
                final_model = f"mn/{final_model}"
            elif "deepseek" in final_model.lower():
                final_model = f"deepseek/{final_model}"
                
        print(f"Routing to Vilao.ai API for annotation with model: {final_model}")
        return OpenAI(api_key=llm_key, base_url="https://api.vilao.ai/v1"), final_model

    else:
        if not deepseek_key:
            print("Error: DEEPSEEK_API_KEY is not set.")
            sys.exit(1)
        target_model = model_name or "deepseek-chat"
        print(f"Routing to DeepSeek API for annotation with model: {target_model}")
        return OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com"), target_model


def main():
    parser = argparse.ArgumentParser(
        description="Annotate raw OCR Vietnamese exams using LLMs"
    )
    parser.add_argument(
        "--input",
        "-i",
        default=str(script_dir / "out"),
        help="Directory containing output markdown files from OCR (default: real_data_annotator/out)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=str(workspace_dir / "output" / "real_exams"),
        help="Directory to save the annotated JSON files (default: output/real_exams)",
    )

    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Limit the number of files to process",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-annotation even if the output file already exists",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model identifier (default: auto-detected based on environment keys)",
    )
    parser.add_argument(
        "--provider",
        choices=["deepseek", "nvidia", "vilao"],
        default=None,
        help="API provider to route requests to",
    )

    args = parser.parse_args()

    # 1. Setup Client based on available keys and requested model/provider
    model_name = args.model
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    llm_key = os.environ.get("LLM_API_KEY")

    client, model_name = get_client_and_model(model_name, deepseek_key, llm_key, provider=args.provider)

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
        md_files = md_files[: args.limit]

    print(f"Found {len(md_files)} file(s) to process.\n")

    success_count = 0
    skipped_count = 0

    for idx, file_path in enumerate(md_files):
        print(
            f"[{idx + 1}/{len(md_files)}] Processing: {file_path.relative_to(input_path)}"
        )

        # Determine subject & grade via auto-detection

        # Create output filename based on a hash of the relative path to avoid collisions
        rel_sig = str(file_path.relative_to(input_path))
        path_hash = hashlib.md5(rel_sig.encode("utf-8")).hexdigest()[:8]
        out_filename = f"real_exam_{path_hash}.json"
        out_file_path = output_path / out_filename

        if out_file_path.exists() and not args.force:
            print(
                f"  Skipping: Annotated output already exists at '{out_file_path.name}'."
            )
            skipped_count += 1
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_ocr_text = f.read()

            if not raw_ocr_text.strip():
                print("  Warning: File is empty. Skipping.")
                continue

            is_nvidia_model = "nvidia" in str(client.base_url) or model_name == "deepseek-ai/deepseek-v4-pro"
            
            max_retries = 2
            success = False
            tagged_text = ""
            raw_text = ""
            spans = []
            
            for attempt in range(max_retries):
                temp = 0.0 if attempt == 0 else 0.2
                if attempt > 0:
                    print(f"  Attempt {attempt + 1}: Retrying with temperature={temp} to break repetition loops...")
                
                try:
                    if is_nvidia_model:
                        print(f"  Calling NVIDIA LLM (non-stream) to annotate text ({len(raw_ocr_text)} chars)...")
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": raw_ocr_text},
                            ],
                            temperature=temp,
                            extra_body={"chat_template_kwargs": {"thinking": True}},
                            stream=False
                        )
                        log_response(response, model=model_name)
                        
                        raw_result = response.choices[0].message.content
                        tagged_text = clean_llm_response(raw_result)
                        
                        usage = getattr(response, "usage", None)
                        if usage:
                            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                            print(f"  Done. Prompt: {prompt_tokens} t, Completion: {completion_tokens} t")
                    else:
                        print(f"  Calling LLM to annotate text ({len(raw_ocr_text)} chars)...")
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": raw_ocr_text},
                            ],
                            temperature=temp,
                            stream=True,
                            stream_options={"include_usage": True},
                        )

                        pbar = tqdm(desc="  LLM thinking", unit=" chars", leave=False)
                        accumulated_content = []
                        accumulated_reasoning = []
                        usage_data = None
                        reasoning_mode = True

                        for chunk in response:
                            if hasattr(chunk, "usage") and chunk.usage is not None:
                                usage_data = chunk.usage

                            if not chunk.choices:
                                continue

                            delta = chunk.choices[0].delta

                            # Check reasoning content
                            reasoning_content = getattr(delta, "reasoning_content", None)
                            content = getattr(delta, "content", None)

                            if reasoning_content:
                                if not reasoning_mode:
                                    reasoning_mode = True
                                    pbar.set_description("  LLM thinking")
                                accumulated_reasoning.append(reasoning_content)
                                pbar.update(len(reasoning_content))

                            if content:
                                if reasoning_mode:
                                    reasoning_mode = False
                                    pbar.set_description("  LLM generating")
                                accumulated_content.append(content)
                                pbar.update(len(content))

                        pbar.close()

                        raw_result = "".join(accumulated_content)

                        tagged_text = clean_llm_response(raw_result)

                        if usage_data:
                            class MockResponse:
                                def __init__(self, usage, model):
                                    self.usage = usage
                                    self.model = model
                            log_response(MockResponse(usage_data, model_name), model=model_name)
                            
                            prompt_tokens = getattr(usage_data, "prompt_tokens", 0) or 0
                            completion_tokens = getattr(usage_data, "completion_tokens", 0) or 0
                            details = getattr(usage_data, "completion_tokens_details", None)
                            reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
                            output_tokens = completion_tokens - reasoning_tokens
                            print(f"  Done. Prompt: {prompt_tokens} t, Reasoning: {reasoning_tokens} t, Output: {output_tokens} t")

                    # Save raw XML file for debugging (successful or ratio mismatch case)
                    xml_out_file_path = output_path / f"real_exam_{path_hash}.xml"
                    with open(xml_out_file_path, "w", encoding="utf-8") as f:
                        f.write(raw_result)
                    print(f"  Saved raw XML for debugging to: {xml_out_file_path.name}")

                    # Parse XML
                    raw_text, spans = parse_xml_annotations(tagged_text)

                    # Simple length validation (should be similar to original text)
                    original_len = len(raw_ocr_text)
                    stripped_len = len(raw_text)
                    ratio = stripped_len / max(1, original_len)

                    if ratio < 0.85 or ratio > 1.15:
                        print(
                            f"  Warning: Text mismatch detected (original: {original_len}, stripped: {stripped_len}, ratio: {ratio:.2f})."
                        )
                        continue  # Trigger retry loop
                    
                    success = True
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    print(f"  Error on attempt {attempt + 1}: {e}")
                    continue
            
            if not success:
                print("  Error: Failed to obtain valid annotations after retries. Skipping file.")
                continue

            # Build and save result
            result_data = {
                "exam_id": f"real_{path_hash}",
                "created_at": datetime.now().isoformat(),
                "is_real": True,
                "raw_text": raw_text,
                "spans": spans,
                "raw_xml": tagged_text,
            }

            with open(out_file_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)

            print(f"  Saved JSON annotation to: {out_file_path.name} ({len(spans)} spans)")
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
