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
workspace_dir = script_dir.parent.parent
sys.path.append(str(script_dir.parent))

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
1. <question_label>...</question_label>: Wrap main question or main sub-question prefix indicators (e.g. "Câu 1:", "Câu 12.", "Question 1:", "C1.", "1.", "2.", "1) ", "2) "), as well as question number prefixes inside reference answer key or explanation blocks (e.g., "<question_label>**Câu 1:**</question_label>", "Question 2.", "(3)"). If the question label is bold like "**Câu 1.**", the bold wrappers "**" must also be wrapped inside the <question_label> tag (e.g., "<question_label>**Câu 1.**</question_label>"). Do NOT wrap sub-question parts like "a)", "b)", "c)" or "a.", "b.", "c." here.
2. <stem>...</stem>: Wrap the main text body of a question or main sub-question (including any ordering items or list of items to order). Note: If a question has sub-parts a), b), c) that are annotated as options, the introductory text (e.g. "Cho tam giác ABC...") should be the <stem> of that question, NOT <context>.
3. <option_label>...</option_label>: Wrap options letters/prefixes or sub-question letters/prefixes (e.g. "A.", "B.", "C.", "D.", "a)", "b)", "c)", "d)", "a.", "b.", "c.", "d."). If the option label is bold like "**A.**", the bold wrappers "**" must also be wrapped inside the <option_label> tag (e.g., "<option_label>**A.**</option_label>"). Do NOT wrap correct answer letters in reference explanation sections here.
4. <option_text>...</option_text>: Wrap the textual content of options or sub-questions (e.g., the requirement/proof statement of part a, b, etc.).
5. <context>...</context>: Wrap the shared passage/context block in group questions (passages, reading texts, shared diagrams description). Do NOT use context for standard question introductions that have parts a), b), c) treated as options.
6. <section>...</section>: Wrap section headers, subheaders, directions, and reference answer/explanation titles (e.g., "PHẦN I. Câu trắc nghiệm...", "Mark the letter A, B, C, or D...", "ĐÁP ÁN THAM KHẢO", "LỜI GIẢI THAM KHẢO").
7. <explanation>...</explanation>: Wrap reference explanations, answers explanation texts, and solutions for questions. Do NOT wrap the reference question number itself here (it must be wrapped in <question_label>, e.g. "**<question_label>Câu 1:</question_label> <explanation>B**\n\nCâu sau chỗ trống nói rằng...</explanation>").

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

---
FEW-SHOT EXAMPLES:

### Example 1: Mathematics (Stems with LaTeX formulas and nested sub-questions)

#### Input:
Câu 1 (2,00 điểm). Cho phương trình $x^2 - 2(m-1)x + m^2 - 6 = 0$.
1. Giải phương trình khi $m = 3$.
2. Tìm $m$ để phương trình có hai nghiệm phân biệt $x_1, x_2$.

#### Output:
<think>
- Math exam with LaTeX. Main question Câu 1 has sub-questions 1. and 2.
- Tag Câu 1... as question_label, intro as stem. Sub-questions are tagged as option_label/option_text.
- Verify LaTeX content is unaltered and tags close sequentially.
</think><question_label>Câu 1 (2,00 điểm).</question_label> <stem>Cho phương trình $x^2 - 2(m-1)x + m^2 - 6 = 0$.</stem>
<option_label>1.</option_label> <option_text>Giải phương trình khi $m = 3$.</option_text>
<option_label>2.</option_label> <option_text>Tìm $m$ để phương trình có hai nghiệm phân biệt $x_1, x_2$.</option_text>

---

### Example 2: English (Pronunciation, passages, and inline options)

#### Input:
**Question 1.** Choose the word whose underlined part is pronounced differently:
    A. fini<u>sh</u>ed    B. expla<u>i</u>ned

*Read the following passage and mark the letter A, B, C, or D on your answer sheet to indicate the correct answer to each of the questions:*
In today's digital age, Vietnamese young people are increasingly depending on the virtual world. While social media platforms offer unprecedented opportunities for connection and learning, they also present significant challenges. Many teenagers spend hours scrolling through their feeds, which can lead to feelings of isolation and sleep deprivation. Experts suggest that parents should encourage outdoor activities to balance screen time.

**Question 2.** What is the main topic of the passage?
A. Technology trends.
B. The impact of the virtual world on young people.
C. How to use social media.
D. The benefits of outdoor activities.

**Question 3.** According to the passage, what is a negative effect of spending too much time on social media?
A. More connections.
B. Higher academic grades.
C. Sleep deprivation.
D. Better health.

**Question 4.** The word "they" in paragraph 1 refers to ________.
A. young people
B. opportunities
C. platforms
D. challenges

#### Output:
<think>
- English exam. Pronunciation Q1 with inline options, reading passage context, and dependent questions Q2-4.
- Tag Question 1 as question_label, choose word... as stem, A./B. as option_label, text as option_text.
- Tag passage instruction as section, passage text as context.
- Tag Questions 2-4 labels, stems, option_labels, and option_texts.
- Ensure bold formatting stars remain INSIDE question_label and option_label tags when labels are bolded.
</think><question_label>**Question 1.**</question_label> <stem>Choose the word whose underlined part is pronounced differently:</stem>
    <option_label>A.</option_label> <option_text>fini<u>sh</u>ed</option_text>    <option_label>B.</option_label> <option_text>expla<u>i</u>ned</option_text>

<section>*Read the following passage and mark the letter A, B, C, or D on your answer sheet to indicate the correct answer to each of the questions:*</section>
<context>In today's digital age, Vietnamese young people are increasingly depending on the virtual world. While social media platforms offer unprecedented opportunities for connection and learning, they also present significant challenges. Many teenagers spend hours scrolling through their feeds, which can lead to feelings of isolation and sleep deprivation. Experts suggest that parents should encourage outdoor activities to balance screen time.</context>

<question_label>**Question 2.**</question_label> <stem>What is the main topic of the passage?</stem>
<option_label>A.</option_label> <option_text>Technology trends.</option_text>
<option_label>B.</option_label> <option_text>The impact of the virtual world on young people.</option_text>
<option_label>C.</option_label> <option_text>How to use social media.</option_text>
<option_label>D.</option_label> <option_text>The benefits of outdoor activities.</option_text>

<question_label>**Question 3.**</question_label> <stem>According to the passage, what is a negative effect of spending too much time on social media?</stem>
<option_label>A.</option_label> <option_text>More connections.</option_text>
<option_label>B.</option_label> <option_text>Higher academic grades.</option_text>
<option_label>C.</option_label> <option_text>Sleep deprivation.</option_text>
<option_label>D.</option_label> <option_text>Better health.</option_text>

<question_label>**Question 4.**</question_label> <stem>The word "they" in paragraph 1 refers to ________.</stem>
<option_label>A.</option_label> <option_text>young people</option_text>
<option_label>B.</option_label> <option_text>opportunities</option_text>
<option_label>C.</option_label> <option_text>platforms</option_text>
<option_label>D.</option_label> <option_text>challenges</option_text>
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
- Markdown tables containing math/English questions.
- Table 1 (Row 1): Inline options. Row 2: Bold option letters. Tag question_label, option_label, option_text.
- Table 2: Header rows untagged. Cell row headers as question_label, option cells as option_text.
- Question 3: Standard question stem with 2x2 options grid.
- Preserve table layout syntax (| and dashes) and markdown stars exactly.
</think>| <question_label>Question 1:</question_label> <option_label>A.</option_label> <option_text><u>fa</u>ce</option_text> | <option_label>B.</option_label> <option_text><u>pa</u>ge</option_text> | <option_label>C.</option_label> <option_text><u>ba</u>ke</option_text> | <option_label>D.</option_label> <option_text><u>mar</u>k</option_text> |
|---|---|---|---|
| <question_label>**Question 2:**</question_label> <option_label>**A.**</option_label> <option_text>cartoon</option_text> | <option_label>**B.**</option_label> <option_text>practice</option_text> | <option_label>**C.**</option_label> <option_text>picture</option_text> | <option_label>**D.**</option_label> <option_text>maintain</option_text> |

| <question_label>**Question 5.**</question_label> | <option_text>a</option_text> | <option_text>the</option_text> | <option_text>an</option_text> | <option_text>Ø (no article)</option_text> |
| <question_label>**Question 6.**</question_label> | <option_text>whom</option_text> | <option_text>who</option_text> | <option_text>whose</option_text> | <option_text>which</option_text> |

<question_label>Câu 3.</question_label> <stem>Kim loại nào dưới đây hoạt động hóa học mạnh nhất?</stem>
| | |
|---|---|
| <option_label>A.</option_label> <option_text>Al</option_text> | <option_label>B.</option_label> <option_text>Fe</option_text> |
| <option_label>C.</option_label> <option_text>Cu</option_text> | <option_label>D.</option_label> <option_text>Zn</option_text> |

--- HẾT ---

---

### Example 4: Ordering / Arrangement Question (câu sắp xếp)

#### Input:
**5.**
Dear Minh,
a. The contest is about whether technology helps or harms human connection.
b. I've seen people at lunch staring at those tiny screens without saying a word.
c. What's your opinion on this? I'm excited to hear from you.
d. I noticed a poster near the school gate about a debate competition.
e. It's an interesting topic, considering how often we text but rarely talk.
Sincerely,
A. d – a – e – b – c     B. c – d – b – a – e     C. e – d – c – a – b     D. d – e – c – b – a

#### Output:
<think>
- English arrangement question with a letter template.
- Scrambled sentences a-e and letter salutations belong entirely to <stem>.
- Choice sequences A-D are real answer options -> option_label and option_text.
- Wrap question number 5. in question_label.
</think><question_label>**5.**</question_label>
<stem>Dear Minh,
a. The contest is about whether technology helps or harms human connection.
b. I've seen people at lunch staring at those tiny screens without saying a word.
c. What's your opinion on this? I'm excited to hear from you.
d. I noticed a poster near the school gate about a debate competition.
e. It's an interesting topic, considering how often we text but rarely talk.
Sincerely,</stem>
<option_label>A.</option_label> <option_text>d – a – e – b – c</option_text>     <option_label>B.</option_label> <option_text>c – d – b – a – e</option_text>     <option_label>C.</option_label> <option_text>e – d – c – a – b</option_text>     <option_label>D.</option_label> <option_text>d – e – c – b – a</option_text>

---

### Example 5: Answer Keys and Reference Explanations at the end

#### Input:
ĐÁP ÁN THAM KHẢO

| Câu | 1 | 2 |
|-----|---|---|
| Đáp án | A | B |

LỜI GIẢI THAM KHẢO

**Câu 1: A**
Đây là giải thích cho câu 1.

#### Output:
<think>
- Reference answers and explanation sections.
- Tag titles ("ĐÁP ÁN...", "LỜI GIẢI...") as section. Reference tables untagged.
- Tag "Câu 1:" as question_label and "A" plus explanation body as explanation.
</think><section>ĐÁP ÁN THAM KHẢO</section>

| Câu | 1 | 2 |
|-----|---|---|
| Đáp án | A | B |

<section>LỜI GIẢI THAM KHẢO</section>

<question_label>**Câu 1:**</question_label> <explanation>A**
Đây là giải thích cho câu 1...</explanation>
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

                    # If this is option_label or question_label, and the text before started with '**' and we're followed by '**',
                    # pull the asterisks into the label span.
                    if tag_name in ["option_label", "question_label"]:
                        # Check if the text before ends with '**' and current raw_chars has enough chars
                        full_current_text = "".join(raw_chars)
                        if full_current_text[tag_start_idx - 2:tag_start_idx] == "**" and tagged_text[end:end+2] == "**":
                            # Shift start back by 2, and we need to append the trailing '**' to the span content
                            tag_start_idx -= 2
                            # Also append '**' to raw_chars so it is counted as part of the tag content
                            raw_chars.append("**")
                            tag_end_idx = len("".join(raw_chars))
                            span_text = "".join(raw_chars)[tag_start_idx:tag_end_idx]
                            # Update pos so we skip these 2 trailing characters in the next loops
                            pos = end + 2
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
                            continue

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
    model_name: str | None,
    deepseek_key: str | None,
    llm_key: str | None,
    provider: str | None = None,
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
                print(
                    "Error: Neither DEEPSEEK_API_KEY nor LLM_API_KEY is configured in the environment."
                )
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
            print(
                "Error: Provider requires NVIDIA_API_KEY or DEEPSEEK_API_KEY but neither is set."
            )
            sys.exit(1)
        target_model = model_name or "deepseek-ai/deepseek-v4-pro"
        if target_model in ["deepseek-v4-pro", "deepseek-ai/deepseek-v4-pro"]:
            target_model = "deepseek-ai/deepseek-v4-pro"
        print(f"Routing to NVIDIA NIM for annotation with model: {target_model}")
        return OpenAI(
            api_key=nvidia_key, base_url="https://integrate.api.nvidia.com/v1"
        ), target_model

    elif use_vilao:
        if not llm_key:
            if deepseek_key:
                print(
                    "Warning: Model might require Vilao.ai (LLM_API_KEY), but only DEEPSEEK_API_KEY is available. Using DeepSeek API."
                )
                return OpenAI(
                    api_key=deepseek_key, base_url="https://api.deepseek.com"
                ), model_name or "deepseek-chat"
            print(
                "Error: Model requires LLM_API_KEY (Vilao.ai) but it is not set in the environment."
            )
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
        return OpenAI(
            api_key=deepseek_key, base_url="https://api.deepseek.com"
        ), target_model


def load_few_shot_messages(example_dir: Path) -> List[Dict[str, str]]:
    """
    Loads few-shot examples from the specified directory as a list of chat messages.
    Expected files are in_1.md/out_1.md, in_2.md/out_2.md, etc.
    """
    messages = []
    if not example_dir.exists():
        return messages

    i = 1
    while True:
        in_file = example_dir / f"in_{i}.md"
        out_file = example_dir / f"out_{i}.md"
        if not (in_file.exists() and out_file.exists()):
            break
        try:
            with open(in_file, "r", encoding="utf-8") as f_in:
                in_content = f_in.read()
            with open(out_file, "r", encoding="utf-8") as f_out:
                out_content = f_out.read()
            messages.append({"role": "user", "content": in_content})
            messages.append({"role": "assistant", "content": out_content})
        except Exception as e:
            print(f"  Warning: Failed to load few-shot pair {i}: {e}")
        i += 1

    if messages:
        print(
            f"  Loaded {len(messages) // 2} few-shot example pair(s) from '{example_dir.name}'."
        )
    return messages


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
        default=str(workspace_dir / "output" / "real-exams"),
        help="Directory to save the annotated JSON files (default: output/real-exams)",
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

    client, model_name = get_client_and_model(
        model_name, deepseek_key, llm_key, provider=args.provider
    )

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

    # Load few-shot examples dynamically from the example directory
    few_shot_messages = load_few_shot_messages(script_dir / "example")

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

            is_nvidia_model = (
                "nvidia" in str(client.base_url)
                or model_name == "deepseek-ai/deepseek-v4-pro"
            )

            max_retries = 2
            success = False
            tagged_text = ""
            raw_text = ""
            spans = []

            for attempt in range(max_retries):
                temp = 0.0 if attempt == 0 else 0.2
                if attempt > 0:
                    print(
                        f"  Attempt {attempt + 1}: Retrying with temperature={temp} to break repetition loops..."
                    )

                try:
                    if is_nvidia_model:
                        print(
                            f"  Calling NVIDIA LLM (non-stream) to annotate text ({len(raw_ocr_text)} chars)..."
                        )
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                            ]
                            + few_shot_messages
                            + [
                                {"role": "user", "content": raw_ocr_text},
                            ],
                            temperature=temp,
                            extra_body={"chat_template_kwargs": {"thinking": True}},
                            stream=False,
                        )
                        log_response(response, model=model_name)

                        raw_result = response.choices[0].message.content
                        tagged_text = clean_llm_response(raw_result)

                        usage = getattr(response, "usage", None)
                        if usage:
                            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                            completion_tokens = (
                                getattr(usage, "completion_tokens", 0) or 0
                            )
                            print(
                                f"  Done. Prompt: {prompt_tokens} t, Completion: {completion_tokens} t"
                            )
                    else:
                        print(
                            f"  Calling LLM to annotate text ({len(raw_ocr_text)} chars)..."
                        )
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                            ]
                            + few_shot_messages
                            + [
                                {"role": "user", "content": raw_ocr_text},
                            ],
                            temperature=temp,
                            stream=True,
                            stream_options={"include_usage": True},
                        )

                        pbar = tqdm(desc="  LLM thinking", unit=" tokens", leave=False)
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
                            reasoning_content = getattr(
                                delta, "reasoning_content", None
                            )
                            content = getattr(delta, "content", None)

                            if reasoning_content:
                                if not reasoning_mode:
                                    reasoning_mode = True
                                    pbar.set_description("  LLM thinking")
                                accumulated_reasoning.append(reasoning_content)
                                pbar.update(1)

                            if content:
                                if reasoning_mode:
                                    reasoning_mode = False
                                    pbar.set_description("  LLM generating")
                                accumulated_content.append(content)
                                pbar.update(1)

                        pbar.close()

                        raw_result = "".join(accumulated_content)

                        tagged_text = clean_llm_response(raw_result)

                        if usage_data:

                            class MockResponse:
                                def __init__(self, usage, model):
                                    self.usage = usage
                                    self.model = model

                            log_response(
                                MockResponse(usage_data, model_name), model=model_name
                            )

                            prompt_tokens = getattr(usage_data, "prompt_tokens", 0) or 0
                            completion_tokens = (
                                getattr(usage_data, "completion_tokens", 0) or 0
                            )
                            details = getattr(
                                usage_data, "completion_tokens_details", None
                            )
                            reasoning_tokens = (
                                getattr(details, "reasoning_tokens", 0) or 0
                            )
                            output_tokens = completion_tokens - reasoning_tokens
                            print(
                                f"  Done. Prompt: {prompt_tokens} t, Reasoning: {reasoning_tokens} t, Output: {output_tokens} t"
                            )

                    # Save raw XML file for debugging (successful or ratio mismatch case)
                    xml_out_file_path = output_path / f"real_exam_{path_hash}.xml"
                    with open(xml_out_file_path, "w", encoding="utf-8") as f:
                        f.write(tagged_text)
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
                print(
                    "  Error: Failed to obtain valid annotations after retries. Skipping file."
                )
                continue

            # Build and save result
            result_data = {
                "exam_id": f"real_{path_hash}",
                "created_at": datetime.now().isoformat(),
                "is_real": True,
                "raw_text": raw_text,
                "spans": spans,
                "raw_xml": tagged_text,
                "annotated": True
            }

            with open(out_file_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)

            print(
                f"  Saved JSON annotation to: {out_file_path.name} ({len(spans)} spans)"
            )
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
