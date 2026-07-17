import argparse
import base64
import os
import random
import re
import sys
from pathlib import Path
from dotenv import load_dotenv
import fitz  # PyMuPDF
from openai import OpenAI

# Reconfigure stdout to use UTF-8 encoding (fixes Windows terminal encoding errors with emojis/unicode)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Load environment variables from .env file
workspace_dir = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=workspace_dir / ".env")

# Access the API key from environment variables
api_key = os.environ.get("LLM_API_KEY")

if not api_key:
    print(
        "Error: LLM_API_KEY environment variable is not set. Please set it or add it to your .env file."
    )
    sys.exit(1)

# Argument parsing
parser = argparse.ArgumentParser(
    description="Recursively perform OCR on PDF files using Minimax-M3."
)
parser.add_argument(
    "--input",
    "-i",
    default=str(workspace_dir / "output" / "ocr_input"),
    help="Directory containing input PDF files (default: output/ocr_input)",
)
parser.add_argument(
    "--output",
    "-o",
    default=str(workspace_dir / "output" / "ocr_out"),
    help="Directory to save output markdown files (default: output/ocr_out)",
)
parser.add_argument(
    "--limit",
    "-l",
    type=int,
    default=None,
    help="Limit the number of PDF files to process",
)
parser.add_argument(
    "--test",
    "-t",
    action="store_true",
    help="Test mode (only processes the first PDF file, same as --limit 1)",
)
parser.add_argument(
    "--force",
    "-f",
    action="store_true",
    help="Force reprocessing even if output file already exists",
)

args = parser.parse_args()

input_dir = args.input
output_dir = args.output
limit = 1 if args.test else args.limit

if not os.path.exists(input_dir):
    print(f"Error: Input directory '{input_dir}' does not exist.")
    sys.exit(1)

# Find all PDF files recursively in the input directory
pdf_files = []
for root, _, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith(".pdf"):
            pdf_files.append(os.path.join(root, file))

if not pdf_files:
    print(f"No PDF files found recursively in '{input_dir}'.")
    sys.exit(0)

# Apply limit if specified
if limit is not None:
    print(
        f"Applying limit: Processing only the first {limit} file(s) out of {len(pdf_files)} found."
    )
    pdf_files = pdf_files[:limit]

print(f"Found {len(pdf_files)} PDF file(s) to process:")
for path in pdf_files:
    print(f"  - {path}")
print()

# Setup OpenAI client
client = OpenAI(api_key=api_key, base_url="https://api.vilao.ai/v1")


# Helper function to remove <think>...</think> blocks from text
def prune_think_tags(text):
    if not text:
        return ""
    # Strip <think>...</think> recursively or standard regex
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Also clean up unclosed <think> if the response was cut off
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


# System instructions to choose from randomly (Markdown vs Plain Text)
SYSTEM_PROMPTS = [
    # Option 1: Markdown Prompt
    (
        "You are an expert OCR assistant. Perform precise OCR on the provided pages. "
        "Extract all text, headings, and structure. Format your output strictly in Markdown. "
        "For any mathematical formulas, variables, and equations, convert them accurately to standard LaTeX format "
        "(e.g. using $...$ for inline or $$...$$ for block formulas) so they are clean and readable. "
        "Use styling tags when relevant to represent visual indicators: "
        "use <u>...</u> to mark underlined text, <mark>...</mark> to represent highlighted/marked text, "
        "**...** for bold text, and *...* for italicized text. "
        "Separate each page using the delimiter '<|page|>' followed by the page indicator 'Page X'. "
        "For example:\n"
        "<|page|>Content of page 1<|page|>Content of page 2"
    ),
    # Option 2: Plain Text Prompt
    (
        "You are an expert OCR assistant. Perform precise OCR on the provided pages. "
        "Extract all text, headings, and structure. Format your output strictly as PLAIN TEXT. "
        "Do not use markdown formatting. For any mathematical formulas, variables, and equations, "
        "convert them accurately to standard LaTeX format (e.g. using $...$ for inline or $$...$$ for block formulas). "
        "Use styling tags when relevant to represent visual indicators: "
        "use <u>...</u> to mark underlined text, <mark>...</mark> to represent highlighted/marked text. "
        "Separate each page using the delimiter '<|page|>' followed by the page indicator 'Page X'. "
        "For example:\n"
        "<|page|>Content of page 1<|page|>Content of page 2"
    ),
]

# Process each PDF file
for pdf_path in pdf_files:
    # Calculate output path keeping the subdirectory structure
    rel_path = os.path.relpath(pdf_path, input_dir)
    out_rel_path = os.path.splitext(rel_path)[0] + ".md"
    output_file_path = os.path.join(output_dir, out_rel_path)

    # Skip if output file already exists (unless --force is specified)
    if os.path.exists(output_file_path) and not args.force:
        print(f"Skipping already processed file (output exists): {pdf_path}")
        continue

    # Ensure output subdirectory exists
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    print(f"==================================================")
    print(f"Processing: {pdf_path}")
    print(f"Output to:  {output_file_path}")
    print(f"==================================================")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Failed to open PDF '{pdf_path}': {e}\n")
        continue

    total_pages = len(doc)
    print(f"Total pages: {total_pages}")

    # Initialize the output file (clear any existing contents)
    with open(output_file_path, "w", encoding="utf-8") as f:
        pass

    # Process in batches of 6 pages
    batch_size = 6
    for start_idx in range(0, total_pages, batch_size):
        end_idx = min(start_idx + batch_size, total_pages)
        print(f"  -> Processing pages {start_idx + 1} to {end_idx}...")

        # Randomly choose markdown or plain text prompt
        system_prompt = random.choice(SYSTEM_PROMPTS)

        # Build the message contents
        content_parts = [{"type": "text", "text": system_prompt}]

        # Process each page in the batch
        for page_idx in range(start_idx, end_idx):
            page_num = page_idx + 1
            page = doc[page_idx]

            # Render page to a JPEG image at 150 DPI
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("jpeg")
            base64_image = base64.b64encode(img_bytes).decode("utf-8")

            content_parts.append(
                {"type": "text", "text": f"--- Document Page {page_num} ---"}
            )
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                }
            )

        # Call the API for the batch
        try:
            response = client.chat.completions.create(
                model="mn/Minimax-M3",
                messages=[{"role": "user", "content": content_parts}],
            )

            raw_result = response.choices[0].message.content

            # Prune <think> tags from the output
            batch_result = prune_think_tags(raw_result)

            # Save to output file
            with open(output_file_path, "a", encoding="utf-8") as f:
                f.write(batch_result)
                f.write("\n\n")

        except Exception as e:
            print(f"  Error calling API for pages {start_idx + 1}-{end_idx}: {e}")

    doc.close()
    print(f"Completed: {pdf_path}\n")

print("All PDF files processed successfully!")
