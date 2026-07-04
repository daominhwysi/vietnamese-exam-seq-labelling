import os
import sys
import json
import re
import hashlib
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Add workspace directory to path
script_dir = Path(__file__).resolve().parent
workspace_dir = script_dir.parent
sys.path.append(str(workspace_dir))

from src.real_data_annotator.annotate_ocr import SYSTEM_PROMPT, clean_llm_response, parse_xml_annotations
from scratch.check_xml_errors import check_alignment

# Load environment
load_dotenv(dotenv_path=workspace_dir / ".env")

EXAMS = [
    {"path": "ta/ta-hp-2.md", "hash": "90871cd4"},
    {"path": "ta/ta-3.md", "hash": "c542fc69"},
    {"path": "ta/ta-hp-ks-12-1.md", "hash": "ed00cf42"},
    {"path": "ta/ta-5.md", "hash": "f20bed2b"},
    {"path": "ta/tpc-1.md", "hash": "d76f06d4"},
    {"path": "ta/ta-hn-kscl.md", "hash": "2f9640d8"},
    {"path": "ta/tpc-2.md", "hash": "ab169ea3"},
    {"path": "ta/ta-dong-thap.md", "hash": "33aad93c"},
    {"path": "ta/ta-10-hp-26-27.md", "hash": "88c1e2d8"},
    {"path": "ta/ta-an-duong-hp-3.md", "hash": "818c22b4"},
]

def main():
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print("Error: NVIDIA_API_KEY is not set.")
        sys.exit(1)

    client = OpenAI(
        api_key=api_key,
        base_url="https://integrate.api.nvidia.com/v1"
    )

    output_dir = workspace_dir / "output" / "real-exams"
    output_dir.mkdir(parents=True, exist_ok=True)

    force = "--force" in sys.argv

    # Use flash model unless pro is explicitly requested
    model_name = "deepseek-ai/deepseek-v4-pro" if "--pro" in sys.argv else "deepseek-ai/deepseek-v4-flash"
    print(f"Using NVIDIA NIM model: {model_name}\n")

    for idx, exam in enumerate(EXAMS, 1):
        path = exam["path"]
        h = exam["hash"]

        print(f"\n==================================================")
        print(f"[{idx}/{len(EXAMS)}] Processing: {path} (Hash: {h})")
        print(f"==================================================")

        xml_path = output_dir / f"real_exam_{h}.xml"
        json_path = output_dir / f"real_exam_{h}.json"

        if xml_path.exists() and json_path.exists() and not force:
            print(f"Annotated XML and JSON already exist. Skipping.")
            continue

        # 1. Read input markdown file
        input_file = workspace_dir / "real_data_annotator" / "out" / path
        if not input_file.exists():
            print(f"Error: Input file {input_file} does not exist.")
            continue

        with open(input_file, "r", encoding="utf-8") as f:
            raw_text = f.read()

        print(f"Loaded {len(raw_text)} chars from {path}")

        # 2. Call LLM with streaming to annotate
        print(f"Calling streaming NVIDIA NIM ({model_name})...")

        success = False
        max_retries = 2
        for attempt in range(max_retries):
            temp = 0.0 if attempt == 0 else 0.2
            if attempt > 0:
                print(f"Attempt {attempt + 1} of {max_retries}...")

            try:
                # Flash model does not support extra thinking parameter
                extra_args = {}
                if "pro" in model_name:
                    extra_args["extra_body"] = {"chat_template_kwargs": {"thinking": True}}

                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": raw_text},
                    ],
                    temperature=temp,
                    stream=True,
                    **extra_args
                )

                accumulated_content = []
                thinking_chars = 0
                content_chars = 0

                print("Streaming output started...")
                for chunk in response:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta

                    reasoning = getattr(delta, "reasoning_content", None)
                    content = getattr(delta, "content", None)

                    if reasoning:
                        thinking_chars += len(reasoning)
                        if thinking_chars % 500 < len(reasoning):
                            print(f"[Thinking: {thinking_chars} chars]")

                    if content:
                        content_chars += len(content)
                        accumulated_content.append(content)
                        if content_chars % 1000 < len(content):
                            print(f"[Generated content: {content_chars} chars]")

                print(f"Done streaming. Thinking chars: {thinking_chars}, Content chars: {content_chars}")

                raw_result = "".join(accumulated_content)
                tagged_xml = clean_llm_response(raw_result)

                # Check XML structure and alignment
                temp_xml_path = xml_path.with_suffix(".tmp.xml")
                with open(temp_xml_path, "w", encoding="utf-8") as f:
                    f.write(tagged_xml)

                align_ok, _ = check_alignment(temp_xml_path, input_file)

                if align_ok:
                    os.rename(temp_xml_path, xml_path)
                    print(f"Saved XML to: {xml_path.name}")
                    success = True
                    break
                else:
                    print("Alignment check FAILED. The LLM modified the text structure.")
                    if attempt < max_retries - 1:
                        print("Retrying...")
                    else:
                        print("Saving the imperfect XML for manual review/fallback.")
                        os.rename(temp_xml_path, xml_path)
                        success = True
            except Exception as e:
                print(f"Exception during LLM call: {e}")
                if attempt < max_retries - 1:
                    print("Retrying...")
                else:
                    print("Failed after all attempts.")

        if not success:
            continue

        # 3. Run parse_xml_to_json.py
        print("Running parse_xml_to_json.py...")
        # Run using direct python interpreter
        cmd = [
            str(workspace_dir / ".pixi" / "envs" / "default" / "bin" / "python"),
            str(workspace_dir / "scratch" / "parse_xml_to_json.py"),
            str(xml_path),
            path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print(res.stdout.strip())
        else:
            print(f"Error running parser:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")

if __name__ == "__main__":
    main()
