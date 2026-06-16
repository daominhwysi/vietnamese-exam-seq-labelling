import sys
import os
import re
import difflib
from pathlib import Path
from typing import Tuple, List, Dict, Any

def parse_xml_annotations(tagged_text: str) -> Tuple[str, List[Dict[str, Any]]]:
    allowed_tags = {"question_label", "stem", "option_label", "option_text", "context", "section"}
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

def check_accuracy():
    # File paths
    original_path = Path("real_data_annotator/out/ta/ta-10-hp-26-27.md")
    xml_path = Path("output/real_exams/real_exam_bc27a60a.xml")
    json_path = Path("output/real_exams/real_exam_bc27a60a.json")

    print(f"Original file exists: {original_path.exists()}")
    print(f"XML file exists: {xml_path.exists()}")
    print(f"JSON file exists: {json_path.exists()}")

    # Read original text
    with open(original_path, "r", encoding="utf-8") as f:
        original_text = f.read()

    # Read XML text
    with open(xml_path, "r", encoding="utf-8") as f:
        xml_text = f.read()

    # Parse XML annotations
    stripped_text, spans = parse_xml_annotations(xml_text)

    # 1. Check exact match
    exact_match = (stripped_text == original_text)
    print(f"\nExact Match: {exact_match}")
    print(f"Original length: {len(original_text)} chars")
    print(f"Stripped length: {len(stripped_text)} chars")

    if not exact_match:
        # Find differences
        diff = list(difflib.unified_diff(
            original_text.splitlines(),
            stripped_text.splitlines(),
            fromfile="original",
            tofile="stripped",
            lineterm=""
        ))
        print(f"\nDifferences found ({len(diff)} lines):")
        for line in diff[:30]:
            print(line)
        if len(diff) > 30:
            print("...")
            
        # Character-level mismatch details
        min_len = min(len(original_text), len(stripped_text))
        mismatches = 0
        first_mismatch_idx = -1
        for i in range(min_len):
            if original_text[i] != stripped_text[i]:
                mismatches += 1
                if first_mismatch_idx == -1:
                    first_mismatch_idx = i
        
        char_accuracy = (min_len - mismatches) / max(len(original_text), len(stripped_text)) * 100
        print(f"\nCharacter-level match rate: {char_accuracy:.6f}% ({mismatches} mismatching characters)")
        if first_mismatch_idx != -1:
            print(f"First mismatch at char index {first_mismatch_idx}:")
            print(f"  Original context: {repr(original_text[max(0, first_mismatch_idx-30):first_mismatch_idx+30])}")
            print(f"  Stripped context: {repr(stripped_text[max(0, first_mismatch_idx-30):first_mismatch_idx+30])}")
    else:
        print("\nCharacter-level match rate: 100.0%")

    # 2. Check XML Tag Validity and Spans Alignment
    print(f"\nTotal spans annotated: {len(spans)}")
    
    # Check for invalid tags or unclosed tags
    unclosed_tags = re.findall(r"<[a-zA-Z_0-9]+>|<\/[a-zA-Z_0-9]+>", stripped_text)
    if unclosed_tags:
        print(f"WARNING: Unclosed/unallowed tags found in stripped text: {unclosed_tags}")
    else:
        print("XML Tagging Integrity: No tag remnants left in stripped text.")

    # Validate that every span's text matches the corresponding raw_text slice
    slice_mismatches = 0
    for s in spans:
        slice_text = stripped_text[s["start"]:s["end"]]
        if slice_text != s["text"]:
            slice_mismatches += 1
            print(f"  Span slice mismatch for {s['label']}: expected {repr(s['text'])} but sliced {repr(slice_text)}")
    if slice_mismatches == 0:
        print("Span Alignment: 100.0% accuracy (all spans exactly match their raw_text slices)")
    else:
        print(f"WARNING: Found {slice_mismatches} span slice mismatches!")

    # Count tags of each type
    tag_counts = {}
    for s in spans:
        tag_counts[s["label"]] = tag_counts.get(s["label"], 0) + 1
    
    print("\nTag distribution:")
    for tag_name, count in sorted(tag_counts.items()):
        print(f"  - {tag_name}: {count}")

    # 3. Question Label Extraction & Completeness
    q_labels = [s for s in spans if s["label"] == "question_label"]
    print(f"Number of question labels: {len(q_labels)}")
    
    # Extract the question numbers found
    q_numbers = []
    for q in q_labels:
        # Match digits in Question X
        match = re.search(r"\b(?:Question|Câu)\s*(\d+)\b", q["text"], re.IGNORECASE)
        if match:
            q_numbers.append(int(match.group(1)))
        else:
            q_numbers.append(q["text"])

    print(f"Annotated question numbers: {q_numbers}")

    # Check if there are any missing questions between 1 and 40
    all_expected = set(range(1, 41))
    annotated_set = {n for n in q_numbers if isinstance(n, int)}
    missing = all_expected - annotated_set
    if missing:
        print(f"WARNING: Missing question numbers in annotations: {sorted(list(missing))}")
    else:
        print("All expected questions (1-40) are annotated!")

if __name__ == "__main__":
    check_accuracy()
