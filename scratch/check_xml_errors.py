#!/usr/bin/env python3
import os
import re
import difflib

ALLOWED_TAGS = {"question_label", "stem", "option_label", "option_text", "context", "section", "explanation"}

# List of single characters or common math/text words we should not treat as tag warnings
EXCLUDE_WORDS = {"p", "div", "span", "br", "hr", "a", "b", "i", "u", "em", "strong", "code", "pre", "h1", "h2", "h3", "h4", "h5", "h6", "x", "y", "z", "t", "n", "k", "m", "g", "d", "c", "s", "o", "f"}

def clean_xml_tags(text):
    # Remove all XML-like tags to get the reconstructed raw text
    return re.sub(r"</?\w+[^>]*>", "", text)

def check_alignment(xml_filepath, txt_filepath):
    print(f"\n[ALIGNMENT CHECK]")
    if not os.path.exists(txt_filepath):
        print(f"  ℹ️ Original text file not found: {os.path.basename(txt_filepath)}")
        return True, 0
        
    with open(xml_filepath, "r", encoding="utf-8") as f:
        xml_content = f.read()
        
    with open(txt_filepath, "r", encoding="utf-8") as f:
        txt_content = f.read()
        
    reconstructed_txt = clean_xml_tags(xml_content)
    
    # Strict check
    if reconstructed_txt == txt_content:
        print("  ✅ Strict alignment check: PASSED (100% exact match)")
        return True, 0
        
    # Check normalized (whitespace collapsed)
    norm_reconstructed = " ".join(reconstructed_txt.split())
    norm_txt = " ".join(txt_content.split())
    
    if norm_reconstructed == norm_txt:
        print("  ⚠️ Whitespace-normalized check: PASSED (Exact match after collapsing whitespace/newlines)")
        return True, 0
        
    # If they are different, calculate differences
    diff_ratio = difflib.SequenceMatcher(None, txt_content, reconstructed_txt).ratio()
    print(f"  ❌ Alignment check: FAILED (Similarity: {diff_ratio:.2%})")
    
    # Print a snippet of the first mismatch
    # Let's find the first character index where they differ
    min_len = min(len(txt_content), len(reconstructed_txt))
    mismatch_idx = -1
    for idx in range(min_len):
        if txt_content[idx] != reconstructed_txt[idx]:
            mismatch_idx = idx
            break
            
    if mismatch_idx != -1:
        line_num = txt_content[:mismatch_idx].count("\n") + 1
        col_num = mismatch_idx - txt_content[:mismatch_idx].rfind("\n")
        print(f"  First mismatch at line {line_num}, col {col_num} (char index {mismatch_idx}):")
        
        start_txt = max(0, mismatch_idx - 40)
        end_txt = min(len(txt_content), mismatch_idx + 40)
        
        start_rec = max(0, mismatch_idx - 40)
        end_rec = min(len(reconstructed_txt), mismatch_idx + 40)
        
        print(f"    Original : ... {repr(txt_content[start_txt:end_txt])} ...")
        print(f"    Annotated: ... {repr(reconstructed_txt[start_rec:end_rec])} ...")
        
    return False, len(txt_content) - len(reconstructed_txt)

def analyze_file(xml_filepath, txt_filepath=None):
    print(f"\n======================================================================")
    print(f"ANALYZING FILE: {os.path.basename(xml_filepath)}")
    print(f"======================================================================")

    with open(xml_filepath, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.splitlines()

    # Find all tags in the text
    tag_regex = re.compile(r"(</?(\w+)[^>]*>)")
    
    tags_found = []
    errors = []
    warnings = []
    
    # 1. First pass: XML tag well-formedness and nesting
    stack = []
    
    for line_idx, line in enumerate(lines, 1):
        for match in tag_regex.finditer(line):
            full_match = match.group(1)
            tag_name = match.group(2)
            is_closing = full_match.startswith("</")
            
            # Check if this is a known/allowed tag
            if tag_name not in ALLOWED_TAGS:
                if tag_name.lower() in EXCLUDE_WORDS or len(tag_name) <= 2:
                    continue
                warnings.append(f"Line {line_idx}: Unknown/suspicious tag name: {full_match}")
                continue
            
            # Process allowed tag
            tags_found.append({
                "name": tag_name,
                "is_closing": is_closing,
                "line": line_idx,
                "full": full_match,
                "span": match.span(),
                "line_text": line
            })
            
            if is_closing:
                if not stack:
                    errors.append(f"Line {line_idx}: Closing tag {full_match} has no matching opening tag.")
                else:
                    open_tag, open_line = stack.pop()
                    if open_tag != tag_name:
                        errors.append(f"Line {line_idx}: Mismatched closing tag {full_match}. Expected </{open_tag}> (opened on line {open_line}).")
            else:
                stack.append((tag_name, line_idx))
                
    while stack:
        open_tag, open_line = stack.pop()
        errors.append(f"Line {open_line}: Unclosed opening tag <{open_tag}>.")

    # 2. Check for empty tags
    for tag_name in ALLOWED_TAGS:
        empty_pattern = re.compile(rf"<{tag_name}>\s*</{tag_name}>")
        for line_idx, line in enumerate(lines, 1):
            for match in empty_pattern.finditer(line):
                warnings.append(f"Line {line_idx}: Empty tag pair <{tag_name}>...</{tag_name}>")

    # 3. Check for specific content anomalies (like * inside option_label)
    label_star_pattern = re.compile(r"<option_label>\s*\*\s*</option_label>")
    for line_idx, line in enumerate(lines, 1):
        for match in label_star_pattern.finditer(line):
            errors.append(f"Line {line_idx}: Bullet point '*' tagged as <option_label>")

    # 4. Check tag sequences
    for i in range(len(tags_found)):
        curr = tags_found[i]
        if curr["is_closing"]:
            continue
            
        next_open = None
        for j in range(i + 1, len(tags_found)):
            if not tags_found[j]["is_closing"]:
                next_open = tags_found[j]
                break
                
        if curr["name"] == "option_label":
            if not next_open:
                warnings.append(f"Line {curr['line']}: <option_label> at end of file (no following tag).")
            elif next_open["name"] != "option_text":
                warnings.append(f"Line {curr['line']}: <option_label> is followed by <{next_open['name']}> (line {next_open['line']}) instead of <option_text>.")
                
        elif curr["name"] == "option_text":
            prev_open = None
            for j in range(i - 1, -1, -1):
                if not tags_found[j]["is_closing"]:
                    prev_open = tags_found[j]
                    break
            if not prev_open or prev_open["name"] != "option_label":
                warnings.append(f"Line {curr['line']}: <option_text> is not preceded by <option_label> (previous is <{prev_open['name'] if prev_open else 'None'}>).")

    # 5. Check for untagged elements in the file
    clean_text = content
    def tag_replacer(match):
        return " " * len(match.group(0))
    
    for tag_name in ALLOWED_TAGS:
        pattern = re.compile(rf"<{tag_name}>.*?</{tag_name}>", re.DOTALL)
        clean_text = pattern.sub(tag_replacer, clean_text)
        
    for tag_name in ALLOWED_TAGS:
        clean_text = re.sub(rf"</?{tag_name}>", lambda m: " " * len(m.group(0)), clean_text)

    clean_lines = clean_text.splitlines()
    
    opt_marker_re = re.compile(r"\b([A-D]\.|[a-d]\))\b")
    for line_idx, clean_line in enumerate(clean_lines, 1):
        original_line = lines[line_idx - 1]
        if original_line.strip().startswith("|") or original_line.strip().startswith("---"):
            continue
            
        for match in opt_marker_re.finditer(clean_line):
            marker = match.group(1)
            char_idx = match.start()
            dollar_count_before = original_line[:char_idx].count("$")
            if dollar_count_before % 2 == 1:
                continue
                
            warnings.append(f"Line {line_idx}: Possible untagged option label '{marker}' found in text: '{original_line.strip()}'")

    quest_marker_re = re.compile(r"\b(Câu\s+\d+|Question\s+\d+)\b", re.IGNORECASE)
    for line_idx, clean_line in enumerate(clean_lines, 1):
        original_line = lines[line_idx - 1]
        for match in quest_marker_re.finditer(clean_line):
            marker = match.group(1)
            warnings.append(f"Line {line_idx}: Possible untagged question label '{marker}' found in text: '{original_line.strip()}'")

    # Count tag occurrences
    tag_counts = {tag: 0 for tag in ALLOWED_TAGS}
    for tag in tags_found:
        if not tag["is_closing"]:
            tag_counts[tag["name"]] += 1

    # Report results
    print(f"\n[TAG COUNTS]")
    for tag, count in tag_counts.items():
        print(f"  - {tag}: {count}")

    # Run alignment check if original text file path is provided
    align_passed = True
    if txt_filepath:
        align_passed, len_diff = check_alignment(xml_filepath, txt_filepath)
        if not align_passed:
            errors.append(f"Alignment check failed: Reconstructed text differs from original raw input text (length difference: {len_diff} chars).")

    print(f"\n[ERRORS FOUND: {len(errors)}]")
    for err in errors:
        print(f"  ❌ {err}")

    print(f"\n[WARNINGS/ANOMALIES FOUND: {len(warnings)}]")
    for warn in warnings:
        print(f"  ⚠️ {warn}")

    return len(errors), len(warnings), tag_counts

# Helper to pass parameter safely
def check_alignment_wrapper(xml_filepath, txt_filepath):
    return check_alignment(xml_filepath, txt_filepath)

def main():
    output_dir = "/home/minh1/project/vietnamese-exam-seq-labelling/output_folder"
    files = [
        ("custom_english_annotated.xml", "custom_english.txt"),
        ("english_annotated.xml", "english.txt"),
        ("random_math_annotated.xml", "random_math.txt")
    ]
    
    total_errors = 0
    total_warnings = 0
    
    for xml_file, txt_file in files:
        xml_path = os.path.join(output_dir, xml_file)
        txt_path = os.path.join(output_dir, txt_file)
        
        if os.path.exists(xml_path):
            err, warn, counts = analyze_file(xml_path, txt_path if os.path.exists(txt_path) else None)
            total_errors += err
            total_warnings += warn
        else:
            print(f"File not found: {xml_path}")
            
    print("\n======================================================================")
    print(f"TOTAL SUMMARY:")
    print(f"  Total parsing/structure/alignment errors: {total_errors}")
    print(f"  Total warnings/labeling anomalies: {total_warnings}")
    print("======================================================================")

if __name__ == "__main__":
    main()
