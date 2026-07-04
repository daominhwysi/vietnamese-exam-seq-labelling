import os
import re
import sys
import subprocess
from pathlib import Path

# Add workspace directory to path
script_dir = Path(__file__).resolve().parent
workspace_dir = script_dir.parent
sys.path.append(str(workspace_dir))

from scratch.check_xml_errors import check_alignment

FILES = [
    ("vl/vl-unknown.md", "4960164c"),
    ("vl/vl-hp-2.md", "301c7331"),
    ("vl/vl-hue.md", "3c558d9d"),
    ("vl/vl-hatinh.md", "80635627"),
    ("vl/vl-caobang.md", "cd39e226"),
    ("vl/vl-phutho.md", "c3ddd7fb"),
    ("vl/vl-hp.md", "ec6c09ad")
]

def tag_content(content):
    lines = content.splitlines()
    output_lines = []
    
    in_stem = False
    last_stem_line_idx = -1
    
    # Patterns
    section_patterns = [
        r"^PHẦN I\b.*",
        r"^PHẦN II\b.*",
        r"^PHẦN III\b.*",
        r"^<b>PHẦN I\b.*",
        r"^<b>PHẦN II\b.*",
        r"^<b>PHẦN III\b.*",
        r"^Phần I\b.*",
        r"^Phần II\b.*",
        r"^Phần III\b.*",
        r"^\*\*PHẦN I\b.*",
        r"^\*\*PHẦN II\b.*",
        r"^\*\*PHẦN III\b.*",
        r"^\*\*Phần I\b.*",
        r"^\*\*Phần II\b.*",
        r"^\*\*Phần III\b.*",
        r"^ĐÁP ÁN.*",
        r"^LỜI GIẢI.*",
        r"^## ĐÁP ÁN.*",
        r"^### PHẦN.*",
        r"^## LỜI GIẢI.*",
        r"^<b>LỜI GIẢI.*",
        r"^<b>PHẦN III.*",
        r"^## ĐÁP ÁN THAM KHẢO.*",
        r"^LỜI GIẢI CHI TIẾT.*"
    ]
    
    q_pattern = r"^(\*\*|<b>)?([Cc]âu|Question)\s+(\d+)\s*([.:])(\*\*|</b>)?(.*)"
    
    # Option pattern: uppercase [A-D] followed by dot, or lowercase [a-d] followed by paren
    # Must be preceded by start of string, multiple spaces, tab, pipe, dash, or formatting tags to prevent false positives like units
    opt_pattern = r'(?:^|([\t|:-]| {2,}| (?=\*\*|<b>)))(?:\*\*|<b>)?(?:([A-D])\.|([a-d])\))(?:\*\*|</b>)?(?=\s|$)'
    
    for line_idx, line in enumerate(lines):
        stripped = line.strip()
        
        # 1. Page breaks or footer/header lines - keep as is (must be outside tags)
        if not stripped or stripped.startswith("<|page|>") or stripped.startswith("thuvienhoclieu.com") or "Trang " in line:
            output_lines.append(line)
            continue
            
        # 2. Section headers
        is_section = False
        for pat in section_patterns:
            if re.match(pat, stripped):
                is_section = True
                break
                
        if is_section:
            if in_stem:
                # Close stem on the last stem line
                if last_stem_line_idx != -1:
                    output_lines[last_stem_line_idx] += "</stem>"
                in_stem = False
            
            lead = line[:len(line) - len(line.lstrip())]
            trail = line[len(line.rstrip()):]
            output_lines.append(f"{lead}<section>{stripped}</section>{trail}")
            continue
            
        # 3. Question labels
        q_match = re.match(q_pattern, stripped)
        if q_match:
            if in_stem:
                if last_stem_line_idx != -1:
                    output_lines[last_stem_line_idx] += "</stem>"
                in_stem = False
                
            lead_stars = q_match.group(1) or ""
            q_word = q_match.group(2)
            q_num = q_match.group(3)
            q_punct = q_match.group(4)
            trail_stars = q_match.group(5) or ""
            rest = q_match.group(6)
            
            lead = line[:len(line) - len(line.lstrip())]
            q_label_text = f"{q_word} {q_num}{q_punct}"
            
            tagged_label = f"{lead_stars}<question_label>{q_label_text}</question_label>{trail_stars}"
            
            if rest.strip():
                # Start stem on the same line
                tagged_line = f"{lead}{tagged_label} <stem>{rest.strip()}"
                in_stem = True
                output_lines.append(tagged_line)
                last_stem_line_idx = len(output_lines) - 1
            else:
                tagged_line = f"{lead}{tagged_label}"
                output_lines.append(tagged_line)
                # We will open stem on next line if it has content
                in_stem = True
                last_stem_line_idx = len(output_lines) - 1
            continue
            
        # 4. Option lines
        opt_matches = list(re.finditer(opt_pattern, stripped))
        if opt_matches:
            if in_stem:
                # Close stem before options
                if last_stem_line_idx != -1:
                    output_lines[last_stem_line_idx] += "</stem>"
                in_stem = False
                
            # Tag options on the line
            line_parts = []
            last_idx = 0
            
            for i, match in enumerate(opt_matches):
                pre_char = match.group(1) or ""
                opt_start = match.start() + len(pre_char)
                pre = stripped[last_idx:opt_start]
                
                if i == 0:
                    line_parts.append(pre)
                else:
                    # Previous option text
                    stripped_pre = pre.strip()
                    lead_space = pre[:len(pre) - len(pre.lstrip())]
                    trail_space = pre[len(pre.rstrip()):]
                    line_parts.append(f"{lead_space}<option_text>{stripped_pre}</option_text>{trail_space}")
                    
                # Tag label
                m_str = stripped[opt_start:match.end()]
                # If m_str contains bold stars (**...**), wrap stars inside option_label
                if m_str.startswith("**") and m_str.endswith("**"):
                    tagged_label = f"<option_label>{m_str}</option_label>"
                else:
                    inner_match = re.search(r'(?:[A-D]\.|[a-d]\))', m_str)
                    if inner_match:
                        inner_start, inner_end = inner_match.span()
                        inner_text = inner_match.group(0)
                        tagged_label = m_str[:inner_start] + f"<option_label>{inner_text}</option_label>" + m_str[inner_end:]
                    else:
                        tagged_label = f"<option_label>{m_str}</option_label>"
                line_parts.append(tagged_label)
                last_idx = match.end()
                
            post = stripped[last_idx:]
            stripped_post = post.strip()
            lead_space = post[:len(post) - len(post.lstrip())]
            trail_space = post[len(post.rstrip()):]
            line_parts.append(f"{lead_space}<option_text>{stripped_post}</option_text>{trail_space}")
            
            lead = line[:len(line) - len(line.lstrip())]
            output_lines.append(lead + "".join(line_parts))
            continue
            
        # 5. Regular line
        if in_stem:
            # We are inside stem, so this line is part of the stem
            output_lines.append(line)
            last_stem_line_idx = len(output_lines) - 1
        else:
            output_lines.append(line)
            
    if in_stem:
        if last_stem_line_idx != -1:
            output_lines[last_stem_line_idx] += "</stem>"
            
    return "\n".join(output_lines)

def verify_alignment_correct(xml_path, orig_path):
    with open(xml_path, "r", encoding="utf-8") as f:
        xml_content = f.read()
    with open(orig_path, "r", encoding="utf-8") as f:
        orig_content = f.read()
        
    allowed_tags = {"question_label", "stem", "option_label", "option_text", "context", "section", "explanation"}
    reconstructed = xml_content
    for tag in allowed_tags:
        reconstructed = re.sub(rf"</?{tag}[^>]*>", "", reconstructed)
        
    if reconstructed == orig_content:
        print("  ✅ Strict alignment check: PASSED (100% exact match)")
        return True
        
    print("  ❌ Alignment check: FAILED")
    min_len = min(len(orig_content), len(reconstructed))
    for idx in range(min_len):
        if orig_content[idx] != reconstructed[idx]:
            print(f"  First mismatch at char index {idx}:")
            print(f"    Original : ... {repr(orig_content[max(0, idx-20):idx+20])} ...")
            print(f"    Reconstructed: ... {repr(reconstructed[max(0, idx-20):idx+20])} ...")
            break
    else:
        print(f"  Length mismatch: original len {len(orig_content)}, reconstructed len {len(reconstructed)}")
    return False

def process_file(rel_path, path_hash):
    print(f"\n==========================================")
    print(f"PROCESSING: {rel_path} (hash: {path_hash})")
    print(f"==========================================")
    
    input_file = workspace_dir / "real_data_annotator" / "out" / rel_path
    if not input_file.exists():
        print(f"Error: Input file {input_file} does not exist.")
        return False
        
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()
        
    tagged = tag_content(content)
    
    # Preserve trailing newlines
    num_trailing = len(content) - len(content.rstrip('\n'))
    tagged = tagged.rstrip('\n') + ('\n' * num_trailing)
    
    output_dir = workspace_dir / "output" / "real-exams"
    output_dir.mkdir(parents=True, exist_ok=True)
    xml_path = output_dir / f"real_exam_{path_hash}.xml"
    
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(tagged)
    print(f"Wrote XML to: {xml_path.name}")
    
    # Run parsing script
    parse_cmd = f"rtk pixi run python scratch/parse_xml_to_json.py output/real-exams/real_exam_{path_hash}.xml {rel_path}"
    print(f"Running parser: {parse_cmd}")
    res = subprocess.run(parse_cmd, shell=True, capture_output=True, text=True, cwd=str(workspace_dir))
    print(f"Parser output:\n{res.stdout}")
    if res.stderr:
        print(f"Parser error:\n{res.stderr}")
    if res.returncode != 0:
        print(f"Parser failed with return code {res.returncode}")
        return False
        
    # Check alignment
    passed = verify_alignment_correct(xml_path, input_file)
    print(f"Strict Alignment Check: {'PASSED' if passed else 'FAILED'}")
    return passed

def main():
    success_count = 0
    failed_files = []
    
    for rel_path, path_hash in FILES:
        try:
            success = process_file(rel_path, path_hash)
            if success:
                success_count += 1
            else:
                failed_files.append(rel_path)
        except Exception as e:
            print(f"Error processing {rel_path}: {e}")
            failed_files.append(rel_path)
            
    print(f"\n==========================================")
    print(f"SUMMARY:")
    print(f"  Successfully processed: {success_count}/{len(FILES)} files")
    if failed_files:
        print(f"  Failed files: {failed_files}")
    else:
        print(f"  All files processed successfully!")
    print(f"==========================================")
    
    if failed_files:
        sys.exit(1)

if __name__ == "__main__":
    main()
