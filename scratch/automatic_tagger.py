import re
import sys
from pathlib import Path

# Add workspace directory to path
script_dir = Path(__file__).resolve().parent
workspace_dir = script_dir.parent
sys.path.append(str(workspace_dir))

from scratch.check_xml_errors import check_alignment

def tag_content(content):
    lines = content.splitlines()
    output_lines = []
    
    # Simple state machine to help determine if we are inside a stem
    in_stem = False
    
    # We will process line by line
    for line_idx, line in enumerate(lines):
        stripped = line.strip()
        
        # 1. Page breaks or empty lines - keep as is (must be outside tags)
        if not stripped or stripped.startswith("<|page|>") or stripped.startswith("thuvienhoclieu.com") or "Trang " in line:
            if in_stem:
                # Page break can interrupt stem, so we close it and reopen it?
                # Actually, let's just close stem before it, and let the state machine handle the rest.
                output_lines.append("</stem>")
                in_stem = False
            output_lines.append(line)
            continue
            
        # 2. Section headers
        section_patterns = [
            r"^PHẦN I\b.*",
            r"^PHẦN II\b.*",
            r"^PHẦN III\b.*",
            r"^\*\*PHẦN I\b.*",
            r"^\*\*PHẦN II\b.*",
            r"^\*\*PHẦN III\b.*",
            r"^ĐÁP ÁN.*",
            r"^LỜI GIẢI.*",
            r"^## ĐÁP ÁN.*",
            r"^### PHẦN.*",
            r"^## LỜI GIẢI.*"
        ]
        
        is_section = False
        for pat in section_patterns:
            if re.match(pat, stripped):
                is_section = True
                break
                
        if is_section:
            if in_stem:
                output_lines.append("</stem>")
                in_stem = False
            # Wrap entire line in section
            # Preserve leading/trailing spaces
            lead = line[:len(line) - len(line.lstrip())]
            trail = line[len(line.rstrip()):]
            output_lines.append(f"{lead}<section>{stripped}</section>{trail}")
            continue
            
        # 3. Question labels
        # Matches: Câu 1., Câu 1:, **Câu 1.**, **Câu 1:**, Question 1., **Question 1.**
        q_pattern = r"^(\*\*|)(Câu \d+|Question \d+)([:.]|)(\*\*|)(.*)"
        q_match = re.match(q_pattern, stripped)
        if q_match:
            if in_stem:
                output_lines[-1] = output_lines[-1] + "</stem>"
                in_stem = False
                
            lead_stars = q_match.group(1)
            q_label = q_match.group(2) + q_match.group(3)
            trail_stars = q_match.group(4)
            rest = q_match.group(5)
            
            lead = line[:len(line) - len(line.lstrip())]
            
            # Wrap question label
            if lead_stars:
                tagged_label = f"**<question_label>{q_label}</question_label>**"
            else:
                tagged_label = f"<question_label>{q_label}</question_label>"
                
            if rest.strip():
                # There is text after the label on the same line, start stem
                tagged_line = f"{lead}{tagged_label} <stem>{rest.strip()}"
                in_stem = True
            else:
                # No text after label on this line
                tagged_line = f"{lead}{tagged_label}"
                
            output_lines.append(tagged_line)
            continue
            
        # 4. Option lines (A, B, C, D or a, b, c, d)
        # Check if line contains option labels.
        # Format 1: - **A.** text
        # Format 2: - **a)** text
        # Format 3: A. text B. text
        # Format 4: **A.** text
        
        # Let's check for options on the line
        # Regex to find option markers: e.g. - **A.**, **A.**, A., - **a)**, a), etc.
        # We need to tag all of them.
        opt_pattern = r"(- \*\*|\*\*|)(- |)([A-D]|[a-d])(\.|\))(\*\*|)"
        
        # Inline options check
        # We find all matches on the line.
        # We want to replace each option marker with <option_label> and wrap the following text up to the next option marker or end of line in <option_text>.
        # Let's tokenize the line using the regex
        matches = list(re.finditer(opt_pattern, stripped))
        if matches:
            if in_stem:
                # Stem ends before the options
                # If there was a previous line in the stem, we close it
                # If the stem was opened on a previous line, we append </stem> to the last line, or insert it.
                # Actually, let's close stem on the previous line or before this line
                # Let's find the last line that had content and append </stem>
                # Let's just prepend </stem> to this line's processing
                # We need to make sure we don't duplicate </stem>
                pass
                
            # Parse options on this line
            line_parts = []
            last_idx = 0
            
            # Helper to close stem if it was open
            stem_closed = False
            
            for i, match in enumerate(matches):
                start, end = match.span()
                # Text before this match (could be spaces or the option text of the previous option)
                pre_text = stripped[last_idx:start]
                
                if i == 0:
                    # Before the first option, if we were in stem, close it
                    if in_stem:
                        line_parts.append(pre_text.rstrip() + "</stem>" + pre_text[len(pre_text.rstrip()):])
                        in_stem = False
                        stem_closed = True
                    else:
                        line_parts.append(pre_text)
                else:
                    # This is option text for the previous option
                    # We wrap it in <option_text>
                    # Be careful to preserve leading/trailing whitespace of the option text
                    stripped_pre = pre_text.strip()
                    lead_space = pre_text[:len(pre_text) - len(pre_text.lstrip())]
                    trail_space = pre_text[len(pre_text.rstrip()):]
                    line_parts.append(f"{lead_space}<option_text>{stripped_pre}</option_text>{trail_space}")
                    
                # Tag the option label
                opt_prefix = match.group(1) # e.g. "- **" or "**"
                bullet = match.group(2) # e.g. "- "
                letter = match.group(3) # e.g. "A"
                punct = match.group(4) # e.g. "." or ")"
                opt_suffix = match.group(5) # e.g. "**"
                
                # Reconstruct with tags
                # Stars belong inside <option_label>
                label_text = f"{letter}{punct}"
                if opt_prefix:
                    if "**" in opt_prefix:
                        tagged_label = f"{bullet}<option_label>**{label_text}**</option_label>"
                    else:
                        tagged_label = f"{bullet}<option_label>{label_text}</option_label>"
                else:
                    tagged_label = f"{bullet}<option_label>{label_text}</option_label>"
                    
                line_parts.append(tagged_label)
                last_idx = end
                
            # Remaining text after the last match is the option_text of the last option
            post_text = stripped[last_idx:]
            stripped_post = post_text.strip()
            lead_space = post_text[:len(post_text) - len(post_text.lstrip())]
            trail_space = post_text[len(post_text.rstrip()):]
            line_parts.append(f"{lead_space}<option_text>{stripped_post}</option_text>{trail_space}")
            
            lead = line[:len(line) - len(line.lstrip())]
            output_lines.append(lead + "".join(line_parts))
            continue
            
        # 5. Regular line: if in_stem, it's part of the stem. Otherwise keep as is.
        if in_stem:
            # We are in stem, so this line is part of stem.
            # But wait! If this line starts with a new question or section (handled above), it would have closed in_stem.
            # Otherwise, it just continues.
            output_lines.append(line)
        else:
            # If not in stem, and not section or question, keep as is
            output_lines.append(line)
            
    # If still in stem at end of file, close it
    if in_stem:
        output_lines.append("</stem>")
        
    return "\n".join(output_lines)

def main():
    if len(sys.argv) < 3:
        print("Usage: python automatic_tagger.py <input_file> <output_file>")
        sys.exit(1)
        
    infile = Path(sys.argv[1])
    outfile = Path(sys.argv[2])
    
    with open(infile, "r", encoding="utf-8") as f:
        content = f.read()
        
    tagged = tag_content(content)
    
    # Write output
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(tagged)
        
    print(f"Tagged XML written to: {outfile}")
    
    # Check alignment
    passed, diff = check_alignment(str(outfile), str(infile))
    print(f"Alignment check: {'PASSED' if passed else 'FAILED'} (diff: {diff} chars)")

if __name__ == "__main__":
    main()
